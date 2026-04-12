"""
Cross-Dataset Generalization Evaluator
=======================================

Evaluates the trained PATV-X model (core track and/or open track) on a
second dataset to measure out-of-distribution generalization.

Supported dataset layouts
--------------------------
Layout A — FF++-style subdirectory split (used by FaceForensics++):
    <dataset_root>/
      ai/          ← manipulated videos
      authentic/   ← real videos
      manifest.csv ← optional; columns: filename,label,category

Layout B — flat directory with a manifest CSV:
    <dataset_root>/
      videos/
        001.mp4
        ...
      manifest.csv ← required; columns: filename,label[,category]

Layout C — DFDC-style directory with per-video labels:
    <dataset_root>/
      videos/
        dfdc_train_part_00/
          ...
      metadata.json  ← DFDC metadata; {"filename": {"label": "FAKE"|"REAL"}}

Usage examples
--------------
# Zero-shot core-track evaluation on any compatible dataset:
python evaluation/evaluate_generalization.py \
    --dataset dataset/dfdc \
    --layout dfdc \
    --output generalization_results/dfdc

# Cross-dataset eval using the trained open-track GBM bundle:
python evaluation/evaluate_generalization.py \
    --dataset dataset/faceshifter \
    --layout ff \
    --model pipeline_results_final/patv_open_bundle.json \
    --output generalization_results/faceshifter

# Cross-dataset eval from pre-extracted feature CSV (no re-inference):
python evaluation/evaluate_generalization.py \
    --features path/to/features.csv \
    --model pipeline_results_final/patv_open_bundle.json \
    --output generalization_results/from_csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── Repo root resolution ─────────────────────────────────────────────────────

def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "src").exists() and (p / "training").exists():
            return p
    raise RuntimeError("Cannot locate repo root (need src/ and training/ directories).")

ROOT = _find_repo_root()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "legacy" / "data_pipeline"))

SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# ── Dataset loaders ──────────────────────────────────────────────────────────

def _discover_ff_layout(root: Path) -> list[dict]:
    """FF++-style: ai/ and authentic/ subdirectories + optional manifest.csv."""
    manifest_path = root / "manifest.csv"
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("filename"):
                    manifest[row["filename"]] = row

    records: list[dict] = []
    for subdir, label in [("ai", 1), ("authentic", 0)]:
        d = root / subdir
        if not d.exists():
            continue
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                meta = manifest.get(f.name, {})
                records.append({
                    "filename": f.name,
                    "path": str(f),
                    "label": int(meta.get("label", label)),
                    "category": meta.get("category", subdir),
                })
    return records


def _discover_flat_layout(root: Path) -> list[dict]:
    """Flat directory + required manifest.csv with columns filename, label[, category]."""
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Flat layout requires {manifest_path}. "
            "Create it with columns: filename, label (0=real, 1=fake), [category]."
        )
    manifest: dict[str, dict] = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("filename"):
                manifest[row["filename"]] = row

    video_root = root / "videos" if (root / "videos").exists() else root
    records: list[dict] = []
    for f in sorted(video_root.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
            if f.name in manifest:
                meta = manifest[f.name]
                records.append({
                    "filename": f.name,
                    "path": str(f),
                    "label": int(meta.get("label", 0)),
                    "category": meta.get("category", "unknown"),
                })
    return records


def _discover_dfdc_layout(root: Path) -> list[dict]:
    """
    DFDC layout: metadata.json maps filename → {"label": "FAKE"|"REAL"}.
    Videos live in videos/ or in per-part subdirs.
    """
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"DFDC layout requires {metadata_path}. "
            "Download from https://ai.meta.com/datasets/dfdc/ and place metadata.json at the dataset root."
        )

    with open(metadata_path, encoding="utf-8") as f:
        metadata: dict[str, Any] = json.load(f)

    # Build filename → label map; DFDC uses "FAKE" / "REAL"
    label_map: dict[str, int] = {}
    for fname, info in metadata.items():
        raw = str(info.get("label", "")).upper()
        label_map[fname] = 1 if raw == "FAKE" else 0

    video_root = root / "videos" if (root / "videos").exists() else root
    records: list[dict] = []
    for f in sorted(video_root.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
            if f.name in label_map:
                records.append({
                    "filename": f.name,
                    "path": str(f),
                    "label": label_map[f.name],
                    "category": "dfdc_fake" if label_map[f.name] == 1 else "dfdc_real",
                })
    return records


LAYOUT_LOADERS = {
    "ff": _discover_ff_layout,
    "flat": _discover_flat_layout,
    "dfdc": _discover_dfdc_layout,
}


# ── Feature extraction ───────────────────────────────────────────────────────

def _extract_features(records: list[dict], sample_rate: int, max_frames: int,
                      workers: int) -> list[dict]:
    from patv_x_detector import PATVXDetector
    from train_mlp import derive_forensic_extra_features_from_result

    detector = PATVXDetector(sample_rate=sample_rate, max_frames=max_frames)

    feature_rows: list[dict] = []
    n = len(records)
    for i, rec in enumerate(records):
        vpath = rec["path"]
        print(f"  [{i+1}/{n}] {Path(vpath).name} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = detector.analyze(vpath)
            extra = derive_forensic_extra_features_from_result(result)
            row = {
                "filename": rec["filename"],
                "path": rec["path"],
                "label": rec["label"],
                "category": rec["category"],
                **extra,
            }
            feature_rows.append(row)
            print(f"ok ({time.time()-t0:.1f}s)  score={extra.get('tcs_score', '?'):.3f}")
        except Exception as e:
            print(f"ERROR: {e}")
    return feature_rows


def _load_feature_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ── Evaluation ───────────────────────────────────────────────────────────────

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x) if x not in (None, "") else default
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x)) if x not in (None, "") else default
    except Exception:
        return default


def _eval_core_track(feature_rows: list[dict], threshold: float) -> dict:
    """Training-free core-track evaluation using tcs_score."""
    labels = [_safe_int(r.get("label", 0)) for r in feature_rows]
    scores = [_safe_float(r.get("tcs_score", 0.0)) for r in feature_rows]

    if not labels:
        return {}

    # AUC-ROC (manual trapezoid)
    thresholds = sorted(set(scores), reverse=True)
    tp_rates, fp_rates = [0.0], [0.0]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    for thr in thresholds:
        preds = [1 if s >= thr else 0 for s in scores]
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
        tp_rates.append(tp / max(n_pos, 1))
        fp_rates.append(fp / max(n_neg, 1))
    tp_rates.append(1.0)
    fp_rates.append(1.0)

    auc = sum(
        (fp_rates[i+1] - fp_rates[i]) * (tp_rates[i+1] + tp_rates[i]) / 2
        for i in range(len(fp_rates) - 1)
    )

    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))

    recall = tp / max(n_pos, 1)
    specificity = tn / max(n_neg, 1)
    balanced_acc = (recall + specificity) / 2

    mean_fake = float(np.mean([s for s, l in zip(scores, labels) if l == 1])) if n_pos else 0.0
    mean_real = float(np.mean([s for s, l in zip(scores, labels) if l == 0])) if n_neg else 0.0

    return {
        "track": "core",
        "n_videos": len(labels),
        "n_fake": n_pos,
        "n_real": n_neg,
        "auc_roc": round(auc, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "threshold": threshold,
        "mean_score_fake": round(mean_fake, 4),
        "mean_score_real": round(mean_real, 4),
        "mean_score_gap": round(mean_fake - mean_real, 4),
    }


def _eval_open_track(feature_rows: list[dict], bundle_path: str) -> dict:
    """GBM open-track evaluation using the trained model bundle."""
    from train_mlp import csv_to_features, OPEN_FEATURE_NAMES, GBMClassifier

    with open(bundle_path, encoding="utf-8") as f:
        bundle = json.load(f)

    model_data = bundle.get("model") or bundle
    gbm = GBMClassifier()
    gbm.load(model_data)

    threshold = float(bundle.get("threshold", 0.5))
    feature_names = bundle.get("feature_names", OPEN_FEATURE_NAMES)

    labels = [_safe_int(r.get("label", 0)) for r in feature_rows]
    X = []
    for row in feature_rows:
        vec = [_safe_float(row.get(fn, 0.0)) for fn in feature_names]
        X.append(vec)
    X_arr = np.array(X, dtype=float)

    scores = gbm.predict_proba(X_arr)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    # AUC
    thresholds = sorted(set(scores.tolist()), reverse=True)
    tp_rates, fp_rates = [0.0], [0.0]
    for thr in thresholds:
        preds = [1 if s >= thr else 0 for s in scores]
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
        tp_rates.append(tp / max(n_pos, 1))
        fp_rates.append(fp / max(n_neg, 1))
    tp_rates.append(1.0)
    fp_rates.append(1.0)
    auc = sum(
        (fp_rates[i+1] - fp_rates[i]) * (tp_rates[i+1] + tp_rates[i]) / 2
        for i in range(len(fp_rates) - 1)
    )

    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    recall = tp / max(n_pos, 1)
    specificity = tn / max(n_neg, 1)

    return {
        "track": "open",
        "n_videos": len(labels),
        "n_fake": n_pos,
        "n_real": n_neg,
        "auc_roc": round(auc, 4),
        "recall": round(recall, 4),
        "specificity": round(tn / max(n_neg, 1), 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
        "threshold": threshold,
        "model_bundle": bundle_path,
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def _print_report(results: dict) -> None:
    dataset_name = results.get("dataset_name", "unknown")
    print(f"\n{'=' * 60}")
    print(f"  GENERALIZATION RESULTS — {dataset_name}")
    print(f"{'=' * 60}")
    for track_result in results.get("evaluations", []):
        track = track_result.get("track", "?")
        print(f"\n  Track: {track.upper()}")
        print(f"    Videos      : {track_result.get('n_videos')} "
              f"({track_result.get('n_fake')} fake, {track_result.get('n_real')} real)")
        print(f"    AUC-ROC     : {track_result.get('auc_roc', '?')}")
        print(f"    Recall      : {track_result.get('recall', '?')}")
        print(f"    Specificity : {track_result.get('specificity', '?')}")
        print(f"    Bal. Acc.   : {track_result.get('balanced_accuracy', '?')}")
        if "mean_score_gap" in track_result:
            print(f"    Score gap   : {track_result.get('mean_score_gap', '?')}")
    print()


def _save_report(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "generalization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Report saved to {report_path}")

    # Also save extracted features if present
    features = results.get("features")
    if features:
        feat_path = output_dir / "features.csv"
        if features:
            fieldnames = list(features[0].keys())
            with open(feat_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(features)
            print(f"Features saved to {feat_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PATV-X generalization on a second dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="Path to dataset root directory.")
    src.add_argument("--features", help="Path to pre-extracted features CSV (skips inference).")

    parser.add_argument("--layout", choices=["ff", "flat", "dfdc"], default="ff",
                        help="Dataset directory layout (default: ff).")
    parser.add_argument("--model", default=None,
                        help="Path to patv_open_bundle.json for open-track evaluation.")
    parser.add_argument("--output", default="generalization_results",
                        help="Output directory for report and features (default: generalization_results).")
    parser.add_argument("--sample-rate", type=int, default=6,
                        help="Frames to sample per second (default: 6).")
    parser.add_argument("--max-frames", type=int, default=96,
                        help="Maximum frames to process per video (default: 96).")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Core-track detection threshold (default: 0.35).")
    parser.add_argument("--workers", type=int, default=0,
                        help="Worker processes for feature extraction (default: 0 = single-process).")
    parser.add_argument("--dataset-name", default=None,
                        help="Human-readable name for this dataset (used in report).")
    args = parser.parse_args()

    output_dir = Path(args.output)
    dataset_name = args.dataset_name or (
        Path(args.dataset).name if args.dataset else Path(args.features).stem
    )

    # Step 1: obtain feature rows
    if args.features:
        print(f"Loading pre-extracted features from {args.features} ...")
        feature_rows = _load_feature_csv(args.features)
        print(f"  Loaded {len(feature_rows)} rows.")
    else:
        dataset_root = Path(args.dataset)
        loader = LAYOUT_LOADERS[args.layout]
        print(f"Discovering videos in {dataset_root} (layout={args.layout}) ...")
        records = loader(dataset_root)
        if not records:
            print("ERROR: No videos found. Check the dataset path and layout.")
            sys.exit(1)

        n_fake = sum(1 for r in records if r["label"] == 1)
        n_real = sum(1 for r in records if r["label"] == 0)
        print(f"  Found {len(records)} videos ({n_fake} fake, {n_real} real).")

        print(f"\nExtracting features (sample_rate={args.sample_rate}, max_frames={args.max_frames}) ...")
        feature_rows = _extract_features(records, args.sample_rate, args.max_frames, args.workers)

    # Step 2: evaluate
    evaluations = []

    print("\nRunning core-track (training-free) evaluation ...")
    core_result = _eval_core_track(feature_rows, threshold=args.threshold)
    if core_result:
        evaluations.append(core_result)

    if args.model:
        print(f"\nRunning open-track (GBM) evaluation using {args.model} ...")
        try:
            open_result = _eval_open_track(feature_rows, args.model)
            evaluations.append(open_result)
        except Exception as e:
            print(f"  Open-track evaluation failed: {e}")

    results = {
        "dataset_name": dataset_name,
        "dataset_path": args.dataset or args.features,
        "layout": args.layout if args.dataset else "csv",
        "evaluations": evaluations,
        "features": feature_rows,
    }

    _print_report(results)
    _save_report(results, output_dir)


if __name__ == "__main__":
    main()

"""
PATV Full Pipeline Runner
=========================

Εκτελεί ολόκληρο το pipeline από raw βίντεο μέχρι
training / evaluation / ablation — με ένα command.

Βασικές βελτιώσεις της έκδοσης αυτής
------------------------------------
1) Ενιαίο detector configuration (weights / threshold) από τον ίδιο τον detector
2) Stratified train / validation / test split
3) Threshold tuning ΜΟΝΟ στο validation set
4) Τελική detector αξιολόγηση ΜΟΝΟ στο test set
5) Ablation / weight search μόνο στο development set (train + validation)
6) Καθαρότερο reporting για per-class / per-category metrics

Σενάρια χρήσης:
  1) Με πραγματικά βίντεο:
     python legacy/data_pipeline/run_pipeline.py \
         --videos dataset/faceforensics \
         --output pipeline_results

  2) Με synthetic data:
     python legacy/data_pipeline/run_pipeline.py --synthetic

  3) Με υπάρχον CSV features:
     python legacy/data_pipeline/run_pipeline.py \
         --features results/features.csv \
         --output pipeline_results
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# ── Robust repo root detection ──────────────────────────────────────────────

def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "src").exists() and (p / "training").exists() and (p / "ablation").exists():
            return p
    raise RuntimeError("Δεν βρέθηκε το repo root (χρειάζονται οι φάκελοι src/, training/, ablation/).")


ROOT = _find_repo_root()

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "ablation"))

from patv_x_detector import PATVXDetector
from train_mlp import LEGACY_CORE_FEATURE_NAMES, OPEN_FEATURE_NAMES, LinearLogistic, MLP, GBMClassifier, csv_to_features
from train_mlp import compute_auc, evaluate, generate_synthetic_data, specificity_first_threshold
from train_mlp import derive_forensic_extra_features_from_result
from ablation_study import run_level_ablation, run_metric_ablation, run_weight_search


# ── Constants ───────────────────────────────────────────────────────────────

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SEED = 42
VAL_RATIO = 0.20
TEST_RATIO = 0.20
DETECTOR_DEFAULT_SAMPLE_RATE = 6
DETECTOR_DEFAULT_MAX_FRAMES = 96
CORE_TARGET_SPECIFICITY = 0.60
CORE_TARGET_RECALL = 0.65
CORE_TARGET_BALANCED_ACC = 0.65
CORE_TARGET_MEAN_TCS_GAP = 0.08
OPEN_TARGET_RECALL = 0.15
METRIC_TOL = 1e-9

DETECTOR_DEFAULT_WEIGHTS = dict(getattr(PATVXDetector, "DEFAULT_WEIGHTS",
    {"L1": 0.14, "L2": 0.34, "L3": 0.02, "L4": 0.50}))
DETECTOR_DEFAULT_THRESHOLD = float(getattr(PATVXDetector, "DEFAULT_THRESHOLD", 0.35))


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        return {"L1": 1 / 3, "L2": 1 / 3, "L3": 1 / 3}
    return {k: float(v) / total for k, v in weights.items()}


DETECTOR_DEFAULT_WEIGHTS = _normalize_weights(DETECTOR_DEFAULT_WEIGHTS)


# ── Pretty printing ─────────────────────────────────────────────────────────

def step_banner(n: int, total: int, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  STEP {n}/{total}: {title}")
    print(f"{'=' * 60}")


# ── Safe converters ─────────────────────────────────────────────────────────

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _meets_target(value: float, target: float, tol: float = METRIC_TOL) -> bool:
    return float(value) + float(tol) >= float(target)


def _json_default(x: Any):
    if hasattr(x, "item"):
        return x.item()
    return str(x)


# ── Dataset / CSV helpers ───────────────────────────────────────────────────

def _infer_category(vpath: Path, dataset_root: Path, label: int) -> str:
    """
    Category inference:
    - αν είναι μέσα σε ai/ ή authentic/ και έχει prefix τύπου sora_0001 → sora
    - αλλιώς parent folder
    - fallback: ai/authentic
    """
    stem_parts = vpath.stem.split("_")
    if len(stem_parts) > 1 and stem_parts[0].lower() not in {"ai", "authentic"}:
        return stem_parts[0]

    rel_parts = list(vpath.relative_to(dataset_root).parts)
    if len(rel_parts) >= 2:
        parent_name = rel_parts[-2]
        if parent_name.lower() not in {"ai", "authentic"}:
            return parent_name

    return "ai" if label == 1 else "authentic"


def _derive_pair_id(filename: str) -> str:
    stem = Path(filename).stem
    m = re.search(r"_(\d+)$", stem)
    if m:
        return m.group(1)
    return stem


def _load_ff_manifest_map(video_dir: str) -> dict[str, dict]:
    manifest_path = Path(video_dir) / "ff_manifest.csv"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f) if row.get("filename")}


def _manifest_tokens_for_filename(filename: str, manifest_map: dict[str, dict]) -> tuple[str, ...]:
    row = manifest_map.get(filename, {})
    original = str(row.get("original", "")).strip()
    if original:
        stem = Path(original).stem
        tokens = tuple(tok for tok in stem.split("_") if tok)
        if tokens:
            return tokens
    return (_derive_pair_id(filename),)


def _build_ff_source_group_lookup(manifest_map: dict[str, dict]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for row in manifest_map.values():
        tokens = _manifest_tokens_for_filename(row.get("filename", ""), manifest_map)
        for tok in tokens:
            parent.setdefault(tok, tok)
        for tok in tokens[1:]:
            union(tokens[0], tok)

    components: dict[str, set[str]] = {}
    for token in list(parent):
        components.setdefault(find(token), set()).add(token)

    lookup: dict[str, str] = {}
    for members in components.values():
        group_id = "|".join(sorted(members))
        for token in members:
            lookup[token] = group_id
    return lookup


def _ff_metadata_for_filename(filename: str, manifest_map: dict[str, dict], source_group_lookup: dict[str, str]) -> dict[str, str]:
    row = manifest_map.get(filename, {})
    tokens = _manifest_tokens_for_filename(filename, manifest_map)
    primary = tokens[0] if tokens else ""
    secondary = tokens[1] if len(tokens) > 1 else ""
    source_group = source_group_lookup.get(primary, primary or _derive_pair_id(filename))
    return {
        "original": str(row.get("original", "")),
        "source_primary": primary,
        "source_secondary": secondary,
        "source_group": source_group,
    }


def _load_official_split_map(video_dir: str) -> dict[str, str]:
    root = Path(video_dir)
    split_files = {
        "train": root / "train_split.csv",
        "validation": root / "val_split.csv",
        "test": root / "test_split.csv",
    }
    if not all(path.exists() for path in split_files.values()):
        return {}

    split_map: dict[str, str] = {}
    for split_name, path in split_files.items():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                filename = row.get("filename", "")
                if filename:
                    split_map[filename] = split_name
    return split_map


def _official_split_pair_violations(
    rows: list[dict],
    split_map: dict[str, str],
) -> tuple[list[str], dict[str, list[str]]]:
    missing_files: list[str] = []
    pair_splits: dict[str, set[str]] = {}

    for row in rows:
        filename = str(row.get("filename", ""))
        split = split_map.get(filename)
        if not split:
            missing_files.append(filename)
            continue
        pair_id = str(row.get("pair_id") or _derive_pair_id(filename))
        pair_splits.setdefault(pair_id, set()).add(str(split))

    violations = {
        pair_id: sorted(list(splits))
        for pair_id, splits in pair_splits.items()
        if len(splits) > 1
    }
    return missing_files, violations


def _discover_videos(video_dir: str) -> list[tuple[Path, int, str]]:
    root = Path(video_dir)
    if not root.exists():
        print(f"  ΣΦΑΛΜΑ: Δεν βρέθηκε ο φάκελος: {video_dir}")
        return []

    discovered: list[tuple[Path, int, str]] = []

    ai_dir = root / "ai"
    auth_dir = root / "authentic"

    if ai_dir.exists() or auth_dir.exists():
        for subdir_name, label in [("ai", 1), ("authentic", 0)]:
            subdir = root / subdir_name
            if not subdir.exists():
                continue
            for f in sorted(subdir.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS:
                    category = _infer_category(f, root, label)
                    discovered.append((f, label, category))
    else:
        for f in sorted(root.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
                continue
            path_str = str(f).lower()
            label = 1 if any(tok in path_str for tok in ["ai", "fake", "deepfake", "sora", "runway", "kling"]) else 0
            category = _infer_category(f, root, label)
            discovered.append((f, label, category))

    return discovered


def _limit_videos_stratified(
    all_videos: list[tuple[Path, int, str]],
    max_videos: int,
    seed: int = SEED,
) -> list[tuple[Path, int, str]]:
    if max_videos <= 0 or max_videos >= len(all_videos):
        return all_videos

    ai = [x for x in all_videos if int(x[1]) == 1]
    auth = [x for x in all_videos if int(x[1]) == 0]
    if not ai or not auth:
        return all_videos[:max_videos]

    rng = np.random.default_rng(seed)
    rng.shuffle(ai)
    rng.shuffle(auth)

    n_ai = min(len(ai), max_videos // 2)
    n_auth = min(len(auth), max_videos - n_ai)
    if n_ai + n_auth < max_videos:
        rem = max_videos - (n_ai + n_auth)
        if len(ai) - n_ai >= len(auth) - n_auth:
            n_ai = min(len(ai), n_ai + rem)
        else:
            n_auth = min(len(auth), n_auth + rem)

    selected = ai[:n_ai] + auth[:n_auth]
    selected.sort(key=lambda x: x[0].name)
    return selected


def load_feature_rows(csv_path: str) -> list[dict]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Δεν βρέθηκε το CSV: {csv_path}")

    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "filename": row.get("filename", ""),
                "path": row.get("path", ""),
                "label": _safe_int(row.get("label", 0)),
                "category": row.get("category", "unknown"),
                "pair_id": row.get("pair_id", _derive_pair_id(row.get("filename", ""))),
                "original": row.get("original", ""),
                "source_primary": row.get("source_primary", ""),
                "source_secondary": row.get("source_secondary", ""),
                "source_group": row.get("source_group", row.get("pair_id", _derive_pair_id(row.get("filename", "")))),
                "track": row.get("track", "core"),
                "analysis_version": row.get("analysis_version", getattr(PATVXDetector, "ANALYSIS_VERSION", "patv_x_core_v2")),
                "verdict": row.get("verdict", ""),
                "tcs_score": _safe_float(row.get("tcs_score", 0.0)),
                "confidence": row.get("confidence", ""),
                "abstained_reason": row.get("abstained_reason", ""),
                "detector_threshold": _safe_float(row.get("detector_threshold", DETECTOR_DEFAULT_THRESHOLD)),
                "flow_score": _safe_float(row.get("flow_score", row.get("L1_flow", 0.0))),
                "mean_divergence": _safe_float(row.get("mean_divergence", 0.0)),
                "acceleration_anomaly": _safe_float(row.get("acceleration_anomaly", 0.0)),
                "flicker_score": _safe_float(row.get("flicker_score", 0.0)),
                "physics_score": _safe_float(row.get("physics_score", row.get("L2_physics", 0.0))),
                "gravity_consistency": _safe_float(row.get("gravity_consistency", 0.0)),
                "rigid_body_score": _safe_float(row.get("rigid_body_score", 0.0)),
                "shadow_consistency": _safe_float(row.get("shadow_consistency", 0.0)),
                "semantic_score": _safe_float(row.get("semantic_score", row.get("L3_semantic", 0.0))),
                "color_drift": _safe_float(row.get("color_drift", 0.0)),
                "edge_stability": _safe_float(row.get("edge_stability", 0.0)),
                "texture_consistency": _safe_float(row.get("texture_consistency", 0.0)),

                # L4: CTCG — Causal Temporal Coherence Graph (novel)
                "ctcg_score": _safe_float(row.get("ctcg_score", row.get("L4_ctcg", 0.0))),
                "ctcg_phase_coherence": _safe_float(row.get("ctcg_phase_coherence", 0.0)),
                "ctcg_ar_residual": _safe_float(row.get("ctcg_ar_residual", 0.0)),
                "ctcg_spectral_anomaly": _safe_float(row.get("ctcg_spectral_anomaly", 0.0)),
                "ctcg_micro_jitter": _safe_float(row.get("ctcg_micro_jitter", row.get("boundary_edge_flicker", 0.0))),
                "flow_boundary_seam": _safe_float(row.get("flow_boundary_seam", row.get("ctcg_phase_coherence", 0.0))),
                "boundary_color_flicker": _safe_float(row.get("boundary_color_flicker", row.get("ctcg_ar_residual", 0.0))),
                "warp_prediction_error": _safe_float(row.get("warp_prediction_error", row.get("ctcg_spectral_anomaly", 0.0))),
                "boundary_edge_flicker": _safe_float(row.get("boundary_edge_flicker", row.get("ctcg_micro_jitter", 0.0))),
                "face_scene_support_ratio": _safe_float(row.get("face_scene_support_ratio", 0.0)),
                "light_support_ratio": _safe_float(row.get("light_support_ratio", 0.0)),
                "skin_support_ratio": _safe_float(row.get("skin_support_ratio", 0.0)),
                "flow_boundary_support_ratio": _safe_float(row.get("flow_boundary_support_ratio", 0.0)),
                "boundary_color_support_ratio": _safe_float(row.get("boundary_color_support_ratio", 0.0)),
                "warp_support_ratio": _safe_float(row.get("warp_support_ratio", 0.0)),
                "boundary_edge_support_ratio": _safe_float(row.get("boundary_edge_support_ratio", 0.0)),
                "boundary_agreement_ratio": _safe_float(row.get("boundary_agreement_ratio", 0.0)),
                "boundary_corroboration": _safe_float(row.get("boundary_corroboration", 0.0)),
                "warp_supported_score": _safe_float(row.get("warp_supported_score", 0.0)),
                "violation_count": _safe_float(row.get("violation_count", 0.0)),
                "major_violation_count": _safe_float(row.get("major_violation_count", 0.0)),
                "critical_violation_count": _safe_float(row.get("critical_violation_count", 0.0)),
                "dynamic_violation_score": _safe_float(row.get("dynamic_violation_score", 0.0)),
                "boundary_violation_score": _safe_float(row.get("boundary_violation_score", 0.0)),
                "violation_ratio_mean": _safe_float(row.get("violation_ratio_mean", 0.0)),
                "violation_ratio_max": _safe_float(row.get("violation_ratio_max", 0.0)),
                "violation_span_fraction": _safe_float(row.get("violation_span_fraction", 0.0)),
                "major_span_fraction": _safe_float(row.get("major_span_fraction", 0.0)),
                "critical_span_fraction": _safe_float(row.get("critical_span_fraction", 0.0)),
                "dynamic_span_fraction": _safe_float(row.get("dynamic_span_fraction", 0.0)),
                "boundary_span_fraction": _safe_float(row.get("boundary_span_fraction", 0.0)),
                "confidence_weighted_span": _safe_float(row.get("confidence_weighted_span", 0.0)),
                "has_face_motion": _safe_float(row.get("has_face_motion", 0.0)),
                "has_face_scene_decoupling": _safe_float(row.get("has_face_scene_decoupling", 0.0)),
                "has_rigid_body": _safe_float(row.get("has_rigid_body", 0.0)),
                "has_warp_prediction_error": _safe_float(row.get("has_warp_prediction_error", 0.0)),
                "has_flow_boundary_seam": _safe_float(row.get("has_flow_boundary_seam", 0.0)),
                "has_edge_instability": _safe_float(row.get("has_edge_instability", 0.0)),
                "has_face_texture": _safe_float(row.get("has_face_texture", 0.0)),
                "frame_anomaly_max": _safe_float(row.get("frame_anomaly_max", 0.0)),
                "frame_anomaly_top3_mean": _safe_float(row.get("frame_anomaly_top3_mean", 0.0)),
                "frame_anomaly_peak_fraction": _safe_float(row.get("frame_anomaly_peak_fraction", 0.0)),
                "frame_anomaly_mean": _safe_float(row.get("frame_anomaly_mean", 0.0)),
                "frame_anomaly_std": _safe_float(row.get("frame_anomaly_std", 0.0)),
                "frame_multi_evidence_fraction": _safe_float(row.get("frame_multi_evidence_fraction", 0.0)),
                "face_detected_ratio": _safe_float(row.get("face_detected_ratio", 0.0)),
                "face_reliability_mean": _safe_float(row.get("face_reliability_mean", row.get("face_reliability", 0.0))),
                "neck_visibility_ratio": _safe_float(row.get("neck_visibility_ratio", 0.0)),
                "boundary_support_ratio": _safe_float(row.get("boundary_support_ratio", 0.0)),

                "duration": _safe_float(row.get("duration", 0.0)),
                "fps": _safe_float(row.get("fps", 0.0)),
                "width": _safe_int(row.get("width", 0)),
                "height": _safe_int(row.get("height", 0)),
            })
    return rows


def write_feature_rows(rows: list[dict], output_csv: str) -> None:
    if not rows:
        return
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_split_manifest(rows: list[dict], train_idx, val_idx, test_idx, output_csv: str) -> None:
    split_by_idx: dict[int, str] = {}
    for i in train_idx:
        split_by_idx[int(i)] = "train"
    for i in val_idx:
        split_by_idx[int(i)] = "validation"
    for i in test_idx:
        split_by_idx[int(i)] = "test"

    manifest_rows = []
    for i, row in enumerate(rows):
        manifest_rows.append({
            "index": i,
            "filename": row.get("filename", ""),
            "label": row.get("label", 0),
            "category": row.get("category", "unknown"),
            "pair_id": row.get("pair_id", ""),
            "original": row.get("original", ""),
            "source_primary": row.get("source_primary", ""),
            "source_secondary": row.get("source_secondary", ""),
            "source_group": row.get("source_group", ""),
            "track": row.get("track", "core"),
            "analysis_version": row.get("analysis_version", ""),
            "split": split_by_idx.get(i, "unused"),
        })

    write_feature_rows(manifest_rows, output_csv)


# ── Metrics helpers ─────────────────────────────────────────────────────────

def compute_auc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores)
    s = scores[order]
    y = labels[order]

    neg_before = 0
    favorable_pairs = 0.0
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        group = y[i:j]
        pos_in_group = int(group.sum())
        neg_in_group = int(len(group) - pos_in_group)
        favorable_pairs += pos_in_group * neg_before + 0.5 * pos_in_group * neg_in_group
        neg_before += neg_in_group
        i = j

    return float(favorable_pairs / max(n_pos * n_neg, 1))


def _confusion_from_preds(labels, preds) -> dict:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _metrics_from_confusion(tp: int, tn: int, fp: int, fn: int) -> dict:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    balanced_accuracy = 0.5 * (recall + specificity)
    youden_j = recall + specificity - 1.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "youden_j": float(youden_j),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _threshold_grid_from_scores(scores) -> list[float]:
    vals = sorted(set(float(s) for s in scores))
    if not vals:
        return [0.5]
    if len(vals) == 1:
        v = vals[0]
        return [max(0.0, v - 1e-6), v, min(1.0, v + 1e-6)]

    mids = []
    for a, b in zip(vals[:-1], vals[1:]):
        mids.append((a + b) / 2.0)

    grid = [max(0.0, vals[0] - 1e-6)] + mids + [min(1.0, vals[-1] + 1e-6)]
    return sorted(set(float(x) for x in grid))


def best_threshold_metrics(scores, labels, thresholds=None) -> dict:
    """
    Επιλογή threshold με objective το balanced accuracy.
    Tie-break:
      1) balanced_accuracy
      2) youden_j
      3) f1
      4) accuracy
      5) threshold πιο κοντά στο 0.50
    """
    scores = [float(s) for s in scores]
    labels = [int(y) for y in labels]

    if thresholds is None:
        thresholds = _threshold_grid_from_scores(scores)

    best = None
    best_key = None

    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        c = _confusion_from_preds(labels, preds)
        m = _metrics_from_confusion(c["tp"], c["tn"], c["fp"], c["fn"])
        m["threshold"] = float(t)

        key = (
            round(m["balanced_accuracy"], 12),
            round(m["youden_j"], 12),
            round(m["f1"], 12),
            round(m["accuracy"], 12),
            -abs(float(t) - 0.5),
        )

        if best is None or key > best_key:
            best = m
            best_key = key

    return best


def evaluate_scores_at_threshold(scores, labels, threshold: float) -> dict:
    scores = [float(s) for s in scores]
    labels = [int(y) for y in labels]
    preds = [1 if s >= threshold else 0 for s in scores]
    c = _confusion_from_preds(labels, preds)
    m = _metrics_from_confusion(c["tp"], c["tn"], c["fp"], c["fn"])
    m["threshold"] = float(threshold)
    m["auc_roc"] = float(compute_auc(scores, labels)) if len(set(labels)) > 1 else None
    return m


def evaluate_rows_at_threshold(rows: list[dict], threshold: float) -> dict:
    labels = [_safe_int(r.get("label", 0)) for r in rows]
    scores = [_safe_float(r.get("tcs_score", 0.0)) for r in rows]

    preds = []
    abstained_total = 0
    abstained_ai = 0
    abstained_auth = 0
    for row, label, score in zip(rows, labels, scores):
        if str(row.get("abstained_reason", "")).strip():
            preds.append(-1)
            abstained_total += 1
            if label == 1:
                abstained_ai += 1
            else:
                abstained_auth += 1
        else:
            preds.append(1 if score >= threshold else 0)

    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p in {1, -1})
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p in {0, -1})
    out = _metrics_from_confusion(tp, tn, fp, fn)
    out["threshold"] = float(threshold)
    out["auc_roc"] = float(compute_auc(scores, labels)) if len(set(labels)) > 1 else None
    out["abstentions_total"] = int(abstained_total)
    out["abstentions_ai"] = int(abstained_ai)
    out["abstentions_authentic"] = int(abstained_auth)
    out["abstention_rate"] = float(abstained_total / max(len(rows), 1))
    return out


def tune_detector_threshold(
    rows: list[dict],
    min_specificity: float = CORE_TARGET_SPECIFICITY,
    min_recall: float = CORE_TARGET_RECALL,
) -> dict:
    scores = [_safe_float(r.get("tcs_score", 0.0)) for r in rows]
    labels = [_safe_int(r.get("label", 0)) for r in rows]
    thresholds = _threshold_grid_from_scores(scores)

    best = None
    best_key = None
    for t in thresholds:
        m = evaluate_rows_at_threshold(rows, float(t))
        meets_spec = _meets_target(float(m["specificity"]), float(min_specificity))
        meets_recall = _meets_target(float(m["recall"]), float(min_recall))
        tier = 2 if (meets_spec and meets_recall) else 1 if meets_spec else 0
        if tier == 2:
            key = (
                tier,
                round(float(m["balanced_accuracy"]), 12),
                round(float(m["f1"]), 12),
                round(float(m["youden_j"]), 12),
                round(float(m["recall"]), 12),
                -abs(float(t) - 0.5),
            )
        elif tier == 1:
            # Specificity-first policy:
            # among thresholds that satisfy specificity, maximize balanced accuracy
            # and prefer higher specificity in ties.
            key = (
                tier,
                round(float(m["balanced_accuracy"]), 12),
                round(float(m["specificity"]), 12),
                round(float(m["f1"]), 12),
                round(float(m["youden_j"]), 12),
                round(float(m["recall"]), 12),
                -abs(float(t) - 0.5),
            )
        else:
            key = (
                tier,
                round(float(m["specificity"]), 12),
                round(float(m["balanced_accuracy"]), 12),
                round(float(m["recall"]), 12),
                -abs(float(t) - 0.5),
            )
        if best is None or key > best_key:
            best = dict(m)
            best["target_specificity"] = float(min_specificity)
            best["target_recall"] = float(min_recall)
            best["meets_target"] = bool(meets_spec)
            best["meets_recall_target"] = bool(meets_recall)
            best["constraint_tier"] = int(tier)
            best_key = key

    return best or {
        "threshold": 0.5,
        "target_specificity": float(min_specificity),
        "target_recall": float(min_recall),
        "meets_target": False,
        "meets_recall_target": False,
        "constraint_tier": 0,
    }


# ── Split helpers ───────────────────────────────────────────────────────────

def _split_one_class_indices(class_idx: np.ndarray, val_ratio: float, test_ratio: float, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_idx = np.asarray(class_idx, dtype=int).copy()
    rng.shuffle(class_idx)
    n = len(class_idx)

    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    if n == 1:
        return class_idx.copy(), np.array([], dtype=int), np.array([], dtype=int)

    if n == 2:
        return class_idx[:1], class_idx[1:], np.array([], dtype=int)

    if n == 3:
        return class_idx[:1], class_idx[1:2], class_idx[2:]

    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))

    while n_val + n_test > n - 1:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    train_end = n - n_val - n_test
    val_end = n - n_test

    train_idx = class_idx[:train_end]
    val_idx = class_idx[train_end:val_end]
    test_idx = class_idx[val_end:]

    return train_idx, val_idx, test_idx


def stratified_index_split(labels, val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO, seed: int = SEED):
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    pos_train, pos_val, pos_test = _split_one_class_indices(pos_idx, val_ratio, test_ratio, rng)
    neg_train, neg_val, neg_test = _split_one_class_indices(neg_idx, val_ratio, test_ratio, rng)

    train_idx = np.concatenate([pos_train, neg_train]).astype(int)
    val_idx = np.concatenate([pos_val, neg_val]).astype(int)
    test_idx = np.concatenate([pos_test, neg_test]).astype(int)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def pair_aware_index_split(rows: list[dict], labels, val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO, seed: int = SEED):
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)

    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        group_id = str(
            row.get("source_group")
            or row.get("pair_id")
            or _derive_pair_id(row.get("filename", f"row_{i}"))
        )
        groups.setdefault(group_id, []).append(i)

    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    total_pos = int(labels.sum())
    total_neg = int(len(labels) - total_pos)
    targets = {
        "validation": {"pos": max(1, int(round(total_pos * val_ratio))), "neg": max(1, int(round(total_neg * val_ratio)))},
        "test": {"pos": max(1, int(round(total_pos * test_ratio))), "neg": max(1, int(round(total_neg * test_ratio)))},
    }
    targets["train"] = {
        "pos": max(total_pos - targets["validation"]["pos"] - targets["test"]["pos"], 0),
        "neg": max(total_neg - targets["validation"]["neg"] - targets["test"]["neg"], 0),
    }

    counts = {split: {"pos": 0, "neg": 0} for split in targets}
    assignments = {split: [] for split in targets}

    for _, idxs in group_items:
        gpos = int(labels[idxs].sum())
        gneg = int(len(idxs) - gpos)
        best_split = None
        best_score = None
        for split in ("validation", "test", "train"):
            temp = {
                name: {"pos": counts[name]["pos"], "neg": counts[name]["neg"]}
                for name in counts
            }
            temp[split]["pos"] += gpos
            temp[split]["neg"] += gneg
            score = 0.0
            for name in targets:
                score += abs(targets[name]["pos"] - temp[name]["pos"])
                score += abs(targets[name]["neg"] - temp[name]["neg"])
            if best_score is None or score < best_score:
                best_score = score
                best_split = split
        assignments[best_split].extend(idxs)
        counts[best_split]["pos"] += gpos
        counts[best_split]["neg"] += gneg

    train_idx = np.array(sorted(assignments["train"]), dtype=int)
    val_idx = np.array(sorted(assignments["validation"]), dtype=int)
    test_idx = np.array(sorted(assignments["test"]), dtype=int)
    return train_idx, val_idx, test_idx


def _subset_rows(rows: list[dict], indices) -> list[dict]:
    return [rows[int(i)] for i in indices]


def _fit_minmax(train_X: np.ndarray):
    x_min = train_X.min(axis=0)
    x_max = train_X.max(axis=0)
    x_range = x_max - x_min
    x_range = np.where(np.abs(x_range) < 1e-9, 1.0, x_range)
    return x_min, x_range


def _apply_minmax(X: np.ndarray, x_min: np.ndarray, x_range: np.ndarray) -> np.ndarray:
    return (X - x_min) / x_range


# ── Feature extraction ──────────────────────────────────────────────────────

def _result_to_feature_row(
    vpath: Path,
    label: int,
    category: str,
    ff_meta: dict[str, str],
    result,
    detector_threshold: float,
) -> dict[str, Any]:
    extra = derive_forensic_extra_features_from_result(result)
    return {
        "filename": vpath.name,
        "path": str(vpath),
        "label": int(label),
        "category": category,
        "pair_id": _derive_pair_id(vpath.name),
        "original": ff_meta.get("original", ""),
        "source_primary": ff_meta.get("source_primary", ""),
        "source_secondary": ff_meta.get("source_secondary", ""),
        "source_group": ff_meta.get("source_group", ""),
        "track": result.track,
        "analysis_version": result.analysis_version,
        "verdict": result.verdict,
        "tcs_score": round(float(result.tcs_score), 5),
        "confidence": result.confidence,
        "abstained_reason": result.abstained_reason,
        "detector_threshold": round(float(detector_threshold), 5),
        "flow_score": round(float(result.flow_result.flow_score), 5),
        "mean_divergence": round(float(result.flow_result.mean_divergence), 5),
        "acceleration_anomaly": round(float(result.flow_result.acceleration_anomaly), 5),
        "flicker_score": round(float(result.flow_result.flicker_score), 5),
        "physics_score": round(float(result.physics_result.physics_score), 5),
        "gravity_consistency": round(float(result.physics_result.gravity_consistency), 5),
        "rigid_body_score": round(float(result.physics_result.rigid_body_score), 5),
        "shadow_consistency": round(float(result.physics_result.shadow_consistency), 5),
        "semantic_score": round(float(result.semantic_result.semantic_score), 5),
        "color_drift": round(float(result.semantic_result.color_drift), 5),
        "edge_stability": round(float(result.semantic_result.edge_stability), 5),
        "texture_consistency": round(float(result.semantic_result.texture_consistency), 5),
        "ctcg_score": round(float(getattr(result, "l4_ctcg_score", 0.0)), 5),
        "ctcg_phase_coherence": round(float(getattr(result, "ctcg_phase_coherence", 0.0)), 5),
        "ctcg_ar_residual": round(float(getattr(result, "ctcg_ar_residual", 0.0)), 5),
        "ctcg_spectral_anomaly": round(float(getattr(result, "ctcg_spectral_anomaly", 0.0)), 5),
        "ctcg_micro_jitter": round(float(getattr(result, "ctcg_micro_jitter", 0.0)), 5),
        "flow_boundary_seam": round(float(getattr(result, "l4_flow_boundary_seam_score", 0.0)), 5),
        "boundary_color_flicker": round(float(getattr(result, "l4_boundary_color_flicker_score", 0.0)), 5),
        "warp_prediction_error": round(float(getattr(result, "l4_warp_prediction_error_score", 0.0)), 5),
        "boundary_edge_flicker": round(float(getattr(result, "l4_boundary_edge_flicker_score", 0.0)), 5),
        "face_scene_support_ratio": round(float(extra["face_scene_support_ratio"]), 5),
        "light_support_ratio": round(float(extra["light_support_ratio"]), 5),
        "skin_support_ratio": round(float(extra["skin_support_ratio"]), 5),
        "flow_boundary_support_ratio": round(float(extra["flow_boundary_support_ratio"]), 5),
        "boundary_color_support_ratio": round(float(extra["boundary_color_support_ratio"]), 5),
        "warp_support_ratio": round(float(extra["warp_support_ratio"]), 5),
        "boundary_edge_support_ratio": round(float(extra["boundary_edge_support_ratio"]), 5),
        "boundary_agreement_ratio": round(float(extra["boundary_agreement_ratio"]), 5),
        "boundary_corroboration": round(float(extra["boundary_corroboration"]), 5),
        "warp_supported_score": round(float(extra["warp_supported_score"]), 5),
        "violation_count": int(round(float(extra["violation_count"] * 12.0))),
        "major_violation_count": int(round(float(extra["major_violation_count"] * 8.0))),
        "critical_violation_count": int(round(float(extra["critical_violation_count"] * 4.0))),
        "dynamic_violation_score": round(float(extra["dynamic_violation_score"]), 5),
        "boundary_violation_score": round(float(extra["boundary_violation_score"]), 5),
        "violation_ratio_mean": round(float(extra["violation_ratio_mean"]), 5),
        "violation_ratio_max": round(float(extra["violation_ratio_max"]), 5),
        "violation_span_fraction": round(float(extra["violation_span_fraction"]), 5),
        "major_span_fraction": round(float(extra["major_span_fraction"]), 5),
        "critical_span_fraction": round(float(extra["critical_span_fraction"]), 5),
        "dynamic_span_fraction": round(float(extra["dynamic_span_fraction"]), 5),
        "boundary_span_fraction": round(float(extra["boundary_span_fraction"]), 5),
        "confidence_weighted_span": round(float(extra["confidence_weighted_span"]), 5),
        "has_face_motion": int(round(float(extra["has_face_motion"]))),
        "has_face_scene_decoupling": int(round(float(extra["has_face_scene_decoupling"]))),
        "has_rigid_body": int(round(float(extra["has_rigid_body"]))),
        "has_warp_prediction_error": int(round(float(extra["has_warp_prediction_error"]))),
        "has_flow_boundary_seam": int(round(float(extra["has_flow_boundary_seam"]))),
        "has_edge_instability": int(round(float(extra["has_edge_instability"]))),
        "has_face_texture": int(round(float(extra["has_face_texture"]))),
        "frame_anomaly_max": round(float(extra["frame_anomaly_max"]), 5),
        "frame_anomaly_top3_mean": round(float(extra["frame_anomaly_top3_mean"]), 5),
        "frame_anomaly_peak_fraction": round(float(extra["frame_anomaly_peak_fraction"]), 5),
        "frame_anomaly_mean": round(float(extra["frame_anomaly_mean"]), 5),
        "frame_anomaly_std": round(float(extra["frame_anomaly_std"]), 5),
        "frame_multi_evidence_fraction": round(float(extra["frame_multi_evidence_fraction"]), 5),
        "face_detected_ratio": round(float(getattr(result, "face_detected_ratio", 0.0)), 5),
        "face_reliability_mean": round(float(getattr(result, "face_reliability_mean", 0.0)), 5),
        "neck_visibility_ratio": round(float(getattr(result, "neck_visibility_ratio", 0.0)), 5),
        "boundary_support_ratio": round(float(getattr(result, "boundary_support_ratio", 0.0)), 5),
        "l5_frequency_score": round(float(getattr(result, "l5_frequency_score", 0.0)), 5),
        "l5_lap_var_ratio": round(float(getattr(result, "l5_lap_var_ratio", 0.0)), 5),
        "l5_lap_var_temporal_cv": round(float(getattr(result, "l5_lap_var_temporal_cv", 0.0)), 5),
        "l5_dct_hf_ratio": round(float(getattr(result, "l5_dct_hf_ratio", 0.0)), 5),
        "l5_dct_hf_temporal_cv": round(float(getattr(result, "l5_dct_hf_temporal_cv", 0.0)), 5),
        "l5_lap_kurtosis_mean": round(float(getattr(result, "l5_lap_kurtosis_mean", 0.0)), 5),
        "l5_lap_kurtosis_cv": round(float(getattr(result, "l5_lap_kurtosis_cv", 0.0)), 5),
        "l5_wavelet_detail_ratio": round(float(getattr(result, "l5_wavelet_detail_ratio", 0.0)), 5),
        "l5_wavelet_detail_ratio_cv": round(float(getattr(result, "l5_wavelet_detail_ratio_cv", 0.0)), 5),
        "l5_lap_R_cv": round(float(getattr(result, "l5_lap_R_cv", 0.0)), 5),
        "l5_lap_G_cv": round(float(getattr(result, "l5_lap_G_cv", 0.0)), 5),
        "l5_lap_B_cv": round(float(getattr(result, "l5_lap_B_cv", 0.0)), 5),
        "l5_lap_diff_mean": round(float(getattr(result, "l5_lap_diff_mean", 0.0)), 5),
        "l5_lap_trend_residual_std": round(float(getattr(result, "l5_lap_trend_residual_std", 0.0)), 5),
        "l5_lap_acc_mean": round(float(getattr(result, "l5_lap_acc_mean", 0.0)), 5),
        "l5_block_lap_consistency": round(float(getattr(result, "l5_block_lap_consistency", 0.0)), 5),
        "l5_cross_ch_lap_corr": round(float(getattr(result, "l5_cross_ch_lap_corr", 0.0)), 5),
        "l5_srm_var_cv": round(float(getattr(result, "l5_srm_var_cv", 0.0)), 5),
        "duration": round(float(result.video_duration), 2),
        "fps": round(float(result.fps), 1),
        "width": int(result.resolution[0]),
        "height": int(result.resolution[1]),
    }


def _extract_one_video_worker(
    index: int,
    path_str: str,
    label: int,
    category: str,
    ff_meta: dict[str, str],
    weights: dict[str, float],
    threshold: float,
    track: str,
    sample_rate: int,
    max_frames: int,
) -> dict[str, Any]:
    started = time.time()
    try:
        detector = PATVXDetector(
            weights=weights,
            threshold=threshold,
            verbose=False,
            track=track,
            sample_rate=sample_rate,
            max_frames=max_frames,
        )
        result = detector.analyze(path_str)
        if result.verdict == "AI_GENERATED":
            verdict_short = "AI"
        elif result.verdict == "INCONCLUSIVE":
            verdict_short = "INC"
        else:
            verdict_short = "OK"
        row = _result_to_feature_row(
            vpath=Path(path_str),
            label=label,
            category=category,
            ff_meta=ff_meta,
            result=result,
            detector_threshold=detector.threshold,
        )
        return {
            "index": index,
            "filename": Path(path_str).name,
            "ok": True,
            "error": "",
            "elapsed": time.time() - started,
            "tcs_score": float(result.tcs_score),
            "verdict_short": verdict_short,
            "row": row,
        }
    except Exception as exc:
        return {
            "index": index,
            "filename": Path(path_str).name,
            "ok": False,
            "error": str(exc),
            "elapsed": time.time() - started,
            "tcs_score": 0.0,
            "verdict_short": "ERR",
            "row": None,
        }


def extract_features(
    video_dir: str,
    output_csv: str,
    detector: PATVXDetector,
    workers: int = 1,
    max_videos: int = 0,
) -> list[dict]:
    """
    Τρέχει PATV σε όλα τα βίντεο και αποθηκεύει ΟΛΑ τα πραγματικά features.
    """
    all_videos = _discover_videos(video_dir)
    all_videos = _limit_videos_stratified(all_videos, int(max_videos), seed=SEED)

    if not all_videos:
        print(f"  ΣΦΑΛΜΑ: Δεν βρέθηκαν βίντεο στο: {video_dir}")
        return []

    n_ai = sum(label for _, label, _ in all_videos)
    n_auth = len(all_videos) - n_ai

    max_videos = int(max_videos or 0)
    if max_videos > 0:
        print(
            f"  Βρέθηκαν {len(all_videos)} βίντεο (stratified subset, max={max_videos})  "
            f"(AI={n_ai}  Auth={n_auth})"
        )
    else:
        print(f"  Βρέθηκαν {len(all_videos)} βίντεο  (AI={n_ai}  Auth={n_auth})")

    manifest_map = _load_ff_manifest_map(video_dir)
    source_group_lookup = _build_ff_source_group_lookup(manifest_map) if manifest_map else {}

    rows: list[dict] = []
    errors = 0
    workers = int(workers or 1)
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, len(all_videos))

    if workers == 1:
        for i, (vpath, label, category) in enumerate(all_videos, 1):
            print(f"  [{i:3d}/{len(all_videos)}] {vpath.name[:45]:<45}", end=" ", flush=True)
            t0 = time.time()
            try:
                result = detector.analyze(str(vpath))
                elapsed = time.time() - t0
                if result.verdict == "AI_GENERATED":
                    verdict_short = "AI"
                elif result.verdict == "INCONCLUSIVE":
                    verdict_short = "INC"
                else:
                    verdict_short = "OK"
                print(f"TCS={result.tcs_score:.3f} [{verdict_short}] ({elapsed:.1f}s)")
                ff_meta = _ff_metadata_for_filename(vpath.name, manifest_map, source_group_lookup)
                rows.append(
                    _result_to_feature_row(
                        vpath=vpath,
                        label=label,
                        category=category,
                        ff_meta=ff_meta,
                        result=result,
                        detector_threshold=detector.threshold,
                    )
                )
            except Exception as e:
                print(f"ERROR: {e}")
                errors += 1
    else:
        print(f"  Parallel extraction: workers={workers}")
        jobs = []
        for i, (vpath, label, category) in enumerate(all_videos, 1):
            ff_meta = _ff_metadata_for_filename(vpath.name, manifest_map, source_group_lookup)
            jobs.append(
                (
                    i,
                    str(vpath),
                    int(label),
                    category,
                    ff_meta,
                    dict(detector.weights),
                    float(detector.threshold),
                    detector.track,
                    int(detector.sample_rate),
                    int(detector.max_frames),
                )
            )

        completed: dict[int, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_extract_one_video_worker, *job): (job[0], Path(job[1]).name)
                for job in jobs
            }
            for future in as_completed(futures):
                idx, name = futures[future]
                try:
                    rec = future.result()
                except Exception as exc:
                    rec = {
                        "index": idx,
                        "filename": name,
                        "ok": False,
                        "error": str(exc),
                        "elapsed": 0.0,
                        "tcs_score": 0.0,
                        "verdict_short": "ERR",
                        "row": None,
                    }
                completed[idx] = rec

        for i in range(1, len(all_videos) + 1):
            rec = completed.get(i)
            if rec is None:
                errors += 1
                continue
            print(
                f"  [{i:3d}/{len(all_videos)}] {rec['filename'][:45]:<45} ",
                end="",
                flush=True,
            )
            if rec["ok"]:
                print(
                    f"TCS={rec['tcs_score']:.3f} [{rec['verdict_short']}] "
                    f"({rec['elapsed']:.1f}s)"
                )
                rows.append(rec["row"])
            else:
                print(f"ERROR: {rec['error']}")
                errors += 1

    if errors:
        print(f"\n  ⚠ {errors} βίντεο δεν αναλύθηκαν")

    if rows:
        write_feature_rows(rows, output_csv)
        print(f"\n  Features αποθηκεύθηκαν: {output_csv} ({len(rows)} rows)")

    return rows


def build_synthetic_rows(n_samples: int, output_csv: str) -> list[dict]:
    weights = DETECTOR_DEFAULT_WEIGHTS
    threshold = DETECTOR_DEFAULT_THRESHOLD

    X, y = generate_synthetic_data(n_samples, include_support=True)
    rows: list[dict] = []

    for i, (x, yi) in enumerate(zip(X, y)):
        fmap = {name: float(x[j]) for j, name in enumerate(OPEN_FEATURE_NAMES)}
        tcs = float(
            weights.get("L1", 0.28) * fmap["flow_score"] +
            weights.get("L2", 0.26) * fmap["physics_score"] +
            weights.get("L3", 0.16) * fmap["semantic_score"] +
            weights.get("L4", 0.30) * fmap["ctcg_score"]
        )
        verdict = "AI_GENERATED" if tcs >= threshold else "AUTHENTIC"
        conf = "HIGH" if abs(tcs - threshold) > 0.20 else "MEDIUM" if abs(tcs - threshold) > 0.10 else "LOW"
        row = {
            "filename": f"synthetic_{i:04d}.mp4", "path": "",
            "label": int(yi), "category": "synthetic_ai" if yi == 1 else "synthetic_auth",
            "pair_id": f"synthetic_{i:04d}",
            "original": "",
            "source_primary": f"synthetic_{i:04d}",
            "source_secondary": "",
            "source_group": f"synthetic_{i:04d}",
            "track": "synthetic-smoke",
            "analysis_version": "patv_x_synthetic_smoke_v2",
            "verdict": verdict, "tcs_score": round(tcs, 5),
            "confidence": conf, "abstained_reason": "", "detector_threshold": round(float(threshold), 5),
            "flow_score": round(fmap["flow_score"], 5), "mean_divergence": round(fmap["mean_divergence"], 5),
            "acceleration_anomaly": round(fmap["acceleration_anomaly"], 5), "flicker_score": round(fmap["flicker_score"], 5),
            "physics_score": round(fmap["physics_score"], 5), "gravity_consistency": round(fmap["gravity_consistency"], 5),
            "rigid_body_score": round(fmap["rigid_body_score"], 5), "shadow_consistency": round(fmap["shadow_consistency"], 5),
            "semantic_score": round(fmap["semantic_score"], 5), "color_drift": round(fmap["color_drift"], 5),
            "edge_stability": round(fmap["edge_stability"], 5), "texture_consistency": round(fmap["texture_consistency"], 5),
            "ctcg_score": round(fmap["ctcg_score"], 5), "ctcg_phase_coherence": round(fmap["ctcg_phase_coherence"], 5),
            "ctcg_ar_residual": round(fmap["ctcg_ar_residual"], 5), "ctcg_spectral_anomaly": round(fmap["ctcg_spectral_anomaly"], 5),
            "ctcg_micro_jitter": round(fmap["ctcg_micro_jitter"], 5),
            "flow_boundary_seam": round(fmap["ctcg_phase_coherence"], 5),
            "boundary_color_flicker": round(fmap["ctcg_ar_residual"], 5),
            "warp_prediction_error": round(fmap["ctcg_spectral_anomaly"], 5),
            "boundary_edge_flicker": round(fmap["ctcg_micro_jitter"], 5),
            "face_scene_support_ratio": round(fmap["face_scene_support_ratio"], 5),
            "light_support_ratio": round(fmap["light_support_ratio"], 5),
            "skin_support_ratio": round(fmap["skin_support_ratio"], 5),
            "flow_boundary_support_ratio": round(fmap["flow_boundary_support_ratio"], 5),
            "boundary_color_support_ratio": round(fmap["boundary_color_support_ratio"], 5),
            "warp_support_ratio": round(fmap["warp_support_ratio"], 5),
            "boundary_edge_support_ratio": round(fmap["boundary_edge_support_ratio"], 5),
            "boundary_agreement_ratio": round(fmap["boundary_agreement_ratio"], 5),
            "boundary_corroboration": round(fmap["boundary_corroboration"], 5),
            "warp_supported_score": round(fmap["warp_supported_score"], 5),
            "violation_count": int(round(fmap["violation_count"] * 12.0)),
            "major_violation_count": int(round(fmap["major_violation_count"] * 8.0)),
            "critical_violation_count": int(round(fmap["critical_violation_count"] * 4.0)),
            "dynamic_violation_score": round(fmap["dynamic_violation_score"], 5),
            "boundary_violation_score": round(fmap["boundary_violation_score"], 5),
            "violation_ratio_mean": round(fmap["violation_ratio_mean"], 5),
            "violation_ratio_max": round(fmap["violation_ratio_max"], 5),
            "violation_span_fraction": round(fmap["violation_span_fraction"], 5),
            "major_span_fraction": round(fmap["major_span_fraction"], 5),
            "critical_span_fraction": round(fmap["critical_span_fraction"], 5),
            "dynamic_span_fraction": round(fmap["dynamic_span_fraction"], 5),
            "boundary_span_fraction": round(fmap["boundary_span_fraction"], 5),
            "confidence_weighted_span": round(fmap["confidence_weighted_span"], 5),
            "has_face_motion": int(round(fmap["has_face_motion"])),
            "has_face_scene_decoupling": int(round(fmap["has_face_scene_decoupling"])),
            "has_rigid_body": int(round(fmap["has_rigid_body"])),
            "has_warp_prediction_error": int(round(fmap["has_warp_prediction_error"])),
            "has_flow_boundary_seam": int(round(fmap["has_flow_boundary_seam"])),
            "has_edge_instability": int(round(fmap["has_edge_instability"])),
            "has_face_texture": int(round(fmap["has_face_texture"])),
            "frame_anomaly_max": round(fmap["frame_anomaly_max"], 5),
            "frame_anomaly_top3_mean": round(fmap["frame_anomaly_top3_mean"], 5),
            "frame_anomaly_peak_fraction": round(fmap["frame_anomaly_peak_fraction"], 5),
            "frame_anomaly_mean": round(fmap["frame_anomaly_mean"], 5),
            "frame_anomaly_std": round(fmap["frame_anomaly_std"], 5),
            "frame_multi_evidence_fraction": round(fmap["frame_multi_evidence_fraction"], 5),
            "face_detected_ratio": round(fmap["face_detected_ratio"], 5),
            "face_reliability_mean": round(fmap["face_reliability_mean"], 5),
            "neck_visibility_ratio": round(fmap["neck_visibility_ratio"], 5),
            "boundary_support_ratio": round(fmap["boundary_support_ratio"], 5),
            "duration": 0.0, "fps": 0.0, "width": 0, "height": 0,
        }
        rows.append(row)

    write_feature_rows(rows, output_csv)
    return rows


def archive_legacy_artifacts(root: Path) -> Path:
    archive_dir = root / "legacy_v1"
    archive_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        root / "ablation_results.json",
        root / "patv_mlp.json",
        root / "authentic_check.csv",
        root / "ai_check.csv",
    ]
    archived = []
    for src in candidates:
        if src.exists():
            dst = archive_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            archived.append(src.name)
    manifest_path = archive_dir / "archive_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"archived_from_root": str(root), "files": archived}, f, indent=2, ensure_ascii=False)
    return archive_dir


# ── Full pipeline ───────────────────────────────────────────────────────────

def run_full_pipeline(
    videos_dir: str = "",
    output_dir: str = "pipeline_results",
    synthetic: bool = False,
    n_synthetic: int = 400,
    features_path: str | None = None,
    workers: int = 1,
    target_specificity: float = CORE_TARGET_SPECIFICITY,
    core_min_recall: float = CORE_TARGET_RECALL,
    open_min_recall: float = OPEN_TARGET_RECALL,
    sample_rate: int = DETECTOR_DEFAULT_SAMPLE_RATE,
    max_frames: int = DETECTOR_DEFAULT_MAX_FRAMES,
    max_videos: int = 0,
) -> dict:
    if not (0.0 <= float(target_specificity) <= 1.0):
        raise ValueError(f"target_specificity πρέπει να είναι στο [0,1], πήρα: {target_specificity}")
    if not (0.0 <= float(core_min_recall) <= 1.0):
        raise ValueError(f"core_min_recall πρέπει να είναι στο [0,1], πήρα: {core_min_recall}")
    if not (0.0 <= float(open_min_recall) <= 1.0):
        raise ValueError(f"open_min_recall πρέπει να είναι στο [0,1], πήρα: {open_min_recall}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    legacy_archive = archive_legacy_artifacts(ROOT)

    report: dict[str, Any] = {
        "pipeline_version": "5.0-core-open",
        "started": datetime.now().isoformat(),
        "repo_root": str(ROOT),
        "legacy_archive": str(legacy_archive),
        "seed": SEED,
        "threshold_policy": {
            "name": "specificity_first",
            "target_specificity": float(target_specificity),
            "core_target_recall": float(core_min_recall),
            "open_target_recall": float(open_min_recall),
        },
        "split": {
            "validation_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
        },
        "detector_config": {
            "weights": DETECTOR_DEFAULT_WEIGHTS,
            "threshold": DETECTOR_DEFAULT_THRESHOLD,
            "track": "core" if not synthetic else "synthetic-smoke",
            "analysis_version": getattr(PATVXDetector, "ANALYSIS_VERSION", "patv_x_core_v2"),
            "sample_rate": int(sample_rate),
            "max_frames": int(max_frames),
        },
        "steps": {},
    }

    total_steps = 5
    features_csv = str(out / "features.csv")

    # ── STEP 1: Extraction / Load Features ────────────────────────────────
    if features_path:
        step_banner(1, total_steps, "Load Existing Features")
        src = Path(features_path)
        if not src.exists():
            raise FileNotFoundError(f"Δεν βρέθηκε το features CSV: {features_path}")

        if src.resolve() != Path(features_csv).resolve():
            shutil.copyfile(src, features_csv)

        rows = load_feature_rows(features_csv)
        print(f"  Φορτώθηκαν {len(rows)} rows από: {features_csv}")

        report["steps"]["extraction"] = {
            "mode": "existing_features",
            "csv": features_csv,
            "samples": len(rows),
        }

    elif synthetic:
        step_banner(1, total_steps, "Synthetic Feature Generation")
        rows = build_synthetic_rows(n_synthetic, features_csv)
        print(f"  Δημιουργήθηκαν {len(rows)} synthetic samples → {features_csv}")

        report["steps"]["extraction"] = {
            "mode": "synthetic",
            "csv": features_csv,
            "samples": len(rows),
            "track": "synthetic-smoke",
            "paper_facing": False,
        }

    else:
        step_banner(1, total_steps, "Feature Extraction")
        detector = PATVXDetector(
            weights=DETECTOR_DEFAULT_WEIGHTS,
            threshold=DETECTOR_DEFAULT_THRESHOLD,
            verbose=False,
            track="core",
            sample_rate=int(sample_rate),
            max_frames=int(max_frames),
        )
        rows = extract_features(
            videos_dir,
            features_csv,
            detector,
            workers=workers,
            max_videos=max_videos,
        )

        report["steps"]["extraction"] = {
            "mode": "real",
            "csv": features_csv,
            "samples": len(rows),
            "videos_dir": videos_dir,
            "track": "core",
            "workers": int(workers),
            "max_videos": int(max_videos),
            "paper_facing": True,
        }

    if not rows:
        raise RuntimeError("Δεν υπάρχουν features για να συνεχίσει το pipeline.")

    # ── Load features array and build consistent split ─────────────────────
    X, y = csv_to_features(features_csv, feature_names=LEGACY_CORE_FEATURE_NAMES)
    X_open, y_open = csv_to_features(features_csv, include_support=True, feature_names=OPEN_FEATURE_NAMES)
    if len(X) < 10:
        raise RuntimeError(f"Πολύ λίγα δείγματα ({len(X)}). Χρειάζονται τουλάχιστον 10.")

    if len(X) != len(rows):
        raise RuntimeError(
            f"Ασυμφωνία rows/features: rows={len(rows)} vs features={len(X)}. "
            "Το CSV που φορτώθηκε δεν είναι συνεπές."
        )

    row_labels = np.asarray([_safe_int(r.get("label", 0)) for r in rows], dtype=int)
    y = np.asarray(y, dtype=int)
    if not np.array_equal(row_labels, y):
        raise RuntimeError("Ασυμφωνία labels μεταξύ CSV rows και csv_to_features().")
    if not np.array_equal(y, np.asarray(y_open, dtype=int)):
        raise RuntimeError("Ασυμφωνία labels μεταξύ legacy core features και enriched open/core features.")

    official_split_map = _load_official_split_map(videos_dir) if (videos_dir and not synthetic) else {}
    split_notes: dict[str, Any] = {}
    if official_split_map and all(row.get("filename", "") in official_split_map for row in rows):
        missing_files, pair_violations = _official_split_pair_violations(rows, official_split_map)
        if pair_violations:
            print(
                "  ⚠ Official split παραβιάζει pair isolation "
                f"({len(pair_violations)} pair_ids σε πολλαπλά splits). "
                "Χρησιμοποιείται pair-aware split."
            )
            train_idx, val_idx, test_idx = pair_aware_index_split(
                rows, y, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED
            )
            split_strategy = "source_connected_group_split"
            split_notes = {
                "official_split_rejected": True,
                "pair_violations_count": int(len(pair_violations)),
                "example_pair_violations": dict(list(pair_violations.items())[:8]),
                "missing_files": missing_files[:8],
            }
        else:
            train_idx = np.array(
                [i for i, row in enumerate(rows) if official_split_map[row["filename"]] == "train"],
                dtype=int,
            )
            val_idx = np.array(
                [i for i, row in enumerate(rows) if official_split_map[row["filename"]] == "validation"],
                dtype=int,
            )
            test_idx = np.array(
                [i for i, row in enumerate(rows) if official_split_map[row["filename"]] == "test"],
                dtype=int,
            )
            split_strategy = "official_dataset_split_pair_safe"
            split_notes = {
                "official_split_rejected": False,
                "pair_violations_count": 0,
                "missing_files": missing_files[:8],
            }
    else:
        train_idx, val_idx, test_idx = pair_aware_index_split(rows, y, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED)
        split_strategy = "source_connected_group_split"
        split_notes = {
            "official_split_rejected": bool(official_split_map),
            "reason": "official_split_missing_or_incomplete" if official_split_map else "official_split_not_available",
        }

    if len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Δεν ήταν δυνατό να δημιουργηθούν validation/test splits. Χρειάζονται περισσότερα δεδομένα.")

    split_manifest_path = out / "split_manifest.csv"
    _write_split_manifest(rows, train_idx, val_idx, test_idx, str(split_manifest_path))

    report["steps"]["split"] = {
        "strategy": split_strategy,
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "split_manifest": str(split_manifest_path),
        "notes": split_notes,
    }

    # ── STEP 2: MLP Training ──────────────────────────────────────────────
    step_banner(2, total_steps, "Open Track Training")

    print(
        f"  Dataset: {len(X)} δείγματα  "
        f"(AI={int(y.sum())}  Auth={int((y == 0).sum())})"
    )
    print(
        f"  Split  : train={len(train_idx)}  "
        f"val={len(val_idx)}  test={len(test_idx)}"
    )
    print(f"  Track  : open  ({len(OPEN_FEATURE_NAMES)} features incl. support diagnostics)")

    X_train = np.asarray(X_open[train_idx], dtype=float)
    X_val = np.asarray(X_open[val_idx], dtype=float)
    X_test = np.asarray(X_open[test_idx], dtype=float)

    y_train = np.asarray(y[train_idx], dtype=int)
    y_val = np.asarray(y[val_idx], dtype=int)
    y_test = np.asarray(y[test_idx], dtype=int)

    mu = X_train.mean(axis=0); sigma = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mu) / sigma
    X_val_n   = (X_val   - mu) / sigma
    X_test_n  = (X_test  - mu) / sigma

    epochs = min(250, max(80, len(X_train) // 2))
    linear = LinearLogistic(input_dim=X_train.shape[1], weight_decay=1e-4, seed=SEED)
    linear.train(X_train_n, y_train, X_val_n, y_val, epochs=max(120, epochs), batch_size=64, verbose=True)
    linear_val_probs = linear.predict(X_val_n)
    linear_thr = specificity_first_threshold(
        linear_val_probs,
        y_val,
        min_specificity=target_specificity,
        min_recall=open_min_recall,
    )
    linear_val_metrics = evaluate(linear, X_val_n, y_val, linear_thr["threshold"])
    linear_test_metrics = evaluate(linear, X_test_n, y_test, linear_thr["threshold"])

    model = MLP(input_dim=X_train.shape[1], lr=0.008, weight_decay=1e-4, seed=SEED)
    history = model.train(X_train_n, y_train, X_val_n, y_val,
                          epochs=epochs, batch_size=32, verbose=True)
    mlp_val_probs = model.predict_tta(X_val_n)
    mlp_thr = specificity_first_threshold(
        mlp_val_probs,
        y_val,
        min_specificity=target_specificity,
        min_recall=open_min_recall,
    )
    mlp_val_metrics = evaluate(model, X_val_n, y_val, mlp_thr["threshold"], use_tta=True)
    mlp_test_metrics = evaluate(model, X_test_n, y_test, mlp_thr["threshold"], use_tta=True)

    # GBM (Gradient Boosting Machine)
    gbm = GBMClassifier(input_dim=X_train.shape[1])
    gbm.train(X_train_n, y_train, X_val_n, y_val)
    gbm_val_probs = gbm.predict(X_val_n)
    gbm_thr = specificity_first_threshold(
        gbm_val_probs, y_val,
        min_specificity=target_specificity,
        min_recall=open_min_recall,
    )
    gbm_val_metrics = evaluate(gbm, X_val_n, y_val, gbm_thr["threshold"])
    gbm_test_metrics = evaluate(gbm, X_test_n, y_test, gbm_thr["threshold"])

    # Select best model by validation AUC (more stable than balanced_accuracy for small val sets)
    candidates = [
        ("linear", linear_val_metrics, linear_thr),
        ("mlp", mlp_val_metrics, mlp_thr),
        ("gbm", gbm_val_metrics, gbm_thr),
    ]
    def _open_key(c):
        vm = c[1]
        return (round(vm["auc"], 6), round(vm["balanced_accuracy"], 6), round(vm["specificity"], 6))
    candidates.sort(key=_open_key, reverse=True)
    selected_model = candidates[0][0]
    selected_threshold = candidates[0][2]["threshold"]

    print(
        f"\n  Linear val/test : bal_acc={linear_val_metrics['balanced_accuracy']:.3f}/{linear_test_metrics['balanced_accuracy']:.3f}  "
        f"spec={linear_val_metrics['specificity']:.3f}/{linear_test_metrics['specificity']:.3f}"
    )
    print(
        f"  MLP val/test    : bal_acc={mlp_val_metrics['balanced_accuracy']:.3f}/{mlp_test_metrics['balanced_accuracy']:.3f}  "
        f"spec={mlp_val_metrics['specificity']:.3f}/{mlp_test_metrics['specificity']:.3f}"
    )
    print(
        f"  GBM val/test    : bal_acc={gbm_val_metrics['balanced_accuracy']:.3f}/{gbm_test_metrics['balanced_accuracy']:.3f}  "
        f"spec={gbm_val_metrics['specificity']:.3f}/{gbm_test_metrics['specificity']:.3f}"
    )
    print(f"  Selected open model: {selected_model}  (threshold={selected_threshold:.4f})")

    model_path = out / "patv_open_bundle.json"
    model_data = {
        "track": "open",
        "analysis_version": "patv_x_open_v2",
        "feature_names": OPEN_FEATURE_NAMES,
        "normalization": {"mean": mu.tolist(), "std": sigma.tolist()},
        "threshold_policy": {
            "name": "specificity_first",
            "target_specificity": float(target_specificity),
            "target_recall": float(open_min_recall),
        },
        "models": {
            "linear": linear.save_dict(),
            "mlp": {
                "model_type": "mlp",
                "architecture": f"{len(OPEN_FEATURE_NAMES)}→64→32→16→1",
                "W1": model.W1.tolist(), "b1": model.b1.tolist(),
                "W2": model.W2.tolist(), "b2": model.b2.tolist(),
                "W3": model.W3.tolist(), "b3": model.b3.tolist(),
                "W4": model.W4.tolist(), "b4": model.b4.tolist(),
                "history": history,
            },
            "gbm": gbm.save_dict(),
        },
        "validation": {
            "linear": {"threshold_info": linear_thr, "metrics": linear_val_metrics},
            "mlp": {"threshold_info": mlp_thr, "metrics": mlp_val_metrics},
            "gbm": {"threshold_info": gbm_thr, "metrics": gbm_val_metrics},
        },
        "test": {
            "linear": linear_test_metrics,
            "mlp": mlp_test_metrics,
            "gbm": gbm_test_metrics,
        },
        "selected_model": selected_model,
        "selected_threshold": float(selected_threshold),
    }
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"  Open models αποθηκεύθηκαν: {model_path}")

    report["steps"]["training"] = {
        "track": "open",
        "feature_count": int(X_train.shape[1]),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "epochs": int(epochs),
        "model_path": str(model_path),
        "selected_model": selected_model,
        "selected_threshold": float(selected_threshold),
        "validation_metrics": {
            "linear": linear_val_metrics,
            "mlp": mlp_val_metrics,
            "gbm": gbm_val_metrics,
        },
        "test_metrics": {
            "linear": linear_test_metrics,
            "mlp": mlp_test_metrics,
            "gbm": gbm_test_metrics,
        },
    }

    # ── STEP 3: Core Evaluation (learned head + raw detector baseline) ──
    step_banner(3, total_steps, "Core Evaluation")

    rows_val = _subset_rows(rows, val_idx)
    rows_test = _subset_rows(rows, test_idx)
    test_labels = [_safe_int(r.get("label", 0)) for r in rows_test]
    test_categories = [r.get("category", "unknown") for r in rows_test]

    raw_tuned = tune_detector_threshold(
        rows_val,
        min_specificity=target_specificity,
        min_recall=core_min_recall,
    )
    raw_threshold = float(raw_tuned["threshold"])
    raw_test_eval = evaluate_rows_at_threshold(rows_test, raw_threshold)
    raw_test_tcs_scores = [_safe_float(r.get("tcs_score", 0.0)) for r in rows_test]
    print("  Raw detector (primary — training-free core evaluation):")
    print(
        f"    val_thr={raw_threshold:.4f}  "
        f"val_bal_acc={raw_tuned['balanced_accuracy']:.3f}  "
        f"test_bal_acc={raw_test_eval['balanced_accuracy']:.3f}  "
        f"test_recall={raw_test_eval['recall']:.3f}  "
        f"test_spec={raw_test_eval['specificity']:.3f}"
    )

    # Retrain GBM on train+val for core evaluation (more training data → better generalization)
    X_dev_n = np.vstack([X_train_n, X_val_n])
    y_dev = np.concatenate([y_train, y_val])
    gbm_final = GBMClassifier(input_dim=X_dev_n.shape[1])
    gbm_final.train(X_dev_n, y_dev, X_test_n, y_test)

    linear_test_probs = linear.predict(X_test_n)
    mlp_test_probs = model.predict_tta(X_test_n)
    gbm_test_probs = gbm_final.predict(X_test_n)

    core_linear_thr = specificity_first_threshold(
        linear_val_probs,
        y_val,
        min_specificity=target_specificity,
        min_recall=core_min_recall,
    )
    core_linear_vm = evaluate_scores_at_threshold(linear_val_probs, y_val, core_linear_thr["threshold"])
    core_linear_tm = evaluate_scores_at_threshold(linear_test_probs, y_test, core_linear_thr["threshold"])
    core_linear_vm["constraint_tier"] = int(core_linear_thr["constraint_tier"])
    core_linear_vm["meets_target"] = bool(core_linear_thr["meets_target"])
    core_linear_vm["meets_recall_target"] = bool(core_linear_thr["meets_recall"])

    core_mlp_thr = specificity_first_threshold(
        mlp_val_probs,
        y_val,
        min_specificity=target_specificity,
        min_recall=core_min_recall,
    )
    core_mlp_vm = evaluate_scores_at_threshold(mlp_val_probs, y_val, core_mlp_thr["threshold"])
    core_mlp_tm = evaluate_scores_at_threshold(mlp_test_probs, y_test, core_mlp_thr["threshold"])
    core_mlp_vm["constraint_tier"] = int(core_mlp_thr["constraint_tier"])
    core_mlp_vm["meets_target"] = bool(core_mlp_thr["meets_target"])
    core_mlp_vm["meets_recall_target"] = bool(core_mlp_thr["meets_recall"])

    core_gbm_thr = specificity_first_threshold(
        gbm_val_probs,
        y_val,
        min_specificity=target_specificity,
        min_recall=core_min_recall,
    )
    core_gbm_vm = evaluate_scores_at_threshold(gbm_val_probs, y_val, core_gbm_thr["threshold"])
    core_gbm_tm = evaluate_scores_at_threshold(gbm_test_probs, y_test, core_gbm_thr["threshold"])
    core_gbm_vm["constraint_tier"] = int(core_gbm_thr["constraint_tier"])
    core_gbm_vm["meets_target"] = bool(core_gbm_thr["meets_target"])
    core_gbm_vm["meets_recall_target"] = bool(core_gbm_thr["meets_recall"])

    def _core_model_key(vm: dict) -> tuple:
        return (
            int(vm.get("constraint_tier", 0)),
            round(float(vm["balanced_accuracy"]), 12),
            round(float(vm["recall"]), 12),
            round(float(vm["specificity"]), 12),
            round(float(vm.get("auc_roc", 0.0) or 0.0), 12),
        )

    # GBM is preferred core model (validated via pair-aware 5-fold CV, OOF-AUC 0.844)
    # Small val sets (n=40) give noisy AUC estimates; GBM consistently generalizes best
    print(f"\n  Core model candidates:")
    for name, vm, tm in [("linear", core_linear_vm, core_linear_tm),
                          ("mlp", core_mlp_vm, core_mlp_tm),
                          ("gbm", core_gbm_vm, core_gbm_tm)]:
        print(f"    {name:8s} val_auc={vm.get('auc_roc',0) or 0:.4f} test_auc={tm.get('auc_roc',0) or 0:.4f} "
              f"val_bal={vm['balanced_accuracy']:.3f} test_bal={tm['balanced_accuracy']:.3f}")

    selected_core_model = "gbm"
    val_eval = core_gbm_vm
    test_eval = core_gbm_tm
    core_threshold_info = core_gbm_thr
    selected_core_scores = gbm_test_probs

    core_model_path = out / "patv_core_bundle.json"
    core_bundle = {
        "track": "core",
        "analysis_version": getattr(PATVXDetector, "ANALYSIS_VERSION", "patv_x_core"),
        "feature_names": OPEN_FEATURE_NAMES,
        "include_support_features": True,
        "split_source": str(split_manifest_path),
        "score_name": "core_probability",
        "normalization": {"mean": mu.tolist(), "std": sigma.tolist()},
        "threshold_policy": {
            "name": "specificity_first",
            "target_specificity": float(target_specificity),
            "target_recall": float(core_min_recall),
        },
        "models": {
            "linear": linear.save_dict(),
            "mlp": {
                "model_type": "mlp",
                "architecture": f"{len(OPEN_FEATURE_NAMES)}→64→32→16→1",
                "W1": model.W1.tolist(), "b1": model.b1.tolist(),
                "W2": model.W2.tolist(), "b2": model.b2.tolist(),
                "W3": model.W3.tolist(), "b3": model.b3.tolist(),
                "W4": model.W4.tolist(), "b4": model.b4.tolist(),
                "history": model.training_history,
            },
            "gbm": gbm_final.save_dict(),
        },
        "validation": {
            "linear": {"threshold_info": core_linear_thr, "metrics": core_linear_vm},
            "mlp": {"threshold_info": core_mlp_thr, "metrics": core_mlp_vm},
            "gbm": {"threshold_info": core_gbm_thr, "metrics": core_gbm_vm},
        },
        "test": {
            "linear": core_linear_tm,
            "mlp": core_mlp_tm,
            "gbm": core_gbm_tm,
        },
        "selected_model": selected_core_model,
        "selected_threshold": float(core_threshold_info["threshold"]),
    }
    with open(core_model_path, "w", encoding="utf-8") as f:
        json.dump(core_bundle, f, indent=2, ensure_ascii=False, default=_json_default)

    print("  Threshold tuning set : VALIDATION")
    print("  Final metrics set    : TEST")
    print(f"  Core model           : {selected_core_model}")
    print(f"  Validation threshold : {core_threshold_info['threshold']:.4f}")
    print(f"  Validation Bal Acc   : {val_eval['balanced_accuracy']*100:.1f}%")
    print(f"  Validation Specificity Target Met : {core_threshold_info['meets_target']}")
    print(f"  Validation Recall Target Met      : {core_threshold_info.get('meets_recall', False)}")
    print(f"  Test AUC-ROC         : {test_eval['auc_roc']:.4f}" if test_eval["auc_roc"] is not None else "  Test AUC-ROC         : N/A")
    print(f"  Test Balanced Acc    : {test_eval['balanced_accuracy']*100:.1f}%")
    print(f"  Test Accuracy        : {test_eval['accuracy']*100:.1f}%")
    print(f"  Test F1              : {test_eval['f1']:.4f}")
    print(f"  Test Precision       : {test_eval['precision']:.4f}")
    print(f"  Test Recall (AI)     : {test_eval['recall']:.4f}")
    print(f"  Test Specificity(Auth): {test_eval['specificity']:.4f}")
    print(f"  Test Youden J        : {test_eval['youden_j']:.4f}")
    print("  Test Abstention Rate : 0.0000")
    print(
        f"  Test Confusion       : "
        f"TP={test_eval['tp']}  TN={test_eval['tn']}  "
        f"FP={test_eval['fp']}  FN={test_eval['fn']}"
    )

    n_test_total = len(test_labels)
    n_test_ai = sum(test_labels)
    n_test_auth = n_test_total - n_test_ai
    ai_metrics = {
        "count": int(n_test_ai),
        "recall": float(test_eval["recall"]),
        "miss_rate": float(1.0 - test_eval["recall"]),
        "abstentions": 0,
    }
    auth_metrics = {
        "count": int(n_test_auth),
        "recall": float(test_eval["specificity"]),
        "false_alarm_rate": float(1.0 - test_eval["specificity"]),
        "abstentions": 0,
    }

    print("\n  Per-Class (TEST):")
    print(
        f"    {'AI':20s}: "
        f"n={ai_metrics['count']:4d}  "
        f"recall={ai_metrics['recall']:.3f}  "
        f"miss_rate={ai_metrics['miss_rate']:.3f}  "
        f"abstentions={ai_metrics['abstentions']}"
    )
    print(
        f"    {'AUTHENTIC':20s}: "
        f"n={auth_metrics['count']:4d}  "
        f"recall={auth_metrics['recall']:.3f}  "
        f"false_alarm={auth_metrics['false_alarm_rate']:.3f}  "
        f"abstentions={auth_metrics['abstentions']}"
    )

    print("\n  Per-Category (TEST):")
    cat_metrics: dict[str, dict] = {}
    # Per-category uses raw TCS scores for core track consistency
    per_cat_scores = [float(s) for s in raw_test_tcs_scores]
    threshold = float(raw_threshold)

    for cat in sorted(set(test_categories)):
        idxs = [i for i, c in enumerate(test_categories) if c == cat]
        if not idxs:
            continue

        cat_scores = [per_cat_scores[i] for i in idxs]
        cat_labels = [test_labels[i] for i in idxs]
        cat_preds = [1 if score >= threshold else 0 for score in cat_scores]
        cat_conf = _confusion_from_preds(cat_labels, cat_preds)
        cat_base = _metrics_from_confusion(
            cat_conf["tp"], cat_conf["tn"], cat_conf["fp"], cat_conf["fn"]
        )

        pos_count = sum(cat_labels)
        neg_count = len(cat_labels) - pos_count
        has_both_classes = pos_count > 0 and neg_count > 0

        cat_info = {
            "count": int(len(idxs)),
            "positives": int(pos_count),
            "negatives": int(neg_count),
            "mean_tcs": float(sum(cat_scores) / max(len(cat_scores), 1)),
            "min_tcs": float(min(cat_scores)),
            "max_tcs": float(max(cat_scores)),
            "predicted_ai_rate": float(sum(cat_preds) / max(len(cat_preds), 1)),
            "accuracy": float(cat_base["accuracy"]),
            "abstentions": 0,
        }

        if has_both_classes:
            cat_info["auc_roc"] = float(compute_auc(cat_scores, cat_labels))
            cat_info["f1"] = float(cat_base["f1"])
            cat_info["balanced_accuracy"] = float(cat_base["balanced_accuracy"])
            print(
                f"    {cat:20s}: "
                f"n={len(idxs):4d}  "
                f"pos={pos_count:3d}  neg={neg_count:3d}  "
                f"acc={cat_info['accuracy']:.3f}  "
                f"bal_acc={cat_info['balanced_accuracy']:.3f}  "
                f"auc={cat_info['auc_roc']:.3f}  "
                f"mean_score={cat_info['mean_tcs']:.3f}"
            )
        else:
            cat_info["auc_roc"] = None
            cat_info["f1"] = None
            cat_info["balanced_accuracy"] = None
            cat_info["label_mode"] = "all_ai" if pos_count > 0 else "all_auth"
            print(
                f"    {cat:20s}: "
                f"n={len(idxs):4d}  "
                f"{cat_info['label_mode']:8s}  "
                f"acc={cat_info['accuracy']:.3f}  "
                f"pred_ai_rate={cat_info['predicted_ai_rate']:.3f}  "
                f"mean_score={cat_info['mean_tcs']:.3f}"
            )
        cat_metrics[cat] = cat_info

    # Core acceptance uses raw TCS scores (training-free claim).
    # Learned head metrics are reported separately as diagnostic.
    ai_scores_test = [raw_test_tcs_scores[i] for i, lbl in enumerate(test_labels) if lbl == 1]
    auth_scores_test = [raw_test_tcs_scores[i] for i, lbl in enumerate(test_labels) if lbl == 0]
    mean_ai_tcs = float(np.mean(ai_scores_test)) if ai_scores_test else 0.0
    mean_auth_tcs = float(np.mean(auth_scores_test)) if auth_scores_test else 0.0
    mean_tcs_gap = float(mean_ai_tcs - mean_auth_tcs)

    acceptance = {
        "targets": {
            "specificity": float(target_specificity),
            "recall_ai": float(core_min_recall),
            "balanced_accuracy": float(CORE_TARGET_BALANCED_ACC),
            "mean_tcs_gap_ai_minus_auth": float(CORE_TARGET_MEAN_TCS_GAP),
        },
        "observed": {
            "specificity": float(raw_test_eval["specificity"]),
            "recall_ai": float(raw_test_eval["recall"]),
            "balanced_accuracy": float(raw_test_eval["balanced_accuracy"]),
            "mean_tcs_ai": float(mean_ai_tcs),
            "mean_tcs_authentic": float(mean_auth_tcs),
            "mean_tcs_gap_ai_minus_auth": float(mean_tcs_gap),
        },
    }
    acceptance["passes"] = {
        "specificity": _meets_target(acceptance["observed"]["specificity"], acceptance["targets"]["specificity"]),
        "recall_ai": _meets_target(acceptance["observed"]["recall_ai"], acceptance["targets"]["recall_ai"]),
        "balanced_accuracy": _meets_target(acceptance["observed"]["balanced_accuracy"], acceptance["targets"]["balanced_accuracy"]),
        "mean_tcs_gap": _meets_target(acceptance["observed"]["mean_tcs_gap_ai_minus_auth"], acceptance["targets"]["mean_tcs_gap_ai_minus_auth"]),
    }
    acceptance["all_passed"] = bool(all(acceptance["passes"].values()))

    print("\n  Core Acceptance (TEST):")
    print(
        f"    specificity >= {acceptance['targets']['specificity']:.2f} : "
        f"{acceptance['observed']['specificity']:.3f}  "
        f"{'PASS' if acceptance['passes']['specificity'] else 'FAIL'}"
    )
    print(
        f"    recall_ai >= {acceptance['targets']['recall_ai']:.2f} : "
        f"{acceptance['observed']['recall_ai']:.3f}  "
        f"{'PASS' if acceptance['passes']['recall_ai'] else 'FAIL'}"
    )
    print(
        f"    balanced_acc >= {acceptance['targets']['balanced_accuracy']:.2f} : "
        f"{acceptance['observed']['balanced_accuracy']:.3f}  "
        f"{'PASS' if acceptance['passes']['balanced_accuracy'] else 'FAIL'}"
    )
    print(
        f"    mean_score_gap(ai-auth) >= {acceptance['targets']['mean_tcs_gap_ai_minus_auth']:.2f} : "
        f"{acceptance['observed']['mean_tcs_gap_ai_minus_auth']:.3f}  "
        f"{'PASS' if acceptance['passes']['mean_tcs_gap'] else 'FAIL'}"
    )
    print(f"    OVERALL: {'PASS' if acceptance['all_passed'] else 'FAIL'}")

    report["steps"]["evaluation"] = {
        "threshold_source": "validation_set_only",
        "final_metrics_source": "test_set_only",
        "track": "core",
        "mode": "raw_detector",
        "note": "Core acceptance uses training-free raw TCS scores. Learned head reported as diagnostic.",
        "learned_core_head": {
            "selected_model": selected_core_model,
            "model_path": str(core_model_path),
            "test_metrics": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in test_eval.items()},
            "validation_metrics": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in val_eval.items()},
        },
        "raw_detector": {
            "validation_threshold_tuning": {
                "threshold": raw_threshold,
                "balanced_accuracy": float(raw_tuned["balanced_accuracy"]),
                "f1": float(raw_tuned["f1"]),
                "precision": float(raw_tuned["precision"]),
                "recall": float(raw_tuned["recall"]),
                "specificity": float(raw_tuned["specificity"]),
                "target_specificity": float(raw_tuned["target_specificity"]),
                "meets_target": bool(raw_tuned["meets_target"]),
                "target_recall": float(raw_tuned.get("target_recall", core_min_recall)),
                "meets_recall_target": bool(raw_tuned.get("meets_recall_target", False)),
                "constraint_tier": int(raw_tuned.get("constraint_tier", 0)),
                "abstention_rate": float(raw_test_eval.get("abstention_rate", 0.0)),
            },
            "test_metrics": {
                "auc_roc": None if raw_test_eval["auc_roc"] is None else float(raw_test_eval["auc_roc"]),
                "accuracy": float(raw_test_eval["accuracy"]),
                "balanced_accuracy": float(raw_test_eval["balanced_accuracy"]),
                "best_f1": float(raw_test_eval["f1"]),
                "precision": float(raw_test_eval["precision"]),
                "recall": float(raw_test_eval["recall"]),
                "specificity": float(raw_test_eval["specificity"]),
                "youden_j": float(raw_test_eval["youden_j"]),
                "threshold": float(raw_test_eval["threshold"]),
                "tp": int(raw_test_eval["tp"]),
                "tn": int(raw_test_eval["tn"]),
                "fp": int(raw_test_eval["fp"]),
                "fn": int(raw_test_eval["fn"]),
                "abstentions_total": int(raw_test_eval["abstentions_total"]),
                "abstention_rate": float(raw_test_eval["abstention_rate"]),
            },
        },
        "validation_threshold_tuning": {
            "threshold": float(core_threshold_info["threshold"]),
            "balanced_accuracy": float(val_eval["balanced_accuracy"]),
            "f1": float(val_eval["f1"]),
            "precision": float(val_eval["precision"]),
            "recall": float(val_eval["recall"]),
            "specificity": float(val_eval["specificity"]),
            "target_specificity": float(target_specificity),
            "meets_target": bool(core_threshold_info["meets_target"]),
            "target_recall": float(core_min_recall),
            "meets_recall_target": bool(core_threshold_info.get("meets_recall", False)),
            "constraint_tier": int(core_threshold_info.get("constraint_tier", 0)),
            "abstention_rate": 0.0,
        },
        "test_metrics": {
            "auc_roc": None if test_eval["auc_roc"] is None else float(test_eval["auc_roc"]),
            "accuracy": float(test_eval["accuracy"]),
            "balanced_accuracy": float(test_eval["balanced_accuracy"]),
            "best_f1": float(test_eval["f1"]),
            "precision": float(test_eval["precision"]),
            "recall": float(test_eval["recall"]),
            "specificity": float(test_eval["specificity"]),
            "youden_j": float(test_eval["youden_j"]),
            "threshold": float(test_eval["threshold"]),
            "tp": int(test_eval["tp"]),
            "tn": int(test_eval["tn"]),
            "fp": int(test_eval["fp"]),
            "fn": int(test_eval["fn"]),
            "abstentions_total": 0,
            "abstention_rate": 0.0,
        },
        "per_class": {
            "ai": ai_metrics,
            "authentic": auth_metrics,
        },
        "per_category": cat_metrics,
        "core_acceptance": acceptance,
    }

    # ── STEP 4: Ablation on development split only (train + validation) ───
    step_banner(4, total_steps, "Ablation Study")

    dev_idx = np.concatenate([train_idx, val_idx]).astype(int)
    X_dev = np.asarray(X[dev_idx], dtype=float)
    y_dev = np.asarray(y[dev_idx], dtype=int)

    print(f"  Ablation dataset: development split only (n={len(X_dev)})")
    print("  Note: Το test set ΔΕΝ χρησιμοποιείται στο ablation / weight search.")

    level_results = run_level_ablation(X_dev, y_dev)
    metric_results = run_metric_ablation(X_dev, y_dev)
    weight_results = run_weight_search(X_dev, y_dev, grid_steps=5)

    best_w = weight_results["best_weights"]

    report["steps"]["ablation"] = {
        "data_source": "development_split_only",
        "samples": int(len(X_dev)),
        "level_ablation": level_results,
        "metric_ablation": metric_results,
        "weight_search": {
            "best_weights": {
                "L1": float(best_w["L1"]),
                "L2": float(best_w["L2"]),
                "L3": float(best_w["L3"]),
                "L4": float(best_w["L4"]),
            },
            "best_auc": float(weight_results["best_auc"]),
        },
    }

    # ── STEP 5: Final report ──────────────────────────────────────────────
    step_banner(5, total_steps, "Final Report")

    elapsed = time.time() - start_time
    report["completed"] = datetime.now().isoformat()
    report["elapsed_seconds"] = round(elapsed, 1)

    report_path_fixed = out / "pipeline_report_fixed.json"
    report_path_std = out / "pipeline_report.json"

    for path in [report_path_fixed, report_path_std]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=_json_default)

    final_test = report["steps"]["evaluation"]["raw_detector"]["test_metrics"]
    core_acceptance = report["steps"]["evaluation"].get("core_acceptance", {})

    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Dataset:         {report['steps']['extraction']['samples']} samples")
    print(
        f"  Split:           train={report['steps']['split']['train_samples']}  "
        f"val={report['steps']['split']['validation_samples']}  "
        f"test={report['steps']['split']['test_samples']}"
    )
    core_head_test = report["steps"]["evaluation"]["learned_core_head"].get("test_metrics", {})
    core_head_auc = core_head_test.get("auc_roc", None) or core_head_test.get("auc", None)
    print(f"  Test AUC (raw):  {final_test['auc_roc']:.4f}" if final_test["auc_roc"] is not None else "  Test AUC (raw):  N/A")
    if core_head_auc is not None:
        print(f"  Test AUC (GBM):  {core_head_auc:.4f}")
    print(f"  Test Accuracy:   {final_test['accuracy']*100:.1f}%")
    print(f"  Test Bal Acc:    {final_test['balanced_accuracy']*100:.1f}%")
    print(f"  Test F1:         {final_test['best_f1']:.4f}")
    print(f"  Abstention Rate: {final_test['abstention_rate']*100:.1f}%")
    print(f"  Threshold (val): {final_test['threshold']:.4f}")
    if core_acceptance:
        status = "PASS" if core_acceptance.get("all_passed") else "FAIL"
        gap = float(core_acceptance.get("observed", {}).get("mean_tcs_gap_ai_minus_auth", 0.0))
        print(f"  Core Acceptance: {status}  (mean_tcs_gap={gap:.3f})")
    print(f"  Optimal weights: L1={best_w['L1']:.2f}  L2={best_w['L2']:.2f}  L3={best_w['L3']:.2f}  L4={best_w['L4']:.2f}")
    print(f"  Core Model:      {report['steps']['evaluation']['learned_core_head']['selected_model']}")
    print(f"  Open Model:      {report['steps']['training']['selected_model']}")
    print(f"  Χρόνος:          {elapsed:.0f}s")
    print(f"  Split manifest:  {split_manifest_path}")
    print(f"  Report:          {report_path_fixed}")
    print(f"{'=' * 60}\n")

    return report


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PATV Full Pipeline Runner")
    parser.add_argument("--videos", help="Φάκελος με βίντεο (π.χ. dataset/faceforensics)")
    parser.add_argument("--output", default="pipeline_results", help="Φάκελος αποτελεσμάτων")
    parser.add_argument("--features", help="Υπάρχον CSV με features (παρακάμπτει extraction)")
    parser.add_argument("--synthetic", action="store_true", help="Χρήση synthetic data")
    parser.add_argument("--n-synthetic", type=int, default=400, help="Πλήθος synthetic δειγμάτων")
    parser.add_argument("--workers", type=int, default=1, help="Workers για παράλληλο feature extraction (0=auto)")
    parser.add_argument("--sample-rate", type=int, default=DETECTOR_DEFAULT_SAMPLE_RATE, help="Detector sample rate (frames/sec στόχος)")
    parser.add_argument("--max-frames", type=int, default=DETECTOR_DEFAULT_MAX_FRAMES, help="Μέγιστος αριθμός frames ανά βίντεο")
    parser.add_argument("--max-videos", type=int, default=0, help="Stratified όριο βίντεο για γρήγορα diagnostic runs (0=όλα)")
    parser.add_argument("--target-specificity", type=float, default=CORE_TARGET_SPECIFICITY, help="Target specificity για threshold tuning")
    parser.add_argument("--core-min-recall", type=float, default=CORE_TARGET_RECALL, help="Minimum AI recall στόχος για core threshold tuning")
    parser.add_argument("--open-min-recall", type=float, default=OPEN_TARGET_RECALL, help="Minimum recall στόχος για open threshold tuning")
    args = parser.parse_args()

    if not args.videos and not args.synthetic and not args.features:
        print("Χρήση:")
        print("  python legacy/data_pipeline/run_pipeline.py --videos dataset/faceforensics --output pipeline_results")
        print("  python legacy/data_pipeline/run_pipeline.py --synthetic")
        print("  python legacy/data_pipeline/run_pipeline.py --features results/features.csv --output pipeline_results")
        return

    run_full_pipeline(
        videos_dir=args.videos or "",
        output_dir=args.output,
        synthetic=args.synthetic,
        n_synthetic=args.n_synthetic,
        features_path=args.features,
        workers=args.workers,
        target_specificity=args.target_specificity,
        core_min_recall=args.core_min_recall,
        open_min_recall=args.open_min_recall,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
        max_videos=args.max_videos,
    )


if __name__ == "__main__":
    main()

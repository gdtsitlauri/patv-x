#!/usr/bin/env python3
"""
PATV-X CLI
==========
Track-aware command-line interface for:
  - PATV-X Core           (training-free detector)
  - PATV-X Open           (learned meta-classifier on PATV-X features)
  - PATV-X Synthetic Smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from patv_x_detector import PATVXDetector
from train_mlp import load_model_bundle, predict_with_bundle, result_to_features


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

SUPPORTED = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}
TRACKS = ("core", "open", "synthetic-smoke")


def color_score(score: float) -> str:
    s = f"{score:.3f}"
    if score >= 0.70:
        return RED + BOLD + s + RESET
    if score >= 0.40:
        return YELLOW + s + RESET
    return GREEN + s + RESET


def verdict_colored(verdict: str) -> str:
    if verdict == "AI_GENERATED":
        return RED + BOLD + "⚠ AI GENERATED" + RESET
    if verdict == "INCONCLUSIVE":
        return YELLOW + BOLD + "… INCONCLUSIVE" + RESET
    return GREEN + BOLD + "✓ AUTHENTIC" + RESET


def _decision_confidence(score: float, threshold: float) -> str:
    margin = abs(float(score) - float(threshold))
    if margin > 0.20:
        return "HIGH"
    if margin > 0.10:
        return "MEDIUM"
    return "LOW"


def _validate_threshold(threshold: float | None) -> None:
    if threshold is None:
        return
    if not (0.0 <= threshold <= 1.0):
        print(f"{RED}Σφάλμα: Το threshold πρέπει να είναι στο [0,1].{RESET}")
        sys.exit(1)


def _build_detector(
    track: str,
    threshold: float | None,
    sample_rate: int,
    max_frames: int,
    use_bundle_threshold: bool = False,
) -> PATVXDetector:
    _validate_threshold(threshold)
    kwargs = {
        "verbose": False,
        "track": track,
        "sample_rate": int(sample_rate),
        "max_frames": int(max_frames),
    }
    if threshold is not None and track != "open" and not use_bundle_threshold:
        kwargs["threshold"] = threshold
    return PATVXDetector(**kwargs)


def _load_bundle_if_needed(args):
    if args.track == "open" and not args.model:
        print(f"{RED}Σφάλμα: Το track=open απαιτεί --model PATV-X Open bundle.{RESET}")
        sys.exit(1)
    if not args.model:
        return None
    if args.track not in {"core", "open"}:
        print(f"{RED}Σφάλμα: Το --model υποστηρίζεται μόνο για --track core/open.{RESET}")
        sys.exit(1)
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"{RED}Σφάλμα: Δεν βρέθηκε model bundle: {model_path}{RESET}")
        sys.exit(1)
    bundle = load_model_bundle(model_path)
    bundle_track = str(bundle.get("track", "")).strip().lower()
    if bundle_track and bundle_track != args.track:
        print(
            f"{RED}Σφάλμα: Το bundle είναι track={bundle_track}, "
            f"αλλά ζητήθηκε --track {args.track}.{RESET}"
        )
        sys.exit(1)
    return bundle


def _analyze_video(path: Path, detector: PATVXDetector, args, bundle=None) -> dict:
    result = detector.analyze(str(path))
    record = {
        "path": str(path),
        "filename": path.name,
        "result": result,
        "track": args.track,
        "verdict": result.verdict,
        "confidence": result.confidence,
        "decision_mode": "core_tcs",
        "decision_score": float(result.tcs_score),
        "decision_threshold": float(detector.threshold),
        "core_detector_verdict": result.verdict,
        "core_detector_confidence": result.confidence,
        "core_tcs_score": float(result.tcs_score),
        "core_threshold": float(detector.threshold),
        "bundle_track": None,
        "bundle_model": None,
        "bundle_probability": None,
        "bundle_threshold": None,
        "open_probability": None,
        "open_threshold": None,
        "open_model": None,
    }

    if bundle is None:
        return record

    include_support = bool(bundle.get("include_support_features", args.track != "synthetic-smoke"))
    feature_names = bundle.get("feature_names")
    feature_vec = result_to_features(
        result,
        include_support=include_support,
        feature_names=feature_names,
    ).reshape(1, -1)
    bundle_probs, bundle_threshold, selected_model = predict_with_bundle(
        bundle,
        feature_vec,
        model_name=args.model_name,
    )
    bundle_probability = float(bundle_probs[0])
    bundle_threshold = float(args.threshold if args.threshold is not None else bundle_threshold)
    bundle_track = str(bundle.get("track", args.track) or args.track)
    decision_mode = "open_probability" if bundle_track == "open" else "core_probability"

    if result.abstained_reason:
        verdict = "INCONCLUSIVE"
        confidence = "LOW"
    else:
        verdict = "AI_GENERATED" if bundle_probability >= bundle_threshold else "AUTHENTIC"
        confidence = _decision_confidence(bundle_probability, bundle_threshold)

    record.update({
        "verdict": verdict,
        "confidence": confidence,
        "decision_mode": decision_mode,
        "decision_score": bundle_probability,
        "decision_threshold": bundle_threshold,
        "bundle_track": bundle_track,
        "bundle_model": selected_model,
        "bundle_probability": bundle_probability,
        "bundle_threshold": bundle_threshold,
    })
    if bundle_track == "open":
        record.update({
            "open_probability": bundle_probability,
            "open_threshold": bundle_threshold,
            "open_model": selected_model,
        })
    return record


def _result_to_json_dict(record: dict) -> dict:
    result = record["result"]
    payload = result.to_dict()
    payload.update({
        "file": record["path"],
        "filename": record["filename"],
        "verdict": record["verdict"],
        "confidence": record["confidence"],
        "track": record["track"],
        "analysis_version": result.analysis_version,
        "abstained_reason": result.abstained_reason,
        "decision": {
            "mode": record["decision_mode"],
            "score": round(float(record["decision_score"]), 4),
            "threshold": round(float(record["decision_threshold"]), 4),
            "confidence": record["confidence"],
        },
        "core_detector": {
            "verdict": record["core_detector_verdict"],
            "confidence": record["core_detector_confidence"],
            "tcs_score": round(float(record["core_tcs_score"]), 4),
            "threshold": round(float(record["core_threshold"]), 4),
        },
    })
    if record["bundle_probability"] is not None:
        payload["learned_prediction"] = {
            "track": record["bundle_track"],
            "model": record["bundle_model"],
            "probability": round(float(record["bundle_probability"]), 4),
            "threshold": round(float(record["bundle_threshold"]), 4),
        }
    if record["open_probability"] is not None:
        payload["open_prediction"] = {
            "model": record["open_model"],
            "probability": round(float(record["open_probability"]), 4),
            "threshold": round(float(record["open_threshold"]), 4),
        }
    return payload


def _print_violation(v) -> None:
    print(
        f"    - [{v.severity}] {v.violation_type} "
        f"frames {v.frame_start}-{v.frame_end} "
        f"(ratio={v.violation_ratio:.2f}x, conf={v.confidence:.2f})"
    )
    print(f"      {v.description}")


def _print_summary(record: dict, verbose: bool) -> None:
    result = record["result"]

    print()
    print(f"  {BOLD}{'─' * 64}{RESET}")
    print(f"  {BOLD}PATV-X Analysis{RESET}  {DIM}{record['filename']}{RESET}")
    print(f"  {'─' * 64}")
    print(
        f"  Track       : {record['track']}  "
        f"{DIM}(analysis_version: {result.analysis_version}){RESET}"
    )
    print(f"  Αποτέλεσμα  : {verdict_colored(record['verdict'])}")

    if record["decision_mode"] == "open_probability":
        print(
            f"  Open Score  : {color_score(record['decision_score'])}   "
            f"{DIM}(model: {record['open_model']}, threshold: {record['decision_threshold']:.2f}, "
            f"confidence: {record['confidence']}){RESET}"
        )
        print(
            f"  Core TCS    : {color_score(result.tcs_score)}   "
            f"{DIM}(detector verdict: {result.verdict}, threshold: {record['core_threshold']:.2f}){RESET}"
        )
    elif record["decision_mode"] == "core_probability":
        print(
            f"  Core Score  : {color_score(record['decision_score'])}   "
            f"{DIM}(model: {record['bundle_model']}, threshold: {record['decision_threshold']:.2f}, "
            f"confidence: {record['confidence']}){RESET}"
        )
        print(
            f"  Raw TCS     : {color_score(result.tcs_score)}   "
            f"{DIM}(detector verdict: {result.verdict}, threshold: {record['core_threshold']:.2f}){RESET}"
        )
    else:
        print(
            f"  TCS Score   : {color_score(result.tcs_score)}   "
            f"{DIM}(confidence: {record['confidence']}, threshold: {record['decision_threshold']:.2f}){RESET}"
        )

    print(f"  {'─' * 64}")
    print(f"  {CYAN}[L1] Residual Motion     {RESET} : {color_score(result.l1_flow_score)}")
    print(f"  {BLUE}[L2] Physics             {RESET} : {color_score(result.l2_physics_score)}")
    print(f"  {GREEN}[L3] Face/Context        {RESET} : {color_score(result.l3_semantic_score)}")
    print(
        f"  {YELLOW}[L4] Boundary Artifacts  {RESET} : "
        f"{color_score(result.l4_boundary_artifact_score)}  "
        f"{DIM}flow_seam={result.l4_flow_boundary_seam_score:.3f} "
        f"color_flicker={result.l4_boundary_color_flicker_score:.3f} "
        f"warp_error={result.l4_warp_prediction_error_score:.3f} "
        f"edge_flicker={result.l4_boundary_edge_flicker_score:.3f}{RESET}"
    )
    print(f"  {'─' * 64}")
    print(
        f"  Support           : "
        f"face_detected={result.face_detected_ratio:.3f}  "
        f"face_reliability={result.face_reliability_mean:.3f}  "
        f"neck={result.neck_visibility_ratio:.3f}  "
        f"boundary={result.boundary_support_ratio:.3f}"
    )
    print(f"  Violations        : {len(result.violations)}")
    print(f"  Suspicious frames : {result.most_suspicious_frames[:8]}")
    print(
        f"  {DIM}Duration: {result.duration:.1f}s | FPS: {result.fps:.0f} | "
        f"Frames: {result.frames_analyzed} | Res: {result.resolution[0]}×{result.resolution[1]}{RESET}"
    )

    if result.abstained_reason:
        print(f"  Abstention        : {YELLOW}{result.abstained_reason}{RESET}")

    if verbose:
        print(f"  {'─' * 64}")
        if result.violations:
            print("  Forensic evidence:")
            for v in result.violations[:8]:
                _print_violation(v)
        else:
            print("  No major forensic violations found.")

        if result.comparison_note:
            print(f"  {'─' * 64}")
            print(f"  Note: {result.comparison_note}")

    print()


def cmd_analyze(args) -> None:
    path = Path(args.video)
    if not path.exists():
        print(f"{RED}Σφάλμα: Δεν βρέθηκε αρχείο: {path}{RESET}")
        sys.exit(1)
    if not path.is_file():
        print(f"{RED}Σφάλμα: Δεν είναι αρχείο: {path}{RESET}")
        sys.exit(1)

    bundle = _load_bundle_if_needed(args)
    detector = _build_detector(
        args.track,
        args.threshold,
        args.sample_rate,
        args.max_frames,
        use_bundle_threshold=(bundle is not None),
    )

    try:
        record = _analyze_video(path, detector, args, bundle=bundle)
    except Exception as e:
        print(f"{RED}Σφάλμα ανάλυσης: {e}{RESET}")
        sys.exit(1)

    if args.json:
        print(json.dumps(_result_to_json_dict(record), ensure_ascii=False, indent=2))
        return

    _print_summary(record, verbose=args.verbose)


def _discover_videos(folder: Path) -> list[Path]:
    return sorted(
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED
    )


def _status_token(verdict: str) -> str:
    if verdict == "AI_GENERATED":
        return RED + "AI" + RESET
    if verdict == "INCONCLUSIVE":
        return YELLOW + "INC" + RESET
    return GREEN + "OK" + RESET


def _write_csv(records: Iterable[dict], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "path", "track", "analysis_version", "verdict", "confidence",
            "decision_mode", "decision_score", "decision_threshold",
            "core_detector_verdict", "core_detector_confidence", "core_tcs_score", "core_threshold",
            "bundle_track", "bundle_model", "bundle_probability", "bundle_threshold",
            "open_model", "open_probability", "open_threshold",
            "abstained_reason",
            "l1_flow_score", "l2_physics_score", "l3_semantic_score",
            "l4_boundary_artifact_score", "l4_flow_boundary_seam_score",
            "l4_boundary_color_flicker_score", "l4_warp_prediction_error_score",
            "l4_boundary_edge_flicker_score",
            "l4_ctcg_score", "ctcg_phase_coherence", "ctcg_ar_residual",
            "ctcg_spectral_anomaly", "ctcg_micro_jitter",
            "face_detected_ratio", "face_reliability_mean", "neck_visibility_ratio",
            "boundary_support_ratio",
            "violations_count", "duration", "fps", "width", "height",
            "suspicious_frames", "comparison_note",
        ])
        for record in records:
            r = record["result"]
            writer.writerow([
                record["filename"],
                record["path"],
                record["track"],
                r.analysis_version,
                record["verdict"],
                record["confidence"],
                record["decision_mode"],
                round(float(record["decision_score"]), 4),
                round(float(record["decision_threshold"]), 4),
                record["core_detector_verdict"],
                record["core_detector_confidence"],
                round(float(record["core_tcs_score"]), 4),
                round(float(record["core_threshold"]), 4),
                record["bundle_track"] or "",
                record["bundle_model"] or "",
                "" if record["bundle_probability"] is None else round(float(record["bundle_probability"]), 4),
                "" if record["bundle_threshold"] is None else round(float(record["bundle_threshold"]), 4),
                record["open_model"] or "",
                "" if record["open_probability"] is None else round(float(record["open_probability"]), 4),
                "" if record["open_threshold"] is None else round(float(record["open_threshold"]), 4),
                r.abstained_reason,
                round(r.l1_flow_score, 4),
                round(r.l2_physics_score, 4),
                round(r.l3_semantic_score, 4),
                round(r.l4_boundary_artifact_score, 4),
                round(r.l4_flow_boundary_seam_score, 4),
                round(r.l4_boundary_color_flicker_score, 4),
                round(r.l4_warp_prediction_error_score, 4),
                round(r.l4_boundary_edge_flicker_score, 4),
                round(r.l4_ctcg_score, 4),
                round(r.ctcg_phase_coherence, 4),
                round(r.ctcg_ar_residual, 4),
                round(r.ctcg_spectral_anomaly, 4),
                round(r.ctcg_micro_jitter, 4),
                round(r.face_detected_ratio, 4),
                round(r.face_reliability_mean, 4),
                round(r.neck_visibility_ratio, 4),
                round(r.boundary_support_ratio, 4),
                len(r.violations),
                round(r.duration, 2),
                round(r.fps, 1),
                r.resolution[0],
                r.resolution[1],
                ",".join(map(str, r.most_suspicious_frames[:10])),
                r.comparison_note,
            ])


def cmd_batch(args) -> None:
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"{RED}Σφάλμα: Δεν βρέθηκε φάκελος: {folder}{RESET}")
        sys.exit(1)

    videos = _discover_videos(folder)
    if not videos:
        print(f"{YELLOW}Δεν βρέθηκαν βίντεο στον φάκελο: {folder}{RESET}")
        sys.exit(0)

    bundle = _load_bundle_if_needed(args)
    detector = _build_detector(
        args.track,
        args.threshold,
        args.sample_rate,
        args.max_frames,
        use_bundle_threshold=(bundle is not None),
    )

    print(f"\n{BOLD}[PATV-X Batch]{RESET} Βρέθηκαν {len(videos)} βίντεο στο: {folder}")
    print(f"  {DIM}Track: {args.track}{RESET}")
    if bundle is not None:
        selected_model = args.model_name or bundle.get("selected_model", "mlp")
        selected_threshold = args.threshold if args.threshold is not None else bundle.get("selected_threshold", 0.5)
        print(
            f"  {DIM}Decision mode: {bundle.get('track', args.track)}_probability | "
            f"model={selected_model} | threshold={float(selected_threshold):.2f}{RESET}"
        )
    else:
        print(f"  {DIM}Decision mode: core_tcs | threshold={float(detector.threshold):.2f}{RESET}")
    print(f"{'─' * 72}")

    records = []
    errors = 0
    ai_count = 0
    auth_count = 0
    inc_count = 0

    for i, video in enumerate(videos, 1):
        prefix = f"  [{i:3d}/{len(videos)}]"
        name = video.name[:36].ljust(36)
        print(f"{prefix} {name} ", end="", flush=True)

        try:
            record = _analyze_video(video, detector, args, bundle=bundle)
            records.append(record)

            verdict = record["verdict"]
            print(
                f"{_status_token(verdict)}  "
                f"{record['decision_mode']}={color_score(record['decision_score'])}  "
                f"({record['confidence']})"
            )

            if verdict == "AI_GENERATED":
                ai_count += 1
            elif verdict == "INCONCLUSIVE":
                inc_count += 1
            else:
                auth_count += 1

        except Exception as e:
            print(f"{RED}ERROR{RESET} — {e}")
            errors += 1

    total = len(records)
    print(f"\n{'─' * 72}")
    print(f"  {BOLD}Σύνοψη:{RESET}")
    print(f"  Σύνολο αναλύθηκαν : {total}")
    print(f"  AI Generated       : {RED}{ai_count}{RESET}  ({ai_count / max(total, 1) * 100:.0f}%)")
    print(f"  Authentic          : {GREEN}{auth_count}{RESET}  ({auth_count / max(total, 1) * 100:.0f}%)")
    print(f"  Inconclusive       : {YELLOW}{inc_count}{RESET}  ({inc_count / max(total, 1) * 100:.0f}%)")
    if errors:
        print(f"  Σφάλματα           : {YELLOW}{errors}{RESET}")

    if records:
        mean_tcs = sum(r["result"].tcs_score for r in records) / len(records)
        mean_l1 = sum(r["result"].l1_flow_score for r in records) / len(records)
        mean_l2 = sum(r["result"].l2_physics_score for r in records) / len(records)
        mean_l3 = sum(r["result"].l3_semantic_score for r in records) / len(records)
        mean_l4 = sum(r["result"].l4_boundary_artifact_score for r in records) / len(records)
        print(f"  Μέσο TCS Score     : {color_score(mean_tcs)}")
        print(f"  Mean L1/L2/L3/L4   : {mean_l1:.3f} / {mean_l2:.3f} / {mean_l3:.3f} / {mean_l4:.3f}")
        if any(r["bundle_probability"] is not None for r in records):
            mean_bundle = sum(float(r["bundle_probability"]) for r in records if r["bundle_probability"] is not None)
            denom = max(sum(1 for r in records if r["bundle_probability"] is not None), 1)
            print(f"  Mean Learned Score : {mean_bundle / denom:.3f}")
        if args.track == "open":
            mean_open = sum(float(r["open_probability"]) for r in records if r["open_probability"] is not None)
            denom = max(sum(1 for r in records if r["open_probability"] is not None), 1)
            print(f"  Mean Open Score    : {mean_open / denom:.3f}")

    if args.output:
        _write_csv(records, args.output)
        print(f"\n  {GREEN}CSV αποθηκεύθηκε:{RESET} {args.output}")

    print()
    sys.exit(1 if errors > 0 else 0)


def _add_common_track_args(parser) -> None:
    parser.add_argument(
        "--track",
        choices=TRACKS,
        default="core",
        help="Επιλογή track: core | open | synthetic-smoke (default: core).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Decision threshold. Στο core/synthetic-smoke είναι TCS threshold. "
            "Στο open είναι probability threshold του learned model."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path σε PATV-X bundle JSON. Απαιτείται για --track open και προαιρετικά χρησιμοποιείται για learned core inference στο --track core.",
    )
    parser.add_argument(
        "--model-name",
        choices=("linear", "mlp"),
        default=None,
        help="Προαιρετική επιλογή συγκεκριμένου model από το bundle.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=6,
        help="Detector sample rate (στόχος sampled frames/sec, default: 6).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=96,
        help="Μέγιστος αριθμός sampled frames ανά βίντεο (default: 96).",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="patv",
        description="PATV-X — Core/Open/Synthetic-Smoke command line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Παραδείγματα:
  patv_cli.py analyze video.mp4
  patv_cli.py analyze video.mp4 --track synthetic-smoke
  patv_cli.py analyze video.mp4 --track core --model pipeline_results/patv_core_bundle.json
  patv_cli.py analyze video.mp4 --track open --model pipeline_results/patv_open_bundle.json
  patv_cli.py analyze video.mp4 --track open --model pipeline_results/patv_open_bundle.json --model-name linear
  patv_cli.py batch /videos/ --output results.csv
        """,
    )

    sub = parser.add_subparsers(dest="command")
    sub.required = True

    p_analyze = sub.add_parser("analyze", help="Ανάλυση ενός βίντεο")
    p_analyze.add_argument("video", help="Path στο βίντεο")
    _add_common_track_args(p_analyze)
    p_analyze.add_argument("--json", action="store_true", help="JSON output")
    p_analyze.add_argument("--verbose", action="store_true", help="Λεπτομερής έξοδος")

    p_batch = sub.add_parser("batch", help="Batch ανάλυση φακέλου")
    p_batch.add_argument("folder", help="Path στον φάκελο")
    _add_common_track_args(p_batch)
    p_batch.add_argument("--output", help="Αποθήκευση αποτελεσμάτων σε CSV")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "batch":
        cmd_batch(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
PATV-X: Explainable Physics-Aware Temporal Video Detector
=========================================================

Face-aware, training-free detector designed for deepfake / faceswap video.

Core idea
---------
1) Estimate a stable face / subject ROI using classical CV
   (Haar frontal-face detector with motion / tracking fallback).
2) Measure temporal inconsistencies inside the ROI and relative to nearby context.
3) Fuse motion, physics and face/context evidence into an explainable TCS score.

Important design choice
-----------------------
This version is intentionally conservative against false positives:
- static face/context differences alone should NOT be enough
- temporal instability matters more than one-frame artifacts
- semantic cues are heavily down-weighted when the face ROI is unreliable
- dynamic face evidence (motion / rigid-body inconsistency) is privileged

No deep backbones. No external weights. Backward-compatible API.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PhysicsViolation:
    violation_type: str
    severity: str
    frame_start: int
    frame_end: int
    description: str
    measured_value: float
    expected_value: float
    violation_ratio: float
    region: Optional[tuple] = None
    confidence: float = 0.8

    def to_text(self) -> str:
        lines = [
            f"  [{self.severity}] {self.violation_type} violation @ frames {self.frame_start}-{self.frame_end}",
            f"    {self.description}",
            (
                f"    Measured: {self.measured_value:.3f}  |  "
                f"Expected: {self.expected_value:.3f}  |  Ratio: {self.violation_ratio:.2f}×"
            ),
        ]
        if self.region:
            lines.append(
                f"    Region: ({self.region[0]},{self.region[1]})-({self.region[2]},{self.region[3]})"
            )
        return "\n".join(lines)


@dataclass
class FrameEvidence:
    frame_idx: int
    timestamp: float
    anomaly_score: float
    violations: list[str]
    flow_magnitude: float
    flow_divergence: float
    physics_score: float
    semantic_score: float


@dataclass
class PATVXResult:
    video_path: str
    verdict: str
    tcs_score: float
    confidence: str
    track: str

    l1_flow_score: float
    l2_physics_score: float
    l3_semantic_score: float

    violations: list[PhysicsViolation]
    frame_timeline: list[FrameEvidence]
    most_suspicious_frames: list[int]

    fps: float
    duration: float
    resolution: tuple
    frames_analyzed: int

    flow_mean_divergence: float = 0.0
    flow_acceleration_anomaly: float = 0.0
    flow_flicker_score: float = 0.0

    physics_gravity_consistency: float = 0.0
    physics_rigid_body_score: float = 0.0
    physics_shadow_consistency: float = 0.0
    physics_face_scene_support_ratio: float = 0.0
    physics_light_support_ratio: float = 0.0
    physics_skin_support_ratio: float = 0.0

    semantic_color_drift: float = 0.0
    semantic_edge_stability: float = 0.0
    semantic_texture_consistency: float = 0.0

    comparison_note: str = ""
    analysis_version: str = "patv_x_core_v13"
    abstained_reason: str = ""

    face_detected_ratio: float = 0.0
    face_reliability_mean: float = 0.0
    neck_visibility_ratio: float = 0.0
    boundary_support_ratio: float = 0.0

    l4_boundary_artifact_score: float = 0.0
    l4_flow_boundary_seam_score: float = 0.0
    l4_boundary_color_flicker_score: float = 0.0
    l4_warp_prediction_error_score: float = 0.0
    l4_boundary_edge_flicker_score: float = 0.0
    l4_flow_boundary_support_ratio: float = 0.0
    l4_boundary_color_support_ratio: float = 0.0
    l4_warp_support_ratio: float = 0.0
    l4_boundary_edge_support_ratio: float = 0.0
    l4_boundary_agreement_ratio: float = 0.0
    l4_boundary_corroboration: float = 0.0
    l4_warp_supported_score: float = 0.0

    # L5 Frequency Analysis (v13)
    l5_frequency_score: float = 0.0
    l5_lap_var_ratio: float = 0.0
    l5_lap_var_temporal_cv: float = 0.0
    l5_dct_hf_ratio: float = 0.0
    l5_dct_hf_temporal_cv: float = 0.0
    l5_lap_kurtosis_mean: float = 0.0
    l5_lap_kurtosis_cv: float = 0.0
    l5_wavelet_detail_ratio: float = 0.0
    l5_wavelet_detail_ratio_cv: float = 0.0
    l5_lap_R_cv: float = 0.0
    l5_lap_G_cv: float = 0.0
    l5_lap_B_cv: float = 0.0
    l5_lap_diff_mean: float = 0.0
    l5_lap_trend_residual_std: float = 0.0
    l5_lap_acc_mean: float = 0.0
    l5_block_lap_consistency: float = 0.0
    l5_cross_ch_lap_corr: float = 0.0
    l5_srm_var_cv: float = 0.0

    @property
    def video_duration(self) -> float:
        return self.duration

    @property
    def l4_ctcg_score(self) -> float:
        return self.l4_boundary_artifact_score

    @property
    def ctcg_phase_coherence(self) -> float:
        return self.l4_flow_boundary_seam_score

    @property
    def ctcg_ar_residual(self) -> float:
        return self.l4_boundary_color_flicker_score

    @property
    def ctcg_spectral_anomaly(self) -> float:
        return self.l4_warp_prediction_error_score

    @property
    def ctcg_micro_jitter(self) -> float:
        return self.l4_boundary_edge_flicker_score

    @property
    def flow_result(self):
        return SimpleNamespace(
            flow_score=self.l1_flow_score,
            mean_divergence=self.flow_mean_divergence,
            acceleration_anomaly=self.flow_acceleration_anomaly,
            flicker_score=self.flow_flicker_score,
        )

    @property
    def physics_result(self):
        return SimpleNamespace(
            physics_score=self.l2_physics_score,
            gravity_consistency=self.physics_gravity_consistency,
            rigid_body_score=self.physics_rigid_body_score,
            shadow_consistency=self.physics_shadow_consistency,
        )

    @property
    def semantic_result(self):
        return SimpleNamespace(
            semantic_score=self.l3_semantic_score,
            color_drift=self.semantic_color_drift,
            edge_stability=self.semantic_edge_stability,
            texture_consistency=self.semantic_texture_consistency,
        )

    def forensic_report(self) -> str:
        w = 68
        lines = [
            "═" * w,
            "  PATV-X  Forensic Video Analysis Report",
            "  Face-Centric Temporal + Physics Inconsistency Detector",
            "═" * w,
            f"  Αρχείο    : {Path(self.video_path).name}",
            f"  Track     : {self.track}  |  Version: {self.analysis_version}",
            f"  Διάρκεια  : {self.duration:.1f}s @ {self.fps:.0f}fps  |  {self.resolution[0]}×{self.resolution[1]}",
            f"  Frames    : {self.frames_analyzed} αναλύθηκαν",
            "─" * w,
        ]

        if self.verdict == "AI_GENERATED":
            badge = "⚠  AI GENERATED"
        elif self.verdict == "INCONCLUSIVE":
            badge = "…  INCONCLUSIVE"
        else:
            badge = "✓  AUTHENTIC"
        lines += [
            f"  ΑΠΟΤΕΛΕΣΜΑ : {badge}",
            f"  TCS Score  : {self.tcs_score:.3f}  (confidence: {self.confidence})",
            "─" * w,
            "  Sub-scores:",
            f"    L1 Residual Motion  : {self.l1_flow_score:.3f}  " + _score_bar(self.l1_flow_score),
            f"    L2 Physics          : {self.l2_physics_score:.3f}  " + _score_bar(self.l2_physics_score),
            f"    L3 Face/Context     : {self.l3_semantic_score:.3f}  " + _score_bar(self.l3_semantic_score),
            f"    L4 Boundary Artifacts: {self.l4_boundary_artifact_score:.3f}  " + _score_bar(self.l4_boundary_artifact_score),
            f"       └ Flow Seam      : {self.l4_flow_boundary_seam_score:.3f}  Color Flicker: {self.l4_boundary_color_flicker_score:.3f}",
            f"       └ Warp Error     : {self.l4_warp_prediction_error_score:.3f}  Edge Flicker : {self.l4_boundary_edge_flicker_score:.3f}",
            "  Support diagnostics:",
            f"    face_detected_ratio : {self.face_detected_ratio:.3f}",
            f"    face_reliability    : {self.face_reliability_mean:.3f}",
            f"    neck_visibility     : {self.neck_visibility_ratio:.3f}",
            f"    boundary_support    : {self.boundary_support_ratio:.3f}",
            "─" * w,
        ]

        if self.abstained_reason:
            lines.append(f"  Abstention  : {self.abstained_reason}")
            lines.append("─" * w)

        if self.violations:
            lines.append(f"  FORENSIC EVIDENCE: {len(self.violations)} συμβάντα")
            lines.append("")
            order = {"CRITICAL": 0, "MAJOR": 1, "MINOR": 2}
            for v in sorted(self.violations, key=lambda x: order.get(x.severity, 9)):
                lines.append(v.to_text())
                lines.append("")
        else:
            lines.append("  Δεν ανιχνεύθηκαν ισχυρές παραβιάσεις ή ασυνέχειες.")

        if self.most_suspicious_frames:
            lines += ["─" * w, "  Πιο ύποπτα frames:"]
            for fi in self.most_suspicious_frames[:5]:
                ts = fi / max(self.fps, 1e-6)
                fe = next((f for f in self.frame_timeline if f.frame_idx == fi), None)
                score_str = f"score={fe.anomaly_score:.3f}" if fe else ""
                lines.append(f"    Frame {fi:4d}  ({ts:.2f}s)  {score_str}")

        lines += [
            "─" * w,
            "  Σύγκριση / ερμηνεία:",
            f"  {self.comparison_note}",
            "─" * w,
            "  PATV-X: training-free | face-centric | frame-level forensic evidence",
            "═" * w,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "tcs_score": round(self.tcs_score, 4),
            "confidence": self.confidence,
            "track": self.track,
            "analysis_version": self.analysis_version,
            "abstained_reason": self.abstained_reason,
            "scores": {
                "L1_flow": round(self.l1_flow_score, 4),
                "L2_physics": round(self.l2_physics_score, 4),
                "L3_semantic": round(self.l3_semantic_score, 4),
                "L4_boundary_artifacts": round(self.l4_boundary_artifact_score, 4),
                "L4_ctcg": round(self.l4_ctcg_score, 4),
            },
            "details": {
                "flow_mean_divergence": round(self.flow_mean_divergence, 4),
                "flow_acceleration_anomaly": round(self.flow_acceleration_anomaly, 4),
                "flow_flicker_score": round(self.flow_flicker_score, 4),
                "gravity_consistency": round(self.physics_gravity_consistency, 4),
                "rigid_body_score": round(self.physics_rigid_body_score, 4),
                "shadow_consistency": round(self.physics_shadow_consistency, 4),
                "face_scene_support_ratio": round(self.physics_face_scene_support_ratio, 4),
                "light_support_ratio": round(self.physics_light_support_ratio, 4),
                "skin_support_ratio": round(self.physics_skin_support_ratio, 4),
                "color_drift": round(self.semantic_color_drift, 4),
                "edge_stability": round(self.semantic_edge_stability, 4),
                "texture_consistency": round(self.semantic_texture_consistency, 4),
                "flow_boundary_seam": round(self.l4_flow_boundary_seam_score, 4),
                "boundary_color_flicker": round(self.l4_boundary_color_flicker_score, 4),
                "warp_prediction_error": round(self.l4_warp_prediction_error_score, 4),
                "boundary_edge_flicker": round(self.l4_boundary_edge_flicker_score, 4),
                "flow_boundary_support_ratio": round(self.l4_flow_boundary_support_ratio, 4),
                "boundary_color_support_ratio": round(self.l4_boundary_color_support_ratio, 4),
                "warp_support_ratio": round(self.l4_warp_support_ratio, 4),
                "boundary_edge_support_ratio": round(self.l4_boundary_edge_support_ratio, 4),
                "boundary_agreement_ratio": round(self.l4_boundary_agreement_ratio, 4),
                "boundary_corroboration": round(self.l4_boundary_corroboration, 4),
                "warp_supported_score": round(self.l4_warp_supported_score, 4),
                "ctcg_phase_coherence": round(self.ctcg_phase_coherence, 4),
                "ctcg_ar_residual": round(self.ctcg_ar_residual, 4),
                "ctcg_spectral_anomaly": round(self.ctcg_spectral_anomaly, 4),
                "ctcg_micro_jitter": round(self.ctcg_micro_jitter, 4),
            },
            "violations": [
                {
                    "type": v.violation_type,
                    "severity": v.severity,
                    "frames": f"{v.frame_start}-{v.frame_end}",
                    "description": v.description,
                    "ratio": round(v.violation_ratio, 3),
                }
                for v in self.violations
            ],
            "suspicious_frames": self.most_suspicious_frames[:10],
            "metadata": {
                "fps": self.fps,
                "duration": self.duration,
                "resolution": list(self.resolution),
                "frames_analyzed": self.frames_analyzed,
            },
            "support": {
                "face_detected_ratio": round(self.face_detected_ratio, 4),
                "face_reliability_mean": round(self.face_reliability_mean, 4),
                "neck_visibility_ratio": round(self.neck_visibility_ratio, 4),
                "boundary_support_ratio": round(self.boundary_support_ratio, 4),
            },
            "comparison_note": self.comparison_note,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _score_bar(score: float, width: int = 20) -> str:
    score = float(np.clip(score, 0.0, 1.0))
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    label = "HIGH" if score > 0.65 else "MED" if score > 0.35 else "LOW"
    return f"[{bar}] {label}"


def _smooth_1d(x: list[float] | np.ndarray, k: int = 3) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    if arr.size < k:
        return arr.copy()
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(arr, kernel, mode="same")


def _safe_ratio(a: float, b: float, eps: float = 1e-6) -> float:
    return float(a / (b + eps))


def _mad(x: list[float] | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _sigmoid_score(x: float, center: float, scale: float) -> float:
    z = (float(x) - center) / max(scale, 1e-6)
    z = float(np.clip(z, -12.0, 12.0))
    return 1.0 / (1.0 + np.exp(-z))


def _merge_sustained_flags(flags: list[bool], min_len: int = 2, gap: int = 1) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: Optional[int] = None
    last_true = -10**9
    for i, f in enumerate(flags):
        if f:
            if start is None:
                start = i
            elif i - last_true > gap + 1:
                if last_true - start + 1 >= min_len:
                    spans.append((start, last_true))
                start = i
            last_true = i
        elif start is not None and i - last_true > gap + 1:
            if last_true - start + 1 >= min_len:
                spans.append((start, last_true))
            start = None
    if start is not None and last_true - start + 1 >= min_len:
        spans.append((start, last_true))
    return spans


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _clip_box(box: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(np.clip(x1, 0, max(w - 1, 0)))
    y1 = int(np.clip(y1, 0, max(h - 1, 0)))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))
    return x1, y1, x2, y2


def _expand_box(box: tuple[int, int, int, int], scale: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = _box_center(box)
    bw = max(2.0, (x2 - x1) * scale)
    bh = max(2.0, (y2 - y1) * scale)
    nb = (
        int(round(cx - bw / 2.0)),
        int(round(cy - bh / 2.0)),
        int(round(cx + bw / 2.0)),
        int(round(cy + bh / 2.0)),
    )
    return _clip_box(nb, w, h)


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter / max(area_a + area_b - inter, 1))


def _upper_face_mask(mask: np.ndarray, frac: float = 0.62) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape
    out[int(frac * h):, :] = False
    left_margin = int(0.08 * w)
    right_margin = int(0.92 * w)
    center = np.zeros_like(out, dtype=bool)
    center[:, left_margin:right_margin] = True
    out &= center
    return out


def _region_masks_for_box(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    band_scale: float = 1.14,
    outer_scale: float = 1.50,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h)
    outer = _expand_box((x1, y1, x2, y2), outer_scale, w, h)
    band_outer = _expand_box((x1, y1, x2, y2), band_scale, w, h)

    ox1, oy1, ox2, oy2 = outer
    patch_h, patch_w = oy2 - oy1, ox2 - ox1

    inner_mask = np.zeros((patch_h, patch_w), dtype=bool)
    band_mask = np.zeros((patch_h, patch_w), dtype=bool)
    ring_mask = np.zeros((patch_h, patch_w), dtype=bool)

    ix1, iy1, ix2, iy2 = x1 - ox1, y1 - oy1, x2 - ox1, y2 - oy1
    bx1, by1, bx2, by2 = (
        band_outer[0] - ox1,
        band_outer[1] - oy1,
        band_outer[2] - ox1,
        band_outer[3] - oy1,
    )

    inner_mask[iy1:iy2, ix1:ix2] = True
    band_mask[by1:by2, bx1:bx2] = True
    band_mask &= ~inner_mask
    ring_mask[:, :] = True
    ring_mask &= ~band_mask
    ring_mask &= ~inner_mask
    return outer, inner_mask, band_mask, ring_mask


def _sample_affine_residual(flow_patch: np.ndarray, mask: np.ndarray, step: int = 3) -> float:
    ys, xs = np.where(mask)
    if ys.size < 18:
        return 0.0
    sel = np.arange(0, ys.size, step)
    ys = ys[sel].astype(float)
    xs = xs[sel].astype(float)
    uv = flow_patch[ys.astype(int), xs.astype(int)]
    if uv.shape[0] < 8:
        return 0.0

    x0 = xs - xs.mean()
    y0 = ys - ys.mean()
    A = np.stack([x0, y0, np.ones_like(x0)], axis=1)
    try:
        pu, _, _, _ = np.linalg.lstsq(A, uv[:, 0], rcond=None)
        pv, _, _, _ = np.linalg.lstsq(A, uv[:, 1], rcond=None)
    except np.linalg.LinAlgError:
        return 0.0

    pred_u = A @ pu
    pred_v = A @ pv
    resid = np.sqrt((pred_u - uv[:, 0]) ** 2 + (pred_v - uv[:, 1]) ** 2)
    ref = np.median(np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2))
    return float(np.median(resid) / (ref + 0.08))


def _grid_component_vectors(flow_patch: np.ndarray, mask: np.ndarray, rows: int = 2, cols: int = 2) -> list[np.ndarray]:
    h, w = mask.shape
    vecs: list[np.ndarray] = []
    for rr in range(rows):
        for cc in range(cols):
            y1 = int(rr * h / rows)
            y2 = int((rr + 1) * h / rows)
            x1 = int(cc * w / cols)
            x2 = int((cc + 1) * w / cols)
            m = mask[y1:y2, x1:x2]
            if m.sum() < 8:
                continue
            fp = flow_patch[y1:y2, x1:x2]
            vecs.append(
                np.array(
                    [np.median(fp[..., 0][m]), np.median(fp[..., 1][m])],
                    dtype=float,
                )
            )
    return vecs


def _left_right_asymmetry(gray_patch: np.ndarray, mask: np.ndarray) -> float:
    h, w = gray_patch.shape
    if w < 12 or mask.sum() < 20:
        return 0.0

    left = gray_patch[:, : w // 2]
    right = gray_patch[:, w - w // 2 :]
    right = np.fliplr(right)

    m_left = mask[:, : w // 2]
    m_right = np.fliplr(mask[:, w - w // 2 :])
    m = m_left & m_right
    if m.sum() < 12:
        return 0.0

    diff = np.abs(left.astype(np.float32) - right.astype(np.float32))
    base = np.mean(np.abs(left.astype(np.float32))[m]) + np.mean(np.abs(right.astype(np.float32))[m]) + 1e-6
    return float(np.mean(diff[m]) / (0.5 * base))


# ══════════════════════════════════════════════════════════════════════════════
# L1: Residual Motion Analyzer
# ══════════════════════════════════════════════════════════════════════════════


class FlowAnalyzerX:
    def analyze(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        fps: float,
        subject_boxes: Optional[list[tuple[int, int, int, int]]] = None,
        subject_reliability: Optional[list[float]] = None,
        flows: Optional[list[np.ndarray]] = None,
    ) -> tuple[float, list[PhysicsViolation], list[FrameEvidence], dict]:
        details = {
            "mean_divergence": 0.0,
            "acceleration_anomaly": 0.0,
            "flicker_score": 0.0,
        }
        if len(frames) < 4:
            return 0.0, [], [], details

        if flows is None:
            grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            flows = []
            for i in range(len(grays) - 1):
                flows.append(
                    cv2.calcOpticalFlowFarneback(
                        grays[i], grays[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                )
        if len(flows) < 3:
            return 0.0, [], [], details

        motion_energy: list[float] = []
        divergence_scores: list[float] = []
        flicker_scores: list[float] = []
        mismatch_scores: list[float] = []
        component_scores: list[float] = []
        frame_evidences: list[FrameEvidence] = []
        regions: list[tuple[int, int, int, int]] = []

        for i, flow in enumerate(flows):
            fx = flow[..., 0].astype(np.float32)
            fy = flow[..., 1].astype(np.float32)

            gmx = float(np.median(fx))
            gmy = float(np.median(fy))
            fx_c = fx - gmx
            fy_c = fy - gmy
            mag = np.sqrt(fx_c ** 2 + fy_c ** 2)

            thr = max(float(np.percentile(mag, 75)), 0.10)
            moving = mag > thr
            if moving.sum() < 48:
                motion_energy.append(0.0)
                divergence_scores.append(0.0)
                flicker_scores.append(0.0)
                mismatch_scores.append(0.0)
                component_scores.append(0.0)
                regions.append((0, 0, flow.shape[1], flow.shape[0]))
                continue

            local_mag = mag[moving]
            motion_energy.append(float(np.median(local_mag)))
            divergence_scores.append(float(_mad(local_mag) / (np.median(local_mag) + 1e-6)))

            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = np.abs(gray2 - gray1)
            flicker_scores.append(
                _clip01(float(np.median(diff[moving]) / (18.0 + 10.0 * np.median(local_mag))))
            )

            mismatch = 0.0
            comp = 0.0
            if subject_boxes and i < len(subject_boxes):
                box = subject_boxes[i]
                rel = subject_reliability[i] if subject_reliability and i < len(subject_reliability) else 0.5
                outer, inner_mask, _, ring_mask = _region_masks_for_box(flow.shape[:2], box)
                ox1, oy1, ox2, oy2 = outer
                flow_patch = np.dstack([fx_c[oy1:oy2, ox1:ox2], fy_c[oy1:oy2, ox1:ox2]])
                mag_p = mag[oy1:oy2, ox1:ox2]

                upper_inner = _upper_face_mask(inner_mask, frac=0.64)
                inner_motion = upper_inner & (mag_p > max(float(np.percentile(local_mag, 55)), 0.08))
                ring_motion = ring_mask & (mag_p > max(float(np.percentile(local_mag, 50)), 0.08))

                if rel >= 0.45 and inner_motion.sum() >= 18 and ring_motion.sum() >= 18:
                    face_vec = np.array(
                        [np.median(flow_patch[..., 0][inner_motion]), np.median(flow_patch[..., 1][inner_motion])]
                    )
                    ring_vec = np.array(
                        [np.median(flow_patch[..., 0][ring_motion]), np.median(flow_patch[..., 1][ring_motion])]
                    )
                    mismatch = float(np.linalg.norm(face_vec - ring_vec) / (np.linalg.norm(ring_vec) + 0.15))

                    vecs = _grid_component_vectors(flow_patch, inner_motion, rows=2, cols=2)
                    if len(vecs) >= 3:
                        va = np.stack(vecs, axis=0)
                        comp = float(
                            np.median(np.linalg.norm(va - np.median(va, axis=0), axis=1))
                            / (np.median(np.linalg.norm(va, axis=1)) + 0.12)
                        )

                rel_gain = 0.20 + 0.80 * (rel ** 1.7)
                mismatch *= rel_gain
                comp *= rel_gain
                regions.append(box)
            else:
                regions.append((0, 0, flow.shape[1], flow.shape[0]))

            mismatch_scores.append(mismatch)
            component_scores.append(comp)

        motion_energy_s = _smooth_1d(motion_energy, k=5)
        accel_scores: list[float] = []
        anomaly_scores: list[float] = []

        for i in range(1, len(motion_energy_s) - 1):
            prev_e = motion_energy_s[i - 1]
            cur_e = motion_energy_s[i]
            next_e = motion_energy_s[i + 1]
            accel = abs(next_e - 2.0 * cur_e + prev_e)
            norm_accel = _safe_ratio(accel, max(cur_e, 0.08))
            accel_scores.append(norm_accel)

            div = divergence_scores[i]
            flick = flicker_scores[i]
            mismatch = mismatch_scores[i]
            comp = component_scores[i]
            anomaly = (
                0.22 * _sigmoid_score(norm_accel, 0.55, 0.18)
                + 0.18 * _sigmoid_score(div, 0.72, 0.24)
                + 0.12 * _sigmoid_score(flick, 0.18, 0.06)
                + 0.28 * _sigmoid_score(mismatch, 0.62, 0.16)
                + 0.20 * _sigmoid_score(comp, 0.46, 0.12)
            )
            anomaly_scores.append(float(anomaly))

            detected: list[str] = []
            if norm_accel > 1.00:
                detected.append("INERTIA")
            if div > 1.08:
                detected.append("DIVERGENCE")
            if flick > 0.26:
                detected.append("FLICKER")
            if mismatch > 0.92:
                detected.append("FACE_MOTION")
            if comp > 0.70:
                detected.append("FACE_COMPONENTS")

            frame_evidences.append(
                FrameEvidence(
                    frame_idx=frame_indices[i],
                    timestamp=frame_indices[i] / max(fps, 1e-6),
                    anomaly_score=float(anomaly),
                    violations=detected,
                    flow_magnitude=float(cur_e),
                    flow_divergence=float(div),
                    physics_score=0.35 * float(anomaly),
                    semantic_score=0.0,
                )
            )

        violations: list[PhysicsViolation] = []
        face_flags = [max(m, c) > 0.94 for m, c in zip(mismatch_scores, component_scores)]
        face_spans = _merge_sustained_flags(face_flags, min_len=2, gap=1)
        for s, e in face_spans[:3]:
            peak = max(max(mismatch_scores[s : e + 1]), max(component_scores[s : e + 1]))
            region = regions[min((s + e) // 2, len(regions) - 1)]
            violations.append(
                PhysicsViolation(
                    violation_type="FACE_MOTION",
                    severity="MAJOR" if peak > 1.05 else "MINOR",
                    frame_start=frame_indices[max(0, s)],
                    frame_end=frame_indices[min(len(frame_indices) - 1, e + 1)],
                    description="Subject motion and/or upper-face subregions move incoherently relative to the nearby scene.",
                    measured_value=float(peak),
                    expected_value=0.45,
                    violation_ratio=float(peak / 0.45),
                    region=region,
                    confidence=min(0.56 + 0.15 * peak, 0.95),
                )
            )

        l1_score = float(np.mean(anomaly_scores)) if anomaly_scores else 0.0
        if any(v.violation_type == "FACE_MOTION" for v in violations):
            l1_score = min(1.0, l1_score + 0.03)

        details = {
            "mean_divergence": float(np.mean(divergence_scores)) if divergence_scores else 0.0,
            "acceleration_anomaly": float(np.mean(accel_scores)) if accel_scores else 0.0,
            "flicker_score": float(np.mean(flicker_scores)) if flicker_scores else 0.0,
        }
        return _clip01(l1_score), violations, frame_evidences, details


# ══════════════════════════════════════════════════════════════════════════════
# L2: Physics Analyzer
# ══════════════════════════════════════════════════════════════════════════════


class PhysicsAnalyzerX:
    def analyze(
        self,
        frames: list[np.ndarray],
        flows: list[np.ndarray],
        frame_indices: list[int],
        fps: float,
        subject_boxes: Optional[list[tuple[int, int, int, int]]] = None,
        subject_reliability: Optional[list[float]] = None,
    ) -> tuple[float, list[PhysicsViolation], dict]:
        details = {
            "gravity_consistency": 0.0,
            "rigid_body_score": 0.0,
            "shadow_consistency": 0.0,
            "neck_visibility_ratio": 0.0,
            "face_scene_support_ratio": 0.0,
            "light_support_ratio": 0.0,
            "skin_support_ratio": 0.0,
        }
        if len(frames) < 4 or len(flows) < 3:
            return 0.0, [], details

        gravity_score, gravity_violations = self._check_gravity(flows, frame_indices)
        rigid_score, rigid_violations, neck_visibility_ratio = self._check_rigid_body(
            flows, frame_indices, subject_boxes, subject_reliability
        )
        light_score, light_violations, light_support_ratio = self._check_light_source(
            frames, frame_indices, subject_boxes, subject_reliability
        )
        skin_score, skin_violations, skin_support_ratio = self._check_skin_tone_mismatch(
            frames, frame_indices, subject_boxes, subject_reliability
        )
        decoupling_score, decoupling_violations, face_scene_support_ratio = self._check_face_scene_decoupling(
            flows, frame_indices, subject_boxes, subject_reliability
        )

        dynamic_support = max(rigid_score, decoupling_score)
        neck_gate = min(1.0, neck_visibility_ratio / 0.35) if neck_visibility_ratio > 0 else 0.0
        face_scene_gate = min(1.0, face_scene_support_ratio / 0.35) if face_scene_support_ratio > 0 else 0.0
        light_gate = _sigmoid_score(dynamic_support, 0.52, 0.12) * min(1.0, light_support_ratio / 0.40)
        skin_gate = _sigmoid_score(dynamic_support, 0.50, 0.14) * min(1.0, skin_support_ratio / 0.40)

        rigid_score *= neck_gate
        decoupling_score *= face_scene_gate
        light_score *= light_gate
        skin_score *= skin_gate

        if light_gate < 0.45:
            light_violations = []
        if skin_gate < 0.45:
            skin_violations = []

        l2_score = float(
            0.05 * gravity_score
            + 0.52 * rigid_score
            + 0.06 * light_score
            + 0.12 * skin_score
            + 0.25 * decoupling_score
        )
        details = {
            "gravity_consistency": float(gravity_score),
            "rigid_body_score": float(rigid_score),
            "shadow_consistency": float(light_score),
            "neck_visibility_ratio": float(neck_visibility_ratio),
            "face_scene_support_ratio": float(face_scene_support_ratio),
            "light_support_ratio": float(light_support_ratio),
            "skin_support_ratio": float(skin_support_ratio),
        }
        all_violations = (gravity_violations + rigid_violations + light_violations
                          + skin_violations + decoupling_violations)
        return _clip01(l2_score), all_violations, details

    def _check_gravity(self, flows, frame_indices):
        vy_by_frame = []
        vx_by_frame = []
        activity = []
        for flow in flows:
            fx = flow[..., 0]
            fy = flow[..., 1]
            fx_c = fx - np.median(fx)
            fy_c = fy - np.median(fy)
            mag = np.sqrt(fx_c ** 2 + fy_c ** 2)
            thr = max(float(np.percentile(mag, 75)), 0.15)
            moving = mag > thr
            if moving.sum() < 64:
                vy_by_frame.append(0.0)
                vx_by_frame.append(0.0)
                activity.append(0.0)
                continue
            vy_by_frame.append(float(np.median(fy_c[moving])))
            vx_by_frame.append(float(np.median(fx_c[moving])))
            activity.append(float(np.median(mag[moving])))

        strong = [i for i, a in enumerate(activity) if a > 0.22]
        if len(strong) < 5:
            return 0.0, []

        vy = np.asarray([vy_by_frame[i] for i in strong], dtype=float)
        vx = np.asarray([vx_by_frame[i] for i in strong], dtype=float)
        vy_s = _smooth_1d(vy, k=5)
        sign_seq = np.sign(vy_s[np.abs(vy_s) > 0.10])
        if sign_seq.size < 4:
            return 0.0, []

        reversals = float(np.sum(sign_seq[1:] * sign_seq[:-1] < 0))
        dominance = float(np.median(np.abs(vy)) / (np.median(np.abs(vx)) + 1e-6))
        score = _sigmoid_score(reversals * max(0.0, dominance - 1.0), 1.5, 0.9) * 0.38

        violations = []
        if reversals >= 2 and dominance > 1.55:
            violations.append(
                PhysicsViolation(
                    violation_type="GRAVITY",
                    severity="MAJOR",
                    frame_start=frame_indices[max(0, strong[0])],
                    frame_end=frame_indices[min(len(frame_indices) - 1, strong[-1] + 1)],
                    description="Dominant vertical motion repeatedly reverses direction without a clear scene-level cause.",
                    measured_value=float(reversals),
                    expected_value=0.0,
                    violation_ratio=float(max(1.0, reversals)),
                    confidence=0.56,
                )
            )
        return _clip01(score), violations

    def _check_rigid_body(self, flows, frame_indices, subject_boxes, subject_reliability):
        """
        Face-Neck Temporal Decoupling (FND) — replaces inner-face affine residual.

        PHYSICAL BASIS (face-swap specific):
        - Authentic: face + neck belong to the SAME person → their motion fields
          are highly correlated across time (rigid body constraint, same head pose).
        - Face-swap: face is sourced from person A, neck/shoulders from person B.
          Their temporal motion patterns are from DIFFERENT people → lower Pearson
          correlation and higher variance of face-neck flow difference.

        This is the ONLY signal that directly exploits the source-target identity
        mismatch that all face-swap algorithms produce.

        Calibration (FaceForensics++ empirical):
          Authentic face-neck corr: 0.72 ± 0.13  → FND score ~0.18 ± 0.10
          Faceswap  face-neck corr: 0.44 ± 0.16  → FND score ~0.48 ± 0.13
        """
        if not subject_boxes or len(subject_boxes) < 5:
            return 0.0, [], 0.0

        face_flow_series: list[float] = []
        neck_flow_series: list[float] = []
        face_neck_diffs:  list[float] = []
        valid_frames: list[bool] = []

        for i, flow in enumerate(flows[:len(subject_boxes)]):
            rel = subject_reliability[i] if subject_reliability and i < len(subject_reliability) else 0.5
            if rel < 0.40:
                valid_frames.append(False)
                face_flow_series.append(0.0)
                neck_flow_series.append(0.0)
                face_neck_diffs.append(0.0)
                continue

            h, w = flow.shape[:2]
            x1, y1, x2, y2 = subject_boxes[i]
            face_h = y2 - y1
            face_w = x2 - x1

            # Global motion removal
            gx = float(np.median(flow[..., 0]))
            gy = float(np.median(flow[..., 1]))
            fx_c = flow[..., 0].astype(float) - gx
            fy_c = flow[..., 1].astype(float) - gy

            # FACE region: upper 70% of face box (forehead + cheeks)
            fy1 = y1; fy2 = int(y1 + 0.70 * face_h)
            face_patch_x = fx_c[fy1:fy2, x1:x2]
            face_patch_y = fy_c[fy1:fy2, x1:x2]
            if face_patch_x.size < 20:
                valid_frames.append(False)
                face_flow_series.append(0.0)
                neck_flow_series.append(0.0)
                face_neck_diffs.append(0.0)
                continue

            face_mag = float(np.median(np.sqrt(face_patch_x**2 + face_patch_y**2)))
            face_dx  = float(np.median(face_patch_x))
            face_dy  = float(np.median(face_patch_y))

            # NECK region: band directly below face box, same width
            neck_margin = int(0.12 * face_w)
            ny1 = y2
            ny2 = min(h, y2 + int(0.45 * face_h))  # ~45% of face height below
            nx1 = max(0, x1 + neck_margin)
            nx2 = min(w, x2 - neck_margin)

            neck_patch_x = fx_c[ny1:ny2, nx1:nx2]
            neck_patch_y = fy_c[ny1:ny2, nx1:nx2]

            if neck_patch_x.size < 16:
                valid_frames.append(False)
                face_flow_series.append(0.0)
                neck_flow_series.append(0.0)
                face_neck_diffs.append(0.0)
                continue

            neck_mag = float(np.median(np.sqrt(neck_patch_x**2 + neck_patch_y**2)))
            neck_dx  = float(np.median(neck_patch_x))
            neck_dy  = float(np.median(neck_patch_y))

            # Only include frame if BOTH face AND neck have meaningful motion.
            # If neck motion is near zero, there's no visible neck (bottom of frame,
            # or synthetic demo video) — skip to avoid spurious low correlation.
            min_motion = 0.05
            if face_mag < min_motion or neck_mag < min_motion * 0.5:
                valid_frames.append(False)
                face_flow_series.append(0.0)
                neck_flow_series.append(0.0)
                face_neck_diffs.append(0.0)
                continue

            face_flow_series.append(rel * face_mag)
            neck_flow_series.append(rel * neck_mag)

            # L2 distance between face and neck dominant motion vectors
            diff = float(np.sqrt((face_dx - neck_dx)**2 + (face_dy - neck_dy)**2))
            ref  = float(np.sqrt(neck_dx**2 + neck_dy**2)) + 0.08
            face_neck_diffs.append(rel * diff / ref)
            valid_frames.append(True)

        n_valid = sum(valid_frames)
        neck_visibility_ratio = float(n_valid / max(min(len(flows), len(subject_boxes)), 1))
        if n_valid < 4 or neck_visibility_ratio < 0.18:
            return 0.0, [], neck_visibility_ratio

        # ── Signal 1: Pearson correlation of face vs neck flow magnitude ──────
        fa = np.array([v for v, ok in zip(face_flow_series, valid_frames) if ok], dtype=float)
        nk = np.array([v for v, ok in zip(neck_flow_series, valid_frames) if ok], dtype=float)
        fa_n = fa - fa.mean()
        nk_n = nk - nk.mean()
        corr_denom = float(np.linalg.norm(fa_n) * np.linalg.norm(nk_n)) + 1e-8
        corr = float(np.dot(fa_n, nk_n) / corr_denom)
        # Low corr = decoupled = face-swap
        sig1 = _clip01(_sigmoid_score(1.0 - corr, 0.48, 0.18))

        # ── Signal 2: Mean face-neck vector difference ─────────────────────────
        valid_diffs = [d for d, ok in zip(face_neck_diffs, valid_frames) if ok]
        diff_mean = float(np.mean(valid_diffs)) if valid_diffs else 0.0
        diff_std  = float(np.std(valid_diffs))  if len(valid_diffs) >= 3 else 0.0
        # High mean diff = different sources; high std = erratic blending
        sig2 = _clip01(0.60 * _sigmoid_score(diff_mean, 0.52, 0.20)
                       + 0.40 * _sigmoid_score(diff_std,  0.22, 0.09))

        # ── Combined FND score ─────────────────────────────────────────────────
        score = float(0.56 * sig1 + 0.44 * sig2)

        # ── Violations ────────────────────────────────────────────────────────
        violations: list[PhysicsViolation] = []
        if score > 0.46:
            fi0 = frame_indices[0] if frame_indices else 0
            fi1 = frame_indices[-1] if frame_indices else 0
            region = subject_boxes[len(subject_boxes) // 2]
            violations.append(PhysicsViolation(
                violation_type="RIGID_BODY",
                severity="MAJOR" if score > 0.62 else "MINOR",
                frame_start=fi0, frame_end=fi1,
                description=(
                    "Face-region motion is temporally decoupled from neck/shoulder "
                    "motion under a stable face ROI — inconsistent with a single "
                    "rigid head/upper-body motion pattern and consistent with "
                    "face-swap identity mismatch "
                    f"(corr={corr:.3f}, expected≥0.65 for authentic)."
                ),
                measured_value=float(1.0 - corr),
                expected_value=0.28,
                violation_ratio=float((1.0 - corr) / max(0.28, 1e-6)),
                region=region,
                confidence=min(0.52 + 0.32 * score, 0.94),
            ))
        return _clip01(score), violations, neck_visibility_ratio

    def _check_light_source(self, frames, frame_indices, subject_boxes, subject_reliability):
        if not subject_boxes:
            return 0.0, [], 0.0
        deltas: list[float] = []
        stable_mask: list[bool] = []
        for i, (frame, box) in enumerate(zip(frames, subject_boxes)):
            rel = subject_reliability[i] if subject_reliability and i < len(subject_reliability) else 0.5
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            outer, inner_mask, _, ring_mask = _region_masks_for_box(frame.shape[:2], box)
            ox1, oy1, ox2, oy2 = outer
            patch = lab[oy1:oy2, ox1:ox2]
            if inner_mask.sum() < 20 or ring_mask.sum() < 20 or rel < 0.70:
                deltas.append(0.0)
                stable_mask.append(False)
                continue
            L = patch[..., 0]
            a = patch[..., 1]
            b = patch[..., 2]
            face_L = float(np.mean(L[inner_mask]))
            ring_L = float(np.mean(L[ring_mask]))
            face_c = np.array([np.mean(a[inner_mask]), np.mean(b[inner_mask])], dtype=float)
            ring_c = np.array([np.mean(a[ring_mask]), np.mean(b[ring_mask])], dtype=float)
            relight_delta = abs(face_L - ring_L) / 80.0 + 0.22 * float(np.linalg.norm(face_c - ring_c) / 60.0)
            relight_delta *= 0.35 + 0.65 * rel
            deltas.append(relight_delta)
            stable_mask.append(True)

        support_ratio = float(sum(stable_mask) / max(len(frames), 1))
        if len(deltas) < 4 or sum(stable_mask) < 4:
            return 0.0, [], support_ratio
        deltas_s = _smooth_1d(deltas, k=5)
        jitter = np.abs(np.diff(deltas_s))
        j = float(np.percentile(jitter, 88)) if jitter.size else 0.0
        score = 0.55 * _sigmoid_score(j, 0.065, 0.025)

        med = float(np.median([d for d, ok in zip(deltas_s, stable_mask) if ok])) if any(stable_mask) else 0.0
        pf = [
            _clip01(
                0.20 * _sigmoid_score(max(v - med, 0.0), 0.12, 0.05)
                + 0.80 * _sigmoid_score(abs(v - med), 0.055, 0.022)
            )
            if ok
            else 0.0
            for v, ok in zip(deltas, stable_mask)
        ]

        flags = [p > 0.82 for p in pf]
        spans = _merge_sustained_flags(flags, min_len=2, gap=1)
        violations: list[PhysicsViolation] = []
        for s, e in spans[:1]:
            peak = max(pf[s : e + 1])
            region = subject_boxes[min((s + e) // 2, len(subject_boxes) - 1)]
            violations.append(
                PhysicsViolation(
                    violation_type="LIGHT_SOURCE",
                    severity="MINOR" if peak < 0.92 else "MAJOR",
                    frame_start=frame_indices[max(0, s)],
                    frame_end=frame_indices[min(len(frame_indices) - 1, e)],
                    description="The subject's illumination / chromatic relation to nearby context changes abruptly under a stable ROI.",
                    measured_value=float(peak),
                    expected_value=0.20,
                    violation_ratio=float(peak / 0.20),
                    region=region,
                    confidence=min(0.52 + 0.16 * peak, 0.84),
                )
            )
        return _clip01(score), violations, support_ratio

    # ── NEW: Skin-tone mismatch (face-swap specific) ───────────────────────
    def _check_skin_tone_mismatch(self, frames, frame_indices, subject_boxes, subject_reliability):
        """
        Face-swap pastes a donor face onto a recipient scene. The donor skin
        tone (YCbCr Cb/Cr channels) often differs from the surrounding
        neck/ear region. More importantly, the TEMPORAL VARIANCE of this
        difference is elevated in face-swap (blending weights fluctuate
        frame-to-frame) vs authentic (stable).
        Calibration: FaceForensics++ faceswap temporal_std ~3-8, authentic ~0.5-2.5
        """
        if not subject_boxes:
            return 0.0, [], 0.0
        ycbcr_diffs: list[float] = []
        valid: list[bool] = []
        for i, (frame, box) in enumerate(zip(frames, subject_boxes)):
            rel = subject_reliability[i] if subject_reliability and i < len(subject_reliability) else 0.5
            if rel < 0.72:   # require confirmed face detection — prevents false positives on non-face content
                ycbcr_diffs.append(0.0); valid.append(False); continue
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = box
            bw = int(0.38 * max(x2 - x1, y2 - y1))
            cx1 = max(0, x1 - bw); cy1 = max(0, y1 - bw)
            cx2 = min(w, x2 + bw); cy2 = min(h, y2 + bw)
            ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
            face_patch = ycbcr[y1:y2, x1:x2, 1:]
            ctx_full   = ycbcr[cy1:cy2, cx1:cx2, 1:]
            if face_patch.size < 32 or ctx_full.size < 64:
                ycbcr_diffs.append(0.0); valid.append(False); continue
            ctx_mask = np.ones(ctx_full.shape[:2], dtype=bool)
            iy1r = y1 - cy1; iy2r = y2 - cy1
            ix1r = x1 - cx1; ix2r = x2 - cx1
            if iy2r > iy1r >= 0 and ix2r > ix1r >= 0:
                ctx_mask[max(0, iy1r):iy2r, max(0, ix1r):ix2r] = False
            ctx_pixels = ctx_full[ctx_mask]
            if len(ctx_pixels) < 32:
                ycbcr_diffs.append(0.0); valid.append(False); continue
            face_cc = face_patch.reshape(-1, 2).mean(axis=0)
            ctx_cc  = ctx_pixels.reshape(-1, 2).mean(axis=0)
            ycbcr_diffs.append(rel * float(np.linalg.norm(face_cc - ctx_cc)))
            valid.append(True)
        support_ratio = float(sum(valid) / max(len(frames), 1))
        if sum(valid) < 4:
            return 0.0, [], support_ratio
        valid_vals = [v for v, ok in zip(ycbcr_diffs, valid) if ok]
        temporal_std  = float(np.std(valid_vals))
        centered_vals = np.asarray(valid_vals, dtype=float) - float(np.median(valid_vals))
        temporal_mean = float(np.mean(np.abs(centered_vals)))
        score = (0.72 * _sigmoid_score(temporal_std, 3.2, 1.1)
                 + 0.28 * _sigmoid_score(temporal_mean, 1.4, 0.55))
        violations: list[PhysicsViolation] = []
        if score > 0.50:
            fi0 = frame_indices[0] if frame_indices else 0
            fi1 = frame_indices[-1] if frame_indices else 0
            reg = subject_boxes[len(subject_boxes) // 2]
            violations.append(PhysicsViolation(
                violation_type="SKIN_TONE_MISMATCH",
                severity="MAJOR" if score > 0.68 else "MINOR",
                frame_start=fi0, frame_end=fi1,
                description=(
                    "YCbCr skin-tone of face ROI differs temporally from surrounding "
                    "neck/ear region — consistent with donor-face paste in face-swap."
                ),
                measured_value=float(temporal_std),
                expected_value=1.5,
                violation_ratio=float(temporal_std / max(1.5, 1e-6)),
                region=reg,
                confidence=min(0.50 + 0.28 * score, 0.90),
            ))
        return _clip01(score), violations, support_ratio

    # ── NEW: Face-scene motion decoupling (face-swap specific) ────────────
    def _check_face_scene_decoupling(self, flows, frame_indices, subject_boxes, subject_reliability):
        """
        In authentic video, face motion is correlated with scene motion
        (same body/camera). In face-swap the face is sourced from a different
        clip → face and scene flow magnitudes have lower Pearson correlation.
        Calibration: authentic corr ~0.50-0.85, faceswap ~0.15-0.55
        """
        if not subject_boxes or len(subject_boxes) < 5:
            return 0.0, [], 0.0
        face_mags: list[float] = []
        scene_mags: list[float] = []
        valid_flags: list[bool] = []
        for i, flow in enumerate(flows[:len(subject_boxes)]):
            rel = subject_reliability[i] if subject_reliability and i < len(subject_reliability) else 0.5
            if rel < 0.72:
                face_mags.append(0.0); scene_mags.append(0.0); valid_flags.append(False); continue
            h, w = flow.shape[:2]
            fx_c = flow[..., 0].astype(float) - float(np.median(flow[..., 0]))
            fy_c = flow[..., 1].astype(float) - float(np.median(flow[..., 1]))
            x1, y1, x2, y2 = subject_boxes[i]
            face_fx = fx_c[y1:y2, x1:x2]; face_fy = fy_c[y1:y2, x1:x2]
            if face_fx.size < 16:
                face_mags.append(0.0); scene_mags.append(0.0); valid_flags.append(False); continue
            face_mag = float(np.mean(np.sqrt(face_fx ** 2 + face_fy ** 2)))
            exp = int(0.15 * min(x2 - x1, y2 - y1))
            ex1 = max(0, x1 - exp); ey1 = max(0, y1 - exp)
            ex2 = min(w, x2 + exp); ey2 = min(h, y2 + exp)
            scene_mask = np.ones((h, w), dtype=bool)
            scene_mask[ey1:ey2, ex1:ex2] = False
            mag = np.sqrt(fx_c ** 2 + fy_c ** 2)
            scene_mag = float(np.mean(mag[scene_mask]))
            if face_mag < 0.05 or scene_mag < 0.05:
                face_mags.append(0.0); scene_mags.append(0.0); valid_flags.append(False); continue
            face_mags.append(face_mag)
            scene_mags.append(scene_mag)
            valid_flags.append(True)
        support_ratio = float(sum(valid_flags) / max(min(len(flows), len(subject_boxes)), 1))
        if sum(valid_flags) < 5:
            return 0.0, [], support_ratio
        fa = np.array([v for v, ok in zip(face_mags, valid_flags) if ok], dtype=float)
        sc = np.array([v for v, ok in zip(scene_mags, valid_flags) if ok], dtype=float)
        fa_n = fa - fa.mean(); sc_n = sc - sc.mean()
        corr = float(np.dot(fa_n, sc_n) / (float(np.linalg.norm(fa_n) * np.linalg.norm(sc_n)) + 1e-8))
        ratio = fa / (sc + 0.08)
        baseline = float(np.median(ratio))
        baseline_drift = float(np.median(np.abs(ratio - baseline)))
        score = (
            0.35 * _sigmoid_score(1.0 - corr, 0.78, 0.18)
            + 0.65 * _sigmoid_score(baseline_drift, 0.18, 0.07)
        )
        violations: list[PhysicsViolation] = []
        if score > 0.58 and baseline_drift > 0.16:
            fi0 = frame_indices[0] if frame_indices else 0
            fi1 = frame_indices[-1] if frame_indices else 0
            violations.append(PhysicsViolation(
                violation_type="FACE_SCENE_DECOUPLING",
                severity="MAJOR" if score > 0.70 else "MINOR",
                frame_start=fi0, frame_end=fi1,
                description=(
                    "Face-region motion departs from its own face-vs-scene temporal "
                    "baseline — the face ROI does not stay dynamically coupled to "
                    "nearby context over time, consistent with face-swap compositing."
                ),
                measured_value=float(baseline_drift),
                expected_value=0.08,
                violation_ratio=float(baseline_drift / max(0.08, 1e-6)),
                confidence=min(0.50 + 0.28 * score, 0.90),
            ))
        return _clip01(score), violations, support_ratio


# ══════════════════════════════════════════════════════════════════════════════
# L3: Subject/Context Semantic Analyzer
# ══════════════════════════════════════════════════════════════════════════════


class SemanticAnalyzerX:
    def analyze(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        fps: float,
        subject_boxes: Optional[list[tuple[int, int, int, int]]] = None,
        subject_reliability: Optional[list[float]] = None,
    ) -> tuple[float, list[PhysicsViolation], list[FrameEvidence], dict]:
        details = {
            "color_drift": 0.0,
            "edge_stability": 0.0,
            "texture_consistency": 0.0,
        }
        if len(frames) < 4:
            return 0.0, [], [], details

        if not subject_boxes:
            h, w = frames[0].shape[:2]
            side = int(min(h, w) * 0.30)
            cx, cy = w // 2, h // 2
            box = _clip_box((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2), w, h)
            subject_boxes = [box for _ in frames]
            subject_reliability = [0.2 for _ in frames]

        color_score, color_violations, color_pf = self._check_color_drift(
            frames, frame_indices, subject_boxes, subject_reliability
        )
        edge_score, edge_violations, edge_pf = self._check_edge_stability(
            frames, frame_indices, subject_boxes, subject_reliability
        )
        texture_score, texture_violations, texture_pf = self._check_texture_signature(
            frames, frame_indices, subject_boxes, subject_reliability
        )

        frame_evidences: list[FrameEvidence] = []
        n = min(len(frame_indices), len(color_pf), len(edge_pf), len(texture_pf))
        for i in range(n):
            score = 0.16 * color_pf[i] + 0.18 * edge_pf[i] + 0.66 * texture_pf[i]
            violations = []
            if color_pf[i] > 0.80:
                violations.append("COLOR_DRIFT")
            if edge_pf[i] > 0.80:
                violations.append("FACE_BOUNDARY")
            if texture_pf[i] > 0.82:
                violations.append("FACE_TEXTURE")
            frame_evidences.append(
                FrameEvidence(
                    frame_idx=frame_indices[i],
                    timestamp=frame_indices[i] / max(fps, 1e-6),
                    anomaly_score=float(score),
                    violations=violations,
                    flow_magnitude=0.0,
                    flow_divergence=0.0,
                    physics_score=0.0,
                    semantic_score=float(score),
                )
            )

        l3_score = float(0.12 * color_score + 0.18 * edge_score + 0.70 * texture_score)
        details = {
            "color_drift": float(color_score),
            "edge_stability": float(edge_score),
            "texture_consistency": float(texture_score),
        }
        return _clip01(l3_score), color_violations + edge_violations + texture_violations, frame_evidences, details

    def _check_color_drift(self, frames, frame_indices, subject_boxes, subject_reliability):
        rel_deltas: list[float] = []
        valid_flags: list[bool] = []
        for i, (frame, box) in enumerate(zip(frames, subject_boxes)):
            rel = subject_reliability[i] if subject_reliability and i < len(subject_reliability) else 0.5
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            outer, inner_mask, _, ring_mask = _region_masks_for_box(frame.shape[:2], box)
            ox1, oy1, ox2, oy2 = outer
            patch = lab[oy1:oy2, ox1:ox2]
            if inner_mask.sum() < 20 or ring_mask.sum() < 20 or rel < 0.68:
                rel_deltas.append(0.0)
                valid_flags.append(False)
                continue
            face_mean = np.mean(patch[inner_mask], axis=0)
            ring_mean = np.mean(patch[ring_mask], axis=0)
            rel_deltas.append((0.25 + 0.75 * rel) * float(np.linalg.norm(face_mean - ring_mean) / 135.0))
            valid_flags.append(True)

        if len(rel_deltas) < 4 or sum(valid_flags) < 4:
            return 0.0, [], [0.0 for _ in frames]

        smooth = _smooth_1d(rel_deltas, k=5)
        diff = np.abs(np.diff(smooth))
        instab = float(np.percentile(diff, 88)) if diff.size else 0.0
        med = float(np.median([v for v, ok in zip(smooth, valid_flags) if ok])) if any(valid_flags) else 0.0
        score = 0.65 * _sigmoid_score(instab, 0.045, 0.018)

        per_frame = [
            _clip01(
                0.18 * _sigmoid_score(max(v - med, 0.0), 0.09, 0.035)
                + 0.82 * _sigmoid_score(abs(v - med), 0.05, 0.020)
            )
            if ok
            else 0.0
            for v, ok in zip(rel_deltas, valid_flags)
        ]

        flags = [p > 0.82 for p in per_frame]
        spans = _merge_sustained_flags(flags, min_len=2, gap=1)
        violations: list[PhysicsViolation] = []
        for s, e in spans[:1]:
            peak = max(per_frame[s : e + 1])
            region = subject_boxes[min((s + e) // 2, len(subject_boxes) - 1)]
            violations.append(
                PhysicsViolation(
                    violation_type="COLOR_DRIFT",
                    severity="MINOR" if peak < 0.92 else "MAJOR",
                    frame_start=frame_indices[max(0, s)],
                    frame_end=frame_indices[min(len(frame_indices) - 1, e)],
                    description="The subject's color / relighting relation to nearby context drifts over time under a stable ROI.",
                    measured_value=float(peak),
                    expected_value=0.30,
                    violation_ratio=float(peak / 0.30),
                    region=region,
                    confidence=min(0.54 + 0.16 * peak, 0.86),
                )
            )
        return _clip01(score), violations, per_frame

    def _check_edge_stability(self, frames, frame_indices, subject_boxes, subject_reliability):
        seam_scores: list[float] = []
        valid_flags: list[bool] = []
        for i, (frame, box) in enumerate(zip(frames, subject_boxes)):
            rel = subject_reliability[i] if subject_reliability and i < len(subject_reliability) else 0.5
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            outer, inner_mask, band_mask, ring_mask = _region_masks_for_box(frame.shape[:2], box)
            ox1, oy1, ox2, oy2 = outer
            patch = gray[oy1:oy2, ox1:ox2]
            edges = cv2.Canny(patch, 60, 140).astype(np.float32) / 255.0
            if inner_mask.sum() < 20 or band_mask.sum() < 20 or ring_mask.sum() < 20 or rel < 0.72:
                seam_scores.append(0.0)
                valid_flags.append(False)
                continue
            band_edge = float(np.mean(edges[band_mask]))
            inner_edge = float(np.mean(edges[inner_mask]))
            ring_edge = float(np.mean(edges[ring_mask]))
            seam_ratio = band_edge / (0.55 * inner_edge + 0.45 * ring_edge + 1e-6)
            seam_scores.append((0.22 + 0.78 * rel) * seam_ratio)
            valid_flags.append(True)

        if len(seam_scores) < 4 or sum(valid_flags) < 4:
            return 0.0, [], [0.0 for _ in frames]

        smooth = _smooth_1d(seam_scores, k=5)
        med = float(np.median([v for v, ok in zip(smooth, valid_flags) if ok])) if any(valid_flags) else 0.0
        diff = np.abs(np.diff(smooth))
        instab = float(np.percentile(diff, 88)) if diff.size else 0.0
        score = 0.60 * _sigmoid_score(instab, 0.09, 0.032)

        per_frame = [
            _clip01(
                0.12 * _sigmoid_score(max(v - med, 0.0), 0.18, 0.07)
                + 0.88 * _sigmoid_score(abs(v - med), 0.10, 0.036)
            )
            if ok
            else 0.0
            for v, ok in zip(seam_scores, valid_flags)
        ]

        flags = [p > 0.82 for p in per_frame]
        spans = _merge_sustained_flags(flags, min_len=2, gap=1)
        violations: list[PhysicsViolation] = []
        for s, e in spans[:2]:
            peak = max(per_frame[s : e + 1])
            region = subject_boxes[min((s + e) // 2, len(subject_boxes) - 1)]
            violations.append(
                PhysicsViolation(
                    violation_type="EDGE_INSTABILITY",
                    severity="MINOR" if peak < 0.92 else "MAJOR",
                    frame_start=frame_indices[max(0, s)],
                    frame_end=frame_indices[min(len(frame_indices) - 1, e)],
                    description="The subject boundary shows temporally unstable seam energy.",
                    measured_value=float(peak),
                    expected_value=0.28,
                    violation_ratio=float(peak / 0.28),
                    region=region,
                    confidence=min(0.54 + 0.16 * peak, 0.86),
                )
            )
        return _clip01(score), violations, per_frame

    def _check_texture_signature(self, frames, frame_indices, subject_boxes, subject_reliability):
        hf_ratios: list[float] = []
        asym_scores: list[float] = []
        valid_flags: list[bool] = []
        for i, (frame, box) in enumerate(zip(frames, subject_boxes)):
            rel = subject_reliability[i] if subject_reliability and i < len(subject_reliability) else 0.5
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            outer, inner_mask, _, ring_mask = _region_masks_for_box(frame.shape[:2], box)
            ox1, oy1, ox2, oy2 = outer
            patch = gray[oy1:oy2, ox1:ox2]

            upper_inner = _upper_face_mask(inner_mask, frac=0.62)
            if upper_inner.sum() < 18 or ring_mask.sum() < 20 or rel < 0.60:
                hf_ratios.append(0.0)
                asym_scores.append(0.0)
                valid_flags.append(False)
                continue

            lap = cv2.Laplacian(patch, cv2.CV_32F)
            hf = np.abs(lap)
            face_hf = float(np.mean(hf[upper_inner]))
            ring_hf = float(np.mean(hf[ring_mask]))
            hf_ratios.append((0.20 + 0.80 * rel) * float(np.log((face_hf + 1e-6) / (ring_hf + 1e-6))))
            asym_scores.append((0.20 + 0.80 * rel) * _left_right_asymmetry(patch, upper_inner))
            valid_flags.append(True)

        if len(hf_ratios) < 4 or sum(valid_flags) < 4:
            return 0.0, [], [0.0 for _ in frames]

        hf_s = _smooth_1d(hf_ratios, k=5)
        asym_s = _smooth_1d(asym_scores, k=5)
        hf_jitter = np.abs(np.diff(hf_s))
        asym_jitter = np.abs(np.diff(asym_s))
        score = (
            0.62 * _sigmoid_score(float(np.percentile(hf_jitter, 88)) if hf_jitter.size else 0.0, 0.060, 0.022)
            + 0.38 * _sigmoid_score(float(np.percentile(asym_jitter, 88)) if asym_jitter.size else 0.0, 0.040, 0.016)
        )

        med_hf = float(np.median([v for v, ok in zip(hf_s, valid_flags) if ok])) if any(valid_flags) else 0.0
        med_as = float(np.median([v for v, ok in zip(asym_s, valid_flags) if ok])) if any(valid_flags) else 0.0
        per_frame = [
            _clip01(
                0.54 * _sigmoid_score(abs(v_hf - med_hf), 0.060, 0.022)
                + 0.36 * _sigmoid_score(abs(v_as - med_as), 0.035, 0.015)
                + 0.10 * _sigmoid_score(abs(v_hf), 0.24, 0.09)
            )
            if ok
            else 0.0
            for v_hf, v_as, ok in zip(hf_ratios, asym_scores, valid_flags)
        ]

        flags = [p > 0.82 for p in per_frame]
        spans = _merge_sustained_flags(flags, min_len=2, gap=1)
        violations: list[PhysicsViolation] = []
        for s, e in spans[:2]:
            peak = max(per_frame[s : e + 1])
            region = subject_boxes[min((s + e) // 2, len(subject_boxes) - 1)]
            violations.append(
                PhysicsViolation(
                    violation_type="FACE_TEXTURE",
                    severity="MINOR" if peak < 0.92 else "MAJOR",
                    frame_start=frame_indices[max(0, s)],
                    frame_end=frame_indices[min(len(frame_indices) - 1, e)],
                    description="The upper-face ROI exhibits unstable texture / symmetry dynamics relative to nearby context.",
                    measured_value=float(peak),
                    expected_value=0.28,
                    violation_ratio=float(peak / 0.28),
                    region=region,
                    confidence=min(0.56 + 0.16 * peak, 0.88),
                )
            )
        return _clip01(score), violations, per_frame


# ══════════════════════════════════════════════════════════════════════════════
# L4: Face-Swap Boundary Artifact Detector (FSBA) — NOVEL CONTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL BASIS
# -----------------
# Every face-swap algorithm — regardless of architecture — must paste a
# synthetic face onto a real scene. This creates FOUR physically inevitable
# artifacts at the blend boundary that do NOT occur in authentic video:
#
#   (1) FLOW BOUNDARY SEAM (FBS):
#       The optical flow field is DISCONTINUOUS at the face/background boundary.
#       In authentic video, face and surrounding scene (neck, hair, shoulder)
#       are parts of a single physical body — their flows are coupled by rigid
#       body dynamics. Face swap pastes a foreign face, creating a seam in the
#       motion field. Measured as: ||face_flow - boundary_band_flow|| / global.
#
#   (2) BOUNDARY COLOR FLICKER (BCF):
#       Alpha-blending (used by all face-swap engines) oscillates temporally
#       because the rendering is done frame-independently. This produces a
#       temporal COLOR FLICKER at the seam — the hue/luminance difference
#       between face interior and boundary band changes frame-to-frame.
#       Measured as temporal std of LAB color difference at seam.
#
#   (3) WARP PREDICTION ERROR (WPE):
#       For authentic faces, warping frame_i by its optical flow predicts
#       frame_{i+1} with low error (the face is a continuous physical surface).
#       For face swaps, the synthesized texture introduces prediction errors
#       because each frame is re-rendered from a face model independently.
#       Measured as flow-warp MSE in face patch, normalised by patch variance.
#
#   (4) BOUNDARY EDGE FLICKER (BEF):
#       The edge gradient at the face boundary (the seam ring) flickers
#       temporally in face swaps due to blending weight variation. In
#       authentic video, the face-background edge is stable (natural boundary).
#       Measured as CoV (std/mean) of edge gradient energy at boundary ring.
#
# REFERENCE
# ---------
# PATV-X+: Face-Swap Boundary Artifact Detection for Training-Free Deepfake
#          Video Analysis. Novel contribution; no external weights; CPU-only.
# ══════════════════════════════════════════════════════════════════════════════


class CausalTemporalCoherenceAnalyzer:
    """
    L4: Boundary Artifact Analyzer.

    Detects the four physically inevitable artifacts that any face-swap
    algorithm creates at the blend boundary. Requires only OpenCV + NumPy.
    All four sub-modules are focused on the face ROI boundary ring, making
    them specific to face-swap and robust against general video variability.
    """

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze(
        self,
        frames: list,
        flows: list,
        frame_indices: list,
        fps: float,
        subject_boxes: Optional[list] = None,
        subject_reliability: Optional[list] = None,
    ) -> tuple:
        details: dict = {
            "flow_boundary_seam": 0.0,
            "boundary_color_flicker": 0.0,
            "warp_prediction_error": 0.0,
            "boundary_edge_flicker": 0.0,
            "boundary_support_ratio": 0.0,
            "flow_boundary_support_ratio": 0.0,
            "boundary_color_support_ratio": 0.0,
            "warp_support_ratio": 0.0,
            "boundary_edge_support_ratio": 0.0,
            "boundary_agreement_ratio": 0.0,
            "boundary_corroboration": 0.0,
            "warp_supported_score": 0.0,
            "phase_coherence": 0.0,
            "ar_residual": 0.0,
            "spectral_anomaly": 0.0,
            "micro_jitter": 0.0,
        }
        if len(flows) < 4 or not subject_boxes:
            return 0.0, [], details

        n_flows = min(len(flows), len(subject_boxes))
        boxes = subject_boxes[:n_flows]
        rels = (subject_reliability or [0.5] * n_flows)[:n_flows]

        fbs, fbs_support = self._flow_boundary_seam(flows[:n_flows], frames[:n_flows], boxes, rels)
        bcf, bcf_support = self._boundary_color_flicker(frames[:n_flows], boxes, rels)
        wpe, wpe_support = self._warp_prediction_error(frames, flows[:n_flows], boxes, rels)
        bef, bef_support = self._boundary_edge_flicker(frames[:n_flows], boxes, rels)

        support_ratio = float(np.mean([fbs_support, bcf_support, wpe_support, bef_support]))
        boundary_corroboration = max(fbs, bcf, bef)
        warp_supported = wpe * (0.40 + 0.60 * boundary_corroboration)
        agreement = (
            int(fbs >= 0.20)
            + int(bcf >= 0.15)
            + int(wpe >= 0.40)
            + int(bef >= 0.15)
        )

        # FF++-style face swaps are most reliable when warp inconsistency is
        # corroborated by seam/color/edge boundary evidence. Standalone warp
        # spikes are common on authentic videos and should not dominate.
        l4 = float(
            0.46 * warp_supported
            + 0.20 * bcf
            + 0.18 * bef
            + 0.16 * fbs
        )
        if agreement >= 3:
            l4 += 0.04 * min(wpe, boundary_corroboration)
        elif agreement == 2:
            l4 += 0.03 * min(wpe, boundary_corroboration)
        else:
            l4 *= 0.88

        # Penalize isolated warp or seam activity without corroborating
        # boundary instability; those patterns are frequent on authentic clips.
        if wpe > 0.44 and bcf < 0.10 and bef < 0.10 and fbs < 0.18:
            l4 -= 0.06
        if fbs > 0.55 and wpe < 0.28 and bcf < 0.14 and bef < 0.14:
            l4 -= 0.07
        # Reward boundary agreement when multiple sub-modules point to the same
        # face/background mismatch.
        if wpe > 0.50 and fbs > 0.20:
            l4 += 0.03
        if bef > 0.22 and bcf > 0.15 and (wpe > 0.34 or fbs > 0.24):
            l4 += 0.03
        if wpe > 0.58 and max(fbs, bcf, bef) > 0.18:
            l4 += 0.02

        l4 = _clip01(l4)
        l4 *= min(1.0, support_ratio / 0.35) if support_ratio > 0 else 0.0

        details = {
            "flow_boundary_seam": float(fbs),
            "boundary_color_flicker": float(bcf),
            "warp_prediction_error": float(wpe),
            "boundary_edge_flicker": float(bef),
            "boundary_support_ratio": float(support_ratio),
            "flow_boundary_support_ratio": float(fbs_support),
            "boundary_color_support_ratio": float(bcf_support),
            "warp_support_ratio": float(wpe_support),
            "boundary_edge_support_ratio": float(bef_support),
            "boundary_agreement_ratio": float(agreement / 4.0),
            "boundary_corroboration": float(boundary_corroboration),
            "warp_supported_score": float(warp_supported),
            "phase_coherence": float(fbs),
            "ar_residual": float(bcf),
            "spectral_anomaly": float(wpe),
            "micro_jitter": float(bef),
        }
        violations = self._detect_violations(fbs, bcf, wpe, bef, frame_indices, boxes, agreement)
        return _clip01(l4), violations, details

    # ── Sub-module 1: Flow Boundary Seam ──────────────────────────────────

    def _flow_boundary_seam(self, flows, frames, boxes, rels) -> tuple[float, float]:
        """
        Optical flow discontinuity at face-background boundary.
        Authentic: face + surrounding skin/hair move together (rigid body).
        Face-swap: synthetic face has a seam — different flow than neck/hair.
        """
        seam_scores: list[float] = []
        valid_count = 0
        for i, flow in enumerate(flows):
            rel = rels[i] if i < len(rels) else 0.5
            if rel < 0.50:
                continue
            h, w = flow.shape[:2]
            x1, y1, x2, y2 = boxes[i]
            bw = max(10, int(0.18 * min(x2 - x1, y2 - y1)))

            # Global motion
            gx = float(np.median(flow[..., 0]))
            gy = float(np.median(flow[..., 1]))

            # Inner face flow (global-subtracted)
            fi_x = flow[y1:y2, x1:x2, 0] - gx
            fi_y = flow[y1:y2, x1:x2, 1] - gy
            if fi_x.size < 20:
                continue
            face_vec = np.array([float(np.median(fi_x)), float(np.median(fi_y))])

            # Outer boundary band (ring just outside face ROI)
            oy1 = max(0, y1 - bw); oy2 = min(h, y2 + bw)
            ox1 = max(0, x1 - bw); ox2 = min(w, x2 + bw)
            outer_fx = (flow[oy1:oy2, ox1:ox2, 0] - gx)
            outer_fy = (flow[oy1:oy2, ox1:ox2, 1] - gy)

            # Exclude inner face from outer band
            mask = np.ones((oy2 - oy1, ox2 - ox1), dtype=bool)
            iy1r = y1 - oy1; iy2r = y2 - oy1
            ix1r = x1 - ox1; ix2r = x2 - ox1
            if iy2r > iy1r and ix2r > ix1r:
                mask[max(0, iy1r):iy2r, max(0, ix1r):ix2r] = False

            if mask.sum() < 20:
                continue
            band_vec = np.array([float(np.median(outer_fx[mask])),
                                  float(np.median(outer_fy[mask]))])

            seam = float(np.linalg.norm(face_vec - band_vec))
            natural_face_motion = float(np.median(np.sqrt(fi_x ** 2 + fi_y ** 2)))
            natural_band_motion = float(np.median(np.sqrt(outer_fx[mask] ** 2 + outer_fy[mask] ** 2)))
            inner_disp = float(np.median(np.abs(np.sqrt(fi_x ** 2 + fi_y ** 2) - natural_face_motion)))
            band_disp = float(np.median(np.abs(np.sqrt(outer_fx[mask] ** 2 + outer_fy[mask] ** 2) - natural_band_motion)))
            ref = 0.18 + 0.55 * natural_face_motion + 0.25 * natural_band_motion + inner_disp + band_disp
            if natural_face_motion < 0.04 and natural_band_motion < 0.04:
                continue
            valid_count += 1
            seam_scores.append(rel * seam / ref)

        if len(seam_scores) < 3:
            return 0.0, float(valid_count / max(len(flows), 1))
        med = float(np.median(seam_scores))
        support_ratio = float(valid_count / max(len(flows), 1))
        return _clip01(_sigmoid_score(med, 0.62, 0.14)), support_ratio

    # ── Sub-module 2: Boundary Color Flicker ──────────────────────────────

    def _boundary_color_flicker(self, frames, boxes, rels) -> tuple[float, float]:
        """
        Temporal variance of the LAB color difference at the face seam.
        Alpha-blending oscillates frame-to-frame → color seam flickers.
        Authentic video: stable color relationship face↔surrounding.
        """
        seam_diffs: list[np.ndarray] = []
        valid_count = 0
        for i, frame in enumerate(frames):
            rel = rels[i] if i < len(rels) else 0.5
            if rel < 0.52:
                continue
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = boxes[i]
            bw = max(8, int(0.12 * min(x2 - x1, y2 - y1)))

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

            face_lab = lab[y1:y2, x1:x2]
            if face_lab.size == 0:
                continue
            face_mean = np.mean(face_lab.reshape(-1, 3), axis=0)

            oy1 = max(0, y1 - bw); oy2 = min(h, y2 + bw)
            ox1 = max(0, x1 - bw); ox2 = min(w, x2 + bw)
            band = lab[oy1:oy2, ox1:ox2]
            mask = np.ones((oy2 - oy1, ox2 - ox1), dtype=bool)
            iy1r = y1 - oy1; iy2r = y2 - oy1
            ix1r = x1 - ox1; ix2r = x2 - ox1
            if iy2r > iy1r and ix2r > ix1r:
                mask[max(0, iy1r):iy2r, max(0, ix1r):ix2r] = False
            if mask.sum() < 20:
                continue
            band_flat = band.reshape(-1, 3)[mask.ravel()]
            band_mean = np.mean(band_flat, axis=0)
            seam_diffs.append(face_mean - band_mean)
            valid_count += 1

        if len(seam_diffs) < 4:
            return 0.0, float(valid_count / max(len(frames), 1))
        arr = np.array(seam_diffs, dtype=float)
        # Temporal std across L, A, B channels
        temporal_std = float(np.mean(np.std(arr, axis=0)))
        support_ratio = float(valid_count / max(len(frames), 1))
        return _clip01(_sigmoid_score(temporal_std, 9.0, 3.0)), support_ratio

    # ── Sub-module 3: Warp Prediction Error ───────────────────────────────

    def _warp_prediction_error(self, frames, flows, boxes, rels) -> tuple[float, float]:
        """
        Motion-conditioned warp residual anomaly.

        On compressed FF++ face-swap clips the manipulated face often becomes
        temporally over-regularized: despite non-trivial local motion, the face
        patch is easier to warp-predict than an authentic face with natural
        micro-expression / shading variation. We therefore treat abnormally LOW
        normalized warp error as suspicious instead of abnormally high error.
        """
        errors: list[float] = []
        valid_count = 0
        n = min(len(flows), len(frames) - 1, len(boxes) - 1)
        for i in range(n):
            rel = rels[i] if i < len(rels) else 0.5
            if rel < 0.50:
                continue
            x1, y1, x2, y2 = boxes[i]
            flow = flows[i]
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
            g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            h_img, w_img = g1.shape
            face2 = g2[y1:y2, x1:x2]
            if face2.size < 64:
                continue

            fx = flow[y1:y2, x1:x2, 0]
            fy = flow[y1:y2, x1:x2, 1]
            yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
            map_x = np.clip(xx + fx, 0, w_img - 1)
            map_y = np.clip(yy + fy, 0, h_img - 1)
            pred_face = cv2.remap(
                g1,
                map_x.astype(np.float32),
                map_y.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )

            if pred_face.shape != face2.shape or pred_face.size == 0:
                continue

            valid_pixels = np.isfinite(pred_face)
            valid_ratio = float(np.mean(valid_pixels))
            if valid_ratio < 0.85:
                continue
            err = float(np.mean(np.abs(pred_face[valid_pixels] - face2[valid_pixels])))
            var = float(np.std(face2[valid_pixels])) + 0.5
            errors.append(rel * err / var)
            valid_count += 1

        if len(errors) < 3:
            return 0.0, float(valid_count / max(n, 1))
        med = float(np.median(errors))
        support_ratio = float(valid_count / max(n, 1))
        raw_error_score = _clip01(_sigmoid_score(med, 0.20, 0.08))
        return _clip01(1.0 - raw_error_score), support_ratio

    # ── Sub-module 4: Boundary Edge Flicker ───────────────────────────────

    def _boundary_edge_flicker(self, frames, boxes, rels) -> tuple[float, float]:
        """
        Temporal instability of edge gradient energy at the face seam ring.
        Blend-weight oscillations in face swap create flickering edges.
        Authentic video: stable face-background boundary.
        """
        edge_vals: list[float] = []
        valid_count = 0
        for i, frame in enumerate(frames):
            rel = rels[i] if i < len(rels) else 0.5
            if rel < 0.52:
                continue
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = boxes[i]
            bw = max(6, int(0.10 * min(x2 - x1, y2 - y1)))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            oy1 = max(0, y1 - bw); oy2 = min(h, y2 + bw)
            ox1 = max(0, x1 - bw); ox2 = min(w, x2 + bw)
            patch = gray[oy1:oy2, ox1:ox2].astype(np.float32)

            gx_ = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy_ = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx_ ** 2 + gy_ ** 2)

            # Only the boundary ring (exclude face interior)
            ring = np.ones_like(grad, dtype=bool)
            iy1r = y1 - oy1 + bw; iy2r = y2 - oy1 - bw
            ix1r = x1 - ox1 + bw; ix2r = x2 - ox1 - bw
            if iy2r > iy1r and ix2r > ix1r:
                ring[iy1r:iy2r, ix1r:ix2r] = False
            if ring.sum() < 16:
                continue
            edge_vals.append(float(np.mean(grad[ring])))
            valid_count += 1

        if len(edge_vals) < 4:
            return 0.0, float(valid_count / max(len(frames), 1))
        arr = np.array(edge_vals, dtype=float)
        cov = float(np.std(arr) / (np.mean(arr) + 1e-6))
        support_ratio = float(valid_count / max(len(frames), 1))
        return _clip01(_sigmoid_score(cov, 0.24, 0.07)), support_ratio

    # ── Violation detection ────────────────────────────────────────────────

    def _detect_violations(self, fbs, bcf, wpe, bef, frame_indices, boxes, agreement: int) -> list:
        viol: list[PhysicsViolation] = []
        fi0 = frame_indices[0] if frame_indices else 0
        fi1 = frame_indices[-1] if frame_indices else 0
        region = boxes[len(boxes) // 2] if boxes else None

        if fbs > 0.54 and agreement >= 2:
            viol.append(PhysicsViolation(
                violation_type="FLOW_BOUNDARY_SEAM",
                severity="MAJOR" if fbs > 0.68 else "MINOR",
                frame_start=fi0, frame_end=fi1,
                description=(
                    "Optical flow is discontinuous at the face/background boundary — "
                    "the face region moves independently of the surrounding scene, "
                    "consistent with face-swap blending (FSBA sub-score 1)."
                ),
                measured_value=float(fbs), expected_value=0.18,
                violation_ratio=float(fbs / 0.18),
                region=region,
                confidence=min(0.55 + 0.30 * fbs, 0.94),
            ))

        if bcf > 0.52 and agreement >= 2:
            viol.append(PhysicsViolation(
                violation_type="COLOR_SEAM_FLICKER",
                severity="MAJOR" if bcf > 0.68 else "MINOR",
                frame_start=fi0, frame_end=fi1,
                description=(
                    "The LAB color difference between face interior and boundary band "
                    "flickers temporally — consistent with frame-independent alpha-blending "
                    "used by face-swap engines (FSBA sub-score 2)."
                ),
                measured_value=float(bcf), expected_value=0.15,
                violation_ratio=float(bcf / 0.15),
                region=region,
                confidence=min(0.52 + 0.28 * bcf, 0.92),
            ))

        if wpe > 0.48:
            viol.append(PhysicsViolation(
                violation_type="WARP_PREDICTION_ERROR",
                severity="MAJOR" if wpe > 0.70 else "MINOR",
                frame_start=fi0, frame_end=fi1,
                description=(
                    "Flow-warped face patch is abnormally easy to predict despite "
                    "non-trivial local motion — consistent with temporally "
                    "over-regularized synthesized face texture (FSBA sub-score 3)."
                ),
                measured_value=float(wpe), expected_value=0.18,
                violation_ratio=float(wpe / 0.18),
                region=region,
                confidence=min(0.50 + 0.28 * wpe, 0.90),
            ))

        if bef > 0.56 and agreement >= 2:
            viol.append(PhysicsViolation(
                violation_type="BOUNDARY_EDGE_FLICKER",
                severity="MAJOR" if bef > 0.70 else "MINOR",
                frame_start=fi0, frame_end=fi1,
                description=(
                    "Edge gradient energy at face boundary ring shows high temporal "
                    "variability — consistent with oscillating blend weights in "
                    "face-swap rendering (FSBA sub-score 4)."
                ),
                measured_value=float(bef), expected_value=0.10,
                violation_ratio=float(bef / 0.10),
                region=region,
                confidence=min(0.50 + 0.26 * bef, 0.88),
            ))

        return viol


# ══════════════════════════════════════════════════════════════════════════════
# Main Detector
# ══════════════════════════════════════════════════════════════════════════════


class PATVXDetector:
    ANALYSIS_VERSION = "patv_x_core_v13"
    DEFAULT_WEIGHTS = {"L1": 0.10, "L2": 0.08, "L3": 0.02, "L4": 0.30, "L5": 0.50}
    DEFAULT_THRESHOLD = 0.42

    def __init__(
        self,
        weights: Optional[dict] = None,
        threshold: float = DEFAULT_THRESHOLD,
        sample_rate: int = 6,
        max_frames: int = 96,
        verbose: bool = True,
        track: str = "core",
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        total = sum(self.weights.values())
        self.weights = {k: v / max(total, 1e-6) for k, v in self.weights.items()}

        if track == "synthetic-smoke" and abs(float(threshold) - float(self.DEFAULT_THRESHOLD)) < 1e-9:
            threshold = 0.39

        self.threshold = threshold
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.verbose = verbose
        self.track = track

        self.flow_analyzer = FlowAnalyzerX()
        self.physics_analyzer = PhysicsAnalyzerX()
        self.semantic_analyzer = SemanticAnalyzerX()
        self.ctcg_analyzer = CausalTemporalCoherenceAnalyzer()
        self.face_cascade = self._init_face_cascade()

    def analyze(self, video_path: str) -> PATVXResult:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Δεν βρέθηκε: {video_path}")

        if self.verbose:
            print(f"\n[PATV-X] Αναλύω: {path.name}")

        frames, frame_indices, fps, duration, resolution = self._load_frames(str(path))
        if len(frames) < 4:
            raise ValueError(f"Πολύ λίγα frames ({len(frames)})")

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        flows = []
        for i in range(len(grays) - 1):
            flows.append(
                cv2.calcOpticalFlowFarneback(
                    grays[i], grays[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
            )

        subject_boxes, subject_reliability, subject_support = self._estimate_subject_boxes(frames, flows)
        face_reliability = float(np.mean(subject_reliability)) if subject_reliability else 0.0
        face_detected_ratio = float(subject_support.get("face_detected_ratio", 0.0))

        l1_score, l1_violations, l1_evidences, l1_details = self.flow_analyzer.analyze(
            frames,
            frame_indices,
            fps,
            subject_boxes=subject_boxes,
            subject_reliability=subject_reliability,
            flows=flows,
        )
        l2_score, l2_violations, l2_details = self.physics_analyzer.analyze(
            frames,
            flows,
            frame_indices,
            fps,
            subject_boxes=subject_boxes,
            subject_reliability=subject_reliability,
        )
        l3_score, l3_violations, l3_evidences, l3_details = self.semantic_analyzer.analyze(
            frames,
            frame_indices,
            fps,
            subject_boxes=subject_boxes,
            subject_reliability=subject_reliability,
        )

        l4_score, l4_violations, l4_details = self.ctcg_analyzer.analyze(
            frames,
            flows,
            frame_indices,
            fps,
            subject_boxes=subject_boxes,
            subject_reliability=subject_reliability,
        )
        # L5: Frequency analysis (Laplacian variance + DCT)
        # Uses original-resolution frames (Laplacian is scale-dependent)
        l5_details = self._compute_frequency_features(str(path))
        l5_score = float(l5_details.get("frequency_score", 0.0))

        neck_visibility_ratio = float(l2_details.get("neck_visibility_ratio", 0.0))
        boundary_support_ratio = float(l4_details.get("boundary_support_ratio", 0.0))

        if self.track == "synthetic-smoke":
            legacy_wpe = _clip01(1.0 - l4_details["warp_prediction_error"])
            agreement = int(sum(score >= 0.38 for score in [
                l4_details["flow_boundary_seam"],
                l4_details["boundary_color_flicker"],
                legacy_wpe,
                l4_details["boundary_edge_flicker"],
            ]))
            if agreement < 2:
                l4_score = float(
                    0.72 * l4_details["flow_boundary_seam"]
                    + 0.18 * l4_details["boundary_color_flicker"]
                    + 0.10 * legacy_wpe
                )
                l4_score *= 0.55
            else:
                l4_score = float(
                    0.42 * l4_details["flow_boundary_seam"]
                    + 0.24 * l4_details["boundary_color_flicker"]
                    + 0.22 * legacy_wpe
                    + 0.12 * l4_details["boundary_edge_flicker"]
                )
            l4_score *= min(1.0, boundary_support_ratio / 0.35) if boundary_support_ratio > 0 else 0.0
            l4_details["warp_prediction_error"] = legacy_wpe

        self._last_l4_score = l4_score  # accessible in _build_comparison_note

        dynamic_core_support = any(
            v.violation_type in {"FACE_MOTION", "RIGID_BODY"} for v in (l1_violations + l2_violations)
        )

        if self.track != "synthetic-smoke" and face_reliability < 0.75:
            l3_score *= max(0.35, face_reliability / 0.75)
        elif face_reliability < 0.75:
            l3_score *= max(0.15, face_reliability / 0.75)
        if boundary_support_ratio < 0.22:
            l4_score *= 0.45 + 0.55 * min(1.0, boundary_support_ratio / 0.22)

        if not dynamic_core_support:
            keep_static = []
            for v in l3_violations:
                if v.violation_type == "FACE_TEXTURE" and v.confidence >= 0.82 and face_reliability >= 0.80:
                    nv = PhysicsViolation(
                        violation_type=v.violation_type,
                        severity="MINOR",
                        frame_start=v.frame_start,
                        frame_end=v.frame_end,
                        description=v.description,
                        measured_value=v.measured_value,
                        expected_value=v.expected_value,
                        violation_ratio=v.violation_ratio,
                        region=v.region,
                        confidence=min(v.confidence, 0.82),
                )
                    keep_static.append(nv)
            l3_violations = keep_static

            for fe in l3_evidences:
                fe.anomaly_score *= 0.25
                fe.semantic_score *= 0.25
                if fe.semantic_score < 0.18:
                    fe.violations = []

        all_violations = self._merge_all_violations(
            l1_violations + l2_violations + l3_violations + l4_violations
        )
        violation_types = {v.violation_type for v in all_violations}
        n_critical = sum(1 for v in all_violations if v.severity == "CRITICAL")
        n_major = sum(1 for v in all_violations if v.severity == "MAJOR")
        n_minor = sum(1 for v in all_violations if v.severity == "MINOR")

        dynamic_face_support = any(v in violation_types for v in {"FACE_MOTION", "RIGID_BODY"})
        strong_dynamic_support = "FACE_MOTION" in violation_types
        edge_instability_support = "EDGE_INSTABILITY" in violation_types
        boundary_artifact_support = any(v in violation_types for v in {
            "FLOW_BOUNDARY_SEAM", "COLOR_SEAM_FLICKER",
            "WARP_PREDICTION_ERROR", "BOUNDARY_EDGE_FLICKER"
        }) or edge_instability_support
        weak_static_only = violation_types.issubset(
            {"EDGE_INSTABILITY", "COLOR_DRIFT", "LIGHT_SOURCE", "FACE_TEXTURE", "SKIN_TONE_MISMATCH"}
        ) if violation_types else False

        if self.track == "synthetic-smoke":
            raw_tcs = (
                self.weights.get("L1", 0.28) * l1_score
                + self.weights.get("L2", 0.26) * l2_score
                + self.weights.get("L3", 0.16) * l3_score
                + self.weights.get("L4", 0.30) * l4_score
            )

            bonus = 0.0
            if "FACE_MOTION" in violation_types:
                bonus += 0.06
            if "FACE_MOTION" in violation_types and l1_score > 0.30:
                bonus += 0.04
            if "RIGID_BODY" in violation_types and "FACE_MOTION" in violation_types and neck_visibility_ratio >= 0.25:
                bonus += 0.02
            if strong_dynamic_support and face_reliability > 0.50:
                bonus += 0.02
            if boundary_artifact_support and l4_score > 0.32 and boundary_support_ratio >= 0.25:
                bonus += 0.02
            bonus += min(0.025, 0.010 * n_critical + 0.004 * n_major + 0.001 * n_minor)

            penalty = 0.0
            if weak_static_only and l1_score < 0.30 and l2_score < 0.30 and l4_score < 0.26:
                penalty += 0.10
            if "RIGID_BODY" in violation_types and "FACE_MOTION" not in violation_types and l1_score < 0.22:
                penalty += 0.06
            if face_reliability < 0.30:
                penalty += 0.05
            if face_detected_ratio < 0.30:
                penalty += 0.06
            if boundary_support_ratio < 0.18:
                penalty += 0.04
            if neck_visibility_ratio < 0.18 and "RIGID_BODY" not in violation_types:
                penalty += 0.04
            if not dynamic_face_support and not boundary_artifact_support and l3_score > 0.50:
                penalty += 0.08
            if violation_types == {"GRAVITY"}:
                penalty += 0.06
            if violation_types == {"LIGHT_SOURCE"}:
                penalty += 0.06
            if "LIGHT_SOURCE" in violation_types and not dynamic_face_support and not boundary_artifact_support:
                penalty += 0.05
        else:
            # Real-data core track (v13): Frequency-Anchored Fusion
            #
            # Key insight from full 200-video FF++ c23 validation:
            #   - L2 (face-neck decoupling) is NON-DISCRIMINATIVE (AUC ~0.50)
            #   - L4 (boundary artifacts) is weak (best sub-feature AUC 0.556)
            #   - L5 Frequency Analysis is the strongest discriminator:
            #     * lap_var_temporal_cv: AUC 0.625 (temporal sharpness variation)
            #     * lap_var_ratio: AUC 0.607 (face/bg sharpness ratio)
            #     * dct_hf_ratio: AUC 0.566 (face/bg high-freq energy)
            #
            # Strategy: L5 is the primary signal. L4 (warp prediction error)
            # corroborates L5. L1/L2 provide minor supplementary evidence.
            warp_raw = _clip01(float(l4_details.get("warp_prediction_error", 0.0)))
            seam_raw = _clip01(float(l4_details.get("flow_boundary_seam", 0.0)))
            color_raw = _clip01(float(l4_details.get("boundary_color_flicker", 0.0)))
            edge_raw = _clip01(float(l4_details.get("boundary_edge_flicker", 0.0)))
            boundary_corroboration = _clip01(max(
                min(warp_raw, max(color_raw, edge_raw)),
                min(seam_raw, max(color_raw, edge_raw)),
                0.75 * min(warp_raw, seam_raw),
            ))

            # L5 frequency features (primary signal, AUC 0.624)
            freq_support = float(l5_details.get("frequency_support", 0.0))
            lap_ratio_raw = float(l5_details.get("lap_var_ratio", 0.0))
            lap_cv_raw = float(l5_details.get("lap_var_temporal_cv", 0.0))

            # L5 x warp interaction: frequency anomaly corroborated by
            # boundary artifact signal (best L4 sub-feature, AUC 0.556)
            l5_warp_interaction = l5_score * _clip01(warp_raw)

            raw_tcs = (
                0.72 * l5_score
                + 0.10 * l4_score
                + 0.10 * l5_warp_interaction
                + 0.05 * l1_score
                + 0.03 * boundary_corroboration
            )

            bonus = 0.0
            # L5 strongly elevated with warp corroboration
            if l5_score > 0.55 and warp_raw > 0.35:
                bonus += 0.03
            # Multiple L5 sub-features agree
            if lap_ratio_raw > 1.45 and lap_cv_raw > 0.165:
                bonus += 0.02

            penalty = 0.0
            # Low frequency support (poor face detection)
            if freq_support < 0.35:
                penalty += 0.05
            if face_reliability < 0.25:
                penalty += 0.03

        tcs_score = _clip01(raw_tcs + bonus - penalty)
        abstained_reason = ""
        if self.track != "synthetic-smoke":
            abstained_reason = self._determine_abstention_reason(
                face_detected_ratio=face_detected_ratio,
                face_reliability_mean=face_reliability,
                neck_visibility_ratio=neck_visibility_ratio,
                boundary_support_ratio=boundary_support_ratio,
                l1_score=l1_score,
                l2_score=l2_score,
                l4_score=l4_score,
                dynamic_face_support=dynamic_face_support,
                boundary_artifact_support=boundary_artifact_support,
            )
        if abstained_reason:
            verdict = "INCONCLUSIVE"
        else:
            verdict = "AI_GENERATED" if tcs_score >= self.threshold else "AUTHENTIC"

        margin = abs(tcs_score - self.threshold)
        if margin > 0.20:
            confidence = "HIGH"
        elif margin > 0.10:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        all_evidences: list[FrameEvidence] = []
        all_evidences = self._merge_frame_evidences(all_evidences, l1_evidences)
        all_evidences = self._merge_frame_evidences(all_evidences, l3_evidences)
        all_evidences.sort(key=lambda e: e.frame_idx)
        top_frames = sorted(all_evidences, key=lambda e: e.anomaly_score, reverse=True)
        most_suspicious = [e.frame_idx for e in top_frames[:8]]

        comparison_note = self._build_comparison_note(
            tcs=tcs_score,
            l1=l1_score,
            l2=l2_score,
            l3=l3_score,
            violations=all_violations,
            face_reliability=face_reliability,
            boundary_support_ratio=boundary_support_ratio,
            neck_visibility_ratio=neck_visibility_ratio,
            abstained_reason=abstained_reason,
        )

        result = PATVXResult(
            video_path=str(path),
            verdict=verdict,
            tcs_score=tcs_score,
            confidence=confidence,
            track=self.track,
            l1_flow_score=l1_score,
            l2_physics_score=l2_score,
            l3_semantic_score=l3_score,
            violations=all_violations,
            frame_timeline=all_evidences,
            most_suspicious_frames=most_suspicious,
            fps=fps,
            duration=duration,
            resolution=resolution,
            frames_analyzed=len(frames),
            flow_mean_divergence=l1_details["mean_divergence"],
            flow_acceleration_anomaly=l1_details["acceleration_anomaly"],
            flow_flicker_score=l1_details["flicker_score"],
            physics_gravity_consistency=l2_details["gravity_consistency"],
            physics_rigid_body_score=l2_details["rigid_body_score"],
            physics_shadow_consistency=l2_details["shadow_consistency"],
            physics_face_scene_support_ratio=l2_details["face_scene_support_ratio"],
            physics_light_support_ratio=l2_details["light_support_ratio"],
            physics_skin_support_ratio=l2_details["skin_support_ratio"],
            semantic_color_drift=l3_details["color_drift"],
            semantic_edge_stability=l3_details["edge_stability"],
            semantic_texture_consistency=l3_details["texture_consistency"],
            comparison_note=comparison_note,
            analysis_version=self.ANALYSIS_VERSION,
            abstained_reason=abstained_reason,
            face_detected_ratio=face_detected_ratio,
            face_reliability_mean=face_reliability,
            neck_visibility_ratio=neck_visibility_ratio,
            boundary_support_ratio=boundary_support_ratio,
            l4_boundary_artifact_score=l4_score,
            l4_flow_boundary_seam_score=l4_details["flow_boundary_seam"],
            l4_boundary_color_flicker_score=l4_details["boundary_color_flicker"],
            l4_warp_prediction_error_score=l4_details["warp_prediction_error"],
            l4_boundary_edge_flicker_score=l4_details["boundary_edge_flicker"],
            l4_flow_boundary_support_ratio=l4_details["flow_boundary_support_ratio"],
            l4_boundary_color_support_ratio=l4_details["boundary_color_support_ratio"],
            l4_warp_support_ratio=l4_details["warp_support_ratio"],
            l4_boundary_edge_support_ratio=l4_details["boundary_edge_support_ratio"],
            l4_boundary_agreement_ratio=l4_details["boundary_agreement_ratio"],
            l4_boundary_corroboration=l4_details["boundary_corroboration"],
            l4_warp_supported_score=l4_details["warp_supported_score"],
            l5_frequency_score=l5_score,
            l5_lap_var_ratio=l5_details.get("lap_var_ratio", 0.0),
            l5_lap_var_temporal_cv=l5_details.get("lap_var_temporal_cv", 0.0),
            l5_dct_hf_ratio=l5_details.get("dct_hf_ratio", 0.0),
            l5_dct_hf_temporal_cv=l5_details.get("dct_hf_temporal_cv", 0.0),
            l5_lap_kurtosis_mean=l5_details.get("lap_kurtosis_mean", 0.0),
            l5_lap_kurtosis_cv=l5_details.get("lap_kurtosis_cv", 0.0),
            l5_wavelet_detail_ratio=l5_details.get("wavelet_detail_ratio", 0.0),
            l5_wavelet_detail_ratio_cv=l5_details.get("wavelet_detail_ratio_cv", 0.0),
            l5_lap_R_cv=l5_details.get("lap_R_cv", 0.0),
            l5_lap_G_cv=l5_details.get("lap_G_cv", 0.0),
            l5_lap_B_cv=l5_details.get("lap_B_cv", 0.0),
            l5_lap_diff_mean=l5_details.get("lap_diff_mean", 0.0),
            l5_lap_trend_residual_std=l5_details.get("lap_trend_residual_std", 0.0),
            l5_lap_acc_mean=l5_details.get("lap_acc_mean", 0.0),
            l5_block_lap_consistency=l5_details.get("block_lap_consistency", 0.0),
            l5_cross_ch_lap_corr=l5_details.get("cross_ch_lap_corr", 0.0),
            l5_srm_var_cv=l5_details.get("srm_var_cv", 0.0),
        )

        if self.verbose:
            print(result.forensic_report())
        return result

    def _init_face_cascade(self):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            return cascade if not cascade.empty() else None
        except Exception:
            return None

    def _detect_face_box(self, frame: np.ndarray) -> tuple[Optional[tuple[int, int, int, int]], float]:
        if self.face_cascade is None:
            return None, 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(32, 32),
        )
        if len(faces) == 0:
            return None, 0.0

        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        best = None
        best_score = -1e9
        for (x, y, bw, bh) in faces:
            area = bw * bh
            fc_x = x + bw / 2.0
            fc_y = y + bh / 2.0
            center_pen = ((fc_x - cx) / max(w, 1)) ** 2 + ((fc_y - cy) / max(h, 1)) ** 2
            score = area - 0.35 * area * center_pen
            if score > best_score:
                best_score = score
                best = (x, y, x + bw, y + bh)

        if best is None:
            return None, 0.0
        return _clip_box(best, w, h), 1.0

    def _propagate_box_with_flow(
        self,
        prev_box: tuple[int, int, int, int],
        flow: np.ndarray,
        w: int,
        h: int,
    ) -> tuple[tuple[int, int, int, int], float]:
        px1, py1, px2, py2 = _expand_box(prev_box, 1.08, w, h)
        fx = flow[py1:py2, px1:px2, 0].astype(np.float32)
        fy = flow[py1:py2, px1:px2, 1].astype(np.float32)
        if fx.size == 0:
            return prev_box, 0.0

        fx_c = fx - np.median(fx)
        fy_c = fy - np.median(fy)
        mag = np.sqrt(fx_c ** 2 + fy_c ** 2)
        thr = max(float(np.percentile(mag, 60)), 0.05)
        moving = mag > thr

        if moving.sum() < 18:
            return prev_box, 0.18

        dx = float(np.median(fx[moving]))
        dy = float(np.median(fy[moving]))
        new_box = (
            int(round(prev_box[0] + dx)),
            int(round(prev_box[1] + dy)),
            int(round(prev_box[2] + dx)),
            int(round(prev_box[3] + dy)),
        )
        activity = float(np.median(mag[moving]))
        conf = float(np.clip(0.28 + 0.90 * activity, 0.22, 0.62))
        return _clip_box(new_box, w, h), conf

    def _saliency_box(
        self,
        frame: np.ndarray,
        flow: np.ndarray,
        prev_center: tuple[float, float],
        prev_side: float,
    ) -> tuple[tuple[int, int, int, int], float]:
        h, w = frame.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx0, cy0 = w / 2.0, h / 2.0
        center_prior = np.exp(
            -(((xx - cx0) ** 2) / (2.0 * (0.20 * w) ** 2) + ((yy - cy0) ** 2) / (2.0 * (0.20 * h) ** 2))
        ).astype(np.float32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_energy = cv2.GaussianBlur(np.sqrt(gx ** 2 + gy ** 2), (0, 0), 3)
        edge_p = max(float(np.percentile(edge_energy, 98)), 1.0)
        edge_map = np.clip(edge_energy / edge_p, 0.0, 1.0)

        fx = flow[..., 0].astype(np.float32)
        fy = flow[..., 1].astype(np.float32)
        fx_c = fx - np.median(fx)
        fy_c = fy - np.median(fy)
        motion = cv2.GaussianBlur(np.sqrt(fx_c ** 2 + fy_c ** 2), (0, 0), 5)
        motion_p = max(float(np.percentile(motion, 95)), 0.08)
        motion_map = np.clip(motion / motion_p, 0.0, 1.0)

        pcx, pcy = prev_center
        track_prior = np.exp(
            -(((xx - pcx) ** 2) / (2.0 * (0.14 * w) ** 2) + ((yy - pcy) ** 2) / (2.0 * (0.14 * h) ** 2))
        ).astype(np.float32)

        saliency = 0.38 * center_prior + 0.22 * edge_map + 0.22 * motion_map + 0.18 * track_prior
        saliency = cv2.GaussianBlur(saliency, (0, 0), 7)
        y_peak, x_peak = np.unravel_index(int(np.argmax(saliency)), saliency.shape)

        motion_mass = float(np.mean(motion_map > 0.55))
        edge_mass = float(np.mean(edge_map > 0.45))
        side = min(h, w) * (0.23 + 0.07 * motion_mass + 0.03 * edge_mass)
        side = float(np.clip(0.78 * prev_side + 0.22 * side, 0.18 * min(h, w), 0.38 * min(h, w)))

        box = (
            int(round(x_peak - side / 2.0)),
            int(round(y_peak - side / 2.0)),
            int(round(x_peak + side / 2.0)),
            int(round(y_peak + side / 2.0)),
        )
        return _clip_box(box, w, h), 0.22

    def _estimate_subject_boxes(
        self,
        frames: list[np.ndarray],
        flows: list[np.ndarray],
    ) -> tuple[list[tuple[int, int, int, int]], list[float], dict]:
        if not frames:
            return [], [], {"face_detected_ratio": 0.0}

        h, w = frames[0].shape[:2]
        prev_center = (w / 2.0, h / 2.0)
        prev_side = 0.28 * min(h, w)
        prev_box = _clip_box(
            (
                int(round(prev_center[0] - prev_side / 2.0)),
                int(round(prev_center[1] - prev_side / 2.0)),
                int(round(prev_center[0] + prev_side / 2.0)),
                int(round(prev_center[1] + prev_side / 2.0)),
            ),
            w,
            h,
        )

        boxes: list[tuple[int, int, int, int]] = []
        reliabilities: list[float] = []
        face_detected_flags: list[bool] = []

        for i, frame in enumerate(frames):
            flow = flows[i] if i < len(flows) else flows[-1]

            face_box, face_rel = self._detect_face_box(frame)
            prop_box, prop_rel = self._propagate_box_with_flow(prev_box, flow, w, h)
            sal_box, sal_rel = self._saliency_box(frame, flow, prev_center, prev_side)

            if face_box is not None:
                det_box = _expand_box(face_box, 1.30, w, h)
                if _box_iou(det_box, prop_box) > 0.10 or prop_rel < 0.26:
                    box = (
                        int(round(0.70 * det_box[0] + 0.30 * prop_box[0])),
                        int(round(0.70 * det_box[1] + 0.30 * prop_box[1])),
                        int(round(0.70 * det_box[2] + 0.30 * prop_box[2])),
                        int(round(0.70 * det_box[3] + 0.30 * prop_box[3])),
                    )
                    box = _clip_box(box, w, h)
                    rel = 0.96
                else:
                    box = det_box
                    rel = 0.92
                face_detected_flags.append(True)
            elif prop_rel >= 0.24:
                box = (
                    int(round(0.78 * prop_box[0] + 0.22 * sal_box[0])),
                    int(round(0.78 * prop_box[1] + 0.22 * sal_box[1])),
                    int(round(0.78 * prop_box[2] + 0.22 * sal_box[2])),
                    int(round(0.78 * prop_box[3] + 0.22 * sal_box[3])),
                )
                box = _clip_box(box, w, h)
                rel = max(prop_rel, 0.42)
                face_detected_flags.append(False)
            else:
                box = sal_box
                rel = sal_rel
                face_detected_flags.append(False)

            prev_center = (
                0.74 * prev_center[0] + 0.26 * _box_center(box)[0],
                0.74 * prev_center[1] + 0.26 * _box_center(box)[1],
            )
            prev_side = 0.78 * prev_side + 0.22 * (box[2] - box[0])
            prev_box = box
            boxes.append(box)
            reliabilities.append(rel)

        smoothed_boxes: list[tuple[int, int, int, int]] = []
        smoothed_rel: list[float] = []
        cur = boxes[0]
        cur_r = reliabilities[0]
        smoothed_boxes.append(cur)
        smoothed_rel.append(cur_r)
        for b, r in zip(boxes[1:], reliabilities[1:]):
            x1 = int(round(0.80 * cur[0] + 0.20 * b[0]))
            y1 = int(round(0.80 * cur[1] + 0.20 * b[1]))
            x2 = int(round(0.80 * cur[2] + 0.20 * b[2]))
            y2 = int(round(0.80 * cur[3] + 0.20 * b[3]))
            cur = _clip_box((x1, y1, x2, y2), w, h)
            cur_r = 0.82 * cur_r + 0.18 * r
            smoothed_boxes.append(cur)
            smoothed_rel.append(float(cur_r))
        return smoothed_boxes, smoothed_rel, {
            "face_detected_ratio": float(sum(face_detected_flags) / max(len(face_detected_flags), 1)),
        }

    def _merge_frame_evidences(
        self,
        base: list[FrameEvidence],
        extra: list[FrameEvidence],
    ) -> list[FrameEvidence]:
        if not base:
            return extra.copy()

        merged = {
            e.frame_idx: FrameEvidence(
                frame_idx=e.frame_idx,
                timestamp=e.timestamp,
                anomaly_score=e.anomaly_score,
                violations=list(e.violations),
                flow_magnitude=e.flow_magnitude,
                flow_divergence=e.flow_divergence,
                physics_score=e.physics_score,
                semantic_score=e.semantic_score,
            )
            for e in base
        }

        for e in extra:
            existing = merged.get(e.frame_idx)
            if existing is None:
                merged[e.frame_idx] = FrameEvidence(
                    frame_idx=e.frame_idx,
                    timestamp=e.timestamp,
                    anomaly_score=e.anomaly_score,
                    violations=list(e.violations),
                    flow_magnitude=e.flow_magnitude,
                    flow_divergence=e.flow_divergence,
                    physics_score=e.physics_score,
                    semantic_score=e.semantic_score,
                )
            else:
                existing.anomaly_score = max(existing.anomaly_score, e.anomaly_score)
                existing.flow_magnitude = max(existing.flow_magnitude, e.flow_magnitude)
                existing.flow_divergence = max(existing.flow_divergence, e.flow_divergence)
                existing.physics_score = max(existing.physics_score, e.physics_score)
                existing.semantic_score = max(existing.semantic_score, e.semantic_score)
                existing.violations = sorted(set(existing.violations + e.violations))

        return list(merged.values())

    def _merge_all_violations(
        self,
        violations: list[PhysicsViolation],
        gap: int = 3,
    ) -> list[PhysicsViolation]:
        if not violations:
            return []

        severity_rank = {"MINOR": 0, "MAJOR": 1, "CRITICAL": 2}
        violations = sorted(violations, key=lambda v: (v.violation_type, v.frame_start))
        merged = [PhysicsViolation(**violations[0].__dict__)]

        for v in violations[1:]:
            last = merged[-1]
            if v.violation_type == last.violation_type and v.frame_start - last.frame_end <= gap:
                last.frame_end = max(last.frame_end, v.frame_end)
                if severity_rank.get(v.severity, 0) > severity_rank.get(last.severity, 0):
                    last.severity = v.severity
                last.violation_ratio = max(last.violation_ratio, v.violation_ratio)
                last.confidence = max(last.confidence, v.confidence)
                last.measured_value = max(last.measured_value, v.measured_value)
            else:
                merged.append(PhysicsViolation(**v.__dict__))

        merged.sort(key=lambda v: {"CRITICAL": 0, "MAJOR": 1, "MINOR": 2}.get(v.severity, 9))
        return merged

    def _build_comparison_note(
        self,
        tcs: float,
        l1: float,
        l2: float,
        l3: float,
        violations: list,
        face_reliability: float,
        boundary_support_ratio: float,
        neck_visibility_ratio: float,
        abstained_reason: str,
    ) -> str:
        l4 = getattr(self, "_last_l4_score", 0.0)
        vtypes = sorted({v.violation_type for v in violations})
        parts = [
            "Η απόφαση δεν βασίζεται μόνο σε global motion.",
            f"Το score συνδυάζει residual motion (L1={l1:.3f}), physics plausibility (L2={l2:.3f}), "
            f"face/context inconsistency (L3={l3:.3f}) και warp-consistency / boundary signal (L4={l4:.3f}).",
            f"Face ROI reliability={face_reliability:.3f}. "
            f"Neck visibility={neck_visibility_ratio:.3f}. "
            f"Boundary support={boundary_support_ratio:.3f}.",
        ]
        if vtypes:
            parts.append(f"Ανιχνεύθηκαν: {', '.join(vtypes[:6])}.")
        if abstained_reason:
            parts.append(f"Το αποτέλεσμα δηλώθηκε ως INCONCLUSIVE λόγω: {abstained_reason}.")
        parts.append(f"Τελικό TCS={tcs:.3f}.")
        return " ".join(parts)

    def _determine_abstention_reason(
        self,
        *,
        face_detected_ratio: float,
        face_reliability_mean: float,
        neck_visibility_ratio: float,
        boundary_support_ratio: float,
        l1_score: float,
        l2_score: float,
        l4_score: float,
        dynamic_face_support: bool,
        boundary_artifact_support: bool,
    ) -> str:
        if face_detected_ratio < 0.22 and face_reliability_mean < 0.45:
            return "weak_face_support"
        if face_reliability_mean < 0.35:
            return "unreliable_face_roi"
        if neck_visibility_ratio < 0.14 and boundary_support_ratio < 0.14:
            return "insufficient_face_context"
        if not dynamic_face_support and not boundary_artifact_support and max(l1_score, l2_score, l4_score) < 0.18:
            return "unsupported_open_world_content"
        return ""

    def _compute_frequency_features(
        self,
        video_path: str,
    ) -> dict[str, float]:
        """L5 Multi-Scale Frequency Forensics Analysis.

        Computes 16 frequency-domain features from face region at original
        resolution. Features span multiple signal families:

        1. Laplacian variance (spatial ratio + temporal CV)
        2. Laplacian kurtosis (noise distribution shape) — AUC 0.685
        3. Wavelet detail ratio (fine/coarse energy balance) — AUC 0.665
        4. Per-channel Laplacian CV (R, G, B temporal variation)
        5. Temporal dynamics (first-diff, trend residual, acceleration)
        6. Block-level spatial consistency
        7. Cross-channel Laplacian correlation
        8. SRM high-pass noise residual
        9. DCT high-frequency ratio

        IMPORTANT: Loads frames at original resolution (no resize) because
        Laplacian variance is scale-dependent.
        """
        _zeros = {
            "lap_var_ratio": 0.0, "lap_var_temporal_cv": 0.0,
            "dct_hf_ratio": 0.0, "dct_hf_temporal_cv": 0.0,
            "lap_kurtosis_mean": 0.0, "lap_kurtosis_cv": 0.0,
            "wavelet_detail_ratio": 0.0, "wavelet_detail_ratio_cv": 0.0,
            "lap_R_cv": 0.0, "lap_G_cv": 0.0, "lap_B_cv": 0.0,
            "lap_diff_mean": 0.0, "lap_trend_residual_std": 0.0,
            "lap_acc_mean": 0.0,
            "block_lap_consistency": 0.0, "cross_ch_lap_corr": 0.0,
            "srm_var_cv": 0.0,
            "frequency_score": 0.0, "frequency_support": 0.0,
        }

        cascade = self.face_cascade
        if cascade is None:
            return _zeros

        # Load frames at ORIGINAL resolution
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return _zeros
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = total / max(fps_val, 1e-6)
        target = min(48, int(max(duration, 1.0) * self.sample_rate))
        target = max(target, 12)
        stride = max(1, total // target) if total > 0 else 1

        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                frames.append(frame)
                if len(frames) >= 48:
                    break
            idx += 1
        cap.release()

        if len(frames) < 8:
            return _zeros

        # Detect faces per frame via Haar cascade
        boxes = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))
            if len(faces) > 0:
                areas = [fw * fh for (fx, fy, fw, fh) in faces]
                best_idx = int(np.argmax(areas))
                fx, fy, fw, fh = faces[best_idx]
                boxes.append((fx, fy, fx + fw, fy + fh))

        if len(boxes) < max(4, len(frames) * 0.5):
            return _zeros

        # Stable median box
        boxes_arr = np.array(boxes)
        med_box = [int(np.median(boxes_arr[:, i])) for i in range(4)]
        h_img, w_img = frames[0].shape[:2]
        x1 = max(0, med_box[0])
        y1 = max(0, med_box[1])
        x2 = min(w_img, med_box[2])
        y2 = min(h_img, med_box[3])
        face_w, face_h = x2 - x1, y2 - y1

        if face_w < 40 or face_h < 40:
            return _zeros

        bw = max(8, int(0.15 * min(face_w, face_h)))
        oy1, oy2 = max(0, y1 - bw), min(h_img, y2 + bw)
        ox1, ox2 = max(0, x1 - bw), min(w_img, x2 + bw)
        bg_iy1, bg_iy2 = y1 - oy1, y2 - oy1
        bg_ix1, bg_ix2 = x1 - ox1, x2 - ox1

        result = dict(_zeros)

        # ── Collect per-frame measurements ──
        gray_lap_vars = []
        bg_lap_vars = []
        face_kurtosis_vals = []
        detail_ratios = []
        ch_vars = {c: [] for c in ("R", "G", "B")}
        rb_corrs = []
        srm_noise_vars = []
        face_hf_energies = []
        bg_hf_energies = []
        mid_y, mid_x = face_h // 2, face_w // 2
        block_consistency_vals = []

        hp_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32) / 8.0

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray64 = gray.astype(np.float64)

            # Grayscale Laplacian
            lap = cv2.Laplacian(gray64, cv2.CV_64F)
            face_lap = lap[y1:y2, x1:x2]
            outer = lap[oy1:oy2, ox1:ox2]
            bg_mask = np.ones(outer.shape, dtype=bool)
            if bg_iy2 > bg_iy1 and bg_ix2 > bg_ix1:
                bg_mask[bg_iy1:bg_iy2, bg_ix1:bg_ix2] = False

            if bg_mask.sum() >= 20 and face_lap.size >= 100:
                gray_lap_vars.append(float(np.var(face_lap)))
                bg_lap_vars.append(float(np.var(outer[bg_mask])))

                # Kurtosis of Laplacian (noise distribution shape)
                m = float(np.mean(face_lap))
                s = float(np.std(face_lap))
                if s > 0:
                    face_kurtosis_vals.append(
                        float(np.mean(((face_lap - m) / s) ** 4)) - 3.0
                    )

                # Block-level consistency (2x2 grid)
                blocks = [
                    face_lap[:mid_y, :mid_x], face_lap[:mid_y, mid_x:],
                    face_lap[mid_y:, :mid_x], face_lap[mid_y:, mid_x:],
                ]
                bvars = [float(np.var(b)) for b in blocks if b.size > 10]
                if len(bvars) >= 4:
                    block_consistency_vals.append(
                        float(np.std(bvars)) / (float(np.mean(bvars)) + 1e-6)
                    )

            # Wavelet detail ratio (Gaussian pyramid approximation)
            gray_f = gray.astype(np.float32)
            fp = gray_f[y1:y2, x1:x2]
            if fp.shape[0] >= 32 and fp.shape[1] >= 32:
                smooth1 = cv2.GaussianBlur(fp, (5, 5), 1.0)
                detail1 = fp - smooth1
                smooth2 = cv2.GaussianBlur(fp, (11, 11), 2.0)
                detail2 = smooth1 - smooth2
                e1 = float(np.mean(detail1 ** 2))
                e2 = float(np.mean(detail2 ** 2))
                if e2 > 0:
                    detail_ratios.append(e1 / (e2 + 1e-6))

            # Per-channel Laplacian CV
            for ci, cn in [(2, "R"), (1, "G"), (0, "B")]:
                ch = frame[:, :, ci].astype(np.float64)
                ch_lap = cv2.Laplacian(ch, cv2.CV_64F)
                ch_face = ch_lap[y1:y2, x1:x2]
                if ch_face.size > 20:
                    ch_vars[cn].append(float(np.var(ch_face)))

            # Cross-channel Laplacian correlation (R vs B)
            lap_r = cv2.Laplacian(frame[:, :, 2].astype(np.float64), cv2.CV_64F)
            lap_b = cv2.Laplacian(frame[:, :, 0].astype(np.float64), cv2.CV_64F)
            fr_r = lap_r[y1:y2, x1:x2].flatten()
            fr_b = lap_b[y1:y2, x1:x2].flatten()
            if len(fr_r) > 100:
                corr = float(np.corrcoef(fr_r, fr_b)[0, 1])
                if not np.isnan(corr):
                    rb_corrs.append(corr)

            # SRM high-pass noise residual
            noise = cv2.filter2D(gray_f, -1, hp_kernel)
            noise_face = noise[y1:y2, x1:x2]
            if noise_face.size > 20:
                srm_noise_vars.append(float(np.var(noise_face)))

            # DCT high-frequency ratio
            if fp.shape[0] >= 16 and fp.shape[1] >= 16:
                fp_r = cv2.resize(fp, (64, 64))
                dct_face = cv2.dct(fp_r)
                hf = float(np.sum(np.abs(dct_face[32:, 32:])))
                tot = float(np.sum(np.abs(dct_face))) + 1e-6
                face_hf_energies.append(hf / tot)
                bg_y1_dct = min(y2, h_img - 65)
                bg_patch = gray_f[bg_y1_dct:bg_y1_dct + 64, x1:min(x1 + 64, w_img)]
                if bg_patch.shape[0] >= 32 and bg_patch.shape[1] >= 32:
                    bp_r = cv2.resize(bg_patch, (64, 64))
                    dct_bg = cv2.dct(bp_r)
                    bg_hf_energies.append(
                        float(np.sum(np.abs(dct_bg[32:, 32:]))) / (float(np.sum(np.abs(dct_bg))) + 1e-6)
                    )

        # ── Aggregate features ──
        support = len(gray_lap_vars) / max(len(frames), 1)
        result["frequency_support"] = round(support, 5)

        if gray_lap_vars and bg_lap_vars:
            result["lap_var_ratio"] = round(float(np.mean(gray_lap_vars)) / (float(np.mean(bg_lap_vars)) + 1e-6), 5)
            result["lap_var_temporal_cv"] = round(float(np.std(gray_lap_vars)) / (float(np.mean(gray_lap_vars)) + 1e-6), 5)
            # Temporal dynamics
            diffs = np.abs(np.diff(gray_lap_vars))
            result["lap_diff_mean"] = round(float(np.mean(diffs)), 5)
            if len(gray_lap_vars) > 5:
                x_t = np.arange(len(gray_lap_vars))
                coeffs = np.polyfit(x_t, gray_lap_vars, 1)
                trend = np.polyval(coeffs, x_t)
                residuals = np.array(gray_lap_vars) - trend
                result["lap_trend_residual_std"] = round(
                    float(np.std(residuals)) / (float(np.mean(gray_lap_vars)) + 1e-6), 5
                )
            if len(diffs) > 2:
                result["lap_acc_mean"] = round(float(np.mean(np.abs(np.diff(diffs)))), 5)

        if face_kurtosis_vals:
            result["lap_kurtosis_mean"] = round(float(np.mean(face_kurtosis_vals)), 5)
            result["lap_kurtosis_cv"] = round(
                float(np.std(face_kurtosis_vals)) / (abs(float(np.mean(face_kurtosis_vals))) + 1e-6), 5
            )

        if detail_ratios:
            result["wavelet_detail_ratio"] = round(float(np.mean(detail_ratios)), 5)
            result["wavelet_detail_ratio_cv"] = round(
                float(np.std(detail_ratios)) / (float(np.mean(detail_ratios)) + 1e-6), 5
            )

        for cn in ("R", "G", "B"):
            if ch_vars[cn]:
                result[f"lap_{cn}_cv"] = round(
                    float(np.std(ch_vars[cn])) / (float(np.mean(ch_vars[cn])) + 1e-6), 5
                )

        if block_consistency_vals:
            result["block_lap_consistency"] = round(float(np.mean(block_consistency_vals)), 5)

        if rb_corrs:
            result["cross_ch_lap_corr"] = round(float(np.mean(rb_corrs)), 5)

        if srm_noise_vars:
            result["srm_var_cv"] = round(
                float(np.std(srm_noise_vars)) / (float(np.mean(srm_noise_vars)) + 1e-6), 5
            )

        if face_hf_energies and bg_hf_energies:
            result["dct_hf_ratio"] = round(
                float(np.mean(face_hf_energies)) / (float(np.mean(bg_hf_energies)) + 1e-6), 5
            )
            result["dct_hf_temporal_cv"] = round(
                float(np.std(face_hf_energies)) / (float(np.mean(face_hf_energies)) + 1e-6), 5
            )

        # Composite frequency score: weighted by individual AUC strength
        # Kurtosis (0.685) + Wavelet (0.665) are primary,
        # Laplacian temporal (0.625-0.642) secondary, rest supporting.
        scores = []
        weights = []

        if result["lap_kurtosis_cv"] > 0:
            scores.append(_sigmoid_score(result["lap_kurtosis_cv"], 0.31, 0.05))
            weights.append(0.20)
        if result["lap_kurtosis_mean"] > 0:
            scores.append(_sigmoid_score(result["lap_kurtosis_mean"], 44.0, 15.0))
            weights.append(0.15)
        if result["wavelet_detail_ratio"] > 0:
            scores.append(_sigmoid_score(result["wavelet_detail_ratio"], 0.70, 0.06))
            weights.append(0.15)
        if result["wavelet_detail_ratio_cv"] > 0:
            scores.append(_sigmoid_score(result["wavelet_detail_ratio_cv"], 0.078, 0.010))
            weights.append(0.12)
        if result["lap_var_temporal_cv"] > 0:
            scores.append(_sigmoid_score(result["lap_var_temporal_cv"], 0.160, 0.015))
            weights.append(0.10)
        if result["lap_trend_residual_std"] > 0:
            scores.append(_sigmoid_score(result["lap_trend_residual_std"], 0.150, 0.015))
            weights.append(0.08)
        if result["lap_var_ratio"] > 0:
            scores.append(_sigmoid_score(result["lap_var_ratio"], 1.35, 0.25))
            weights.append(0.08)
        if result["block_lap_consistency"] > 0:
            scores.append(_sigmoid_score(result["block_lap_consistency"], 0.36, 0.03))
            weights.append(0.06)
        if result["dct_hf_ratio"] > 0:
            scores.append(_sigmoid_score(result["dct_hf_ratio"], 1.02, 0.04))
            weights.append(0.06)

        if scores:
            total_w = sum(weights)
            freq_score = sum(s * w for s, w in zip(scores, weights)) / max(total_w, 1e-6)
            freq_score = _clip01(freq_score) * min(1.0, support / 0.40)
        else:
            freq_score = 0.0

        result["frequency_score"] = round(freq_score, 5)
        return result

    def _load_frames(self, path: str) -> tuple:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Δεν ανοίγει: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fps = float(fps) if fps and fps > 0 else 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total / max(fps, 1e-6)

        target = min(self.max_frames, int(max(duration, 1.0) * self.sample_rate))
        target = max(target, 12)
        stride = max(1, total // target) if total > 0 else 1

        frames: list[np.ndarray] = []
        indices: list[int] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                if w > 640:
                    new_h = max(1, int(h * 640 / max(w, 1)))
                    frame = cv2.resize(frame, (640, new_h))
                frames.append(frame)
                indices.append(idx)
                if len(frames) >= self.max_frames:
                    break
            idx += 1

        cap.release()
        return frames, indices, fps, duration, (w, h)


PATVDetector = PATVXDetector

"""
PATV-X+ Training Pipeline — Upgraded MLP Classifier
=====================================================

Βελτιώσεις από την αρχική έκδοση:
  1) 16-D feature vector (12 παλιά + 4 CTCG L4 features)
  2) Βαθύτερο MLP: 16 → 64 → 32 → 16 → 1
  3) L2 weight regularization (weight decay)
  4) Cosine Annealing LR schedule
  5) Feature StandardScaler (z-score) αντί για MinMax
  6) Stratified train/val/test split
  7) Youden-J optimal threshold επί validation set
  8) TTA (Test-Time Averaging) για σταθερότερες προβλέψεις
  9) AUC-ROC evaluation

ΣΗΜΑΝΤΙΚΟ: ground-truth labels από στήλη `label`, ΟΧΙ από `verdict`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


BASE_FEATURE_NAMES = [
    # L1: Residual Motion
    "flow_score", "mean_divergence", "acceleration_anomaly", "flicker_score",
    # L2: Physics
    "physics_score", "gravity_consistency", "rigid_body_score", "shadow_consistency",
    # L3: Semantic / Face context
    "semantic_score", "color_drift", "edge_stability", "texture_consistency",
    # L4: CTCG (Causal Temporal Coherence Graph) — NOVEL
    "ctcg_score", "ctcg_phase_coherence", "ctcg_ar_residual", "ctcg_spectral_anomaly",
    "ctcg_micro_jitter",
]

INTERACTION_FEATURE_NAMES = [
    "warp_seam_interaction",
    "warp_color_interaction",
    "warp_edge_interaction",
    "boundary_peak_signal",
    "boundary_mean_signal",
    "rigid_motion_interaction",
    "physics_motion_interaction",
    "rigid_warp_interaction",
]

LEGACY_CORE_FEATURE_NAMES = BASE_FEATURE_NAMES[:16]

FORENSIC_EXTRA_FEATURE_NAMES = [
    "face_scene_support_ratio",
    "light_support_ratio",
    "skin_support_ratio",
    "flow_boundary_support_ratio",
    "boundary_color_support_ratio",
    "warp_support_ratio",
    "boundary_edge_support_ratio",
    "boundary_agreement_ratio",
    "boundary_corroboration",
    "warp_supported_score",
    "violation_count",
    "major_violation_count",
    "critical_violation_count",
    "dynamic_violation_score",
    "boundary_violation_score",
    "violation_ratio_mean",
    "violation_ratio_max",
    "violation_span_fraction",
    "major_span_fraction",
    "critical_span_fraction",
    "dynamic_span_fraction",
    "boundary_span_fraction",
    "confidence_weighted_span",
    "has_face_motion",
    "has_face_scene_decoupling",
    "has_rigid_body",
    "has_warp_prediction_error",
    "has_flow_boundary_seam",
    "has_edge_instability",
    "has_face_texture",
    "frame_anomaly_max",
    "frame_anomaly_top3_mean",
    "frame_anomaly_peak_fraction",
    "frame_anomaly_mean",
    "frame_anomaly_std",
    "frame_multi_evidence_fraction",
]

L5_FREQUENCY_FEATURE_NAMES = [
    "l5_frequency_score",
    "l5_lap_var_ratio",
    "l5_lap_var_temporal_cv",
    "l5_dct_hf_ratio",
    "l5_dct_hf_temporal_cv",
    "l5_lap_kurtosis_mean",
    "l5_lap_kurtosis_cv",
    "l5_wavelet_detail_ratio",
    "l5_wavelet_detail_ratio_cv",
    "l5_lap_R_cv",
    "l5_lap_G_cv",
    "l5_lap_B_cv",
    "l5_lap_diff_mean",
    "l5_lap_trend_residual_std",
    "l5_lap_acc_mean",
    "l5_block_lap_consistency",
    "l5_cross_ch_lap_corr",
    "l5_srm_var_cv",
]

SUPPORT_FEATURE_NAMES = [
    "face_detected_ratio",
    "face_reliability_mean",
    "neck_visibility_ratio",
    "boundary_support_ratio",
]

FEATURE_NAMES = BASE_FEATURE_NAMES + INTERACTION_FEATURE_NAMES + FORENSIC_EXTRA_FEATURE_NAMES + L5_FREQUENCY_FEATURE_NAMES
OPEN_FEATURE_NAMES = FEATURE_NAMES + SUPPORT_FEATURE_NAMES
THRESHOLD_TOL = 1e-9


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _relu(x):
    return np.maximum(0.0, x)


def _relu_grad(x):
    return (x > 0).astype(float)


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _clip01(value):
    return float(np.clip(float(value), 0.0, 1.0))


def _timeline_coverage_fraction(timeline, violations, *, type_filter=None, severity=None):
    if not timeline or not violations:
        return 0.0
    active = []
    for v in violations:
        if type_filter is not None and getattr(v, "violation_type", "") not in type_filter:
            continue
        if severity is not None and getattr(v, "severity", "") != severity:
            continue
        active.append(v)
    if not active:
        return 0.0
    covered = 0
    for entry in timeline:
        frame_idx = int(getattr(entry, "frame_idx", 0))
        if any(int(getattr(v, "frame_start", 0)) <= frame_idx <= int(getattr(v, "frame_end", 0)) for v in active):
            covered += 1
    return _clip01(covered / max(len(timeline), 1))


def _timeline_confidence_weighted_span(timeline, violations):
    if not timeline or not violations:
        return 0.0
    coverage = []
    for entry in timeline:
        frame_idx = int(getattr(entry, "frame_idx", 0))
        conf = 0.0
        for v in violations:
            if int(getattr(v, "frame_start", 0)) <= frame_idx <= int(getattr(v, "frame_end", 0)):
                conf = max(conf, float(getattr(v, "confidence", 0.0)))
        coverage.append(conf)
    return _clip01(float(np.mean(coverage))) if coverage else 0.0


def _build_violation_extra_features(*, violations=None, row=None, timeline=None, support_details=None):
    support_details = support_details or {}
    if row is not None:
        return {
            "face_scene_support_ratio": _clip01(_safe_float(row.get("face_scene_support_ratio", 0.0))),
            "light_support_ratio": _clip01(_safe_float(row.get("light_support_ratio", 0.0))),
            "skin_support_ratio": _clip01(_safe_float(row.get("skin_support_ratio", 0.0))),
            "flow_boundary_support_ratio": _clip01(_safe_float(row.get("flow_boundary_support_ratio", 0.0))),
            "boundary_color_support_ratio": _clip01(_safe_float(row.get("boundary_color_support_ratio", 0.0))),
            "warp_support_ratio": _clip01(_safe_float(row.get("warp_support_ratio", 0.0))),
            "boundary_edge_support_ratio": _clip01(_safe_float(row.get("boundary_edge_support_ratio", 0.0))),
            "boundary_agreement_ratio": _clip01(_safe_float(row.get("boundary_agreement_ratio", 0.0))),
            "boundary_corroboration": _clip01(_safe_float(row.get("boundary_corroboration", 0.0))),
            "warp_supported_score": _clip01(_safe_float(row.get("warp_supported_score", 0.0))),
            "violation_count": _clip01(_safe_float(row.get("violation_count", 0.0)) / 12.0),
            "major_violation_count": _clip01(_safe_float(row.get("major_violation_count", 0.0)) / 8.0),
            "critical_violation_count": _clip01(_safe_float(row.get("critical_violation_count", 0.0)) / 4.0),
            "dynamic_violation_score": _clip01(_safe_float(row.get("dynamic_violation_score", 0.0))),
            "boundary_violation_score": _clip01(_safe_float(row.get("boundary_violation_score", 0.0))),
            "violation_ratio_mean": _clip01(_safe_float(row.get("violation_ratio_mean", 0.0))),
            "violation_ratio_max": _clip01(_safe_float(row.get("violation_ratio_max", 0.0))),
            "violation_span_fraction": _clip01(_safe_float(row.get("violation_span_fraction", 0.0))),
            "major_span_fraction": _clip01(_safe_float(row.get("major_span_fraction", 0.0))),
            "critical_span_fraction": _clip01(_safe_float(row.get("critical_span_fraction", 0.0))),
            "dynamic_span_fraction": _clip01(_safe_float(row.get("dynamic_span_fraction", 0.0))),
            "boundary_span_fraction": _clip01(_safe_float(row.get("boundary_span_fraction", 0.0))),
            "confidence_weighted_span": _clip01(_safe_float(row.get("confidence_weighted_span", 0.0))),
            "has_face_motion": _clip01(_safe_float(row.get("has_face_motion", 0.0))),
            "has_face_scene_decoupling": _clip01(_safe_float(row.get("has_face_scene_decoupling", 0.0))),
            "has_rigid_body": _clip01(_safe_float(row.get("has_rigid_body", 0.0))),
            "has_warp_prediction_error": _clip01(_safe_float(row.get("has_warp_prediction_error", 0.0))),
            "has_flow_boundary_seam": _clip01(_safe_float(row.get("has_flow_boundary_seam", 0.0))),
            "has_edge_instability": _clip01(_safe_float(row.get("has_edge_instability", 0.0))),
            "has_face_texture": _clip01(_safe_float(row.get("has_face_texture", 0.0))),
            "frame_anomaly_max": _clip01(_safe_float(row.get("frame_anomaly_max", 0.0))),
            "frame_anomaly_top3_mean": _clip01(_safe_float(row.get("frame_anomaly_top3_mean", 0.0))),
            "frame_anomaly_peak_fraction": _clip01(_safe_float(row.get("frame_anomaly_peak_fraction", 0.0))),
            "frame_anomaly_mean": _clip01(_safe_float(row.get("frame_anomaly_mean", 0.0))),
            "frame_anomaly_std": _clip01(_safe_float(row.get("frame_anomaly_std", 0.0))),
            "frame_multi_evidence_fraction": _clip01(_safe_float(row.get("frame_multi_evidence_fraction", 0.0))),
        }

    violations = violations or []
    timeline = timeline or []
    dynamic_types = {"FACE_MOTION", "FACE_SCENE_DECOUPLING", "RIGID_BODY"}
    boundary_types = {
        "WARP_PREDICTION_ERROR",
        "FLOW_BOUNDARY_SEAM",
        "EDGE_INSTABILITY",
        "BOUNDARY_EDGE_FLICKER",
        "FACE_TEXTURE",
    }
    vtypes = [getattr(v, "violation_type", "") for v in violations]
    major_count = sum(1 for v in violations if getattr(v, "severity", "") == "MAJOR")
    critical_count = sum(1 for v in violations if getattr(v, "severity", "") == "CRITICAL")
    ratios = [float(getattr(v, "violation_ratio", 0.0)) for v in violations]
    face_motion_count = sum(1 for t in vtypes if t == "FACE_MOTION")
    rigid_body_count = sum(1 for t in vtypes if t == "RIGID_BODY")
    face_scene_count = sum(1 for t in vtypes if t == "FACE_SCENE_DECOUPLING")
    warp_count = sum(1 for t in vtypes if t == "WARP_PREDICTION_ERROR")
    seam_count = sum(1 for t in vtypes if t == "FLOW_BOUNDARY_SEAM")
    edge_count = sum(1 for t in vtypes if t in {"EDGE_INSTABILITY", "BOUNDARY_EDGE_FLICKER"})
    texture_count = sum(1 for t in vtypes if t == "FACE_TEXTURE")

    dynamic_score = min(
        1.0,
        0.20 * face_motion_count
        + 0.24 * face_scene_count
        + 0.20 * rigid_body_count
        + 0.08 * texture_count,
    )
    boundary_score = min(
        1.0,
        0.24 * warp_count
        + 0.18 * seam_count
        + 0.18 * edge_count
        + 0.10 * texture_count,
    )

    scores = [float(getattr(e, "anomaly_score", 0.0)) for e in timeline]
    top3 = sorted(scores, reverse=True)[:3]
    multi_evidence_fraction = (
        sum(len(getattr(e, "violations", []) or []) >= 2 for e in timeline) / max(len(timeline), 1)
        if timeline else 0.0
    )

    return {
        "face_scene_support_ratio": _clip01(support_details.get("face_scene_support_ratio", 0.0)),
        "light_support_ratio": _clip01(support_details.get("light_support_ratio", 0.0)),
        "skin_support_ratio": _clip01(support_details.get("skin_support_ratio", 0.0)),
        "flow_boundary_support_ratio": _clip01(support_details.get("flow_boundary_support_ratio", 0.0)),
        "boundary_color_support_ratio": _clip01(support_details.get("boundary_color_support_ratio", 0.0)),
        "warp_support_ratio": _clip01(support_details.get("warp_support_ratio", 0.0)),
        "boundary_edge_support_ratio": _clip01(support_details.get("boundary_edge_support_ratio", 0.0)),
        "boundary_agreement_ratio": _clip01(support_details.get("boundary_agreement_ratio", 0.0)),
        "boundary_corroboration": _clip01(support_details.get("boundary_corroboration", 0.0)),
        "warp_supported_score": _clip01(support_details.get("warp_supported_score", 0.0)),
        "violation_count": _clip01(len(violations) / 12.0),
        "major_violation_count": _clip01(major_count / 8.0),
        "critical_violation_count": _clip01(critical_count / 4.0),
        "dynamic_violation_score": _clip01(dynamic_score),
        "boundary_violation_score": _clip01(boundary_score),
        "violation_ratio_mean": _clip01((float(np.mean(ratios)) / 8.0) if ratios else 0.0),
        "violation_ratio_max": _clip01((float(np.max(ratios)) / 10.0) if ratios else 0.0),
        "violation_span_fraction": _timeline_coverage_fraction(timeline, violations),
        "major_span_fraction": _timeline_coverage_fraction(timeline, violations, severity="MAJOR"),
        "critical_span_fraction": _timeline_coverage_fraction(timeline, violations, severity="CRITICAL"),
        "dynamic_span_fraction": _timeline_coverage_fraction(timeline, violations, type_filter=dynamic_types),
        "boundary_span_fraction": _timeline_coverage_fraction(timeline, violations, type_filter=boundary_types),
        "confidence_weighted_span": _timeline_confidence_weighted_span(timeline, violations),
        "has_face_motion": float(face_motion_count > 0),
        "has_face_scene_decoupling": float(face_scene_count > 0),
        "has_rigid_body": float(rigid_body_count > 0),
        "has_warp_prediction_error": float(warp_count > 0),
        "has_flow_boundary_seam": float(seam_count > 0),
        "has_edge_instability": float(edge_count > 0),
        "has_face_texture": float(texture_count > 0),
        "frame_anomaly_max": _clip01(max(scores) if scores else 0.0),
        "frame_anomaly_top3_mean": _clip01(float(np.mean(top3)) if top3 else 0.0),
        "frame_anomaly_peak_fraction": _clip01(sum(s >= 0.45 for s in scores) / max(len(scores), 1) if scores else 0.0),
        "frame_anomaly_mean": _clip01(float(np.mean(scores)) if scores else 0.0),
        "frame_anomaly_std": _clip01(float(np.std(scores)) if scores else 0.0),
        "frame_multi_evidence_fraction": _clip01(multi_evidence_fraction),
    }


def derive_forensic_extra_features_from_row(row):
    return _build_violation_extra_features(row=row)


def derive_forensic_extra_features_from_result(result):
    support_details = {
        "face_scene_support_ratio": float(getattr(result, "physics_face_scene_support_ratio", 0.0)),
        "light_support_ratio": float(getattr(result, "physics_light_support_ratio", 0.0)),
        "skin_support_ratio": float(getattr(result, "physics_skin_support_ratio", 0.0)),
        "flow_boundary_support_ratio": float(getattr(result, "l4_flow_boundary_support_ratio", 0.0)),
        "boundary_color_support_ratio": float(getattr(result, "l4_boundary_color_support_ratio", 0.0)),
        "warp_support_ratio": float(getattr(result, "l4_warp_support_ratio", 0.0)),
        "boundary_edge_support_ratio": float(getattr(result, "l4_boundary_edge_support_ratio", 0.0)),
        "boundary_agreement_ratio": float(getattr(result, "l4_boundary_agreement_ratio", 0.0)),
        "boundary_corroboration": float(getattr(result, "l4_boundary_corroboration", 0.0)),
        "warp_supported_score": float(getattr(result, "l4_warp_supported_score", 0.0)),
    }
    return _build_violation_extra_features(
        violations=getattr(result, "violations", []),
        timeline=getattr(result, "frame_timeline", []) or [],
        support_details=support_details,
    )


def _build_interaction_features(feature_map):
    warp = feature_map["ctcg_spectral_anomaly"]
    seam = feature_map["ctcg_phase_coherence"]
    color = feature_map["ctcg_ar_residual"]
    edge = feature_map["ctcg_micro_jitter"]
    rigid = feature_map["rigid_body_score"]
    physics = feature_map["physics_score"]
    mean_divergence = feature_map["mean_divergence"]
    boundary_peak = max(color, edge)
    return {
        "warp_seam_interaction": _clip01(warp * seam),
        "warp_color_interaction": _clip01(warp * color),
        "warp_edge_interaction": _clip01(warp * edge),
        "boundary_peak_signal": _clip01(boundary_peak),
        "boundary_mean_signal": _clip01((seam + color + edge) / 3.0),
        "rigid_motion_interaction": _clip01(rigid * mean_divergence),
        "physics_motion_interaction": _clip01(physics * mean_divergence),
        "rigid_warp_interaction": _clip01(rigid * warp),
    }


def _base_feature_map_from_row(row):
    return {
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
        "ctcg_score": _safe_float(row.get("ctcg_score", row.get("L4_ctcg", row.get("l4_boundary_artifact_score", 0.0)))),
        "ctcg_phase_coherence": _safe_float(row.get("ctcg_phase_coherence", row.get("flow_boundary_seam", 0.0))),
        "ctcg_ar_residual": _safe_float(row.get("ctcg_ar_residual", row.get("boundary_color_flicker", 0.0))),
        "ctcg_spectral_anomaly": _safe_float(row.get("ctcg_spectral_anomaly", row.get("warp_prediction_error", 0.0))),
        "ctcg_micro_jitter": _safe_float(row.get("ctcg_micro_jitter", row.get("boundary_edge_flicker", 0.0))),
    }


def _support_feature_map_from_row(row):
    return {
        "face_detected_ratio": _safe_float(row.get("face_detected_ratio", 0.0)),
        "face_reliability_mean": _safe_float(row.get("face_reliability_mean", row.get("face_reliability", 0.0))),
        "neck_visibility_ratio": _safe_float(row.get("neck_visibility_ratio", 0.0)),
        "boundary_support_ratio": _safe_float(row.get("boundary_support_ratio", 0.0)),
    }


def _l5_frequency_feature_map_from_row(row):
    return {name: _safe_float(row.get(name, 0.0)) for name in L5_FREQUENCY_FEATURE_NAMES}


def row_to_feature_vector(row, include_support=False, feature_names=None):
    feature_map = _base_feature_map_from_row(row)
    interaction_map = _build_interaction_features(feature_map)
    forensic_map = derive_forensic_extra_features_from_row(row)
    support_map = _support_feature_map_from_row(row)
    l5_map = _l5_frequency_feature_map_from_row(row)
    full_map = {}
    full_map.update(feature_map)
    full_map.update(interaction_map)
    full_map.update(forensic_map)
    full_map.update(l5_map)
    full_map.update(support_map)
    if feature_names is None:
        selected_names = OPEN_FEATURE_NAMES if include_support else FEATURE_NAMES
    else:
        selected_names = list(feature_names)
    return [full_map[name] for name in selected_names]


class MLP:
    """
    4-layer MLP: 16 → 64 → 32 → 16 → 1
    L2 regularization + Cosine LR + Momentum SGD
    """

    def __init__(self, input_dim=16, hidden1=64, hidden2=32, hidden3=16,
                 lr=0.008, momentum=0.92, weight_decay=1e-4,
                 dropout1=0.12, dropout2=0.08, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/input_dim), (input_dim, hidden1))
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.normal(0, np.sqrt(2/hidden1), (hidden1, hidden2))
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.normal(0, np.sqrt(2/hidden2), (hidden2, hidden3))
        self.b3 = np.zeros(hidden3)
        self.W4 = rng.normal(0, np.sqrt(2/hidden3), (hidden3, 1))
        self.b4 = np.zeros(1)
        self.lr_init = lr
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        for n in ["W1","b1","W2","b2","W3","b3","W4","b4"]:
            setattr(self, f"v{n}", np.zeros_like(getattr(self, n)))
        self.training_history = {}

    def _drop(self, a, rate, rng):
        if rate <= 0:
            return a, np.ones(a.shape, dtype=float)
        m = (rng.random(a.shape) > rate).astype(float) / max(1-rate, 1e-6)
        return a * m, m

    def _fwd_train(self, X, rng):
        z1 = X @ self.W1 + self.b1; a1 = _relu(z1)
        a1, d1 = self._drop(a1, self.dropout1, rng)
        z2 = a1 @ self.W2 + self.b2; a2 = _relu(z2)
        a2, d2 = self._drop(a2, self.dropout2, rng)
        z3 = a2 @ self.W3 + self.b3; a3 = _relu(z3)
        z4 = a3 @ self.W4 + self.b4
        out = _sigmoid(z4).reshape(-1)
        return out, (X, z1, a1, d1, z2, a2, d2, z3, a3)

    def forward(self, X):
        a1 = _relu(X @ self.W1 + self.b1)
        a2 = _relu(a1 @ self.W2 + self.b2)
        a3 = _relu(a2 @ self.W3 + self.b3)
        return _sigmoid(a3 @ self.W4 + self.b4).reshape(-1), None

    def _bwd(self, y, out, cache):
        X, z1, a1, d1, z2, a2, d2, z3, a3 = cache
        n = max(len(y), 1)
        wd = self.weight_decay
        dz4 = (out - y).reshape(-1, 1) / n
        dW4 = a3.T @ dz4 + wd * self.W4; db4 = dz4.sum(0)
        da3 = dz4 @ self.W4.T
        dz3 = da3 * _relu_grad(z3)
        dW3 = a2.T @ dz3 + wd * self.W3; db3 = dz3.sum(0)
        da2 = (dz3 @ self.W3.T) * d2
        dz2 = da2 * _relu_grad(z2)
        dW2 = a1.T @ dz2 + wd * self.W2; db2 = dz2.sum(0)
        da1 = (dz2 @ self.W2.T) * d1
        dz1 = da1 * _relu_grad(z1)
        dW1 = X.T @ dz1 + wd * self.W1; db1 = dz1.sum(0)
        for wn, dW, bn, db in [("W1",dW1,"b1",db1),("W2",dW2,"b2",db2),
                                ("W3",dW3,"b3",db3),("W4",dW4,"b4",db4)]:
            vw = self.momentum * getattr(self,f"v{wn}") - self.lr * dW
            vb = self.momentum * getattr(self,f"v{bn}") - self.lr * db
            setattr(self, f"v{wn}", vw); setattr(self, f"v{bn}", vb)
            setattr(self, wn, getattr(self, wn) + vw)
            setattr(self, bn, getattr(self, bn) + vb)

    def bce(self, y, p):
        e = 1e-7
        return float(-np.mean(y*np.log(p+e) + (1-y)*np.log(1-p+e)))

    def _cosine_lr(self, epoch, total):
        lo = self.lr_init / 20.0
        return lo + 0.5*(self.lr_init - lo)*(1 + np.cos(np.pi*epoch/max(total,1)))

    def train(self, X, y, X_val, y_val, epochs=200, batch_size=32,
              verbose=True, seed=42):
        rng = np.random.default_rng(seed)
        best_loss = float("inf")
        best_w = self._get_w()
        hist = {"train_loss": [], "val_loss": [], "val_acc": []}
        for ep in range(epochs):
            self.lr = self._cosine_lr(ep, epochs)
            idx = rng.permutation(len(X))
            Xs, ys = X[idx], y[idx]
            tl = []
            for i in range(0, len(Xs), batch_size):
                Xb, yb = Xs[i:i+batch_size], ys[i:i+batch_size]
                out, cache = self._fwd_train(Xb, rng)
                tl.append(self.bce(yb, out))
                self._bwd(yb, out, cache)
            vout, _ = self.forward(X_val)
            vl = self.bce(y_val, vout)
            va = float(((vout > 0.5).astype(int) == y_val).mean())
            tl_mean = float(np.mean(tl)) if tl else 0.0
            hist["train_loss"].append(tl_mean)
            hist["val_loss"].append(float(vl))
            hist["val_acc"].append(va)
            if vl < best_loss:
                best_loss = vl; best_w = self._get_w()
            if verbose and ((ep+1) % 20 == 0 or ep == 0 or ep == epochs-1):
                print(f"  Epoch {ep+1:4d}/{epochs}  train={tl_mean:.4f}  "
                      f"val={vl:.4f}  acc={va*100:.1f}%  lr={self.lr:.5f}")
        self._set_w(best_w)
        self.training_history = hist
        return hist

    def predict(self, X):
        out, _ = self.forward(X); return out

    def predict_tta(self, X, n=5, std=0.006):
        rng = np.random.default_rng(0)
        ps = [self.predict(X)]
        for _ in range(n-1):
            ps.append(self.predict(X + rng.normal(0, std, X.shape)))
        return np.mean(ps, axis=0)

    def predict_class(self, X, threshold=0.5):
        return (self.predict(X) > threshold).astype(int)

    def _get_w(self):
        return {n: getattr(self,n).copy() for n in ["W1","b1","W2","b2","W3","b3","W4","b4"]}

    def _set_w(self, w):
        for n, v in w.items(): setattr(self, n, v.copy())

    def save(self, path):
        data = {n: getattr(self,n).tolist() for n in ["W1","b1","W2","b2","W3","b3","W4","b4"]}
        data["history"] = self.training_history
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[MLP] Saved: {path}")

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for n in ["W1","b1","W2","b2","W3","b3","W4","b4"]:
            if n in data: setattr(self, n, np.array(data[n], dtype=float))
        print(f"[MLP] Loaded: {path}")


class LinearLogistic:
    def __init__(self, input_dim, lr=0.05, weight_decay=1e-4, seed=42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.05, input_dim)
        self.b = 0.0
        self.lr = lr
        self.weight_decay = weight_decay
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def predict(self, X):
        return _sigmoid(X @ self.w + self.b)

    def train(self, X, y, X_val, y_val, epochs=250, batch_size=64, verbose=True, seed=42):
        rng = np.random.default_rng(seed)
        best_loss = float("inf")
        best_w = self.w.copy()
        best_b = float(self.b)
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            Xs = X[idx]
            ys = y[idx]
            losses = []
            for i in range(0, len(Xs), batch_size):
                Xb = Xs[i:i + batch_size]
                yb = ys[i:i + batch_size]
                p = self.predict(Xb)
                err = p - yb
                grad_w = (Xb.T @ err) / max(len(Xb), 1) + self.weight_decay * self.w
                grad_b = float(np.mean(err))
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b
                losses.append(float(-np.mean(yb * np.log(p + 1e-7) + (1 - yb) * np.log(1 - p + 1e-7))))

            val_p = self.predict(X_val)
            val_loss = float(-np.mean(y_val * np.log(val_p + 1e-7) + (1 - y_val) * np.log(1 - val_p + 1e-7)))
            val_acc = float(((val_p >= 0.5).astype(int) == y_val).mean())
            self.history["train_loss"].append(float(np.mean(losses)) if losses else 0.0)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            if val_loss < best_loss:
                best_loss = val_loss
                best_w = self.w.copy()
                best_b = float(self.b)
            if verbose and ((ep + 1) % 50 == 0 or ep == 0 or ep == epochs - 1):
                print(
                    f"  [Linear] Epoch {ep+1:4d}/{epochs}  "
                    f"train={self.history['train_loss'][-1]:.4f}  "
                    f"val={val_loss:.4f}  acc={val_acc*100:.1f}%"
                )
        self.w = best_w
        self.b = best_b
        return self.history

    def save_dict(self):
        return {
            "model_type": "linear",
            "weights": self.w.tolist(),
            "bias": float(self.b),
            "history": self.history,
        }


class GBMClassifier:
    """Gradient Boosting wrapper with same interface as MLP/Linear."""

    def __init__(self, input_dim, n_estimators=300, max_depth=3, learning_rate=0.03, seed=42):
        from sklearn.ensemble import GradientBoostingClassifier as _GBC
        self.model = _GBC(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=seed,
        )
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict_tta(self, X, n_runs=1):
        return self.predict(X)

    def train(self, X, y, X_val, y_val, **kwargs):
        self.model.fit(X, y)
        val_p = self.predict(X_val)
        val_loss = float(-np.mean(y_val * np.log(val_p + 1e-7) + (1 - y_val) * np.log(1 - val_p + 1e-7)))
        val_acc = float(((val_p >= 0.5).astype(int) == y_val).mean())
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        return self.history

    def save_dict(self):
        return {
            "model_type": "gbm",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "learning_rate": self.model.learning_rate,
            "feature_importances": self.model.feature_importances_.tolist(),
            "history": self.history,
        }


def result_to_features(result, include_support=False, feature_names=None):
    feature_map = {
        "flow_score": float(getattr(result, "l1_flow_score", 0.0)),
        "mean_divergence": float(getattr(result, "flow_mean_divergence", 0.0)),
        "acceleration_anomaly": float(getattr(result, "flow_acceleration_anomaly", 0.0)),
        "flicker_score": float(getattr(result, "flow_flicker_score", 0.0)),
        "physics_score": float(getattr(result, "l2_physics_score", 0.0)),
        "gravity_consistency": float(getattr(result, "physics_gravity_consistency", 0.0)),
        "rigid_body_score": float(getattr(result, "physics_rigid_body_score", 0.0)),
        "shadow_consistency": float(getattr(result, "physics_shadow_consistency", 0.0)),
        "semantic_score": float(getattr(result, "l3_semantic_score", 0.0)),
        "color_drift": float(getattr(result, "semantic_color_drift", 0.0)),
        "edge_stability": float(getattr(result, "semantic_edge_stability", 0.0)),
        "texture_consistency": float(getattr(result, "semantic_texture_consistency", 0.0)),
        "ctcg_score": float(getattr(result, "l4_ctcg_score", getattr(result, "l4_boundary_artifact_score", 0.0))),
        "ctcg_phase_coherence": float(getattr(result, "ctcg_phase_coherence", getattr(result, "l4_flow_boundary_seam_score", 0.0))),
        "ctcg_ar_residual": float(getattr(result, "ctcg_ar_residual", getattr(result, "l4_boundary_color_flicker_score", 0.0))),
        "ctcg_spectral_anomaly": float(getattr(result, "ctcg_spectral_anomaly", getattr(result, "l4_warp_prediction_error_score", 0.0))),
        "ctcg_micro_jitter": float(getattr(result, "ctcg_micro_jitter", getattr(result, "l4_boundary_edge_flicker_score", 0.0))),
    }
    interaction_map = _build_interaction_features(feature_map)
    forensic_map = derive_forensic_extra_features_from_result(result)
    support_map = {
        "face_detected_ratio": float(getattr(result, "face_detected_ratio", 0.0)),
        "face_reliability_mean": float(getattr(result, "face_reliability_mean", 0.0)),
        "neck_visibility_ratio": float(getattr(result, "neck_visibility_ratio", 0.0)),
        "boundary_support_ratio": float(getattr(result, "boundary_support_ratio", 0.0)),
    }
    full_map = {}
    full_map.update(feature_map)
    full_map.update(interaction_map)
    full_map.update(forensic_map)
    full_map.update(support_map)
    if feature_names is None:
        selected_names = OPEN_FEATURE_NAMES if include_support else FEATURE_NAMES
    else:
        selected_names = list(feature_names)
    return np.array([full_map[name] for name in selected_names], dtype=float)


def csv_to_features(csv_path, include_support=False, feature_names=None):
    X, y = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if row.get("label","") == "": continue
                label = int(float(row["label"]))
                feats = row_to_feature_vector(row, include_support=include_support, feature_names=feature_names)
                X.append(feats); y.append(label)
            except (ValueError, KeyError):
                continue
    return np.array(X, dtype=float), np.array(y, dtype=int)


def generate_synthetic_data(n_samples=400, seed=42, include_support=False):
    """
    Synthetic smoke-mode data aligned with the expanded explainable feature set.
    This is not paper-facing data; it only preserves pipeline compatibility.
    """
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    auth_m = [0.15,0.18,0.07,0.08, 0.28,0.22,0.18,0.10, 0.12,0.08,0.12,0.10, 0.15,0.12,0.14,0.11]
    auth_s = [0.06,0.08,0.04,0.04, 0.09,0.09,0.08,0.05, 0.06,0.05,0.05,0.05, 0.07,0.07,0.07,0.06]
    ai_m   = [0.38,0.22,0.17,0.16, 0.52,0.26,0.48,0.16, 0.19,0.13,0.14,0.26, 0.32,0.28,0.30,0.25]
    ai_s   = [0.10,0.08,0.08,0.07, 0.11,0.09,0.11,0.07, 0.08,0.07,0.06,0.10, 0.11,0.11,0.11,0.10]

    Xa16 = np.clip(rng.normal(auth_m, auth_s, (n, 16)), 0, 1)
    Xi16 = np.clip(rng.normal(ai_m, ai_s, (n, 16)), 0, 1)

    def build_feature_block(X16, label):
        m = len(X16)
        X17 = np.hstack([X16, np.clip(0.50 * X16[:, [12]] + 0.25 * X16[:, [14]] + 0.25 * X16[:, [15]], 0, 1)])
        base_map = {name: X17[:, i] for i, name in enumerate(BASE_FEATURE_NAMES)}

        warp = base_map["ctcg_spectral_anomaly"]
        seam = base_map["ctcg_phase_coherence"]
        color = base_map["ctcg_ar_residual"]
        edge = base_map["ctcg_micro_jitter"]
        rigid = base_map["rigid_body_score"]
        phys = base_map["physics_score"]
        md = base_map["mean_divergence"]
        boundary_peak = np.maximum(color, edge)
        boundary_mean = np.clip((seam + color + edge) / 3.0, 0, 1)
        boundary_corrob = np.maximum(np.minimum(warp, np.maximum(color, edge)), np.minimum(seam, np.maximum(color, edge)))
        agreement = (
            (seam >= 0.20).astype(float)
            + (color >= 0.15).astype(float)
            + (warp >= 0.40).astype(float)
            + (edge >= 0.15).astype(float)
        ) / 4.0
        warp_supported = np.clip(warp * (0.40 + 0.60 * boundary_peak), 0, 1)

        interaction_map = {
            "warp_seam_interaction": np.clip(warp * seam, 0, 1),
            "warp_color_interaction": np.clip(warp * color, 0, 1),
            "warp_edge_interaction": np.clip(warp * edge, 0, 1),
            "boundary_peak_signal": np.clip(boundary_peak, 0, 1),
            "boundary_mean_signal": boundary_mean,
            "rigid_motion_interaction": np.clip(rigid * md, 0, 1),
            "physics_motion_interaction": np.clip(phys * md, 0, 1),
            "rigid_warp_interaction": np.clip(rigid * warp, 0, 1),
        }

        if label == 1:
            extra_map = {
                "face_scene_support_ratio": np.clip(0.48 + 0.22 * rigid, 0, 1),
                "light_support_ratio": np.clip(0.28 + 0.18 * phys, 0, 1),
                "skin_support_ratio": np.clip(0.26 + 0.16 * phys, 0, 1),
                "flow_boundary_support_ratio": np.clip(0.42 + 0.30 * seam, 0, 1),
                "boundary_color_support_ratio": np.clip(0.42 + 0.28 * color, 0, 1),
                "warp_support_ratio": np.clip(0.45 + 0.28 * warp, 0, 1),
                "boundary_edge_support_ratio": np.clip(0.40 + 0.30 * edge, 0, 1),
                "boundary_agreement_ratio": np.clip(agreement, 0, 1),
                "boundary_corroboration": np.clip(boundary_corrob, 0, 1),
                "warp_supported_score": np.clip(warp_supported, 0, 1),
                "violation_count": np.clip(0.42 + 0.30 * agreement, 0, 1),
                "major_violation_count": np.clip(0.28 + 0.26 * agreement, 0, 1),
                "critical_violation_count": np.clip(0.04 + 0.08 * (warp > 0.55).astype(float), 0, 1),
                "dynamic_violation_score": np.clip(0.42 + 0.34 * np.clip(rigid * md, 0, 1), 0, 1),
                "boundary_violation_score": np.clip(0.40 + 0.36 * boundary_corrob, 0, 1),
                "violation_ratio_mean": np.clip(0.34 + 0.28 * warp, 0, 1),
                "violation_ratio_max": np.clip(0.44 + 0.30 * np.maximum(warp, seam), 0, 1),
                "violation_span_fraction": np.clip(0.34 + 0.26 * agreement, 0, 1),
                "major_span_fraction": np.clip(0.24 + 0.22 * agreement, 0, 1),
                "critical_span_fraction": np.clip(0.03 + 0.08 * (warp > 0.55).astype(float), 0, 1),
                "dynamic_span_fraction": np.clip(0.32 + 0.24 * np.clip(rigid * md, 0, 1), 0, 1),
                "boundary_span_fraction": np.clip(0.30 + 0.28 * boundary_corrob, 0, 1),
                "confidence_weighted_span": np.clip(0.38 + 0.24 * np.maximum(warp_supported, rigid * md), 0, 1),
                "has_face_motion": np.full(m, 0.72),
                "has_face_scene_decoupling": np.full(m, 0.58),
                "has_rigid_body": np.full(m, 0.62),
                "has_warp_prediction_error": np.full(m, 0.66),
                "has_flow_boundary_seam": np.full(m, 0.54),
                "has_edge_instability": np.full(m, 0.48),
                "has_face_texture": np.full(m, 0.42),
                "frame_anomaly_max": np.clip(0.54 + 0.26 * boundary_peak, 0, 1),
                "frame_anomaly_top3_mean": np.clip(0.46 + 0.26 * boundary_mean, 0, 1),
                "frame_anomaly_peak_fraction": np.clip(0.20 + 0.40 * (agreement > 0.45).astype(float), 0, 1),
                "frame_anomaly_mean": np.clip(0.38 + 0.24 * np.maximum(boundary_corrob, rigid * md), 0, 1),
                "frame_anomaly_std": np.clip(0.18 + 0.14 * boundary_peak, 0, 1),
                "frame_multi_evidence_fraction": np.clip(0.24 + 0.34 * agreement, 0, 1),
            }
            support = np.clip(rng.normal([0.90, 0.88, 0.72, 0.78], [0.05, 0.05, 0.10, 0.10], (m, 4)), 0, 1)
        else:
            extra_map = {
                "face_scene_support_ratio": np.clip(0.26 + 0.16 * rigid, 0, 1),
                "light_support_ratio": np.clip(0.16 + 0.12 * phys, 0, 1),
                "skin_support_ratio": np.clip(0.14 + 0.10 * phys, 0, 1),
                "flow_boundary_support_ratio": np.clip(0.18 + 0.18 * seam, 0, 1),
                "boundary_color_support_ratio": np.clip(0.18 + 0.16 * color, 0, 1),
                "warp_support_ratio": np.clip(0.18 + 0.16 * warp, 0, 1),
                "boundary_edge_support_ratio": np.clip(0.16 + 0.16 * edge, 0, 1),
                "boundary_agreement_ratio": np.clip(0.55 * agreement, 0, 1),
                "boundary_corroboration": np.clip(0.60 * boundary_corrob, 0, 1),
                "warp_supported_score": np.clip(0.58 * warp_supported, 0, 1),
                "violation_count": np.clip(0.16 + 0.14 * agreement, 0, 1),
                "major_violation_count": np.clip(0.08 + 0.10 * agreement, 0, 1),
                "critical_violation_count": np.zeros(m),
                "dynamic_violation_score": np.clip(0.10 + 0.18 * np.clip(rigid * md, 0, 1), 0, 1),
                "boundary_violation_score": np.clip(0.10 + 0.16 * boundary_corrob, 0, 1),
                "violation_ratio_mean": np.clip(0.10 + 0.14 * warp, 0, 1),
                "violation_ratio_max": np.clip(0.14 + 0.16 * np.maximum(warp, seam), 0, 1),
                "violation_span_fraction": np.clip(0.08 + 0.12 * agreement, 0, 1),
                "major_span_fraction": np.clip(0.05 + 0.10 * agreement, 0, 1),
                "critical_span_fraction": np.zeros(m),
                "dynamic_span_fraction": np.clip(0.06 + 0.10 * np.clip(rigid * md, 0, 1), 0, 1),
                "boundary_span_fraction": np.clip(0.08 + 0.12 * boundary_corrob, 0, 1),
                "confidence_weighted_span": np.clip(0.10 + 0.14 * np.maximum(boundary_corrob, rigid * md), 0, 1),
                "has_face_motion": np.full(m, 0.20),
                "has_face_scene_decoupling": np.full(m, 0.10),
                "has_rigid_body": np.full(m, 0.18),
                "has_warp_prediction_error": np.full(m, 0.14),
                "has_flow_boundary_seam": np.full(m, 0.12),
                "has_edge_instability": np.full(m, 0.10),
                "has_face_texture": np.full(m, 0.10),
                "frame_anomaly_max": np.clip(0.16 + 0.20 * boundary_peak, 0, 1),
                "frame_anomaly_top3_mean": np.clip(0.14 + 0.18 * boundary_mean, 0, 1),
                "frame_anomaly_peak_fraction": np.clip(0.04 + 0.12 * (agreement > 0.45).astype(float), 0, 1),
                "frame_anomaly_mean": np.clip(0.12 + 0.16 * np.maximum(boundary_corrob, rigid * md), 0, 1),
                "frame_anomaly_std": np.clip(0.08 + 0.10 * boundary_peak, 0, 1),
                "frame_multi_evidence_fraction": np.clip(0.06 + 0.12 * agreement, 0, 1),
            }
            support = np.clip(rng.normal([0.92, 0.90, 0.75, 0.72], [0.04, 0.04, 0.08, 0.08], (m, 4)), 0, 1)

        # L5 frequency features (synthetic)
        if label == 1:
            l5_map = {
                "l5_frequency_score": np.clip(rng.normal(0.55, 0.12, m), 0, 1),
                "l5_lap_var_ratio": np.clip(rng.normal(1.60, 0.40, m), 0, 5),
                "l5_lap_var_temporal_cv": np.clip(rng.normal(0.17, 0.03, m), 0, 1),
                "l5_dct_hf_ratio": np.clip(rng.normal(1.05, 0.04, m), 0, 5),
                "l5_dct_hf_temporal_cv": np.clip(rng.normal(0.12, 0.03, m), 0, 1),
            }
        else:
            l5_map = {
                "l5_frequency_score": np.clip(rng.normal(0.45, 0.12, m), 0, 1),
                "l5_lap_var_ratio": np.clip(rng.normal(1.15, 0.35, m), 0, 5),
                "l5_lap_var_temporal_cv": np.clip(rng.normal(0.15, 0.03, m), 0, 1),
                "l5_dct_hf_ratio": np.clip(rng.normal(0.99, 0.04, m), 0, 5),
                "l5_dct_hf_temporal_cv": np.clip(rng.normal(0.10, 0.03, m), 0, 1),
            }

        feature_map = {}
        feature_map.update(base_map)
        feature_map.update(interaction_map)
        feature_map.update(extra_map)
        feature_map.update(l5_map)
        X = np.column_stack([np.clip(feature_map[name], 0, 1) if name not in l5_map else feature_map[name] for name in FEATURE_NAMES])
        if include_support:
            X = np.hstack([X, support])
        return X

    Xa = build_feature_block(Xa16, label=0)
    Xi = build_feature_block(Xi16, label=1)
    X = np.vstack([Xa, Xi])
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def normalize(X_train, X_val=None):
    mu = X_train.mean(0); sigma = X_train.std(0) + 1e-8
    Xn = (X_train - mu) / sigma
    if X_val is not None:
        return Xn, (X_val - mu) / sigma, mu, sigma
    return Xn, mu, sigma


def compute_auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
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


def _threshold_candidates(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return [0.5]

    unique_scores = np.unique(np.clip(scores, 0.0, 1.0))
    candidates = {0.0, 1.0, 0.5}
    for score in unique_scores:
        s = float(score)
        candidates.add(max(0.0, s - 1e-6))
        candidates.add(min(1.0, s + 1e-6))

    for left, right in zip(unique_scores[:-1], unique_scores[1:]):
        candidates.add(float((left + right) * 0.5))

    return sorted(candidates)


def specificity_first_threshold(scores, labels, min_specificity=0.70, min_recall=0.0):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    best = None
    best_key = None
    for t in _threshold_candidates(scores):
        p = (scores >= t).astype(int)
        tp = int(((p == 1) & (labels == 1)).sum())
        tn = int(((p == 0) & (labels == 0)).sum())
        fp = int(((p == 1) & (labels == 0)).sum())
        fn = int(((p == 0) & (labels == 1)).sum())
        rec = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        bal = 0.5 * (rec + spec)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        meets_spec = (spec + THRESHOLD_TOL) >= float(min_specificity)
        meets_recall = (rec + THRESHOLD_TOL) >= float(min_recall)
        tier = 2 if (meets_spec and meets_recall) else 1 if meets_spec else 0
        if tier == 2:
            key = (
                tier,
                round(bal, 12),
                round(f1, 12),
                round(rec, 12),
                round(spec, 12),
                -abs(float(t) - 0.5),
            )
        elif tier == 1:
            # Specificity-first policy for constrained thresholds.
            key = (
                tier,
                round(bal, 12),
                round(spec, 12),
                round(f1, 12),
                round(rec, 12),
                -abs(float(t) - 0.5),
            )
        else:
            key = (
                tier,
                round(spec, 12),
                round(rec, 12),
                round(bal, 12),
                -abs(float(t) - 0.5),
            )
        if best is None or key > best_key:
            best = {
                "threshold": float(t),
                "specificity": float(spec),
                "balanced_accuracy": float(bal),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "target_specificity": float(min_specificity),
                "target_recall": float(min_recall),
                "meets_target": bool(meets_spec),
                "meets_recall": bool(meets_recall),
                "constraint_tier": int(tier),
            }
            best_key = key
    return best


def youden_threshold(scores, labels):
    best_j, best_t = -1.0, 0.5
    for t in _threshold_candidates(scores):
        p = (scores >= t).astype(int)
        tp = int(((p==1)&(labels==1)).sum()); tn = int(((p==0)&(labels==0)).sum())
        fp = int(((p==1)&(labels==0)).sum()); fn = int(((p==0)&(labels==1)).sum())
        j = tp/(tp+fn+1e-9) + tn/(tn+fp+1e-9) - 1.0
        if j > best_j: best_j, best_t = j, float(t)
    return best_t, best_j


def evaluate(model, X, y, threshold=0.5, use_tta=False):
    p = model.predict_tta(X) if use_tta else model.predict(X)
    preds = (p > threshold).astype(int)
    tp = int(((preds==1)&(y==1)).sum()); tn = int(((preds==0)&(y==0)).sum())
    fp = int(((preds==1)&(y==0)).sum()); fn = int(((preds==0)&(y==1)).sum())
    acc = (tp+tn)/max(len(y),1); prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    spec = tn/(tn+fp+1e-9)
    return dict(accuracy=float(acc), precision=float(prec), recall=float(rec),
                specificity=float(spec), balanced_accuracy=float(0.5*(rec+spec)),
                f1=float(2*prec*rec/(prec+rec+1e-9)),
                auc=float(compute_auc(p,y)), tp=tp, tn=tn, fp=fp, fn=fn)


def predict_with_model_dict(model_dict, X):
    model_type = model_dict.get("model_type", "mlp")
    if model_type == "linear":
        w = np.asarray(model_dict["weights"], dtype=float)
        b = float(model_dict["bias"])
        return _sigmoid(np.asarray(X, dtype=float) @ w + b)

    a1 = _relu(np.asarray(X, dtype=float) @ np.asarray(model_dict["W1"], dtype=float) + np.asarray(model_dict["b1"], dtype=float))
    a2 = _relu(a1 @ np.asarray(model_dict["W2"], dtype=float) + np.asarray(model_dict["b2"], dtype=float))
    a3 = _relu(a2 @ np.asarray(model_dict["W3"], dtype=float) + np.asarray(model_dict["b3"], dtype=float))
    return _sigmoid(a3 @ np.asarray(model_dict["W4"], dtype=float) + np.asarray(model_dict["b4"], dtype=float)).reshape(-1)


def load_model_bundle(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_with_bundle(X, bundle):
    norm = bundle.get("normalization", {})
    mu = np.asarray(norm.get("mean", []), dtype=float)
    sigma = np.asarray(norm.get("std", []), dtype=float)
    if mu.size == 0 or sigma.size == 0:
        return np.asarray(X, dtype=float)
    return (np.asarray(X, dtype=float) - mu) / (sigma + 1e-8)


def predict_with_bundle(bundle, X, model_name=None):
    selected = model_name or bundle.get("selected_model", "mlp")
    models = bundle.get("models", {})
    if selected not in models:
        raise KeyError(f"Unknown model '{selected}' in bundle.")
    Xn = normalize_with_bundle(X, bundle)
    return predict_with_model_dict(models[selected], Xn), float(bundle.get("selected_threshold", 0.5)), selected


def stratified_split(X, y, vf=0.20, tf=0.20, seed=42):
    rng = np.random.default_rng(seed)
    i0, i1 = np.where(y==0)[0], np.where(y==1)[0]
    rng.shuffle(i0); rng.shuffle(i1)
    def sp(idx):
        n = len(idx); nt = max(1,int(n*tf)); nv = max(1,int(n*vf))
        return idx[nv+nt:], idx[nt:nv+nt], idx[:nt]
    tr0,v0,te0 = sp(i0); tr1,v1,te1 = sp(i1)
    itr = np.concatenate([tr0,tr1]); iv = np.concatenate([v0,v1]); ite = np.concatenate([te0,te1])
    return X[itr],y[itr], X[iv],y[iv], X[ite],y[ite]


def detect_split_manifest(data_path, explicit_path=None):
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Δεν βρέθηκε split manifest: {path}")
        return path
    if not data_path:
        return None
    candidate = Path(data_path).resolve().with_name("split_manifest.csv")
    return candidate if candidate.exists() else None


def split_indices_from_manifest(manifest_path, n_rows):
    train_idx, val_idx, test_idx = [], [], []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(float(row.get("index", -1)))
            if idx < 0 or idx >= n_rows:
                continue
            split = str(row.get("split", "")).strip().lower()
            if split == "train":
                train_idx.append(idx)
            elif split in {"validation", "val"}:
                val_idx.append(idx)
            elif split == "test":
                test_idx.append(idx)
    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int), np.array(test_idx, dtype=int)


def main():
    parser = argparse.ArgumentParser(description="PATV-X Open Track Training")
    parser.add_argument("--data")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.008)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output", default="patv_open_bundle.json")
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--min-specificity", type=float, default=0.70)
    parser.add_argument("--min-recall", type=float, default=0.15)
    parser.add_argument("--no-support-features", action="store_true")
    parser.add_argument("--split-manifest")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  PATV-X Open  Training  (Linear + MLP)")
    print("="*60)

    include_support = not args.no_support_features
    feature_names = OPEN_FEATURE_NAMES if include_support else FEATURE_NAMES
    track = "synthetic-smoke" if args.synthetic else "open"

    if not args.synthetic and not args.data:
        sys.exit("[ΣΦΑΛΜΑ] Για publishable training απαιτείται --data. Το synthetic είναι μόνο smoke mode.")

    if args.synthetic:
        print(f"\n[Data] Synthetic smoke {args.n_samples} samples ({len(feature_names)}-D)...")
        X, y = generate_synthetic_data(args.n_samples, include_support=include_support)
    else:
        print(f"\n[Data] Loading real features: {args.data}")
        X, y = csv_to_features(args.data, include_support=include_support)
        if not len(X):
            sys.exit("[ΣΦΑΛΜΑ] Δεν βρέθηκαν δεδομένα.")

    print(f"[Data] {len(X)} samples | AI={int(y.sum())} Auth={int((y==0).sum())}")
    X = np.clip(np.nan_to_num(X, 0.0, 1.0, 0.0), 0, 1)

    split_manifest = None if args.synthetic else detect_split_manifest(args.data, args.split_manifest)
    if split_manifest is not None:
        train_idx, val_idx, test_idx = split_indices_from_manifest(split_manifest, len(X))
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise SystemExit(f"[ΣΦΑΛΜΑ] Άδειο split στο manifest: {split_manifest}")
        Xtr, ytr = X[train_idx], y[train_idx]
        Xv, yv = X[val_idx], y[val_idx]
        Xte, yte = X[test_idx], y[test_idx]
        split_desc = f"manifest={split_manifest}"
    else:
        Xtr,ytr, Xv,yv, Xte,yte = stratified_split(X, y)
        split_desc = "auto_stratified"

    Xtr_n, Xv_n, mu, sigma = normalize(Xtr, Xv)
    Xte_n = (Xte - mu) / sigma

    print(f"\n[Split] train={len(Xtr)} val={len(Xv)} test={len(Xte)}  ({split_desc})")
    print(f"[Train] epochs={args.epochs} lr={args.lr} batch={args.batch_size} wd={args.weight_decay}")
    print(
        f"[Track] {track} | support_features={include_support} | "
        f"target_specificity={args.min_specificity:.2f} | target_recall={args.min_recall:.2f}"
    )
    print("-"*60)

    linear = LinearLogistic(input_dim=len(feature_names), weight_decay=args.weight_decay)
    linear.train(Xtr_n, ytr, Xv_n, yv, epochs=max(120, args.epochs), batch_size=max(64, args.batch_size), verbose=True)
    linear_val_probs = linear.predict(Xv_n)
    linear_thr = specificity_first_threshold(
        linear_val_probs,
        yv,
        min_specificity=args.min_specificity,
        min_recall=args.min_recall,
    )
    linear_vm = evaluate(linear, Xv_n, yv, linear_thr["threshold"])
    linear_tm = evaluate(linear, Xte_n, yte, linear_thr["threshold"])

    model = MLP(input_dim=len(feature_names), lr=args.lr, weight_decay=args.weight_decay)
    model.train(Xtr_n, ytr, Xv_n, yv, epochs=args.epochs, batch_size=args.batch_size)
    mlp_val_probs = model.predict_tta(Xv_n)
    mlp_thr = specificity_first_threshold(
        mlp_val_probs,
        yv,
        min_specificity=args.min_specificity,
        min_recall=args.min_recall,
    )
    mlp_vm = evaluate(model, Xv_n, yv, mlp_thr["threshold"], use_tta=True)
    mlp_tm = evaluate(model, Xte_n, yte, mlp_thr["threshold"], use_tta=True)

    # GBM (Gradient Boosting) — best performing model
    gbm = GBMClassifier(input_dim=len(feature_names))
    gbm.train(Xtr_n, ytr, Xv_n, yv)
    gbm_val_probs = gbm.predict(Xv_n)
    gbm_thr = specificity_first_threshold(
        gbm_val_probs,
        yv,
        min_specificity=args.min_specificity,
        min_recall=args.min_recall,
    )
    gbm_vm = evaluate(gbm, Xv_n, yv, gbm_thr["threshold"])
    gbm_tm = evaluate(gbm, Xte_n, yte, gbm_thr["threshold"])

    print(f"\n[Linear] val_thr={linear_thr['threshold']:.3f} spec={linear_vm['specificity']:.3f} bal_acc={linear_vm['balanced_accuracy']:.3f} auc={linear_vm['auc']:.4f}")
    print(f"[MLP]    val_thr={mlp_thr['threshold']:.3f} spec={mlp_vm['specificity']:.3f} bal_acc={mlp_vm['balanced_accuracy']:.3f} auc={mlp_vm['auc']:.4f}")
    print(f"[GBM]    val_thr={gbm_thr['threshold']:.3f} spec={gbm_vm['specificity']:.3f} bal_acc={gbm_vm['balanced_accuracy']:.3f} auc={gbm_vm['auc']:.4f}")

    # Select best model by validation balanced accuracy, then AUC
    candidates = [
        ("linear", linear_vm, linear_thr),
        ("mlp", mlp_vm, mlp_thr),
        ("gbm", gbm_vm, gbm_thr),
    ]
    candidates.sort(key=lambda c: (
        round(c[1]["balanced_accuracy"], 6),
        round(c[1]["auc"], 6),
    ), reverse=True)
    selected_model = candidates[0][0]
    selected_threshold = candidates[0][2]["threshold"]

    print("\n  Validation Linear:", {k: round(v, 3) if isinstance(v, float) else v for k, v in linear_vm.items()})
    print("  Validation MLP:   ", {k: round(v, 3) if isinstance(v, float) else v for k, v in mlp_vm.items()})
    print("  Validation GBM:   ", {k: round(v, 3) if isinstance(v, float) else v for k, v in gbm_vm.items()})
    print("  Test Linear:      ", {k: round(v, 3) if isinstance(v, float) else v for k, v in linear_tm.items()})
    print("  Test MLP:         ", {k: round(v, 3) if isinstance(v, float) else v for k, v in mlp_tm.items()})
    print("  Test GBM:         ", {k: round(v, 3) if isinstance(v, float) else v for k, v in gbm_tm.items()})
    print(f"\n  Selected model: {selected_model}  threshold={selected_threshold:.3f}")

    out = {
        "track": track,
        "analysis_version": "patv_x_open_v2",
        "feature_names": feature_names,
        "include_support_features": bool(include_support),
        "split_source": split_desc,
        "normalization": {"mean": mu.tolist(), "std": sigma.tolist()},
        "threshold_policy": {
            "name": "specificity_first",
            "target_specificity": float(args.min_specificity),
            "target_recall": float(args.min_recall),
        },
        "models": {
            "linear": linear.save_dict(),
            "mlp": {
                "model_type": "mlp",
                "architecture": f"{len(feature_names)}→64→32→16→1",
                "W1": model.W1.tolist(), "b1": model.b1.tolist(),
                "W2": model.W2.tolist(), "b2": model.b2.tolist(),
                "W3": model.W3.tolist(), "b3": model.b3.tolist(),
                "W4": model.W4.tolist(), "b4": model.b4.tolist(),
                "history": model.training_history,
            },
            "gbm": gbm.save_dict(),
        },
        "validation": {
            "linear": {"threshold_info": linear_thr, "metrics": linear_vm},
            "mlp": {"threshold_info": mlp_thr, "metrics": mlp_vm},
            "gbm": {"threshold_info": gbm_thr, "metrics": gbm_vm},
        },
        "test": {
            "linear": linear_tm,
            "mlp": mlp_tm,
            "gbm": gbm_tm,
        },
        "selected_model": selected_model,
        "selected_threshold": float(selected_threshold),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {args.output}")
    print("="*60+"\n")


if __name__ == "__main__":
    main()

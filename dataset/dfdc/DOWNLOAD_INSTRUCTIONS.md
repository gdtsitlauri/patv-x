# DFDC (DeepFake Detection Challenge) Dataset

## About

The DeepFake Detection Challenge (DFDC) dataset contains ~100,000 videos across
diverse subjects, backgrounds, lighting, and five face-swap / face-reenactment
methods. It is the largest public deepfake benchmark and provides strong
cross-method generalization signal.

- **Size**: ~470 GB (full); preview set ~10 GB
- **License**: [DFDC Terms of Service](https://ai.meta.com/datasets/dfdc/) —
  requires acceptance of Meta AI's research use terms
- **Download**: https://ai.meta.com/datasets/dfdc/
- **Paper**: Dolhansky et al., "The DeepFake Detection Challenge Dataset",
  arXiv:2006.07397

## Expected Directory Layout

After downloading, organize files as follows so `evaluate_generalization.py`
can find them:

```
dataset/dfdc/
  metadata.json        ← per-video labels from DFDC ({"filename": {"label": "FAKE"|"REAL"}})
  videos/
    dfdc_train_part_00/
      xxxxxxxx.mp4
      ...
    dfdc_train_part_01/
      ...
```

## Running Evaluation

```bash
# Core-track (training-free) evaluation:
python evaluation/evaluate_generalization.py \
    --dataset dataset/dfdc \
    --layout dfdc \
    --dataset-name DFDC \
    --output generalization_results/dfdc

# With open-track GBM model:
python evaluation/evaluate_generalization.py \
    --dataset dataset/dfdc \
    --layout dfdc \
    --model pipeline_results_final/patv_open_bundle.json \
    --dataset-name DFDC \
    --output generalization_results/dfdc
```

## Expected Generalization Behavior

PATV-X was trained exclusively on FF++ c23 faceswap. On DFDC, expect:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| Core AUC-ROC | 0.60–0.72 | DFDC methods include neural synthesis; Laplacian features remain partly discriminative |
| Core Recall | 0.55–0.75 | High-quality neural fakes may produce fewer Laplacian anomalies |
| Core Specificity | 0.55–0.70 | DFDC real videos have more compression than FF++ raw |

The core detector's physics-based and frequency-domain features generalize to
methods that introduce face-boundary artifacts and frequency inconsistencies.
Neural synthesis methods (e.g., NTH, FSGAN) that preserve high-frequency
texture may score closer to authentic video.

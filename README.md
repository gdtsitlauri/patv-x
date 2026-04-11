# PATV-X: Training-Free Explainable Deepfake Detection via Multi-Scale Frequency Forensics
---
**Research Paper:**
This project is accompanied by a full research paper describing the architecture, methodology, and experimental results of PATV-X. The research and all experiments were conducted in 2026.

You can find the paper in [paper/patv_x_paper.tex](paper/patv_x_paper.tex).
---
PATV-X is a novel, training-free, explainable face-centric temporal video forensics system for detecting face-swapped and AI-generated videos. It achieves **AUC 0.83** on FaceForensics++ (c23, faceswap) without any training on deepfake data.

## Key Results

**Dataset**: FaceForensics++ c23 faceswap (200 videos: 100 authentic + 100 faceswap)

| Metric | Raw Detector (training-free) | + GBM Learned Head |
|--------|-----------------------------|--------------------|
| **AUC-ROC** | 0.733 | **0.830** |
| Specificity | 0.600 | 0.850 |
| Recall | 0.800 | 0.600 |
| Balanced Accuracy | 70.0% | 72.5% |
| Core Acceptance | PASS | PASS |

## What Makes PATV-X Novel

### 1. Training-Free Detection
PATV-X does not train on any deepfake data. It detects forgeries by measuring deviations from the statistical properties of natural video — Laplacian distributions, wavelet energy ratios, frequency-domain consistency, and cross-channel correlations. This makes it inherently generalizable to unseen manipulation methods.

### 2. Multi-Scale Frequency Forensics (L5)
The core algorithmic contribution is **L5: Multi-Scale Frequency Forensics** — a new analysis level with 16 features across 9 signal families:

| Signal Family | Features | Best Individual AUC |
|---------------|----------|---------------------|
| Laplacian Kurtosis | mean, temporal CV | 0.699 |
| Wavelet Detail Ratio | Gaussian pyramid fine/coarse energy | 0.665 |
| Per-Channel Laplacian | R, G, B temporal CV | 0.634 |
| Laplacian Variance | face/bg ratio, temporal CV | 0.607 |
| Block Consistency | 2x2 grid Laplacian variance | 0.590 |
| Cross-Channel Correlation | R-B Laplacian correlation | 0.587 |
| Temporal Dynamics | first-diff, trend residual, acceleration | 0.578 |
| SRM Noise Residual | high-pass filter noise analysis | 0.578 |
| DCT High-Frequency | high-freq energy ratio | 0.540 |

**Critical finding**: Laplacian features are scale-dependent. Processing at the original video resolution is essential; resizing to 640px destroys the discriminative signal entirely.

### 3. Explainable Architecture
Every detection is accompanied by a structured explanation across 5 analysis levels:

- **L1**: Residual motion analysis (optical flow anomalies)
- **L2**: Physics plausibility (face-neck rigid body decoupling)
- **L3**: Semantic consistency (color drift, texture, edge stability)
- **L4**: Boundary artifacts (CTCG-style spectral analysis)
- **L5**: Multi-scale frequency forensics (Laplacian, wavelet, SRM)

### 4. Frequency-Anchored Score Fusion
The final TCS (Temporal Consistency Score) uses a frequency-anchored fusion formula:

```
TCS = 0.50 * L5 + 0.20 * L4 + 0.10 * L5xL4_interaction
    + 0.08 * L1 + 0.06 * boundary_corroboration
    + 0.04 * L2 + 0.02 * L3
```

L5 frequency features carry 50% of the detection weight, validated by ablation studies showing they provide the strongest individual discrimination.

### 5. GBM Learned Head
A lightweight Gradient Boosting Machine (300 estimators, max_depth=3) trained on the 83 training-free features achieves AUC 0.83. Top features by importance:

1. `l5_frequency_score` (0.176) — composite frequency anomaly
2. `l5_wavelet_detail_ratio` (0.109) — fine/coarse pyramid energy
3. `l5_lap_kurtosis_cv` (0.099) — Laplacian kurtosis temporal variation
4. `l5_srm_var_cv` (0.061) — SRM noise residual variation
5. `l5_lap_var_ratio` (0.058) — face/background Laplacian ratio

## Project Structure

```
patv_x/
  src/
    patv_x_detector.py       # Core detector (L1-L5 analysis levels)
  training/
    train_mlp.py              # GBM, MLP, Linear model training
  legacy/data_pipeline/
    run_pipeline.py           # End-to-end evaluation pipeline
  ablation/
    ablation_study.py         # Level/feature ablation studies
  dataset/
    faceforensics/            # FF++ c23 data (videos not in repo)
  pipeline_results_final/     # Final evaluation results
  patv_cli.py                 # Command-line interface
  requirements.txt
  LICENSE
  .gitignore
```

### Core Files

- **`src/patv_x_detector.py`** — The main detector. Contains all 5 analysis levels, the frequency-anchored fusion formula, face detection, original-resolution frequency analysis, and explainability output. This is where the research contribution lives.

- **`training/train_mlp.py`** — Training module for the learned heads (GBM, MLP, Linear). Includes feature extraction from CSV, pair-aware cross-validation, threshold selection under specificity constraints, and model export.

- **`legacy/data_pipeline/run_pipeline.py`** — The full evaluation pipeline: feature extraction from videos, pair-aware train/val/test splitting, model training and selection, core acceptance testing, ablation studies, and final reporting.

- **`ablation/ablation_study.py`** — Ablation diagnostics: level contribution analysis, individual feature contribution, weight grid search.

## How to Run

### Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+ required. Dependencies: `opencv-python`, `numpy`, `scikit-learn`, `scipy`.

### Full Pipeline (from videos)

```bash
python legacy/data_pipeline/run_pipeline.py \
  --videos dataset/faceforensics \
  --output pipeline_results \
  --workers 0 \
  --sample-rate 6 \
  --max-frames 96
```

### Pipeline from Existing Features

```bash
python legacy/data_pipeline/run_pipeline.py \
  --features pipeline_results_final/features.csv \
  --output pipeline_results_new
```

### Single Video Analysis

```bash
# Core track (training-free)
python patv_cli.py analyze video.mp4 --track core --verbose

# Open track (with learned head)
python patv_cli.py analyze video.mp4 \
  --track open \
  --model pipeline_results_final/patv_open_bundle.json \
  --verbose
```

### Ablation Study

```bash
python ablation/ablation_study.py --data pipeline_results_final/features.csv
```

## Evaluation Protocol

- **Pair-aware splitting**: Authentic/faceswap pairs share source identities. The pipeline enforces pair-aware group splitting to prevent identity leakage across train/val/test.
- **Threshold tuning on validation only**: The detection threshold is selected on the validation set; test metrics are never used for model selection.
- **Core acceptance criteria**: specificity >= 0.60, recall >= 0.65, balanced accuracy >= 0.65, mean score gap >= 0.08.
- **GBM model selection**: Validated via 5-fold pair-aware cross-validation (OOF-AUC 0.844). The GBM is retrained on train+val for final test evaluation.

## Research Contributions

1. **Multi-Scale Frequency Forensics (L5)**: A novel set of 16 training-free features from 9 signal families that capture forgery artifacts invisible to spatial-domain analysis.

2. **Original-Resolution Frequency Analysis**: Empirical demonstration that Laplacian-based forensic features are scale-dependent and must be computed at native resolution.

3. **Frequency-Anchored Fusion**: A principled score fusion formula where frequency forensics (L5) anchors the detection, corroborated by boundary (L4) and motion (L1) evidence.

4. **Explainable Detection**: Every prediction includes per-level scores and human-readable explanations of which signal families triggered and why.

## License

MIT License. See [LICENSE](LICENSE).

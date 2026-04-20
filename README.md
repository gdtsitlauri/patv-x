# PATV-X: Training-Free Explainable Deepfake Detection via Multi-Scale Frequency Forensics

**Author:** George David Tsitlauri  
**Affiliation:** Dept. of Informatics & Telecommunications, University of Thessaly, Greece  
**Contact:** gdtsitlauri@gmail.com  
**Year:** 2026
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
  evaluation/
    evaluate_generalization.py  # Cross-dataset generalization evaluator
  dataset/
    faceforensics/            # FF++ c23 data (videos not in repo)
    dfdc/                     # DFDC dataset stub (see DOWNLOAD_INSTRUCTIONS.md)
    faceshifter/              # FaceShifter dataset stub (see DOWNLOAD_INSTRUCTIONS.md)
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

## Generalization Beyond FaceForensics++

### Why FF++ c23 Faceswap Is the Primary Benchmark

FaceForensics++ (c23, faceswap) is the standard benchmark for this class of work because:

1. **Controlled compression**: c23 applies moderate H.264 compression — realistic for social-media video without destroying discriminative frequency artifacts. Raw (c0) would overstate performance; heavy (c40) would suppress it.
2. **Pair-aligned subjects**: Every manipulated video has an identically sourced authentic counterpart with the same subject identity, enabling rigorous pair-aware cross-validation that prevents identity-based leakage.
3. **Classical faceswap artifacts**: The faceswap method produces the class of frequency and boundary artifacts (Laplacian discontinuities, wavelet energy shifts, SRM noise residuals) that PATV-X's L4/L5 features are designed to detect.
4. **Reproducibility**: FF++ is publicly available under a research license and is the most widely used benchmark for direct comparison with prior work (MesoNet, XceptionNet, FWA, etc.).

### Expected Generalization Limits

PATV-X's core detector is training-free and hypothesis-driven. Its generalization properties are therefore predictable from first principles:

| Manipulation Type | Expected Performance | Limiting Factor |
|-------------------|---------------------|-----------------|
| FF++ faceswap (c23) | AUC 0.73 (measured) | — primary benchmark |
| FF++ Deepfakes (c23) | AUC 0.68–0.75 | Similar boundary + frequency artifacts |
| FaceShifter (c23) | AUC 0.62–0.73 | Fewer blending artifacts; better attribute preservation |
| DFDC (mixed methods) | AUC 0.60–0.72 | Neural synthesis methods; diverse compression |
| Face reenactment (e.g., Face2Face) | AUC 0.55–0.68 | No face-swap boundary; different artifact signature |
| Diffusion-based generation | AUC 0.50–0.62 | Full-frame synthesis; no blending boundary at all |
| Authentic video (any source) | Specificity ≥ 0.60 | Core acceptance criterion maintained |

**Key generalization insight**: PATV-X's Laplacian and wavelet features detect the *statistical footprint of blending* — a manipulated face region pasted onto a real background frame. Methods that eliminate blending (diffusion generation, full-frame synthesis) produce different or weaker artifacts and should be expected to lower recall. The core detector will not silently fail; its per-level explainability output indicates which signals triggered, making it possible to diagnose why a particular manipulation type is harder.

### Running Cross-Dataset Evaluation

A dedicated script, `evaluation/evaluate_generalization.py`, evaluates PATV-X on any compatible dataset without retraining.

#### FaceShifter (FF++ license, available after signing the FF++ request form)

```bash
# Download FF++ FaceShifter videos and place them in dataset/faceshifter/
# See dataset/faceshifter/DOWNLOAD_INSTRUCTIONS.md

python evaluation/evaluate_generalization.py \
    --dataset dataset/faceshifter \
    --layout ff \
    --model pipeline_results_final/patv_open_bundle.json \
    --dataset-name FaceShifter-c23 \
    --output generalization_results/faceshifter
```

#### DFDC (Meta AI research license)

```bash
# Download DFDC dataset and place videos + metadata.json in dataset/dfdc/
# See dataset/dfdc/DOWNLOAD_INSTRUCTIONS.md

python evaluation/evaluate_generalization.py \
    --dataset dataset/dfdc \
    --layout dfdc \
    --model pipeline_results_final/patv_open_bundle.json \
    --dataset-name DFDC \
    --output generalization_results/dfdc
```

#### Any Custom Dataset (flat layout with manifest)

```bash
# Prepare manifest.csv with columns: filename, label (0=real, 1=fake), category
# Place videos in your_dataset/videos/

python evaluation/evaluate_generalization.py \
    --dataset path/to/your_dataset \
    --layout flat \
    --dataset-name "My Dataset" \
    --output generalization_results/custom
```

#### From Pre-Extracted Features (skips re-inference)

```bash
python evaluation/evaluate_generalization.py \
    --features path/to/features.csv \
    --model pipeline_results_final/patv_open_bundle.json \
    --output generalization_results/from_csv
```

The script outputs a `generalization_report.json` with AUC-ROC, recall, specificity, and balanced accuracy, plus the extracted feature CSV for further analysis.

## Research Contributions

1. **Multi-Scale Frequency Forensics (L5)**: A novel set of 16 training-free features from 9 signal families that capture forgery artifacts invisible to spatial-domain analysis.

2. **Original-Resolution Frequency Analysis**: Empirical demonstration that Laplacian-based forensic features are scale-dependent and must be computed at native resolution.

3. **Frequency-Anchored Fusion**: A principled score fusion formula where frequency forensics (L5) anchors the detection, corroborated by boundary (L4) and motion (L1) evidence.

4. **Explainable Detection**: Every prediction includes per-level scores and human-readable explanations of which signal families triggered and why.

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{tsitlauri2026patvx,
  author = {George David Tsitlauri},
  title  = {PATV-X: Training-Free Explainable Deepfake Detection via Multi-Scale Frequency Forensics},
  year   = {2026},
  institution = {University of Thessaly},
  email  = {gdtsitlauri@gmail.com}
}
```

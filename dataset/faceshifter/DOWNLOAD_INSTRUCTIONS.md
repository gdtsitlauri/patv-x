# FaceShifter Dataset

## About

FaceShifter (Li et al., 2020) is a high-fidelity identity-preserving face-swap
method that produces significantly more realistic swaps than the classic
FaceForensics++ methods. Its artifacts differ from FF++ faceswap:

- Uses a two-stage synthesis pipeline (AEI-Net + HEAR)
- Better preserves target facial attributes (pose, expression, illumination)
- Fewer blending boundary artifacts than classical faceswap
- Included in FaceForensics++ as the `FaceShifter` manipulation category

- **License**: FaceShifter videos are distributed as part of the FF++ dataset
  under the [FF++ Terms of Use](https://github.com/ondyari/FaceForensics)
  (requires research agreement with the dataset authors)
- **Download**: https://github.com/ondyari/FaceForensics (requires signing the
  request form at the linked Google form)
- **Paper**: Li et al., "FaceShifter: Towards High Fidelity And Occlusion Aware
  Face Swapping", CVPR 2020

## Expected Directory Layout

```
dataset/faceshifter/
  manifest.csv         ← required; columns: filename, label, category
  ai/
    *.mp4              ← FaceShifter manipulated videos (c23 compression)
  authentic/
    *.mp4              ← Corresponding authentic videos (same subjects as FF++)
```

### manifest.csv format

```csv
filename,label,category
0000_0001.mp4,1,faceshifter
0001_0000.mp4,1,faceshifter
0000.mp4,0,authentic
0001.mp4,0,authentic
```

## Running Evaluation

```bash
# Core-track (training-free) evaluation:
python evaluation/evaluate_generalization.py \
    --dataset dataset/faceshifter \
    --layout ff \
    --dataset-name FaceShifter-c23 \
    --output generalization_results/faceshifter

# With open-track GBM model:
python evaluation/evaluate_generalization.py \
    --dataset dataset/faceshifter \
    --layout ff \
    --model pipeline_results_final/patv_open_bundle.json \
    --dataset-name FaceShifter-c23 \
    --output generalization_results/faceshifter
```

## Expected Generalization Behavior

FaceShifter is a harder target than classical faceswap because it:
1. Produces fewer blending-boundary artifacts at face edges (L4 signal weaker)
2. Better preserves facial texture (Laplacian kurtosis closer to authentic)
3. Still shows cross-channel frequency inconsistencies (L5 partially effective)

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| Core AUC-ROC | 0.62–0.73 | Fewer boundary artifacts reduces L4 contribution |
| Core Recall | 0.55–0.70 | High-fidelity synthesis is harder for frequency forensics |
| GBM AUC-ROC | 0.65–0.76 | GBM re-weights; wavelet and SRM features remain informative |

Performance drop relative to FF++ faceswap is expected and informative: it
quantifies the gap introduced by improved synthesis quality.

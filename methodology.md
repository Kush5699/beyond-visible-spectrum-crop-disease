# Solution Write-Up: Beyond Visible Spectrum — AI for Agriculture 2026 (Task 2)

**Author:** Kush Patel  
**Kaggle Username:** @kushp3690  
**Final Ranking:** 5th Place  
**Private LB:** 0.79166 | **Public LB:** 0.81250

---

## 1. Problem Overview

Task 2 of the ICPR 2026 "Beyond Visible Spectrum: AI for Agriculture" competition challenges participants to classify crop diseases from 12-band Sentinel-2 satellite imagery. The four target categories are **Aphid, Blast, RPH, and Rust**. The labeled training dataset contains approximately 900 samples distributed across these classes with significant class imbalance (RPH dominant, Blast minority), while the test set has 40 samples scored in a 40/60 public/private split.

The core challenge lies in the domain gap: high-dimensional multi-spectral satellite data (12 bands covering visible, near-infrared, and SWIR regions) with a very limited number of labeled samples.

## 2. Approach

### 2.1 Strategy: ImageNet Transfer Learning with Spectral Adaptation

Rather than training from scratch or attempting full SSL pre-training (which would require extensive compute for the 126 GB unlabeled dataset), I used ImageNet-pretrained backbones and adapted them for 12-channel Sentinel-2 input.

The key technical insight is that modern vision models (especially Vision Transformers like Swin) learn generalizable spatial features from ImageNet that transfer well to overhead satellite imagery, even though the spectral domain is different. The `timm` library's `in_chans` parameter handles the channel adaptation by reinitializing the first-layer weights from 3 channels (RGB) to 12 channels (Sentinel-2 bands).

### 2.2 Backbone Selection

I evaluated four architectures in a systematic 3-fold, 20-epoch comparison:

| Backbone | Family | Accuracy | Macro F1 | Time |
|---|---|---|---|---|
| **Swin-Tiny** | Vision Transformer | **0.8933 ± 0.012** | **0.7820** | 77.6 min |
| ConvNeXt-Tiny | Modern CNN | 0.8300 ± 0.012 | 0.6766 | 20.7 min |
| ResNet-50 | Classic CNN | 0.5178 ± 0.045 | 0.4565 | 20.8 min |
| EfficientNet-B2 | Efficient CNN | 0.5000 ± 0.033 | 0.3878 | 20.1 min |

**Swin-Tiny** (`swin_tiny_patch4_window7_224`) was the clear winner. Its shifted-window self-attention mechanism is well-suited for capturing both local spectral signatures and broader spatial patterns in satellite imagery. The CNN-based architectures struggled significantly, likely because the small dataset size limited their ability to learn meaningful features from scratch on the adapted 12-channel input.

Swin Transformer was run at its native 224×224 resolution, giving it access to more spatial detail — a significant advantage for distinguishing visually similar disease symptoms.

### 2.3 Data Preprocessing

**Band loading:** Each sample consists of 12 individual GeoTIFF files (one per Sentinel-2 band: B1–B12). These are loaded using `rasterio`, resized to 224×224 via bilinear interpolation, and stacked into a (12, H, W) tensor.

**Normalization:** Percentile-clipped normalization (2nd–98th percentile) per band. This is more robust than simple min-max for satellite imagery where extreme outlier reflectance values are common due to atmospheric effects, sensor noise, and cloud edges.

**Augmentation:** Geometric augmentations only — random horizontal/vertical flips and 90° rotations. These are physically valid for overhead satellite imagery since there is no inherent "up" direction. Augmentations were kept simple to reduce overfitting risk on the small dataset.

### 2.4 Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Well-suited for transformers |
| Learning rate | 1e-4 | Conservative for fine-tuning pretrained weights |
| Weight decay | 1e-4 | Mild regularization |
| Scheduler | Warmup (5 epochs) + cosine decay | Prevents early divergence, smooth convergence |
| Loss | CrossEntropyLoss | Standard multi-class classification |
| Label smoothing | 0.1 | Prevents overconfident predictions, improves generalization |
| Class weights | Balanced: N / (4 × count) | Compensates for class imbalance (RPH >> Blast) |
| Gradient clipping | max_norm = 1.0 | Prevents training instability |
| Epochs | 40 | Sufficient for convergence without overfitting |
| Cross-validation | 5-fold stratified | Maximizes training data usage |

### 2.5 Inference: 5-Fold Ensemble

For final predictions, all 5 fold models are loaded, each produces softmax probabilities on the test set, and the probabilities are averaged across folds before taking argmax. This ensemble averaging reduces variance and produces more robust predictions compared to any single fold.

## 3. Results

### Cross-Validation Performance (5-fold):
| Fold | Accuracy |
|------|----------|
| Fold 1 | 0.8944 |
| Fold 2 | 0.9056 |
| Fold 3 | 0.9278 |
| Fold 4 | 0.9222 |
| Fold 5 | 0.9000 |
| **Mean** | **0.9100 ± 0.0129** |
| **Overall Macro F1** | **0.7914** |

### Leaderboard Performance:
- **Private LB (60% of test):** 0.79166 → **5th Place**
- Public LB (40% of test): 0.81250

### Prediction Distribution (40 test samples):
| Category | Count |
|----------|-------|
| RPH | 17 |
| Aphid | 11 |
| Rust | 8 |
| Blast | 4 |

### Confusion Matrix Analysis:
The main challenge was the **Blast** class (minority, ~35% recall in CV). The model frequently confused Blast with RPH due to similar spectral signatures. Despite this, the ensemble approach produced 4 correct Blast predictions on the test set, contributing to the strong private LB score.

## 4. What Worked

1. **Swin Transformer >> CNNs** for this task — attention handles multi-spectral correlations better than convolutions
2. **Percentile normalization** instead of min-max — crucial for stable satellite data preprocessing
3. **Warmup + cosine LR schedule** — prevented early training collapse observed with other schedulers
4. **5-fold ensemble** — more robust predictions than any single model
5. **Balanced class weights + label smoothing** — helped the model learn minority classes (Blast, Rust)
6. **Keeping it simple** — minimal augmentation, standard loss, no complex multi-stage pipeline

## 5. What Could Be Improved

1. **SSL pre-training** on the 126 GB unlabeled data would likely provide a significant accuracy boost
2. **Test-time augmentation (TTA)** — averaging predictions over rotations/flips at inference
3. **Multi-modal fusion** (combining HS, MS, RGB data) could capture complementary information
4. **Larger Swin variants** (Swin-Small, Swin-Base) with more training epochs
5. **Multi-backbone ensemble** (combining Swin + ConvNeXt predictions)

## 6. Reproducibility

The code has been verified to reproduce the winning private LB score (0.79166) on Kaggle's T4 GPU environment. Due to inherent non-determinism in CUDA operations (GPU parallelism, cuDNN algorithm selection), exact per-sample predictions may vary slightly between runs, but the overall methodology consistently produces:
- CV accuracy: **0.90–0.92**
- Private LB: **0.75–0.80**

**Environment:**
- Kaggle Notebooks with NVIDIA T4 GPU
- Python 3.10+, PyTorch 2.x, timm, rasterio, scikit-learn

## 7. Code & Weights

All source code, model weights, and this methodology document are available at:

**GitHub Repository:** [INSERT YOUR GITHUB REPO LINK HERE]

- `final_notebook.py` — Complete Kaggle notebook (single-file)
- `methodology.md` — This write-up
- `best_fold0.pth` through `best_fold4.pth` — Trained model weights (~55 MB each)

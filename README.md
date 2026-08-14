# Beyond Visible Spectrum: AI for Agriculture 2026 - Task 2

**3rd Place Solution** for the ICPR 2026 "Beyond Visible Spectrum: AI for Agriculture" competition.

**Author:** Kush Ashvinbhai Patel (@kushp3690)  
**Private LB:** 0.79166 | **Public LB:** 0.81250

## Approach

Transfer learning with **Swin Transformer** (swin_tiny_patch4_window7_224) pretrained on ImageNet, adapted for 12-band Sentinel-2 satellite input. 5-fold cross-validation with softmax ensemble averaging.

See [methodology.md](methodology.md) for the full write-up.

## Files

| File | Description |
|------|-------------|
| `final_notebook.py` | Complete Kaggle notebook (single-file) |
| `methodology.md` | Detailed solution write-up |
| `weights/*.pth` | Trained model weights (5-fold) |
| `confusion_matrix.png` | CV confusion matrix |
| `submission.csv` | Final submission predictions |

## Quick Start

1. Upload `final_notebook.py` to Kaggle as a notebook
2. Add the competition dataset as input
3. Run all cells — takes ~2 hours on T4 GPU
4. `submission.csv` will be generated in `/kaggle/working/`

## Results

- CV Accuracy: **0.9100 ± 0.0129**
- CV Macro F1: **0.7914**
- Private LB: **0.79166** (5th Place)

###############################################################################
# Beyond Visible Spectrum: AI for Agriculture 2026 — Task 2 (SSL)
# Crop Disease Classification using Sentinel-2 Satellite Imagery
#
# Author: Kush Patel (@kushp3690)
# Approach: Transfer learning with ImageNet-pretrained Swin Transformer,
#           adapted for 12-band Sentinel-2 input via timm's in_chans.
#           5-fold cross-validation with ensemble averaging for final predictions.
###############################################################################


# ─── Cell 1: Setup & installs ───────────────────────────────────────────
# !pip install -q rasterio timm scikit-learn


# ─── Cell 2: Imports & configuration ────────────────────────────────────

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
import rasterio
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"  GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {gpu.total_memory / 1e9:.1f} GB")

# --- paths (Kaggle layout) ---
BASE_PATH = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2026"
TRAIN_DIR = os.path.join(BASE_PATH, "ICPR02", "kaggle")
EVAL_DIR  = os.path.join(BASE_PATH, "ICPR02", "kaggle", "evaluation")
OUT_DIR   = "/kaggle/working"

# the 12 Sentinel-2 spectral bands we have per sample
BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6",
              "B7", "B8", "B8A", "B9", "B11", "B12"]
NUM_BANDS  = len(BAND_NAMES)
IMG_SIZE   = 64     # default for most backbones; swin uses 224 separately

# disease categories for Task 2
CLASSES   = ["Aphid", "Blast", "RPH", "Rust"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}

print(f"Train dir : {TRAIN_DIR}")
print(f"Eval dir  : {EVAL_DIR}")
print(f"Classes   : {CLASSES}")


# ─── Cell 3: Gather labeled and test file paths ─────────────────────────

def collect_training_samples():
    """Walk through class folders and collect (path, label) pairs."""
    paths, labels = [], []
    for cls_name in CLASSES:
        folder = os.path.join(TRAIN_DIR, cls_name)
        if not os.path.isdir(folder):
            print(f"  [warn] folder not found: {folder}")
            continue
        for item in sorted(os.listdir(folder)):
            paths.append(os.path.join(folder, item))
            labels.append(CLASS2IDX[cls_name])
    labels = np.array(labels)

    print(f"\nTraining data: {len(paths)} samples total")
    for c in CLASSES:
        n = (labels == CLASS2IDX[c]).sum()
        print(f"  {c:>8s}: {n:4d} samples")
    return paths, labels


def collect_test_samples():
    """List everything in the evaluation folder."""
    paths, ids = [], []
    if not os.path.isdir(EVAL_DIR):
        print(f"  [warn] eval dir missing: {EVAL_DIR}")
        return paths, ids
    for item in sorted(os.listdir(EVAL_DIR)):
        paths.append(os.path.join(EVAL_DIR, item))
        ids.append(item)
    print(f"\nTest data: {len(paths)} samples")
    return paths, ids


train_paths, train_labels = collect_training_samples()
test_paths, test_ids = collect_test_samples()


# ─── Cell 4: Dataset class ──────────────────────────────────────────────

class SentinelCropDataset(Dataset):

    def __init__(self, file_paths, labels=None, img_size=IMG_SIZE, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def _read_bands(self, path):
        bands = []

        if os.path.isdir(path):
            for band_name in BAND_NAMES:
                tif_path = os.path.join(path, f"{band_name}.tif")
                if os.path.exists(tif_path):
                    with rasterio.open(tif_path) as src:
                        bands.append(src.read(1).astype(np.float32))
                else:
                    bands.append(None)
        else:
            fpath = path if os.path.isfile(path) else path + ".tif"
            if os.path.exists(fpath):
                with rasterio.open(fpath) as src:
                    data = src.read().astype(np.float32)
                    for i in range(data.shape[0]):
                        bands.append(data[i])
            while len(bands) < NUM_BANDS:
                bands.append(None)

        return bands[:NUM_BANDS]

    def _preprocess(self, bands):
        """Resize + percentile-clip normalize each band."""
        processed = []
        for b in bands:
            if b is not None:
                t = torch.from_numpy(b).unsqueeze(0).unsqueeze(0)
                t = F.interpolate(t, size=(self.img_size, self.img_size),
                                  mode="bilinear", align_corners=False)
                processed.append(t.squeeze())
            else:
                processed.append(torch.zeros(self.img_size, self.img_size))

        img = torch.stack(processed, dim=0)  # shape: (12, H, W)

        # percentile clipping — more robust than simple min-max because
        # satellite imagery often has extreme outlier pixels
        for ch in range(img.shape[0]):
            band = img[ch]
            valid_pixels = band[band > 0]
            if len(valid_pixels) > 0:
                lo = torch.quantile(valid_pixels, 0.02)
                hi = torch.quantile(valid_pixels, 0.98)
                if hi > lo:
                    img[ch] = torch.clamp((band - lo) / (hi - lo), 0.0, 1.0)
                else:
                    img[ch] = torch.zeros_like(band)
            else:
                img[ch] = torch.zeros_like(band)

        return img

    def __getitem__(self, idx):
        bands = self._read_bands(self.file_paths[idx])
        img = self._preprocess(bands)

        # data augmentation — flips and rotations are "free" transforms
        # for overhead satellite imagery (no fixed orientation)
        if self.augment:
            if torch.rand(1) > 0.5:
                img = torch.flip(img, [2])     # horizontal flip
            if torch.rand(1) > 0.5:
                img = torch.flip(img, [1])     # vertical flip
            k = torch.randint(0, 4, (1,)).item()
            img = torch.rot90(img, k, [1, 2])  # random 90° rotation

        if self.labels is not None:
            return img, self.labels[idx]
        return img


# quick sanity check
print("\n--- sanity check ---")
_test_ds = SentinelCropDataset(train_paths[:1])
_sample = _test_ds[0]
print(f"shape: {_sample.shape}  dtype: {_sample.dtype}")
for i, bn in enumerate(BAND_NAMES):
    b = _sample[i]
    print(f"  {bn:4s} | min={b.min():.3f}  max={b.max():.3f}  mean={b.mean():.3f}  nonzero={(b>0).sum().item()}")
del _test_ds, _sample
print("looks good!\n")


# ─── Cell 5: Model definition ───────────────────────────────────────────

class DiseaseClassifier(nn.Module):
    def __init__(self, backbone_name="swin_tiny_patch4_window7_224",
                 num_classes=4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=NUM_BANDS
        )

    def forward(self, x):
        return self.backbone(x)


# verify model works
_m = DiseaseClassifier(pretrained=False)
_x = torch.randn(2, NUM_BANDS, 224, 224)
_y = _m(_x)
print(f"model check: input {_x.shape} -> output {_y.shape}")
del _m, _x, _y


# ─── Cell 6: Backbone comparison (quick 3-fold scan) ────────────────────
# We tried 4 different pretrained backbones to see which one works best
# for our specific satellite data.
#
# UNCOMMENT the block below to re-run the full comparison (~1 hour).
# For reproducibility, we skip it and use the known winner directly.

# BACKBONES = {
#     "resnet50":        {"img_size": 64,  "bs": 32, "lr": 3e-4},
#     "efficientnet_b2": {"img_size": 64,  "bs": 32, "lr": 3e-4},
#     "convnext_tiny":   {"img_size": 64,  "bs": 16, "lr": 1e-4},
#     "swin_tiny_patch4_window7_224": {"img_size": 224, "bs": 8, "lr": 1e-4},
# }
#
# SCAN_EPOCHS = 20
# SCAN_FOLDS  = 3
# scan_results = {}
#
# for backbone, cfg in BACKBONES.items():
#     print(f"\n{'='*60}")
#     print(f"  Trying: {backbone}")
#     print(f"  img={cfg['img_size']}px  batch={cfg['bs']}  lr={cfg['lr']}")
#     print(f"{'='*60}")
#
#     kfold = StratifiedKFold(n_splits=SCAN_FOLDS, shuffle=True, random_state=42)
#     fold_accs, fold_f1s = [], []
#     t0 = time.time()
#
#     for fold_idx, (tr_idx, va_idx) in enumerate(kfold.split(train_paths, train_labels)):
#         print(f"\n  fold {fold_idx + 1}/{SCAN_FOLDS}")
#
#         tr_p = [train_paths[i] for i in tr_idx]
#         va_p = [train_paths[i] for i in va_idx]
#         tr_l, va_l = train_labels[tr_idx], train_labels[va_idx]
#
#         tr_loader = DataLoader(
#             SentinelCropDataset(tr_p, tr_l, img_size=cfg["img_size"], augment=True),
#             batch_size=cfg["bs"], shuffle=True, num_workers=2, pin_memory=True)
#         va_loader = DataLoader(
#             SentinelCropDataset(va_p, va_l, img_size=cfg["img_size"]),
#             batch_size=cfg["bs"], num_workers=2, pin_memory=True)
#
#         try:
#             model = DiseaseClassifier(backbone, pretrained=True).to(DEVICE)
#         except Exception as e:
#             print(f"  failed: {e}"); break
#
#         counts = np.bincount(tr_l, minlength=4).astype(np.float32)
#         w = torch.tensor(counts.sum() / (4 * counts + 1e-6)).to(DEVICE)
#         loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
#
#         optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
#         warmup = 3
#         def make_lr_schedule(epoch):
#             if epoch < warmup:
#                 return (epoch + 1) / warmup
#             return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (SCAN_EPOCHS - warmup)))
#         scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_schedule)
#
#         best_acc, best_f1 = 0, 0
#         for ep in range(SCAN_EPOCHS):
#             model.train()
#             running_loss = 0
#             for imgs, lbls in tr_loader:
#                 imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
#                 loss = loss_fn(model(imgs), lbls)
#                 optimizer.zero_grad(); loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step(); running_loss += loss.item()
#             scheduler.step()
#
#             model.eval()
#             preds, trues = [], []
#             with torch.no_grad():
#                 for imgs, lbls in va_loader:
#                     preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
#                     trues.extend(lbls.numpy())
#             acc = accuracy_score(trues, preds)
#             f1 = f1_score(trues, preds, average="macro")
#             if acc > best_acc: best_acc, best_f1 = acc, f1
#             if (ep+1) % 5 == 0:
#                 print(f"    ep {ep+1}/{SCAN_EPOCHS}  loss={running_loss/len(tr_loader):.4f}  "
#                       f"acc={acc:.4f}  f1={f1:.4f}")
#
#         fold_accs.append(best_acc); fold_f1s.append(best_f1)
#         print(f"  fold {fold_idx+1} done | acc={best_acc:.4f}  f1={best_f1:.4f}")
#         del model; torch.cuda.empty_cache()
#
#     elapsed = time.time() - t0
#     if fold_accs:
#         scan_results[backbone] = {
#             "mean_acc": np.mean(fold_accs), "std_acc": np.std(fold_accs),
#             "mean_f1": np.mean(fold_f1s), "time_min": elapsed / 60, "cfg": cfg,
#         }
#
# # print comparison
# print(f"\n{'='*65}")
# print(f"  BACKBONE COMPARISON  ({SCAN_FOLDS}-fold, {SCAN_EPOCHS} epochs)")
# print(f"{'='*65}")
# winner, top_f1 = None, 0
# for name, res in sorted(scan_results.items(), key=lambda x: -x[1]["mean_f1"]):
#     print(f"  {name:<33s}  acc={res['mean_acc']:.4f}  f1={res['mean_f1']:.4f}")
#     if res["mean_f1"] > top_f1: top_f1, winner = res["mean_f1"], name
# print(f"\n  >>> Best: {winner}")

# Results from the comparison (already run):
#   swin_tiny_patch4_window7_224     0.8933 acc   0.7820 F1
#   convnext_tiny                    0.8300 acc   0.6766 F1
#   resnet50                         0.5178 acc   0.4565 F1
#   efficientnet_b2                  0.5000 acc   0.3878 F1

winner = "swin_tiny_patch4_window7_224"
scan_results = {
    winner: {"cfg": {"img_size": 224, "bs": 8, "lr": 1e-4}}
}
print(f"Selected backbone: {winner}")


# ─── Cell 7: Full training with the winning backbone (5-fold CV) ────────

BEST_BACKBONE = winner
BEST_CFG      = scan_results[BEST_BACKBONE]["cfg"]
FULL_EPOCHS   = 40
N_FOLDS       = 5

print(f"\nFull training with {BEST_BACKBONE}")
print(f"  img_size={BEST_CFG['img_size']}  batch={BEST_CFG['bs']}  lr={BEST_CFG['lr']}")
print(f"  {N_FOLDS} folds x {FULL_EPOCHS} epochs\n")

kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
cv_scores = []
oof_preds, oof_true = [], []

for fold_idx, (tr_idx, va_idx) in enumerate(kfold.split(train_paths, train_labels)):
    print(f"\n{'='*50}")
    print(f"  FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"{'='*50}")

    tr_p = [train_paths[i] for i in tr_idx]
    va_p = [train_paths[i] for i in va_idx]
    tr_l, va_l = train_labels[tr_idx], train_labels[va_idx]

    tr_loader = DataLoader(
        SentinelCropDataset(tr_p, tr_l, img_size=BEST_CFG["img_size"], augment=True),
        batch_size=BEST_CFG["bs"], shuffle=True, num_workers=2, pin_memory=True
    )
    va_loader = DataLoader(
        SentinelCropDataset(va_p, va_l, img_size=BEST_CFG["img_size"]),
        batch_size=BEST_CFG["bs"], num_workers=2, pin_memory=True
    )

    model = DiseaseClassifier(BEST_BACKBONE, pretrained=True).to(DEVICE)

    # weighted loss for class imbalance
    counts = np.bincount(tr_l, minlength=4).astype(np.float32)
    w = torch.tensor(counts.sum() / (4 * counts + 1e-6)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=BEST_CFG["lr"], weight_decay=1e-4)

    # warmup + cosine schedule
    warmup_ep = 5
    def lr_lambda(epoch):
        if epoch < warmup_ep:
            return (epoch + 1) / warmup_ep
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_ep) / (FULL_EPOCHS - warmup_ep)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0
    save_path = os.path.join(OUT_DIR, f"best_fold{fold_idx}.pth")

    for ep in range(FULL_EPOCHS):
        model.train()
        train_loss = 0
        for imgs, lbls in tr_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            loss = loss_fn(model(imgs), lbls)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # evaluate
        model.eval()
        preds_list, true_list = [], []
        with torch.no_grad():
            for imgs, lbls in va_loader:
                preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
                preds_list.extend(preds)
                true_list.extend(lbls.numpy())

        acc = accuracy_score(true_list, preds_list)
        f1 = f1_score(true_list, preds_list, average="macro")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1:2d}/{FULL_EPOCHS}  loss={train_loss/len(tr_loader):.4f}  "
                  f"acc={acc:.4f}  f1={f1:.4f}  best={best_acc:.4f}")

    cv_scores.append(best_acc)
    oof_preds.extend(preds_list)
    oof_true.extend(true_list)

    print(f"\n  fold {fold_idx+1} best accuracy: {best_acc:.4f}")
    print(classification_report(true_list, preds_list, target_names=CLASSES))
    del model
    torch.cuda.empty_cache()

# summary
print(f"\n{'='*50}")
print(f"  CV RESULTS ({BEST_BACKBONE})")
print(f"{'='*50}")
for i, s in enumerate(cv_scores):
    print(f"  fold {i+1}: {s:.4f}")
mean_cv = np.mean(cv_scores)
std_cv  = np.std(cv_scores)
print(f"  mean:  {mean_cv:.4f} +/- {std_cv:.4f}")
overall_f1 = f1_score(oof_true, oof_preds, average="macro")
print(f"  overall macro F1: {overall_f1:.4f}")


# ─── Cell 8: Confusion matrix ───────────────────────────────────────────

cm = confusion_matrix(oof_true, oof_preds)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title(f"Confusion Matrix — {BEST_BACKBONE}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.show()


# ─── Cell 9: Generate submission using ensemble of all 5 folds ──────────

print("Generating submission predictions...\n")

if len(test_paths) > 0:
    test_ds = SentinelCropDataset(test_paths, img_size=BEST_CFG["img_size"])
    fold_predictions = []

    for fold_idx in range(N_FOLDS):
        weight_path = os.path.join(OUT_DIR, f"best_fold{fold_idx}.pth")
        if not os.path.exists(weight_path):
            print(f"  [warn] missing weights for fold {fold_idx}")
            continue

        model = DiseaseClassifier(BEST_BACKBONE, pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.eval()

        preds = []
        with torch.no_grad():
            loader = DataLoader(test_ds, batch_size=BEST_CFG["bs"], num_workers=2)
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                # softmax probabilities for smoother ensemble
                probs = F.softmax(model(batch.to(DEVICE)), dim=1)
                preds.append(probs.cpu().numpy())

        fold_predictions.append(np.concatenate(preds, axis=0))
        del model
        torch.cuda.empty_cache()
        print(f"  fold {fold_idx + 1} inference done")

    # average probabilities and take argmax
    ensemble_probs = np.mean(fold_predictions, axis=0)
    predicted_classes = ensemble_probs.argmax(axis=1)

    submission = pd.DataFrame({
        "Id": test_ids,
        "Category": [IDX2CLASS[c] for c in predicted_classes]
    })
    sub_path = os.path.join(OUT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)

    print(f"\nSaved {sub_path}  ({len(submission)} predictions)")
    print(f"\nPrediction distribution:")
    print(submission["Category"].value_counts().to_string())
    print(f"\nFirst few predictions:")
    print(submission.head(10).to_string(index=False))

    # confidence stats
    max_conf = ensemble_probs.max(axis=1)
    print(f"\nEnsemble confidence: mean={max_conf.mean():.3f}  "
          f"min={max_conf.min():.3f}  max={max_conf.max():.3f}")
else:
    print("No test data found — skipping submission generation.")

print("\nAll done!")

"""Train custom CRNN OCR model for hand-drawn circuit text recognition.

Architecture: CNN feature extractor → BiLSTM → CTC loss
Only needs to recognize ~30 characters (digits + circuit unit symbols).
11,642 training crops from Digitize-HCD dataset.
"""

import sys, os, json, random, time
from pathlib import Path
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/cghd_text")
BATCH_SIZE = 64
IMG_H = 32          # resize all crops to this height
IMG_W = 128         # max width after resize
EPOCHS = 30
LR = 0.0005  # lower LR for continued training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 1
RESUME = False  # train from scratch on CGHD text data

# ---------------------------------------------------------------------------
# Character mapping
# ---------------------------------------------------------------------------
# Circuit-notation characters we need to recognize
CHARS = "0123456789.kKmMΩμunpFV AHz-+"
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = blank (CTC)
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
IDX_TO_CHAR[0] = ""  # blank token
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank

print(f"Characters: {len(CHARS)} -> '{CHARS}'")
print(f"Classes: {NUM_CLASSES} (including CTC blank)")
print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CircuitTextDataset(Dataset):
    def __init__(self, label_file: Path, augment: bool = False):
        self.samples = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "\t" not in line:
                    continue
                path, text = line.split("\t", 1)
                if os.path.isfile(path):
                    self.samples.append((path, text))
        self.augment = augment
        print(f"  Loaded {len(self.samples)} samples from {label_file.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

        # Resize preserving aspect ratio
        h, w = img.shape
        scale = IMG_H / max(h, 1)
        new_w = max(int(w * scale), 4)
        img = cv2.resize(img, (new_w, IMG_H))

        # Pad to IMG_W
        if new_w > IMG_W:
            img = img[:, :IMG_W]
        else:
            pad = np.zeros((IMG_H, IMG_W - new_w), dtype=np.uint8)
            img = np.hstack([img, pad])

        # Normalize + invert (dark text on light bg → light text on dark bg for CNN)
        img = 255 - img  # invert
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # normalize to [-1, 1]

        # Data augmentation
        if self.augment:
            if random.random() < 0.3:
                # slight brightness jitter
                img = img * (0.8 + 0.4 * random.random())
            if random.random() < 0.2:
                # slight noise
                img += np.random.normal(0, 0.05, img.shape)

        img = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)

        # Encode text to indices
        label = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
        if not label:
            label = [0]
        return img, torch.LongTensor(label), len(label)


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images, 0)
    # Concatenate all labels for CTC
    labels_cat = torch.cat(labels, 0)
    label_lengths = torch.LongTensor(lengths)
    return images, labels_cat, label_lengths


# ---------------------------------------------------------------------------
# Model: CRNN (CNN + BiLSTM)
# ---------------------------------------------------------------------------
class CRNN(nn.Module):
    def __init__(self, num_classes, input_h=IMG_H):
        super().__init__()
        self.num_classes = num_classes

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),                                    # 64 x H/2 x W/2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),                                    # 128 x H/4 x W/4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),                         # 256 x H/8 x W/4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),                         # 512 x H/16 x W/4
            nn.AdaptiveAvgPool2d((1, None)),                       # collapse H to 1
        )

        # CNN output H=1 after AdaptiveAvgPool2d
        self.cnn_out_h = 1
        self.cnn_out_c = 512 * self.cnn_out_h

        # BiLSTM sequence modeling
        self.rnn = nn.LSTM(self.cnn_out_c, 256, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(512, num_classes)  # 256*2 (bidirectional)

    def forward(self, x):
        # x: (B, 1, H, W)
        features = self.cnn(x)  # (B, 512, H/16, W/4)
        # Reshape: (B, C*H, W) → (B, W, C*H)
        B, C, H, W = features.shape
        features = features.reshape(B, C * H, W).permute(0, 2, 1)  # (B, W, C*H)
        rnn_out, _ = self.rnn(features)  # (B, W, 512)
        logits = self.fc(rnn_out)  # (B, W, num_classes)
        return logits  # (B, T, C) for CTC loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    train_ds = CircuitTextDataset(DATA_DIR / "train_labels.txt", augment=True)
    val_ds = CircuitTextDataset(DATA_DIR / "val_labels.txt", augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    model = CRNN(NUM_CLASSES).to(DEVICE)
    start_epoch = 0
    best_val_loss = float("inf")

    model_path = Path("runs/ocr_crnn/best.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    if RESUME and model_path.exists():
        try:
            ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model"])
            best_val_loss = ckpt.get("val_loss", float("inf"))
            start_epoch = ckpt.get("epoch", 0)
            print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        except Exception as e:
            print(f"Could not resume: {e}, starting fresh")

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n{'='*60}")
    print(f"  Training CRNN OCR Model")
    print(f"  Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
    print(f"  Epochs: {EPOCHS} (from epoch {start_epoch})  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch + 1, start_epoch + EPOCHS + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        start = time.time()
        for images, labels, label_lengths in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)  # (B, T, C)
            # CTC expects (T, B, C)
            logits_ctc = logits.permute(1, 0, 2)  # (T, B, C)
            input_lengths = torch.full((images.size(0),), logits_ctc.size(0),
                                       dtype=torch.long, device=DEVICE)

            loss = ctc_loss(logits_ctc, labels, input_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        scheduler.step()

        # --- Val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, label_lengths in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                label_lengths = label_lengths.to(DEVICE)
                logits = model(images)
                logits_ctc = logits.permute(1, 0, 2)
                input_lengths = torch.full((images.size(0),), logits_ctc.size(0),
                                           dtype=torch.long, device=DEVICE)
                loss = ctc_loss(logits_ctc, labels, input_lengths, label_lengths)
                val_loss += loss.item()
        avg_val_loss = val_loss / max(len(val_loader), 1)

        elapsed = time.time() - start
        improved = "*" if avg_val_loss < best_val_loss else " "
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({"model": model.state_dict(), "chars": CHARS, "img_h": IMG_H,
                        "val_loss": best_val_loss, "epoch": epoch},
                       model_path)

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"  Epoch {epoch:2d}/{EPOCHS} | "
                  f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f} | {elapsed:.0f}s {improved}", flush=True)

    print(f"\nBest val loss: {best_val_loss:.4f} (final epoch: {start_epoch + EPOCHS})")
    print(f"Model saved: {model_path}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def load_trained_model(model_path="runs/ocr_crnn/best.pt"):
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    chars = ckpt["chars"]
    img_h = ckpt.get("img_h", IMG_H)
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    idx_to_char[0] = ""

    model = CRNN(len(chars) + 1, img_h).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, chars, char_to_idx, idx_to_char, img_h


def predict(model, img_gray, chars, idx_to_char, img_h=IMG_H):
    """Recognize text from a grayscale image crop."""
    if img_gray is None or img_gray.size == 0:
        return ""

    # Preprocess
    h, w = img_gray.shape
    scale = img_h / max(h, 1)
    new_w = max(int(w * scale), 4)
    img = cv2.resize(img_gray, (new_w, img_h))
    if new_w > IMG_W:
        img = img[:, :IMG_W]
    else:
        pad = np.zeros((img_h, IMG_W - new_w), dtype=np.uint8)
        img = np.hstack([img, pad])

    img = 255 - img
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)  # (1, T, C)
        logits = logits.squeeze(0)  # (T, C)
        preds = logits.argmax(dim=1).cpu().numpy()

    # CTC greedy decode
    result = []
    prev = 0
    for p in preds:
        if p != prev and p != 0:
            result.append(idx_to_char.get(p, ""))
        prev = p
    return "".join(result)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "--train" in sys.argv:
        train()
    elif "--test" in sys.argv:
        # Quick test on a few validation samples
        model, chars, c2i, i2c, img_h = load_trained_model()
        val_file = DATA_DIR / "val_labels.txt"
        if val_file.exists():
            samples = []
            with open(val_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "\t" in line:
                        samples.append(line.strip().split("\t", 1))
            random.shuffle(samples)
            correct = 0
            for path, gt in samples[:20]:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                pred = predict(model, img, chars, i2c, img_h)
                ok = "OK" if pred == gt else f"FAIL (gt={gt})"
                print(f"  pred='{pred}'  {ok}")
                if pred == gt:
                    correct += 1
            print(f"\nAccuracy: {correct}/20")
    else:
        print("Usage: python train_ocr.py --train  |  --test")
        print(f"Data ready: {DATA_DIR}")
        print(f"Train: 10,477  Val: 1,165")

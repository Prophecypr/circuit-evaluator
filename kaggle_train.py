"""Standalone CRNN OCR training script for Kaggle GPU.

Upload cghd_text_data.zip to Kaggle first, then run this notebook/script.
"""
import os, random, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Config
# ============================================================
DATA_DIR  = "/kaggle/input/datasets/prophecypr/cghd-text/cghd_text"
BATCH     = 128
IMG_H     = 32
IMG_W     = 128
EPOCHS    = 40
LR        = 0.001
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_OUT = "/kaggle/working/best.pt"

CHARS = "0123456789.kKmMΩμnpFV AHz-+/"
CHAR_TO_IDX = {c: i+1 for i,c in enumerate(CHARS)}
IDX_TO_CHAR = {i+1:c for i,c in enumerate(CHARS)}
IDX_TO_CHAR[0] = ""
NUM_CLASSES = len(CHARS) + 1

print(f"Device: {DEVICE}")
print(f"Chars: {len(CHARS)}  Classes: {NUM_CLASSES}")

# ============================================================
# Dataset
# ============================================================
class TextDataset(Dataset):
    def __init__(self, label_file, data_dir, augment=False):
        self.samples = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "\t" not in line: continue
                p, t = line.split("\t", 1)
                # Fix path: label file has local paths, resolve to Kaggle data_dir
                fname = os.path.basename(p.replace("\\", "/"))
                full_path = os.path.join(data_dir, fname)
                t = "".join(c for c in t if c in CHAR_TO_IDX)
                if t and os.path.isfile(full_path):
                    self.samples.append((full_path, t))
        self.augment = augment
        print(f"  {len(self.samples)} samples from {os.path.basename(label_file)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

        h, w = img.shape
        scale = IMG_H / max(h, 1)
        new_w = max(int(w * scale), 4)
        img = cv2.resize(img, (new_w, IMG_H))

        if new_w > IMG_W:
            img = img[:, :IMG_W]

        # Pad to IMG_W
        canvas = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        canvas[:, :img.shape[1]] = img

        # Normalize
        img = canvas.astype(np.float32) / 255.0

        # Encode label
        label = [CHAR_TO_IDX[c] for c in text]
        return torch.tensor(img).unsqueeze(0), torch.tensor(label, dtype=torch.long), len(label)

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels = torch.cat(labels)
    input_lengths = torch.tensor([images.shape[3] // 4] * len(images), dtype=torch.long)
    target_lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, input_lengths, target_lengths

# ============================================================
# Model (CNN -> BiLSTM -> CTC)
# ============================================================
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.AdaptiveAvgPool2d((1, None)),  # collapse height to 1
        )
        self.rnn = nn.LSTM(512, 256, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)          # (B, 512, H', W')
        x = x.squeeze(2)         # (B, 512, W')
        x = x.permute(0, 2, 1)   # (B, W', 512)
        x, _ = self.rnn(x)       # (B, W', 512)
        x = self.fc(x)           # (B, W', NC)
        return F.log_softmax(x, dim=2).permute(1, 0, 2)  # (W', B, NC) for CTC

# ============================================================
# Train
# ============================================================
def train():
    train_ds = TextDataset(os.path.join(DATA_DIR, "train_labels.txt"), DATA_DIR)
    val_ds   = TextDataset(os.path.join(DATA_DIR, "val_labels.txt"), DATA_DIR)
    train_loader = DataLoader(train_ds, BATCH, shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   BATCH, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = CRNN(NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    best_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"  Epochs: {EPOCHS}  Batch: {BATCH}  LR: {LR}")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for batch_idx, (images, labels, input_lengths, target_lengths) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = ctc_loss(output, labels, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"\r  Epoch {epoch+1:2d}/{EPOCHS} | batch {batch_idx:4d} | loss={loss.item():.4f}", end="")

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, input_lengths, target_lengths in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                output = model(images)
                val_loss += ctc_loss(output, labels, input_lengths, target_lengths).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - t0
        print(f"\r  Epoch {epoch+1:2d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.6f} | {elapsed:.0f}s  ")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model": model.state_dict(), "chars": CHARS, "val_loss": val_loss, "img_h": IMG_H}, MODEL_OUT)
            print(f"  -> Best model saved (val_loss={val_loss:.4f})")

    # Test
    print(f"\nBest val_loss: {best_loss:.4f}")
    print("Testing 10 samples:")
    ckpt = torch.load(MODEL_OUT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    correct = 0
    samples = random.sample(val_ds.samples, min(10, len(val_ds.samples)))
    for path, gt in samples:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        h, w = img.shape
        scale = IMG_H / max(h, 1)
        new_w = max(int(w*scale), 4)
        img = cv2.resize(img, (min(new_w, IMG_W), IMG_H))
        canvas = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        canvas[:, :img.shape[1]] = img
        tensor = torch.tensor(canvas.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(tensor)
            _, preds = out.max(2)
            preds = preds.squeeze(1).cpu().numpy()
        result = []
        prev = 0
        for p in preds:
            if p != prev and p != 0:
                result.append(IDX_TO_CHAR.get(p, ""))
            prev = p
        pred = "".join(result)
        ok = "OK" if pred == gt else f"FAIL (gt={gt})"
        print(f"  pred='{pred}'  {ok}")
        if pred == gt: correct += 1
    print(f"\nAccuracy: {correct}/10")

if __name__ == "__main__":
    train()

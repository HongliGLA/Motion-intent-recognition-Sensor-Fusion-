# ============================================================
# sanity_check.py
# DB6 Same-session Sanity Check
#
# Purpose:
#   Only use S1_D1_T1_200Hz_merged.csv
#   Randomly split windows into train/val
#
# If this accuracy is high:
#   preprocessing + labels + model are basically correct
#
# If this accuracy is still low:
#   there may be preprocessing / label / model issue
# ============================================================

import os
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import butter, filtfilt


# ============================================================
# 1. Config
# ============================================================

DATA_PATH = "../data/DB6/S1_D1_T1_200Hz_merged.csv"

FS = 200

LOWCUT = 20
HIGHCUT = 90
FILTER_ORDER = 4
ENVELOPE_CUTOFF = 5

WINDOW_SIZE = 40
STRIDE = 10

BATCH_SIZE = 128
EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.3

SEED = 42


# ============================================================
# 2. Seed
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. Load merged CSV
# ============================================================

def load_numeric_csv(path):
    df = pd.read_csv(path, header=None, low_memory=False)
    df = df.replace(["-", " ", ""], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)
    return df.values.astype(np.float32)


def load_data():
    data = load_numeric_csv(DATA_PATH)

    emg = data[:, 0:16].astype(np.float32)
    acc = data[:, 16:64].astype(np.float32)
    labels = data[:, 64].astype(np.int64)

    print("========== RAW SESSION INFO ==========")
    print("Data shape:", data.shape)
    print("EMG shape:", emg.shape)
    print("ACC shape:", acc.shape)
    print("Labels:", np.unique(labels))

    return emg, acc, labels


# ============================================================
# 4. EMG preprocessing
# ============================================================

def bandpass_filter_emg(emg):
    nyquist = FS / 2.0
    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist

    b, a = butter(
        FILTER_ORDER,
        [low, high],
        btype="bandpass"
    )

    return filtfilt(b, a, emg, axis=0).astype(np.float32)


def rectify_emg(emg):
    return np.abs(emg).astype(np.float32)


def extract_envelope(emg):
    nyquist = FS / 2.0
    cutoff = ENVELOPE_CUTOFF / nyquist

    b, a = butter(
        FILTER_ORDER,
        cutoff,
        btype="lowpass"
    )

    return filtfilt(b, a, emg, axis=0).astype(np.float32)


def preprocess_emg(emg):
    emg = bandpass_filter_emg(emg)
    emg = rectify_emg(emg)
    emg = extract_envelope(emg)
    return emg


# ============================================================
# 5. Remove empty ACC columns
# ============================================================

def remove_empty_acc_columns(acc):
    std = acc.std(axis=0)
    valid_cols = np.where(std > 1e-8)[0]

    print("\n========== ACC CHECK ==========")
    print("Original ACC columns:", acc.shape[1])
    print("Valid ACC columns:", len(valid_cols))
    print("Removed columns:", acc.shape[1] - len(valid_cols))

    return acc[:, valid_cols], valid_cols


# ============================================================
# 6. Create windows
# ============================================================

def create_windows(x, labels):
    X = []
    y = []

    for start in range(0, len(labels) - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE

        window_labels = labels[start:end]

        if np.all(window_labels == window_labels[0]):
            X.append(x[start:end, :])
            y.append(window_labels[0])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y


def remap_labels(y):
    original_labels = sorted(np.unique(y))

    label_to_new = {
        old_label: new_label
        for new_label, old_label in enumerate(original_labels)
    }

    y_new = np.array([label_to_new[x] for x in y], dtype=np.int64)

    print("\n========== LABEL MAPPING ==========")
    print("Original labels:", original_labels)
    print("Label to new:", label_to_new)
    print("Number of classes:", len(original_labels))

    return y_new, label_to_new


# ============================================================
# 7. Dataset
# ============================================================

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 8. Model
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            ) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SensorTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        window_size=40,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.3
    ):
        super().__init__()

        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=window_size
        )

        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.feature_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.pos_encoding(x)
        x = self.input_norm(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.feature_norm(x)
        logits = self.classifier(x)
        return logits


# ============================================================
# 9. Train / Eval
# ============================================================

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            batch_size = X.size(0)

            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    acc = 100.0 * total_correct / total_samples

    return avg_loss, acc


def run_sanity(mode="emg"):
    set_seed(SEED)

    emg, acc, labels = load_data()

    # remove rest
    mask = labels != 0
    emg = emg[mask]
    acc = acc[mask]
    labels = labels[mask]

    print("\n========== AFTER REMOVE REST ==========")
    print("EMG:", emg.shape)
    print("ACC:", acc.shape)
    print("Labels:", np.unique(labels))

    # preprocess
    emg = preprocess_emg(emg)
    acc, valid_cols = remove_empty_acc_columns(acc)

    if mode == "emg":
        x = emg
    elif mode == "acc":
        x = acc
    elif mode == "fusion":
        x = np.concatenate([emg, acc], axis=1)
    else:
        raise ValueError("mode must be emg, acc, or fusion")

    # normalize using the same session only
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    x = ((x - mean) / std).astype(np.float32)

    X, y = create_windows(x, labels)
    y, label_to_new = remap_labels(y)

    print("\n========== WINDOW DATA ==========")
    print("Mode:", mode)
    print("X:", X.shape)
    print("y:", y.shape)

    dataset = WindowDataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    input_dim = X.shape[2]
    num_classes = len(label_to_new)

    model = SensorTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        window_size=WINDOW_SIZE,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_acc = 0.0

    print("\n========== START SANITY TRAINING ==========")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)

            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += batch_size

        train_loss = total_loss / total_samples
        train_acc = 100.0 * total_correct / total_samples

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Best Val: {best_val_acc:.2f}%"
        )

    print("\n========== SANITY CHECK RESULT ==========")
    print("Mode:", mode)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="emg",
        choices=["emg", "acc", "fusion"]
    )

    args = parser.parse_args()

    run_sanity(mode=args.mode)
# ============================================================
# train.py
# Train DB6 EMG / ACC / Fusion Transformer
#
# Usage:
#   python train.py --mode emg
#   python train.py --mode acc
#   python train.py --mode fusion
# ============================================================

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from model import SensorTransformer


# ============================================================
# 1. Settings
# ============================================================

DATA_PATH = "../processed/db6_s3_emg_acc_200hz.npz"

RESULT_DIR = "../results/S3/aug"

SEED = 42

BATCH_SIZE = 128
EPOCHS = 80
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3

D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.3

EARLY_STOPPING_PATIENCE = 12


# ============================================================
# 2. Seed
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. Dataset
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
# 4. Load data by mode
# ============================================================

def load_data(mode):
    data = np.load(DATA_PATH)

    if mode == "emg":
        X_train = data["X_emg_train"]
        X_val = data["X_emg_val"]
        X_test = data["X_emg_test"]

    elif mode == "acc":
        X_train = data["X_acc_train"]
        X_val = data["X_acc_val"]
        X_test = data["X_acc_test"]

    elif mode == "fusion":
        X_train = data["X_fusion_train"]
        X_val = data["X_fusion_val"]
        X_test = data["X_fusion_test"]

    else:
        raise ValueError("mode must be one of: emg, acc, fusion")

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    num_classes = int(data["num_classes"])

    print("========== LOADED DATA ==========")
    print("Mode:", mode)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    print("Number of classes:", num_classes)

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes


# ============================================================
# 5. Evaluation
# ============================================================

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

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

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total_samples
    acc = 100.0 * total_correct / total_samples

    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


# ============================================================
# 6. Plot training curves
# ============================================================

def plot_curves(history, save_path, mode):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{mode.upper()} Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Acc")
    plt.plot(epochs, history["val_acc"], marker="s", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{mode.upper()} Accuracy Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved training curves to:", save_path)


# ============================================================
# 7. Confusion Matrix
# ============================================================

def plot_confusion_matrix(y_true, y_pred, num_classes, save_path, title):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    ticks = np.arange(num_classes)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    threshold = cm.max() / 2.0

    for i in range(num_classes):
        for j in range(num_classes):
            value = cm[i, j]
            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved confusion matrix to:", save_path)

    return cm


# ============================================================
# 8. Per-class Accuracy
# ============================================================

def compute_per_class_accuracy(cm):
    correct = np.diag(cm)
    total = cm.sum(axis=1)

    acc = []
    for i in range(len(total)):
        if total[i] == 0:
            acc.append(0.0)
        else:
            acc.append(100.0 * correct[i] / total[i])

    return np.array(acc)


def save_per_class_accuracy(per_class_acc, csv_path, fig_path, mode, split_name):
    with open(csv_path, "w") as f:
        f.write("class,accuracy\n")
        for i, acc in enumerate(per_class_acc):
            f.write(f"{i},{acc:.4f}\n")

    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(per_class_acc)), per_class_acc)
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{mode.upper()} {split_name} Per-class Accuracy")
    plt.xticks(np.arange(len(per_class_acc)))
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("Saved per-class accuracy CSV to:", csv_path)
    print("Saved per-class accuracy figure to:", fig_path)


# ============================================================
# 9. Main training
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="fusion",
        choices=["emg", "acc", "fusion"],
        help="Choose input modality: emg, acc, fusion"
    )
    args = parser.parse_args()

    mode = args.mode

    set_seed(SEED)

    os.makedirs(RESULT_DIR, exist_ok=True)

    mode_dir = os.path.join(RESULT_DIR, mode)
    os.makedirs(mode_dir, exist_ok=True)

    model_save_path = os.path.join(mode_dir, f"best_{mode}_transformer.pth")
    curve_save_path = os.path.join(mode_dir, f"{mode}_training_curves.png")

    val_cm_path = os.path.join(mode_dir, f"{mode}_val_confusion_matrix.png")
    test_cm_path = os.path.join(mode_dir, f"{mode}_test_confusion_matrix.png")

    val_pc_csv = os.path.join(mode_dir, f"{mode}_val_per_class_accuracy.csv")
    val_pc_fig = os.path.join(mode_dir, f"{mode}_val_per_class_accuracy.png")

    test_pc_csv = os.path.join(mode_dir, f"{mode}_test_per_class_accuracy.csv")
    test_pc_fig = os.path.join(mode_dir, f"{mode}_test_per_class_accuracy.png")

    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = load_data(mode)


    window_size = X_train.shape[1]
    input_dim = X_train.shape[2]

    train_dataset = WindowDataset(X_train, y_train)
    val_dataset = WindowDataset(X_val, y_val)
    test_dataset = WindowDataset(X_test, y_test)

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = SensorTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        window_size=window_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)

    print("\n========== MODEL ==========")
    print(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    print("\n========== START TRAINING ==========")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            batch_size = X.size(0)

            train_loss_sum += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += batch_size

        train_loss = train_loss_sum / train_total
        train_acc = 100.0 * train_correct / train_total

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mode": mode,
                    "input_dim": input_dim,
                    "window_size": window_size,
                    "num_classes": num_classes,
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch
                },
                model_save_path
            )

            print("Saved best model.")

        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break

    plot_curves(history, curve_save_path, mode)

    print("\n========== LOAD BEST MODEL ==========")

    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Best Epoch: {checkpoint['best_epoch']}")
    print(f"Best Val Acc: {checkpoint['best_val_acc']:.2f}%")

    # ========================================================
    # Final validation evaluation
    # ========================================================

    val_loss, val_acc, val_preds, val_labels = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    print("\n========== FINAL VALIDATION ==========")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.2f}%")

    val_cm = plot_confusion_matrix(
        y_true=val_labels,
        y_pred=val_preds,
        num_classes=num_classes,
        save_path=val_cm_path,
        title=f"{mode.upper()} Validation Confusion Matrix"
    )

    val_per_class_acc = compute_per_class_accuracy(val_cm)

    print("\nValidation Per-class Accuracy:")
    for i, acc in enumerate(val_per_class_acc):
        print(f"Class {i}: {acc:.2f}%")

    save_per_class_accuracy(
        val_per_class_acc,
        val_pc_csv,
        val_pc_fig,
        mode,
        "Validation"
    )

    # ========================================================
    # Final test evaluation
    # ========================================================

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print("\n========== FINAL TEST ==========")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.2f}%")

    test_cm = plot_confusion_matrix(
        y_true=test_labels,
        y_pred=test_preds,
        num_classes=num_classes,
        save_path=test_cm_path,
        title=f"{mode.upper()} Test Confusion Matrix"
    )

    test_per_class_acc = compute_per_class_accuracy(test_cm)

    print("\nTest Per-class Accuracy:")
    for i, acc in enumerate(test_per_class_acc):
        print(f"Class {i}: {acc:.2f}%")

    save_per_class_accuracy(
        test_per_class_acc,
        test_pc_csv,
        test_pc_fig,
        mode,
        "Test"
    )

    print("\n========== SUMMARY ==========")
    print("Mode:", mode)
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    print(f"Final Val Acc: {val_acc:.2f}%")
    print(f"Final Test Acc: {test_acc:.2f}%")
    print("Results saved to:", mode_dir)


if __name__ == "__main__":
    main()
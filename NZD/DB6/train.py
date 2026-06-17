import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

import config as cfg
from preprocessing import prepare_subject_data, resolve_many_paths
from dataset import ACCEMGWindowDataset
from model import PureTransformerClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def create_experiment_dirs():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = f"{timestamp}_{cfg.EXPERIMENT_NAME}"

    run_dir = cfg.OUTPUT_ROOT / run_name

    dirs = {
        "run": run_dir,
        "confusion_matrix": run_dir / "confusion_matrix",
        "curves": run_dir / "curves",
        "models": run_dir / "models",
        "preprocessing": run_dir / "preprocessing",
        "results": run_dir / "results",
    }

    for path in dirs.values():
        path.mkdir(
            parents=True,
            exist_ok=True
        )

    return dirs


def make_dataloaders(
    train_x,
    train_y,
    val_x,
    val_y,
    test_x,
    test_y
):
    train_dataset = ACCEMGWindowDataset(
        train_x,
        train_y
    )

    val_dataset = ACCEMGWindowDataset(
        val_x,
        val_y
    )

    test_dataset = ACCEMGWindowDataset(
        test_x,
        test_y
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(cfg.DEVICE)
        y = y.to(cfg.DEVICE)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(
            logits,
            dim=1
        )

        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    for x, y in loader:
        x = x.to(cfg.DEVICE)
        y = y.to(cfg.DEVICE)

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        probs = torch.softmax(
            logits,
            dim=1
        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        total_loss += loss.item() * x.size(0)
        correct += (preds == y).sum().item()
        total += y.size(0)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            y.cpu().numpy()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

    avg_loss = total_loss / total
    acc = correct / total

    return (
        avg_loss,
        acc,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )


def save_training_curve(
    history,
    subject_name,
    dirs
):
    history_df = pd.DataFrame(history)

    history_path = dirs["results"] / f"{subject_name}_training_history.csv"

    history_df.to_csv(
        history_path,
        index=False
    )

    epochs = history["epoch"]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Val Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{subject_name} Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)

    plt.plot(
        epochs,
        history["train_acc"],
        label="Train Acc"
    )

    plt.plot(
        epochs,
        history["val_acc"],
        label="Val Acc"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{subject_name} Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    curve_path = dirs["curves"] / f"{subject_name}_train_val_curve.png"

    plt.savefig(
        curve_path,
        dpi=300
    )

    plt.close()


def save_report_and_cm(
    y_true,
    y_pred,
    y_prob,
    subject_name,
    split_name,
    dirs
):
    labels = np.arange(cfg.NUM_CLASSES)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=cfg.CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=cfg.CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    base_name = f"{subject_name}_{split_name}"

    report_csv_path = dirs["results"] / f"{base_name}_classification_report.csv"

    pd.DataFrame(
        report_dict
    ).transpose().to_csv(
        report_csv_path
    )

    report_txt_path = dirs["results"] / f"{base_name}_classification_report.txt"

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    pred_df = pd.DataFrame(
        {
            "true": y_true,
            "pred": y_pred
        }
    )

    for i, class_name in enumerate(cfg.CLASS_NAMES):
        pred_df[f"prob_{class_name}"] = y_prob[:, i]

    pred_path = dirs["results"] / f"{base_name}_predictions.csv"

    pred_df.to_csv(
        pred_path,
        index=False
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=cfg.CLASS_NAMES,
        columns=cfg.CLASS_NAMES
    )

    cm_csv_path = dirs["confusion_matrix"] / f"{base_name}_confusion_matrix.csv"

    cm_df.to_csv(
        cm_csv_path
    )

    plt.figure(figsize=(8, 7))

    plt.imshow(cm)

    plt.title(f"{base_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(
        labels,
        cfg.CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        labels,
        cfg.CLASS_NAMES
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    cm_png_path = dirs["confusion_matrix"] / f"{base_name}_confusion_matrix.png"

    plt.savefig(
        cm_png_path,
        dpi=300
    )

    plt.close()

    return report_text


def train_one_subject(
    subject_cfg,
    dirs,
    log_lines
):
    subject_name = subject_cfg["name"]

    print("\n" + "#" * 80)
    print(f"Start subject: {subject_name}")
    print("#" * 80)

    set_seed(cfg.RANDOM_SEED)

    (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        prep_info
    ) = prepare_subject_data(
        subject_cfg,
        output_dir=dirs["preprocessing"]
    )

    train_loader, val_loader, test_loader = make_dataloaders(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y
    )

    model = PureTransformerClassifier().to(
        cfg.DEVICE
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=6
    )

    best_model_path = dirs["models"] / f"{subject_name}_best_model.pth"

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val_acc
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        msg = (
            f"{subject_name} | "
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        print(msg)

        log_lines.append(msg)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "subject": subject_name,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                    "config": {
                        "window_size": cfg.WINDOW_SIZE,
                        "stride": cfg.STRIDE,
                        "input_dim": cfg.INPUT_DIM,
                        "num_classes": cfg.NUM_CLASSES,
                        "acc_channels": cfg.ACC_CHANNELS,
                        "emg_channels": cfg.EMG_CHANNELS,
                        "d_model": cfg.D_MODEL,
                        "nhead": cfg.NHEAD,
                        "qk_feature_mode": cfg.QK_FEATURE_MODE,
                        "qk_pool_size": cfg.QK_POOL_SIZE,
                        "qk_use_softmax": cfg.QK_USE_SOFTMAX,
                        "train_augment": cfg.TRAIN_AUGMENT,
                        "train_Eaugment": cfg.TRAIN_EAUGMENT,
                        "raw_labels": cfg.RAW_LABELS,
                        "train_raw_labels": cfg.TRAIN_RAW_LABELS,
                        "signal_preprocessing": cfg.ENABLE_SIGNAL_PREPROCESSING,
                        "raw_sampling_rate": cfg.RAW_SAMPLING_RATE,
                        "emg_bandpass_low": cfg.EMG_BANDPASS_LOW,
                        "emg_bandpass_high": cfg.EMG_BANDPASS_HIGH,
                        "emg_envelope_cutoff": cfg.EMG_ENVELOPE_CUTOFF,
                        "acc_lowpass_cutoff": cfg.ACC_LOWPASS_CUTOFF,
                    },
                },
                best_model_path
            )

            print(f"{subject_name}: saved best model.")

        else:
            patience_counter += 1

        if patience_counter >= cfg.PATIENCE:
            print(f"\n{subject_name}: Early stopping triggered.")
            break

    save_training_curve(
        history,
        subject_name,
        dirs
    )

    checkpoint = torch.load(
        best_model_path,
        map_location=cfg.DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
        model,
        val_loader,
        criterion
    )

    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
        model,
        test_loader,
        criterion
    )

    val_report_text = save_report_and_cm(
        val_labels,
        val_preds,
        val_probs,
        subject_name,
        "validation",
        dirs
    )

    test_report_text = save_report_and_cm(
        test_labels,
        test_preds,
        test_probs,
        subject_name,
        "test",
        dirs
    )

    val_f1_macro = f1_score(
        val_labels,
        val_preds,
        average="macro",
        zero_division=0
    )

    test_f1_macro = f1_score(
        test_labels,
        test_preds,
        average="macro",
        zero_division=0
    )

    test_f1_weighted = f1_score(
        test_labels,
        test_preds,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print(f"{subject_name} Final Result")
    print("=" * 70)
    print(f"Best Epoch   : {best_epoch}")
    print(f"Best Val Acc : {best_val_acc * 100:.2f}%")
    print(f"Final Val Acc: {val_acc * 100:.2f}%")
    print(f"Test Acc     : {test_acc * 100:.2f}%")

    print("\nValidation Classification Report")
    print(val_report_text)

    print("\nTest Classification Report")
    print(test_report_text)

    result = {
        "subject": subject_name,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_val_loss": val_loss,
        "final_val_acc": val_acc,
        "final_val_f1_macro": val_f1_macro,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1_macro": test_f1_macro,
        "test_f1_weighted": test_f1_weighted,
        "train_windows": int(len(train_x)),
        "val_windows": int(len(val_x)),
        "test_windows": int(len(test_x)),
        "best_model_path": str(best_model_path),
    }

    return result


def subject_files_exist(subject_cfg):
    try:
        resolve_many_paths(subject_cfg["train"])
        resolve_many_paths(subject_cfg["val"])
        resolve_many_paths(subject_cfg["test"])
        return True
    except FileNotFoundError as e:
        print(e)
        return False


def main():
    dirs = create_experiment_dirs()

    log_lines = []

    header_lines = [
        "Run QK-only EMG/ACC block attention classifier with train augmentation",
        f"Run dir: {dirs['run']}",
        f"Device: {cfg.DEVICE}",
        f"Raw DB6 EMG columns: {cfg.DB6_EMG_RAW_COLS}",
        f"Raw DB6 ACC columns: {cfg.DB6_ACC_RAW_COLS}",
        f"Model input order: ACC first ({cfg.ACC_DIM}), EMG second ({cfg.EMG_DIM})",
        f"ACC channels after reorder: {cfg.ACC_CHANNELS}",
        f"EMG channels after reorder: {cfg.EMG_CHANNELS}",
        f"Window size: {cfg.WINDOW_SIZE}",
        f"Stride recorded only: {cfg.STRIDE}",
        f"Raw labels: {cfg.RAW_LABELS}",
        f"Train raw labels: {cfg.TRAIN_RAW_LABELS}",
        f"Num classes: {cfg.NUM_CLASSES}",
        f"Remove rest: {cfg.REMOVE_REST}",
        f"Signal preprocessing: {cfg.ENABLE_SIGNAL_PREPROCESSING}",
        f"EMG preprocess: mean={cfg.EMG_REMOVE_MEAN}, notch={cfg.EMG_NOTCH}, bandpass=({cfg.EMG_BANDPASS_LOW}, {cfg.EMG_BANDPASS_HIGH}), rectify={cfg.EMG_RECTIFY}, envelope={cfg.EMG_ENVELOPE}",
        f"ACC preprocess: lowpass={cfg.ACC_LOWPASS}, cutoff={cfg.ACC_LOWPASS_CUTOFF}",
        f"Save preprocessed signal plots: {cfg.SAVE_PREPROCESSED_SIGNAL_PLOTS}",
        f"Preprocess plot max points: {cfg.PREPROCESS_PLOT_MAX_POINTS}",
        f"Train augment: {cfg.TRAIN_AUGMENT}",
        f"Train Eaugment: {cfg.TRAIN_EAUGMENT}",
        f"Val/Test augment: {cfg.VAL_AUGMENT}, {cfg.TEST_AUGMENT}",
        f"QK feature mode: {cfg.QK_FEATURE_MODE}",
        f"QK use softmax: {cfg.QK_USE_SOFTMAX}",
    ]

    for line in header_lines:
        print(line)
        log_lines.append(line)

    all_results = []

    for subject_cfg in cfg.SUBJECTS:
        if not subject_files_exist(subject_cfg):
            if cfg.SKIP_MISSING_SUBJECTS:
                print(f"Skip {subject_cfg['name']} because files are missing.")
                continue
            else:
                raise FileNotFoundError(
                    f"{subject_cfg['name']} 文件不完整。"
                )

        result = train_one_subject(
            subject_cfg,
            dirs,
            log_lines
        )

        all_results.append(
            result
        )

        pd.DataFrame(all_results).to_csv(
            dirs["results"] / cfg.RESULT_CSV_NAME,
            index=False
        )

    if len(all_results) == 0:
        raise RuntimeError(
            "没有任何受试者被训练，请检查 CSV 文件路径。"
        )

    results_df = pd.DataFrame(all_results)

    results_path = dirs["results"] / cfg.RESULT_CSV_NAME

    results_df.to_csv(
        results_path,
        index=False
    )

    summary = {
        "num_subjects": int(len(results_df)),
        "mean_test_acc": float(results_df["test_acc"].mean()),
        "std_test_acc": float(results_df["test_acc"].std(ddof=0)),
        "mean_test_f1_macro": float(results_df["test_f1_macro"].mean()),
        "std_test_f1_macro": float(results_df["test_f1_macro"].std(ddof=0)),
        "mean_best_val_acc": float(results_df["best_val_acc"].mean()),
    }

    summary_path = dirs["results"] / "summary_statistics.csv"

    pd.DataFrame([summary]).to_csv(
        summary_path,
        index=False
    )

    print("\n" + "#" * 80)
    print("All Subjects Summary")
    print("#" * 80)

    print(
        results_df[
            [
                "subject",
                "best_val_acc",
                "final_val_acc",
                "test_acc",
                "test_f1_macro",
            ]
        ]
    )

    print("\nSummary:")
    print(f"Number of subjects      : {summary['num_subjects']}")
    print(f"Mean Test Accuracy      : {summary['mean_test_acc'] * 100:.2f}%")
    print(f"Std Test Accuracy       : {summary['std_test_acc'] * 100:.2f}%")
    print(f"Mean Test Macro F1      : {summary['mean_test_f1_macro'] * 100:.2f}%")
    print(f"Std Test Macro F1       : {summary['std_test_f1_macro'] * 100:.2f}%")
    print(f"Mean Best Val Accuracy  : {summary['mean_best_val_acc'] * 100:.2f}%")

    print("\nResults saved to:")
    print(dirs["results"])

    log_path = dirs["results"] / cfg.LOGBOOK_NAME

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
        f.write("\n\nFinal results:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nSummary:\n")
        f.write(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2
            )
        )


if __name__ == "__main__":
    main()

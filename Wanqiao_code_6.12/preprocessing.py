# ============================================================
# preprocessing.py
# DB6 EMG + ACC Fusion Preprocessing
#
# Split:
#   Train: Day1 + Day2
#   Val:   Day3
#   Test:  Day4 + Day5
#
# Output:
#   processed/db6_s2_emg_acc_200hz.npz
#
# Important:
#   If USE_TEACHER_AUGMENTATION = True:
#       training set will use teacher's sequence_generator
#       and should become about 3 times larger.
#   If USE_TEACHER_AUGMENTATION = False:
#       training set will use normal sliding window.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

from config import (
    DATA_DIR,
    PROCESSED_DIR,
    FIG_DIR,
    SAVE_PATH,
    TRAIN_SESSIONS,
    VAL_SESSIONS,
    TEST_SESSIONS,
    FS,
    LOWCUT,
    HIGHCUT,
    FILTER_ORDER,
    ENVELOPE_CUTOFF,
    WINDOW_SIZE,
    STRIDE,
    REMOVE_REST,
)

from augmentation import sequence_generator


# ============================================================
# 0. Augmentation switch
# ============================================================

# True  = use teacher augmentation for training set
# False = no augmentation
USE_TEACHER_AUGMENTATION = True


# ============================================================
# 1. Load CSV
# ============================================================

def load_numeric_csv(path, dtype=np.float32):
    df = pd.read_csv(path, header=None, low_memory=False)
    df = df.replace(["-", " ", ""], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)
    return df.values.astype(dtype)


def load_session(session_name):
    merged_path = os.path.join(DATA_DIR, f"{session_name}_merged.csv")

    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"Cannot find file: {merged_path}")

    data = load_numeric_csv(merged_path, dtype=np.float32)

    # merged csv columns:
    # 0-15   : EMG
    # 16-63  : ACC
    # 64     : restimulus
    # 65     : rerepetition
    emg = data[:, 0:16].astype(np.float32)
    acc = data[:, 16:64].astype(np.float32)
    labels = data[:, 64].astype(np.int64)
    reps = data[:, 65].astype(np.int64)

    print(f"\nLoaded {session_name}")
    print("Merged shape:", data.shape)
    print("EMG shape:", emg.shape)
    print("ACC shape:", acc.shape)
    print("Labels:", np.unique(labels))
    print("Repetitions:", np.unique(reps))

    return emg, acc, labels, reps


# ============================================================
# 2. EMG preprocessing
# ============================================================

def bandpass_filter_emg(emg):
    nyquist = FS / 2.0
    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist

    if high >= 1.0:
        raise ValueError("HIGHCUT must be smaller than Nyquist frequency.")

    b, a = butter(
        FILTER_ORDER,
        [low, high],
        btype="bandpass"
    )

    filtered = filtfilt(b, a, emg, axis=0)
    return filtered.astype(np.float32)


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

    envelope = filtfilt(b, a, emg, axis=0)
    return envelope.astype(np.float32)


def preprocess_emg(emg):
    emg_filtered = bandpass_filter_emg(emg)
    emg_rectified = rectify_emg(emg_filtered)
    emg_envelope = extract_envelope(emg_rectified)
    return emg_envelope


# ============================================================
# 3. ACC valid column detection
# ============================================================

def find_valid_acc_columns(train_acc_list):
    train_acc_all = np.concatenate(train_acc_list, axis=0)
    std = train_acc_all.std(axis=0)
    valid_cols = np.where(std > 1e-8)[0]

    print("\n========== ACC COLUMN CHECK ==========")
    print("Original ACC columns:", train_acc_all.shape[1])
    print("Valid ACC columns:", len(valid_cols))
    print("Removed empty columns:", train_acc_all.shape[1] - len(valid_cols))
    print("Valid column indices:", valid_cols)

    return valid_cols


# ============================================================
# 4. Normalization
# ============================================================

def compute_mean_std(data_list):
    data_all = np.concatenate(data_list, axis=0)
    mean = data_all.mean(axis=0, keepdims=True)
    std = data_all.std(axis=0, keepdims=True) + 1e-8

    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalization(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


# ============================================================
# 5. Normal sliding window
# ============================================================

def create_windows(emg, acc, labels):
    X_emg = []
    X_acc = []
    y = []

    for start in range(0, len(labels) - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE

        window_labels = labels[start:end]

        # Only keep pure-label windows
        if np.all(window_labels == window_labels[0]):
            X_emg.append(emg[start:end, :])
            X_acc.append(acc[start:end, :])
            y.append(window_labels[0])

    X_emg = np.array(X_emg, dtype=np.float32)
    X_acc = np.array(X_acc, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X_emg, X_acc, y


def window_group(emg_list, acc_list, label_list):
    X_emg_all = []
    X_acc_all = []
    y_all = []

    for emg, acc, labels in zip(emg_list, acc_list, label_list):
        X_emg, X_acc, y = create_windows(emg, acc, labels)

        if len(y) == 0:
            continue

        X_emg_all.append(X_emg)
        X_acc_all.append(X_acc)
        y_all.append(y)

    if len(X_emg_all) == 0:
        raise RuntimeError("No windows were generated in window_group().")

    X_emg_all = np.concatenate(X_emg_all, axis=0)
    X_acc_all = np.concatenate(X_acc_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return X_emg_all, X_acc_all, y_all


# ============================================================
# 6. Teacher augmentation sliding window
# ============================================================

def teacher_sequence_group_fusion(emg_list, acc_list, label_list):
    """
    Use teacher's sequence_generator to generate augmented fusion windows.

    Only used for training set.

    Input continuous data:
        EMG:   [T, 16]
        ACC:   [T, 42]
        label: [T]

    DataFrame format:
        columns 0:16   = EMG
        columns 16:58  = ACC
        column 58      = label

    Output:
        X_fusion: [N, WINDOW_SIZE, 58]
        y:        [N]
    """

    X_fusion_all = []
    y_all = []

    for emg, acc, labels in zip(emg_list, acc_list, label_list):

        fusion = np.concatenate([emg, acc], axis=1).astype(np.float32)

        # ----------------------------------------------------
        # Important:
        # Split continuous data by label first.
        # This avoids windows crossing different gestures.
        # ----------------------------------------------------

        start = 0

        while start < len(labels):
            current_label = labels[start]
            end = start + 1

            while end < len(labels) and labels[end] == current_label:
                end += 1

            segment_len = end - start

            if segment_len > WINDOW_SIZE:
                segment_fusion = fusion[start:end]
                segment_labels = labels[start:end].reshape(-1, 1)

                df_array = np.concatenate(
                    [segment_fusion, segment_labels],
                    axis=1
                )

                df = pd.DataFrame(df_array)

                X_part, y_part = sequence_generator(
                    data=df,
                    L=WINDOW_SIZE,

                    # EMG columns
                    CH0=0,
                    CH1=16,

                    # ACC columns
                    CH2=16,
                    CH3=58,

                    # Label column index
                    W=58,

                    D=0,
                    O=1,

                    # Total feature channels
                    c=58,

                    # EMG channels
                    e=16,

                    # Turn on teacher augmentation
                    augment=1,
                    Eaugment=1,

                    # Use our original stride
                    stride=STRIDE
                )

                if len(y_part) > 0:
                    X_fusion_all.append(X_part.astype(np.float32))
                    y_all.append(y_part.reshape(-1).astype(np.int64))

            start = end

    if len(X_fusion_all) == 0:
        raise RuntimeError("No augmented windows were generated.")

    X_fusion_all = np.concatenate(X_fusion_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return X_fusion_all, y_all


# ============================================================
# 7. Label remapping
# ============================================================

def remap_labels(y_train, y_val, y_test):
    all_labels = np.concatenate([y_train, y_val, y_test])
    original_labels = sorted(np.unique(all_labels))

    label_to_new = {
        old_label: new_label
        for new_label, old_label in enumerate(original_labels)
    }

    y_train_new = np.array([label_to_new[x] for x in y_train], dtype=np.int64)
    y_val_new = np.array([label_to_new[x] for x in y_val], dtype=np.int64)
    y_test_new = np.array([label_to_new[x] for x in y_test], dtype=np.int64)

    print("\n========== LABEL MAPPING ==========")
    print("Original labels:", original_labels)
    print("Label to new:", label_to_new)
    print("Number of classes:", len(original_labels))

    return y_train_new, y_val_new, y_test_new, label_to_new


# ============================================================
# 8. Plot example signal
# ============================================================

def plot_example_signal(emg, acc, labels):
    os.makedirs(FIG_DIR, exist_ok=True)

    length = min(2000, len(labels))
    t = np.arange(length) / FS

    plt.figure(figsize=(14, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, emg[:length, 0])
    plt.title("Example EMG Channel 1 after Preprocessing")
    plt.xlabel("Time (s)")
    plt.ylabel("EMG")

    plt.subplot(3, 1, 2)
    plt.plot(t, acc[:length, 0])
    plt.title("Example ACC Channel 1")
    plt.xlabel("Time (s)")
    plt.ylabel("ACC")

    plt.subplot(3, 1, 3)
    plt.plot(t, labels[:length])
    plt.title("Restimulus Label")
    plt.xlabel("Time (s)")
    plt.ylabel("Label")

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "db6_signal_example.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved example figure:", save_path)


# ============================================================
# 9. Load and preprocess sessions
# ============================================================

def load_and_preprocess_sessions(session_list):
    emg_list = []
    acc_list = []
    label_list = []

    for session in session_list:
        emg, acc, labels, reps = load_session(session)

        emg = preprocess_emg(emg)

        if REMOVE_REST:
            mask = labels != 0
            emg = emg[mask]
            acc = acc[mask]
            labels = labels[mask]

        emg_list.append(emg)
        acc_list.append(acc)
        label_list.append(labels)

    return emg_list, acc_list, label_list


# ============================================================
# 10. Main
# ============================================================

def main():

    print("========== PATH CHECK ==========")
    print("Current working directory:", os.getcwd())
    print("DATA_DIR:", DATA_DIR)
    print("Absolute DATA_DIR:", os.path.abspath(DATA_DIR))

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load train / val / test sessions
    # --------------------------------------------------------

    print("========== LOAD AND PREPROCESS TRAIN ==========")
    train_emg_list, train_acc_list, train_label_list = load_and_preprocess_sessions(
        TRAIN_SESSIONS
    )

    print("\n========== LOAD AND PREPROCESS VAL ==========")
    val_emg_list, val_acc_list, val_label_list = load_and_preprocess_sessions(
        VAL_SESSIONS
    )

    print("\n========== LOAD AND PREPROCESS TEST ==========")
    test_emg_list, test_acc_list, test_label_list = load_and_preprocess_sessions(
        TEST_SESSIONS
    )

    # --------------------------------------------------------
    # Remove invalid ACC columns using train set only
    # --------------------------------------------------------

    valid_acc_cols = find_valid_acc_columns(train_acc_list)

    train_acc_list = [x[:, valid_acc_cols] for x in train_acc_list]
    val_acc_list = [x[:, valid_acc_cols] for x in val_acc_list]
    test_acc_list = [x[:, valid_acc_cols] for x in test_acc_list]

    # --------------------------------------------------------
    # Normalize using train mean/std only
    # --------------------------------------------------------

    emg_mean, emg_std = compute_mean_std(train_emg_list)
    acc_mean, acc_std = compute_mean_std(train_acc_list)

    train_emg_list = [
        apply_normalization(x, emg_mean, emg_std)
        for x in train_emg_list
    ]

    val_emg_list = [
        apply_normalization(x, emg_mean, emg_std)
        for x in val_emg_list
    ]

    test_emg_list = [
        apply_normalization(x, emg_mean, emg_std)
        for x in test_emg_list
    ]

    train_acc_list = [
        apply_normalization(x, acc_mean, acc_std)
        for x in train_acc_list
    ]

    val_acc_list = [
        apply_normalization(x, acc_mean, acc_std)
        for x in val_acc_list
    ]

    test_acc_list = [
        apply_normalization(x, acc_mean, acc_std)
        for x in test_acc_list
    ]

    # --------------------------------------------------------
    # Plot example
    # --------------------------------------------------------

    plot_example_signal(
        train_emg_list[0],
        train_acc_list[0],
        train_label_list[0]
    )

    # --------------------------------------------------------
    # Generate train windows
    # --------------------------------------------------------

    if USE_TEACHER_AUGMENTATION:
        print("\n========== USE TEACHER AUGMENTATION FOR TRAIN ==========")

        X_fusion_train, y_train = teacher_sequence_group_fusion(
            train_emg_list,
            train_acc_list,
            train_label_list
        )

        # Split augmented fusion data back into EMG and ACC
        # Fusion shape: [N, 40, 58]
        # EMG: columns 0-15
        # ACC: columns 16-57
        X_emg_train = X_fusion_train[:, :, 0:16]
        X_acc_train = X_fusion_train[:, :, 16:58]

    else:
        print("\n========== NO AUGMENTATION FOR TRAIN ==========")

        X_emg_train, X_acc_train, y_train = window_group(
            train_emg_list,
            train_acc_list,
            train_label_list
        )

        X_fusion_train = np.concatenate(
            [X_emg_train, X_acc_train],
            axis=2
        )

    # --------------------------------------------------------
    # Generate val/test windows
    # No augmentation for val/test
    # --------------------------------------------------------

    X_emg_val, X_acc_val, y_val = window_group(
        val_emg_list,
        val_acc_list,
        val_label_list
    )

    X_emg_test, X_acc_test, y_test = window_group(
        test_emg_list,
        test_acc_list,
        test_label_list
    )

    X_fusion_val = np.concatenate([X_emg_val, X_acc_val], axis=2)
    X_fusion_test = np.concatenate([X_emg_test, X_acc_test], axis=2)

    # --------------------------------------------------------
    # Remap labels after window generation
    # --------------------------------------------------------

    y_train, y_val, y_test, label_to_new = remap_labels(
        y_train,
        y_val,
        y_test
    )

    num_classes = len(label_to_new)

    # --------------------------------------------------------
    # Print final shapes
    # --------------------------------------------------------

    print("\n========== FINAL DATA SHAPES ==========")
    print("X_emg_train:", X_emg_train.shape)
    print("X_acc_train:", X_acc_train.shape)
    print("X_fusion_train:", X_fusion_train.shape)
    print("y_train:", y_train.shape)

    print("X_emg_val:", X_emg_val.shape)
    print("X_acc_val:", X_acc_val.shape)
    print("X_fusion_val:", X_fusion_val.shape)
    print("y_val:", y_val.shape)

    print("X_emg_test:", X_emg_test.shape)
    print("X_acc_test:", X_acc_test.shape)
    print("X_fusion_test:", X_fusion_test.shape)
    print("y_test:", y_test.shape)

    print("Number of classes:", num_classes)

    # --------------------------------------------------------
    # Save processed data
    # --------------------------------------------------------

    np.savez(
        SAVE_PATH,

        X_emg_train=X_emg_train,
        X_acc_train=X_acc_train,
        X_fusion_train=X_fusion_train,
        y_train=y_train,

        X_emg_val=X_emg_val,
        X_acc_val=X_acc_val,
        X_fusion_val=X_fusion_val,
        y_val=y_val,

        X_emg_test=X_emg_test,
        X_acc_test=X_acc_test,
        X_fusion_test=X_fusion_test,
        y_test=y_test,

        num_classes=num_classes,
        valid_acc_cols=valid_acc_cols,

        emg_mean=emg_mean,
        emg_std=emg_std,
        acc_mean=acc_mean,
        acc_std=acc_std,
    )

    print("\n========== SAVED ==========")
    print("Saved processed data to:", SAVE_PATH)


if __name__ == "__main__":
    main()
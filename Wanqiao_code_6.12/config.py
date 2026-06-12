# ============================================================
# config.py
# DB6 EMG + ACC Fusion
# ============================================================
'''
import os

# =========================
# Path
# =========================
DATA_DIR = "../data/DB6"
PROCESSED_DIR = "../processed"
FIG_DIR = "../figures"
SAVE_PATH = "../processed/db6_s1_emg_acc_200hz.npz"

SAVE_PATH = os.path.join(PROCESSED_DIR, "db6_s1_emg_acc_200hz.npz")

# =========================
# Dataset split
# =========================

TRAIN_SESSIONS = [
    "S1_D1_T1_200Hz",
    "S1_D1_T2_200Hz",
    "S1_D2_T1_200Hz",
    "S1_D2_T2_200Hz",
]

VAL_SESSIONS = [
    "S1_D3_T1_200Hz",
    "S1_D3_T2_200Hz",
]

TEST_SESSIONS = [
    "S1_D4_T1_200Hz",
    "S1_D4_T2_200Hz",
    "S1_D5_T1_200Hz",
    "S1_D5_T2_200Hz",
]


# =========================
# Sampling
# =========================
FS = 200

# =========================
# EMG preprocessing
# =========================
LOWCUT = 20
HIGHCUT = 90
FILTER_ORDER = 4
ENVELOPE_CUTOFF = 5

# =========================
# Sliding window
# =========================
WINDOW_SIZE = 40   # 200 ms at 200 Hz
STRIDE = 10        # 50 ms at 200 Hz

REMOVE_REST = True
'''

'''
import os

DATA_DIR = "../data/DB6"
PROCESSED_DIR = "../processed"
FIG_DIR = "../figures"

# 注意这里改成 S2，避免覆盖 S1 的 npz
SAVE_PATH = "../processed/db6_s2_emg_acc_200hz.npz"

TRAIN_SESSIONS = [
    "S2_D1_T1_200Hz",
    "S2_D1_T2_200Hz",
    "S2_D2_T1_200Hz",
    
]

VAL_SESSIONS = [
    "S2_D3_T1_200Hz",
    "S2_D3_T2_200Hz",
]

TEST_SESSIONS = [
    "S2_D4_T1_200Hz",
    "S2_D4_T2_200Hz",
    "S2_D5_T1_200Hz",
    "S2_D5_T2_200Hz",
]

FS = 200

LOWCUT = 20
HIGHCUT = 90
FILTER_ORDER = 4
ENVELOPE_CUTOFF = 5

WINDOW_SIZE = 40
STRIDE = 10

REMOVE_REST = True
'''
import os

DATA_DIR = "../data/DB6"
PROCESSED_DIR = "../processed"
FIG_DIR = "../figures"

SAVE_PATH = "../processed/db6_s3_emg_acc_200hz.npz"

TRAIN_SESSIONS = [
    "S3_D1_T1_200Hz",
    "S3_D1_T2_200Hz",
    "S3_D2_T1_200Hz",
    "S3_D2_T2_200Hz",
]

VAL_SESSIONS = [
    "S3_D3_T1_200Hz",
    "S3_D3_T2_200Hz",
]

TEST_SESSIONS = [
    "S3_D4_T1_200Hz",
    "S3_D4_T2_200Hz",
    "S3_D5_T1_200Hz",
    "S3_D5_T2_200Hz",
]

FS = 200

LOWCUT = 20
HIGHCUT = 90
FILTER_ORDER = 4
ENVELOPE_CUTOFF = 5

WINDOW_SIZE = 40
STRIDE = 10

REMOVE_REST = True
from pathlib import Path
import torch

# ============================================================
# 路径和输出
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "output"

# 把你上传的 zip 放在代码同级目录；如果已经解压，也可以直接放 CSV。
DB6_ZIP_PATH = None
DB6_DATA_DIR = PROJECT_ROOT

EXPERIMENT_NAME = "db6_emg_acc_qk"
RESULT_CSV_NAME = "results.csv"
LOGBOOK_NAME = "log.txt"

# ============================================================
# DB6 当前处理后 CSV 格式
# 0~15  : EMG, 16 通道
# 16~63 : ACC, 48 通道
# 64    : 动作标签，0 为 rest
# 65    : repetition / 序号，1~12
# ============================================================

DB6_EMG_RAW_COLS = list(range(0, 16))
DB6_ACC_RAW_COLS = list(range(16, 64))
DB6_LABEL_COL = 64
DB6_REPETITION_COL = 65
DB6_EXPECTED_MIN_COLS = 66

EMG_DIM = len(DB6_EMG_RAW_COLS)
ACC_DIM = len(DB6_ACC_RAW_COLS)

# 进入 sequence_generator 和模型之前，统一重排为：ACC first, EMG second。
# 这样原来的 Eaugment 仍然作用在第二块 EMG 上，增强方法不需要改。
FEATURE_START_COL = 0
FEATURE_END_COL = ACC_DIM + EMG_DIM
INPUT_DIM = FEATURE_END_COL - FEATURE_START_COL

# 这个 zip 中第 64 列出现的原始动作标签。
# 0 是 rest；非 rest 标签不是连续编号，所以后续会用映射转换为训练标签 0~6。
RAW_LABELS = [0, 1, 3, 4, 6, 9, 10, 11]
RAW_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(RAW_LABELS)}
RAW_NUM_CLASSES = len(RAW_LABELS)

LABEL_START_COL = FEATURE_END_COL
LABEL_END_COL = LABEL_START_COL + RAW_NUM_CLASSES
RAW_COLUMNS = list(range(LABEL_END_COL))

FEATURE_COLUMNS = [f"ACC_{i + 1}" for i in range(ACC_DIM)] + [f"EMG_{i + 1}" for i in range(EMG_DIM)]


# ============================================================
# DB6 原始信号预处理
# 说明：这些步骤发生在窗口切分和 train mean/std 标准化之前。
# 不会改变你的数据增强函数、模型结构或 attention 计算方式。
# ============================================================

RAW_SAMPLING_RATE = 200.0
ENABLE_SIGNAL_PREPROCESSING = True

# EMG: 去均值 -> 工频陷波 -> 带通 -> 全波整流 -> 低通平滑成包络
EMG_REMOVE_MEAN = True
EMG_NOTCH = True
EMG_NOTCH_FREQ = 50.0
EMG_NOTCH_Q = 30.0
EMG_BANDPASS = True
EMG_BANDPASS_LOW = 20.0
EMG_BANDPASS_HIGH = 95.0
EMG_FILTER_ORDER = 4
EMG_RECTIFY = True
EMG_ENVELOPE = True
EMG_ENVELOPE_CUTOFF = 10.0

# ACC: 低通去高频噪声，保留姿态/动作变化中的低频成分
ACC_LOWPASS = True
ACC_LOWPASS_CUTOFF = 10.0
ACC_FILTER_ORDER = 4

# 是否把每个原始 CSV 预处理后的 ACC+EMG+label+repetition 另存出来。
# 默认 False，避免第一次训练时额外写出很大的 CSV；需要检查预处理结果时可改成 True。
SAVE_PREPROCESSED_CSV = False
DB6_PREPROCESSED_DIR = PROJECT_ROOT / "db6_preprocessed"
PREPROCESSED_SUFFIX = "_preprocessed_acc_emg.csv"

# 是否在预处理后自动画图，图会保存到 output/<run>/preprocessing/preprocessed_signal_plots。
# 只画预处理后的预览图，不影响训练数据、增强逻辑和模型方法。
SAVE_PREPROCESSED_SIGNAL_PLOTS = True
PREPROCESS_PLOT_MAX_POINTS = 3000
PREPROCESS_PLOT_EMG_CHANNELS = [0, 1, 2, 3]
PREPROCESS_PLOT_ACC_CHANNELS = [0, 1, 2, 3, 4, 5]
PREPROCESS_PLOT_DPI = 200

# ============================================================
# attention / 模型里的通道索引
# 注意：这是“重排后的窗口张量”的通道索引，不是原始 CSV 索引。
# 窗口张量顺序：前 48 列 ACC，后 16 列 EMG。
# ============================================================

ACC_CHANNELS = list(range(0, ACC_DIM))
EMG_CHANNELS = list(range(ACC_DIM, ACC_DIM + EMG_DIM))

# 为了兼容旧 checkpoint 或旧打印字段，保留别名；新代码不用 FMG 名称。
FMG_CHANNELS = ACC_CHANNELS

# ============================================================
# sequence_generator 参数
# ACC first, EMG second
# ============================================================

CH0 = 0
CH1 = ACC_DIM          # ACC: 0~47

CH2 = ACC_DIM
CH3 = ACC_DIM + EMG_DIM  # EMG: 48~63

D_OFFSET = 0

C_CHANNELS = ACC_DIM
E_CHANNELS = EMG_DIM

# 训练集增强：保持你原来的开关不变
TRAIN_AUGMENT = 1
TRAIN_EAUGMENT = 1

# 验证集和测试集不增强
VAL_AUGMENT = None
VAL_EAUGMENT = None

TEST_AUGMENT = None
TEST_EAUGMENT = None

# ============================================================
# 是否删除 rest 类
# ============================================================

REMOVE_REST = True
REST_LABEL = 0

if REMOVE_REST:
    TRAIN_RAW_LABELS = [label for label in RAW_LABELS if label != REST_LABEL]
else:
    TRAIN_RAW_LABELS = RAW_LABELS.copy()

TRAIN_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(TRAIN_RAW_LABELS)}
INDEX_TO_TRAIN_LABEL = {idx: label for label, idx in TRAIN_LABEL_TO_INDEX.items()}

NUM_CLASSES = len(TRAIN_RAW_LABELS)
CLASS_NAMES = [f"Gesture_{label}" for label in TRAIN_RAW_LABELS]

# ============================================================
# 数据划分
# 当前策略：按天划分，T1 和 T2 都纳入对应 split。
# train: 第 1 天 + 第 2 天，即 D1、D2 的 T1/T2
# val  : 第 3 天，即 D3 的 T1/T2
# test : 第 4 天 + 第 5 天，即 D4、D5 的 T1/T2
# 不再按 repetition 切分；每个 split 使用对应文件的全部 repetition。
# ============================================================

def day_files(days):
    files = []

    for day in days:
        files.append(Path(f"S1_D{day}_T1_200Hz_merged.csv"))
        files.append(Path(f"S1_D{day}_T2_200Hz_merged.csv"))

    return files

TRAIN_DAYS = [1, 2]
VAL_DAYS = [3]
TEST_DAYS = [4, 5]

SUBJECTS = [
    {
        "name": "S1_DB6",
        "train": day_files(TRAIN_DAYS),
        "val": day_files(VAL_DAYS),
        "test": day_files(TEST_DAYS),
        "train_repetitions": None,
        "val_repetitions": None,
        "test_repetitions": None,
    }
]

SKIP_MISSING_SUBJECTS = False

# ============================================================
# 标准化
# ============================================================

NORMALIZE_BY_TRAIN = True

# ============================================================
# 窗口
# 注意：sequence_generator 原方法仍然是逐点滑动窗口。
# STRIDE 这里只用于记录，不参与 sequence_generator。
# ============================================================

WINDOW_SIZE = 40
STRIDE = 10

# ============================================================
# QK-only Transformer 参数：保持原模型方法不变
# ============================================================

D_MODEL = 64
NHEAD = 4
DROPOUT = 0.15

QK_FEATURE_MODE = "pool"      # "pool" 或 "flatten"
QK_POOL_SIZE = 4
QK_USE_SOFTMAX = False
QK_CLASSIFIER_HIDDEN = 128

# ============================================================
# 训练参数
# ============================================================

BATCH_SIZE = 128
EPOCHS = 80
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 20

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

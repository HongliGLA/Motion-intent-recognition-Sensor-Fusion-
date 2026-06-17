import json
import random
import re
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import signal
except Exception:  # scipy 不可用时，预处理会自动退化为基础 numpy 版本
    signal = None

import config as cfg


# =====================================================================
# 0. 你的数据增强代码：保持不改
# =====================================================================

def sequence_generator(data,L,CH0,CH1,CH2,CH3,W,D,O,c,e,augment = None,Eaugment = None):
  sequence_group = []
  for i in range(len(data)-L-D):
    sequence_group.append(data.iloc[i:i+L+D])
  data_= np.array([df.values for df in sequence_group])
  input1 = data_[:,:L,CH0:CH1]
  input2 = data_[:,:L,CH2:CH3]
  input = np.concatenate((input1,input2),axis=2)
  target = data_[:,-1,W:W+O]
  #input = (input-input.mean(axis=0))/input.std(axis=0)
  #input = (input-input.mean(axis=0))/input.std(axis=0)
  if augment ==1:
      augment_data = input.copy()
      augment_data2 = input.copy()
      augment_data3 = input.copy()
      increase_percentages = np.random.uniform(low=0.1, high=0.9, size=(input.shape[0], int(c/2)))
      decrease_percentages = np.random.uniform(low=0.1, high=0.9, size=(input.shape[0], c-int(c/2)))
      #increase_percentages = 0.8 * np.random.beta(2, 5,(input.shape[0], int(c/2)) ) + 0.1
      #decrease_percentages = 0.8 * np.random.beta(2, 5,(input.shape[0], int(c/2)) ) + 0.1
      for i in range(input.shape[0]):
        np.random.shuffle(increase_percentages[i])
        np.random.shuffle(decrease_percentages[i])
      col_indices = [random.sample(range(0,c), c) for i in range(input.shape[0])]
      for i in range(input.shape[0]):
        for j, col in enumerate(col_indices[i][:int(c/2)]):
          augment_data[i, :, col] = input[i, :, col] * (1 + increase_percentages[i, j])+np.random.uniform(low=-0.1, high=0.1)
        for j, col in enumerate(col_indices[i][int(c/2):]):
          augment_data[i, :, col] = input[i, :, col] * (1 - decrease_percentages[i, j])+np.random.uniform(low=-0.1, high=0.1)
        #emg aug
        if Eaugment == 1:
          mean_values = augment_data2[i,:,c:c+e].mean(axis=0)
          max_channel_index = np.argmax(mean_values)
          chosen_number = random.choice([0, 1])
          crange = [8,9,10,11,12,13,14,15]
          crange_e = crange+crange[:3]
          prange = crange_e[max_channel_index:max_channel_index+4]
        #comp = np.random.uniform(low=0, high=0.3)

          for j in range(c,augment_data2.shape[2]):
            if j == max_channel_index+c:
              if chosen_number == 0:
                augment_data2[i, :, j] *= (1+np.random.uniform(low=0, high=0.3))
              else:
                augment_data2[i, :, j] *= (1-np.random.uniform(low=0, high=0.3))
            else:
              if chosen_number == 0:
                augment_data2[i, :, j] *= (1-np.random.uniform(low=0, high=0.3))
              else:
                augment_data2[i, :, j] *= (1+np.random.uniform(low=0, high=0.3))
          augment_data3[i, :, :8] = augment_data[i, :, :8]
          augment_data3[i, :, 8:16] = augment_data2[i, :, 8:16]

      if Eaugment == 1:
        input = np.concatenate((input,augment_data,augment_data2),axis=0)
        target = np.concatenate((target,target,target),axis=0)
      else:
        input = np.concatenate((input,augment_data),axis=0)
        target = np.concatenate((target,target),axis=0)

  if augment == 2:
    augment_data2 = input.copy()
    mean_values = augment_data2[i,:,:].mean(axis=0)
    max_channel_index = np.argmax(mean_values)
    chosen_number = random.choice([0, 1])
    crange = [8,9,10,11,12,13,14,15]
    crange_e = crange+crange[:3]
    prange = crange_e[max_channel_index:max_channel_index+4]
        #comp = np.random.uniform(low=0, high=0.3)

    for j in range(c,augment_data2.shape[2]):
      if j == max_channel_index:
        if chosen_number == 0:
          augment_data2[i, :, j] *= (1+np.random.uniform(low=0, high=0.3))
        else:
          augment_data2[i, :, j] *= (1-np.random.uniform(low=0, high=0.3))
      else:
        if chosen_number == 0:
          augment_data2[i, :, j] *= (1-np.random.uniform(low=0, high=0.3))
        else:
          augment_data2[i, :, j] *= (1+np.random.uniform(low=0, high=0.3))
    input = np.concatenate((input,augment_data2),axis=0)
    target = np.concatenate((target,target),axis=0)
      #del_indices = np.random.rand(input.shape[0]) < 0.5
      #input = input[del_indices]
      #target = target[del_indices]

  return input,target


# =====================================================================
# 1. 路径自动查找 / zip 自动解压
# =====================================================================

def _path_candidates(path_item):
    path = Path(path_item)
    return [
        path,
        cfg.PROJECT_ROOT / path,
        cfg.DB6_DATA_DIR / path.name,
        cfg.PROJECT_ROOT / path.name,
    ]


def _find_existing_path(path_item):
    for path in _path_candidates(path_item):
        if path.exists():
            return path

    matches = list(cfg.PROJECT_ROOT.rglob(Path(path_item).name))
    if len(matches) > 0:
        return matches[0]

    return None


def ensure_db6_zip_extracted():
    # 如果所有 CSV 已经在目录里，就不解压。
    if cfg.DB6_DATA_DIR.exists() and any(cfg.DB6_DATA_DIR.glob("*.csv")):
        return

    zip_path = _find_existing_path(cfg.DB6_ZIP_PATH)

    if zip_path is None:
        return

    cfg.DB6_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Extract DB6 zip:", zip_path)
    print("Extract to     :", cfg.DB6_DATA_DIR)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cfg.DB6_DATA_DIR)


def resolve_path(path_or_candidates):
    # 这里 list/tuple 仍按“候选路径”处理；多文件 split 用 resolve_many_paths。
    if isinstance(path_or_candidates, (str, Path)):
        candidates = [path_or_candidates]
    else:
        candidates = path_or_candidates

    searched = []

    for item in candidates:
        for path in _path_candidates(item):
            searched.append(str(path))
            if path.exists():
                return path

        matches = list(cfg.PROJECT_ROOT.rglob(Path(item).name))
        if len(matches) > 0:
            return matches[0]

    ensure_db6_zip_extracted()

    for item in candidates:
        for path in _path_candidates(item):
            if path.exists():
                return path

        matches = list(cfg.PROJECT_ROOT.rglob(Path(item).name))
        if len(matches) > 0:
            return matches[0]

    raise FileNotFoundError(
        "找不到文件，已经尝试：\n"
        + "\n".join(searched)
        + "\n请确认 CSV 已经解压，或 zip 文件和代码在同一目录下。"
    )


def resolve_many_paths(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]

    return [resolve_path(path) for path in paths]



# =====================================================================
# 2. DB6 原始信号预处理
# =====================================================================

def _as_float32_array(values):
    arr = np.asarray(values, dtype=np.float32)

    if not np.isfinite(arr).all():
        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(np.float32)

    return arr


def _valid_cutoff(cutoff, fs):
    nyq = fs / 2.0
    cutoff = float(cutoff)
    cutoff = max(cutoff, 1e-6)
    cutoff = min(cutoff, nyq * 0.99)
    return cutoff


def _zero_phase_sos_filter(x, sos):
    """
    对每个通道做零相位滤波。
    如果序列太短导致 filtfilt 失败，则退化为普通前向滤波。
    """

    if signal is None:
        return x

    if x.shape[0] < 8:
        return x

    try:
        return signal.sosfiltfilt(
            sos,
            x,
            axis=0
        ).astype(np.float32)
    except ValueError:
        return signal.sosfilt(
            sos,
            x,
            axis=0
        ).astype(np.float32)


def _zero_phase_ba_filter(x, b, a):
    if signal is None:
        return x

    if x.shape[0] < 8:
        return x

    try:
        return signal.filtfilt(
            b,
            a,
            x,
            axis=0
        ).astype(np.float32)
    except ValueError:
        return signal.lfilter(
            b,
            a,
            x,
            axis=0
        ).astype(np.float32)


def preprocess_emg(emg):
    """
    EMG 预处理顺序：
        1) float32 + 非法值清零
        2) 去均值 / 去 DC
        3) 50Hz 工频陷波
        4) 20~95Hz 带通
        5) 全波整流
        6) 10Hz 低通平滑得到 envelope

    所有开关和截止频率都在 config.py 里。
    """

    emg = _as_float32_array(emg)

    if not cfg.ENABLE_SIGNAL_PREPROCESSING:
        return emg

    fs = float(cfg.RAW_SAMPLING_RATE)

    if cfg.EMG_REMOVE_MEAN:
        emg = emg - emg.mean(
            axis=0,
            keepdims=True
        )

    if signal is not None and cfg.EMG_NOTCH:
        notch_freq = float(cfg.EMG_NOTCH_FREQ)
        if 0 < notch_freq < fs / 2.0:
            b, a = signal.iirnotch(
                w0=notch_freq,
                Q=float(cfg.EMG_NOTCH_Q),
                fs=fs
            )
            emg = _zero_phase_ba_filter(
                emg,
                b,
                a
            )

    if signal is not None and cfg.EMG_BANDPASS:
        low = _valid_cutoff(
            cfg.EMG_BANDPASS_LOW,
            fs
        )
        high = _valid_cutoff(
            cfg.EMG_BANDPASS_HIGH,
            fs
        )

        if low >= high:
            low = max(1e-6, high * 0.5)

        sos = signal.butter(
            int(cfg.EMG_FILTER_ORDER),
            [low, high],
            btype="bandpass",
            fs=fs,
            output="sos"
        )
        emg = _zero_phase_sos_filter(
            emg,
            sos
        )

    if cfg.EMG_RECTIFY:
        emg = np.abs(emg)

    if signal is not None and cfg.EMG_ENVELOPE:
        cutoff = _valid_cutoff(
            cfg.EMG_ENVELOPE_CUTOFF,
            fs
        )
        sos = signal.butter(
            int(cfg.EMG_FILTER_ORDER),
            cutoff,
            btype="lowpass",
            fs=fs,
            output="sos"
        )
        emg = _zero_phase_sos_filter(
            emg,
            sos
        )

        # 零相位滤波在边界可能有很小的负值；envelope 保持非负。
        emg = np.maximum(
            emg,
            0.0
        )

    return emg.astype(np.float32)


def preprocess_acc(acc):
    """
    ACC 预处理：低通去除高频噪声，保留重力/姿态/动作低频信息。
    """

    acc = _as_float32_array(acc)

    if not cfg.ENABLE_SIGNAL_PREPROCESSING:
        return acc

    if signal is not None and cfg.ACC_LOWPASS:
        fs = float(cfg.RAW_SAMPLING_RATE)
        cutoff = _valid_cutoff(
            cfg.ACC_LOWPASS_CUTOFF,
            fs
        )
        sos = signal.butter(
            int(cfg.ACC_FILTER_ORDER),
            cutoff,
            btype="lowpass",
            fs=fs,
            output="sos"
        )
        acc = _zero_phase_sos_filter(
            acc,
            sos
        )

    return acc.astype(np.float32)


def preprocess_raw_signals(emg_raw, acc_raw):
    emg = preprocess_emg(emg_raw)
    acc = preprocess_acc(acc_raw)

    return emg, acc


def maybe_save_preprocessed_csv(csv_path, acc, emg, raw_labels, repetitions):
    if not cfg.SAVE_PREPROCESSED_CSV:
        return None

    out_dir = Path(cfg.DB6_PREPROCESSED_DIR)
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = Path(csv_path)
    out_path = out_dir / f"{csv_path.stem}{cfg.PREPROCESSED_SUFFIX}"

    acc_cols = [f"ACC_{i + 1}" for i in range(acc.shape[1])]
    emg_cols = [f"EMG_{i + 1}" for i in range(emg.shape[1])]

    out_df = pd.DataFrame(
        np.concatenate([acc, emg], axis=1),
        columns=acc_cols + emg_cols
    )
    out_df["label"] = raw_labels.astype(np.int64)
    out_df["repetition"] = repetitions.astype(np.int64)

    out_df.to_csv(
        out_path,
        index=False
    )

    return str(out_path)


def _safe_filename(text):
    text = str(text)
    text = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:120] if len(text) > 120 else text


def _select_preview_range(raw_labels, max_points):
    """
    优先选择第一个非 rest 的连续片段来画图；如果没有非 rest，就画开头一段。
    返回 [start, end) 的连续范围，避免画图跨动作边界。
    """

    n = len(raw_labels)
    if n == 0:
        return 0, 0

    max_points = max(1, int(max_points))
    raw_labels = np.asarray(raw_labels)

    change = np.ones(n, dtype=bool)
    change[1:] = raw_labels[1:] != raw_labels[:-1]
    starts = np.where(change)[0]
    ends = np.r_[starts[1:], n]

    # 先找非 rest 片段。
    for start, end in zip(starts, ends):
        if int(raw_labels[start]) != int(cfg.REST_LABEL):
            return int(start), int(min(end, start + max_points))

    return 0, int(min(n, max_points))


def _downsample_indices(start, end, max_points):
    length = max(0, end - start)

    if length == 0:
        return np.array([], dtype=np.int64)

    max_points = max(1, int(max_points))

    if length <= max_points:
        return np.arange(start, end, dtype=np.int64)

    return np.linspace(
        start,
        end - 1,
        max_points,
        dtype=np.int64
    )


def _valid_channel_list(channels, max_channels):
    valid = []

    for channel in channels:
        channel = int(channel)
        if 0 <= channel < max_channels:
            valid.append(channel)

    if len(valid) == 0:
        valid = list(range(min(max_channels, 4)))

    return valid


def maybe_save_preprocessed_signal_plots(
    csv_path,
    split_name,
    emg,
    acc,
    raw_labels,
    repetitions,
    plot_dir=None
):
    """
    保存预处理后的信号预览图。
    图只用于检查预处理后波形，不会参与训练，也不会改变数据增强和模型方法。
    """

    if not cfg.SAVE_PREPROCESSED_SIGNAL_PLOTS:
        return {}

    if plot_dir is None:
        plot_dir = cfg.OUTPUT_ROOT / "preprocessed_signal_plots"

    plot_dir = Path(plot_dir) / "preprocessed_signal_plots"
    plot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = Path(csv_path)
    base_name = _safe_filename(f"{split_name}_{csv_path.stem}")

    max_points = int(cfg.PREPROCESS_PLOT_MAX_POINTS)
    start, end = _select_preview_range(
        raw_labels,
        max_points=max_points
    )
    idx = _downsample_indices(
        start,
        end,
        max_points=max_points
    )

    if len(idx) == 0:
        return {}

    fs = float(cfg.RAW_SAMPLING_RATE)
    time_sec = (idx - idx[0]) / fs

    emg_channels = _valid_channel_list(
        cfg.PREPROCESS_PLOT_EMG_CHANNELS,
        emg.shape[1]
    )
    acc_channels = _valid_channel_list(
        cfg.PREPROCESS_PLOT_ACC_CHANNELS,
        acc.shape[1]
    )

    label_values = np.asarray(raw_labels)[idx]
    repetition_values = np.asarray(repetitions)[idx]

    paths = {}

    waveform_path = plot_dir / f"{base_name}_preprocessed_waveform.png"

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True
    )

    for ch in emg_channels:
        axes[0].plot(
            time_sec,
            emg[idx, ch],
            linewidth=0.9,
            label=f"EMG {ch + 1}"
        )

    axes[0].set_title(
        f"{csv_path.name} | preprocessed EMG envelope"
    )
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for ch in acc_channels:
        axes[1].plot(
            time_sec,
            acc[idx, ch],
            linewidth=0.9,
            label=f"ACC {ch + 1}"
        )

    axes[1].set_title("Preprocessed ACC low-pass signal")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        time_sec,
        label_values,
        linewidth=1.0,
        label="Raw label"
    )
    axes[2].plot(
        time_sec,
        repetition_values,
        linewidth=1.0,
        label="Repetition"
    )
    axes[2].set_title("Label / repetition of preview range")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Value")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Preprocessed signal preview | rows {start}~{end - 1}",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(
        waveform_path,
        dpi=int(cfg.PREPROCESS_PLOT_DPI)
    )
    plt.close(fig)

    paths["waveform"] = str(waveform_path)

    heatmap_path = plot_dir / f"{base_name}_preprocessed_heatmap.png"

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True
    )

    emg_img = axes[0].imshow(
        emg[idx, :].T,
        aspect="auto",
        origin="lower",
        extent=[time_sec[0], time_sec[-1], 1, emg.shape[1]]
    )
    axes[0].set_title("Preprocessed EMG envelope heatmap")
    axes[0].set_ylabel("EMG channel")
    fig.colorbar(emg_img, ax=axes[0], fraction=0.018, pad=0.01)

    acc_img = axes[1].imshow(
        acc[idx, :].T,
        aspect="auto",
        origin="lower",
        extent=[time_sec[0], time_sec[-1], 1, acc.shape[1]]
    )
    axes[1].set_title("Preprocessed ACC low-pass heatmap")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("ACC channel")
    fig.colorbar(acc_img, ax=axes[1], fraction=0.018, pad=0.01)

    fig.suptitle(
        f"Preprocessed signal shape | {csv_path.name} | rows {start}~{end - 1}",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(
        heatmap_path,
        dpi=int(cfg.PREPROCESS_PLOT_DPI)
    )
    plt.close(fig)

    paths["heatmap"] = str(heatmap_path)

    return paths


# =====================================================================
# 2. one-hot / 标签转换
# =====================================================================

def raw_labels_to_onehot(raw_labels):
    raw_labels = np.asarray(raw_labels, dtype=np.int64)

    unknown_labels = sorted(set(raw_labels.tolist()) - set(cfg.RAW_LABELS))

    if len(unknown_labels) > 0:
        raise ValueError(
            f"发现 config.RAW_LABELS 未包含的标签: {unknown_labels}。"
            f"请先在 config.py 中更新 RAW_LABELS。"
        )

    onehot = np.zeros(
        (len(raw_labels), cfg.RAW_NUM_CLASSES),
        dtype=np.float32
    )

    mapped = np.array(
        [cfg.RAW_LABEL_TO_INDEX[int(label)] for label in raw_labels],
        dtype=np.int64
    )

    onehot[np.arange(len(raw_labels)), mapped] = 1.0

    return onehot


def decode_onehot_labels(onehot):
    """
    onehot: [N, RAW_NUM_CLASSES]
    返回原始标签值，例如 0/1/3/4/6/9/10/11。
    """

    labels = []

    multi_hot_count = 0
    empty_count = 0

    for row in onehot:
        active = np.where(row > 0.5)[0]

        if len(active) == 0:
            labels.append(-1)
            empty_count += 1
            continue

        if len(active) > 1:
            multi_hot_count += 1

            rest_index = cfg.RAW_LABEL_TO_INDEX[cfg.REST_LABEL]

            non_rest = [
                x for x in active
                if x != rest_index
            ]

            if len(non_rest) > 0:
                labels.append(int(cfg.RAW_LABELS[int(non_rest[0])]))
            else:
                labels.append(int(cfg.RAW_LABELS[int(active[0])]))
        else:
            labels.append(int(cfg.RAW_LABELS[int(active[0])]))

    labels = np.array(labels, dtype=np.int64)

    print("Multi-hot rows:", multi_hot_count)
    print("Empty-label rows:", empty_count)

    if empty_count > 0:
        raise ValueError("存在没有标签的行，请检查标签列。")

    return labels


# =====================================================================
# 3. 读取一个 DB6 CSV
# =====================================================================

def _normalize_repetition_filter(repetition_filter):
    if repetition_filter is None:
        return None
    return set(int(x) for x in repetition_filter)


def load_one_csv(csv_path, split_name, repetition_filter=None, plot_dir=None):
    csv_path = resolve_path(csv_path)
    repetition_filter = _normalize_repetition_filter(repetition_filter)

    print("\n" + "=" * 70)
    print(f"Loading {split_name}")
    print("=" * 70)
    print("Path:", csv_path)
    print("Repetition filter:", None if repetition_filter is None else sorted(repetition_filter))
    print("Signal preprocessing:", cfg.ENABLE_SIGNAL_PREPROCESSING)

    df_raw = pd.read_csv(
        csv_path,
        header=None
    )

    print(f"{split_name} raw shape:", df_raw.shape)

    if df_raw.shape[1] < cfg.DB6_EXPECTED_MIN_COLS:
        raise ValueError(
            f"{split_name} CSV 列数不足，当前 {df_raw.shape[1]} 列，"
            f"至少需要 {cfg.DB6_EXPECTED_MIN_COLS} 列。"
        )

    raw_labels = df_raw.iloc[:, cfg.DB6_LABEL_COL].values.astype(np.int64)
    repetitions = df_raw.iloc[:, cfg.DB6_REPETITION_COL].values.astype(np.int64)

    if repetition_filter is not None:
        keep = np.isin(repetitions, list(repetition_filter))
        df_raw = df_raw.loc[keep].reset_index(drop=True)
        raw_labels = raw_labels[keep]
        repetitions = repetitions[keep]

    if len(df_raw) == 0:
        raise ValueError(f"{split_name} 根据 repetition 过滤后没有数据。")

    # 原始 CSV 顺序：0~15 EMG，16~63 ACC。
    # 先做原始信号预处理，再重排成模型输入顺序：ACC first, EMG second。
    emg_raw = df_raw.iloc[:, cfg.DB6_EMG_RAW_COLS].values.astype(np.float32)
    acc_raw = df_raw.iloc[:, cfg.DB6_ACC_RAW_COLS].values.astype(np.float32)

    emg, acc = preprocess_raw_signals(
        emg_raw,
        acc_raw
    )

    saved_preprocessed_path = maybe_save_preprocessed_csv(
        csv_path,
        acc,
        emg,
        raw_labels,
        repetitions
    )

    saved_plot_paths = maybe_save_preprocessed_signal_plots(
        csv_path=csv_path,
        split_name=split_name,
        emg=emg,
        acc=acc,
        raw_labels=raw_labels,
        repetitions=repetitions,
        plot_dir=plot_dir
    )

    feature_df = pd.DataFrame(
        np.concatenate([acc, emg], axis=1),
        columns=list(range(cfg.FEATURE_START_COL, cfg.FEATURE_END_COL))
    )

    y_onehot = raw_labels_to_onehot(raw_labels)

    label_df = pd.DataFrame(
        y_onehot,
        columns=list(range(cfg.LABEL_START_COL, cfg.LABEL_END_COL))
    )

    df = pd.concat(
        [feature_df, label_df],
        axis=1
    )

    df["raw_label"] = raw_labels
    df["repetition"] = repetitions

    # 根据 raw_label 或 repetition 变化生成 segment_id，防止窗口跨 rest / 动作 / repetition 边界。
    label_or_rep_change = np.ones(len(raw_labels), dtype=bool)
    label_or_rep_change[1:] = (
        (raw_labels[1:] != raw_labels[:-1])
        | (repetitions[1:] != repetitions[:-1])
    )
    segment_ids = np.cumsum(label_or_rep_change)

    df["segment_id"] = segment_ids

    print("Preprocessed EMG shape:", emg.shape)
    print("Preprocessed ACC shape:", acc.shape)
    if saved_preprocessed_path is not None:
        print("Saved preprocessed CSV:", saved_preprocessed_path)
    if len(saved_plot_paths) > 0:
        print("Saved preprocessed plots:", saved_plot_paths)

    print(f"\n{split_name} raw label counts:")

    unique_labels, counts = np.unique(
        raw_labels,
        return_counts=True
    )

    raw_label_counts = {}

    for label, count in zip(unique_labels, counts):
        raw_label_counts[int(label)] = int(count)
        print(f"Raw label {label:2d}: {count}")

    info = {
        "split": split_name,
        "path": str(csv_path),
        "raw_shape": list(df_raw.shape),
        "standardized_shape": list(df.shape),
        "raw_label_counts": raw_label_counts,
        "repetition_filter": None if repetition_filter is None else sorted(repetition_filter),
        "raw_feature_order": {
            "emg_cols": cfg.DB6_EMG_RAW_COLS,
            "acc_cols": cfg.DB6_ACC_RAW_COLS,
            "label_col": cfg.DB6_LABEL_COL,
            "repetition_col": cfg.DB6_REPETITION_COL,
        },
        "signal_preprocessing": {
            "enabled": cfg.ENABLE_SIGNAL_PREPROCESSING,
            "fs": cfg.RAW_SAMPLING_RATE,
            "emg_remove_mean": cfg.EMG_REMOVE_MEAN,
            "emg_notch": cfg.EMG_NOTCH,
            "emg_notch_freq": cfg.EMG_NOTCH_FREQ,
            "emg_bandpass": cfg.EMG_BANDPASS,
            "emg_bandpass_low": cfg.EMG_BANDPASS_LOW,
            "emg_bandpass_high": cfg.EMG_BANDPASS_HIGH,
            "emg_rectify": cfg.EMG_RECTIFY,
            "emg_envelope": cfg.EMG_ENVELOPE,
            "emg_envelope_cutoff": cfg.EMG_ENVELOPE_CUTOFF,
            "acc_lowpass": cfg.ACC_LOWPASS,
            "acc_lowpass_cutoff": cfg.ACC_LOWPASS_CUTOFF,
            "scipy_available": signal is not None,
            "saved_preprocessed_csv": saved_preprocessed_path,
            "saved_preprocessed_plots": saved_plot_paths,
        },
        "model_input_order": "ACC first, EMG second",
    }

    return df, info


def load_split_csvs(csv_paths, split_name, repetition_filter=None, plot_dir=None):
    resolved_paths = resolve_many_paths(csv_paths)

    all_dfs = []
    file_infos = []
    segment_offset = 0

    for path in resolved_paths:
        df, info = load_one_csv(
            path,
            split_name=f"{split_name} | {Path(path).name}",
            repetition_filter=repetition_filter,
            plot_dir=plot_dir
        )

        df = df.copy()
        df["segment_id"] = df["segment_id"] + segment_offset
        segment_offset = int(df["segment_id"].max())

        all_dfs.append(df)
        file_infos.append(info)

    split_df = pd.concat(
        all_dfs,
        axis=0,
        ignore_index=True
    )

    combined_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(split_df["raw_label"].values, return_counts=True))
    }

    info = {
        "split": split_name,
        "num_files": len(resolved_paths),
        "paths": [str(path) for path in resolved_paths],
        "combined_shape": list(split_df.shape),
        "combined_raw_label_counts": combined_counts,
        "files": file_infos,
    }

    print("\n" + "=" * 70)
    print(f"{split_name} combined")
    print("=" * 70)
    print("Files:", len(resolved_paths))
    print("Combined shape:", split_df.shape)

    return split_df, info


# =====================================================================
# 4. 读取某一个受试者
# =====================================================================

def load_subject_data(subject_cfg, output_dir=None):
    print("\n" + "#" * 70)
    print(f"Loading subject: {subject_cfg['name']}")
    print("#" * 70)

    train_df, train_info = load_split_csvs(
        subject_cfg["train"],
        split_name=f"{subject_cfg['name']} TRAIN",
        repetition_filter=subject_cfg.get("train_repetitions"),
        plot_dir=output_dir
    )

    val_df, val_info = load_split_csvs(
        subject_cfg["val"],
        split_name=f"{subject_cfg['name']} VAL",
        repetition_filter=subject_cfg.get("val_repetitions"),
        plot_dir=output_dir
    )

    test_df, test_info = load_split_csvs(
        subject_cfg["test"],
        split_name=f"{subject_cfg['name']} TEST",
        repetition_filter=subject_cfg.get("test_repetitions"),
        plot_dir=output_dir
    )

    info = {
        "train": train_info,
        "val": val_info,
        "test": test_info,
    }

    return train_df, val_df, test_df, info


# =====================================================================
# 5. 只用 train mean/std 标准化
# =====================================================================

def normalize_by_train(train_df, val_df, test_df):
    if not cfg.NORMALIZE_BY_TRAIN:
        print("\nSkip normalization because NORMALIZE_BY_TRAIN=False")
        info = {
            "normalized": False
        }
        return train_df, val_df, test_df, info

    print("\n" + "=" * 70)
    print("Normalization by this subject TRAIN mean/std")
    print("=" * 70)

    # 如果删除 rest，则只用 train 中非 rest 的行计算 mean/std
    if cfg.REMOVE_REST:
        norm_df = train_df[
            train_df["raw_label"] != cfg.REST_LABEL
        ].copy()
    else:
        norm_df = train_df.copy()

    train_x = norm_df.iloc[
        :,
        cfg.FEATURE_START_COL:cfg.FEATURE_END_COL
    ].values.astype(np.float32)

    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)

    std[std == 0] = 1.0

    print("Mean shape:", mean.shape)
    print("Std shape :", std.shape)

    def normalize_df(df):
        df = df.copy()

        x = df.iloc[
            :,
            cfg.FEATURE_START_COL:cfg.FEATURE_END_COL
        ].values.astype(np.float32)

        x = (x - mean) / std

        df.iloc[
            :,
            cfg.FEATURE_START_COL:cfg.FEATURE_END_COL
        ] = x

        return df

    train_df = normalize_df(train_df)
    val_df = normalize_df(val_df)
    test_df = normalize_df(test_df)

    info = {
        "normalized": True,
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
    }

    return train_df, val_df, test_df, info


# =====================================================================
# 6. 由 one-hot target 转训练标签
# =====================================================================

def convert_target_to_train_label(target_onehot):
    """
    target_onehot: [N, RAW_NUM_CLASSES]

    输出:
        原始标签 0/1/3/4/6/9/10/11 -> 训练标签 0~6 或 0~7。
        REMOVE_REST=True 时删除 rest，并用 TRAIN_LABEL_TO_INDEX 做非连续标签映射。
    """

    raw_labels = decode_onehot_labels(target_onehot)

    if cfg.REMOVE_REST:
        keep_mask = raw_labels != cfg.REST_LABEL
        raw_labels = raw_labels[keep_mask]
    else:
        keep_mask = np.ones_like(raw_labels, dtype=bool)

    missing = sorted(set(raw_labels.tolist()) - set(cfg.TRAIN_LABEL_TO_INDEX.keys()))

    if len(missing) > 0:
        raise ValueError(
            f"存在不能映射为训练类别的标签: {missing}。"
            f"请检查 REMOVE_REST 或 config.TRAIN_RAW_LABELS。"
        )

    train_labels = np.array(
        [cfg.TRAIN_LABEL_TO_INDEX[int(label)] for label in raw_labels],
        dtype=np.int64
    )

    return train_labels, keep_mask


# =====================================================================
# 7. 用你的 sequence_generator 生成窗口
# =====================================================================

def make_windows_from_df(
    df,
    split_name,
    augment=None,
    Eaugment=None
):
    """
    每个窗口只在同一个 segment_id 内生成。

    训练集:
        augment=1, Eaugment=1

    验证集/测试集:
        augment=None, Eaugment=None
    """

    all_x = []
    all_y = []

    grouped = df.groupby(
        "segment_id",
        sort=False
    )

    for seg_id, seg_df in grouped:
        raw_label = int(seg_df["raw_label"].iloc[0])

        # 删除 rest
        if cfg.REMOVE_REST and raw_label == cfg.REST_LABEL:
            continue

        if len(seg_df) <= cfg.WINDOW_SIZE + cfg.D_OFFSET:
            continue

        # 传入 sequence_generator 的数据必须只包含 0~LABEL_END_COL-1 列
        # 否则 W=LABEL_START_COL 的标签位置会乱
        seg_data = seg_df[cfg.RAW_COLUMNS].copy()

        x_seg, target_onehot = sequence_generator(
            data=seg_data,
            L=cfg.WINDOW_SIZE,
            CH0=cfg.CH0,
            CH1=cfg.CH1,
            CH2=cfg.CH2,
            CH3=cfg.CH3,
            W=cfg.LABEL_START_COL,
            D=cfg.D_OFFSET,
            O=cfg.RAW_NUM_CLASSES,
            c=cfg.C_CHANNELS,
            e=cfg.E_CHANNELS,
            augment=augment,
            Eaugment=Eaugment
        )

        y_seg, keep_mask = convert_target_to_train_label(
            target_onehot
        )

        x_seg = x_seg[keep_mask]

        if len(x_seg) == 0:
            continue

        all_x.append(
            x_seg.astype(np.float32)
        )

        all_y.append(
            y_seg.astype(np.int64)
        )

    if len(all_x) == 0:
        raise ValueError(
            f"{split_name} 没有生成任何窗口，请检查 WINDOW_SIZE、repetition 划分或数据。"
        )

    windows = np.concatenate(
        all_x,
        axis=0
    ).astype(np.float32)

    labels = np.concatenate(
        all_y,
        axis=0
    ).astype(np.int64)

    print("\n" + "=" * 70)
    print(f"{split_name} window generation")
    print("=" * 70)
    print("Augment:", augment)
    print("Eaugment:", Eaugment)
    print(f"{split_name} windows:", windows.shape)
    print(f"{split_name} labels :", labels.shape)

    print(f"\n{split_name} window label counts:")

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    label_counts = {}

    for label, count in zip(unique_labels, counts):
        label_counts[int(label)] = int(count)
        original_label = cfg.INDEX_TO_TRAIN_LABEL[int(label)]

        print(
            f"Train label {label:2d} "
            f"(Original label {original_label:2d}): {count}"
        )

    return windows, labels, label_counts


# =====================================================================
# 8. 准备某一个受试者的数据
# =====================================================================

def prepare_subject_data(subject_cfg, output_dir=None):
    train_df, val_df, test_df, csv_info = load_subject_data(
        subject_cfg,
        output_dir=output_dir
    )

    train_df, val_df, test_df, norm_info = normalize_by_train(
        train_df,
        val_df,
        test_df
    )

    # ================================================================
    # 训练集：做数据增强
    # ================================================================
    train_x, train_y, train_counts = make_windows_from_df(
        train_df,
        split_name=f"{subject_cfg['name']} TRAIN",
        augment=cfg.TRAIN_AUGMENT,
        Eaugment=cfg.TRAIN_EAUGMENT
    )

    # ================================================================
    # 验证集：不做数据增强
    # ================================================================
    val_x, val_y, val_counts = make_windows_from_df(
        val_df,
        split_name=f"{subject_cfg['name']} VAL",
        augment=cfg.VAL_AUGMENT,
        Eaugment=cfg.VAL_EAUGMENT
    )

    # ================================================================
    # 测试集：不做数据增强
    # ================================================================
    test_x, test_y, test_counts = make_windows_from_df(
        test_df,
        split_name=f"{subject_cfg['name']} TEST",
        augment=cfg.TEST_AUGMENT,
        Eaugment=cfg.TEST_EAUGMENT
    )

    info = {
        "subject": subject_cfg["name"],
        "csv_info": csv_info,
        "normalization": norm_info,
        "augmentation": {
            "train_augment": cfg.TRAIN_AUGMENT,
            "train_Eaugment": cfg.TRAIN_EAUGMENT,
            "val_augment": cfg.VAL_AUGMENT,
            "val_Eaugment": cfg.VAL_EAUGMENT,
            "test_augment": cfg.TEST_AUGMENT,
            "test_Eaugment": cfg.TEST_EAUGMENT,
        },
        "window_info": {
            "window_size": cfg.WINDOW_SIZE,
            "D_OFFSET": cfg.D_OFFSET,
            "train_windows": int(len(train_x)),
            "val_windows": int(len(val_x)),
            "test_windows": int(len(test_x)),
            "train_label_counts": train_counts,
            "val_label_counts": val_counts,
            "test_label_counts": test_counts,
        },
        "channel_info": {
            "RAW_DB6_EMG_COLS": cfg.DB6_EMG_RAW_COLS,
            "RAW_DB6_ACC_COLS": cfg.DB6_ACC_RAW_COLS,
            "ACC_CHANNELS_AFTER_REORDER": cfg.ACC_CHANNELS,
            "EMG_CHANNELS_AFTER_REORDER": cfg.EMG_CHANNELS,
            "sequence_generator_input_order": "ACC first, EMG second",
            "attention_blocks": ["S_ee", "S_ea", "S_ae", "S_aa"],
        },
        "label_info": {
            "raw_labels": cfg.RAW_LABELS,
            "train_raw_labels": cfg.TRAIN_RAW_LABELS,
            "train_label_to_original_label": cfg.INDEX_TO_TRAIN_LABEL,
        },
    }

    if output_dir is not None:
        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        info_path = output_dir / f"{subject_cfg['name']}_preprocessing.json"

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(
                info,
                f,
                ensure_ascii=False,
                indent=2
            )

    print("\n" + "=" * 70)
    print(f"{subject_cfg['name']} final prepared data")
    print("=" * 70)

    print("Train X:", train_x.shape)
    print("Train y:", train_y.shape)
    print("Val X  :", val_x.shape)
    print("Val y  :", val_y.shape)
    print("Test X :", test_x.shape)
    print("Test y :", test_y.shape)

    return (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        info
    )

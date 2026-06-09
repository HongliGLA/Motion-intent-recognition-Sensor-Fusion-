import numpy as np
import mne
import pandas as pd
# pip install mne pyqt5 matplotlib pandas
# ================= 1. 准备数据 =================
# 这里我们生成一段模拟的 EMG 数据。
# 实际应用中，你可以用 pd.read_csv() 读取你的数据
sfreq = 1000  # 采样率：1000Hz
duration = 10 # 总时长：10秒
times = np.linspace(0, duration, sfreq * duration)
emg_data = np.random.normal(0, 0.1, len(times))
emg_data[2000:3000] += np.random.normal(0, 1.5, 1000) # 模拟第一次肌肉收缩
emg_data[6000:7500] += np.random.normal(0, 1.2, 1500) # 模拟第二次肌肉收缩

# MNE 要求的输入数据形状必须是 (通道数, 采样点数)
# 我们只有1个通道的 EMG，所以使用 reshape(1, -1)
data = emg_data.reshape(1, -1)

# ================= 2. 构建 MNE 数据结构 =================
# 创建 Info 对象 (记录元数据：通道名、采样率、通道类型)
info = mne.create_info(ch_names=['My_EMG_Channel'], sfreq=sfreq, ch_types=['emg'])

# 创建 Raw 对象
raw = mne.io.RawArray(data, info)

# ================= 3. 启动交互式标注界面 =================
print("请在弹出的波形图中操作。")
print("快捷键提示：按 'a' 调出标注面板。")

# 调用 plot。
# block=True 非常重要，它会让代码暂停运行，直到你关闭绘图窗口。
fig = raw.plot(duration=10, 
               title='EMG 标注器 (按 "a" 开启打标签)', 
               scalings={'emg': 5}, # 调整纵轴缩放比例，让波形更明显
               block=True)

# ================= 4. 导出并保存标签 =================
# 当你关闭绘图窗口后，代码会继续往下执行
annotations = raw.annotations

if len(annotations) > 0:
    # 将 MNE 的标注对象直接转换为 Pandas DataFrame
    df_labels = annotations.to_data_frame()
    # MNE 默认的列是 onset(起点时间), duration(持续时间), description(标签名)
    # 我们可以计算出 offset(终点时间)
    df_labels['offset'] = df_labels['onset'] + pd.to_timedelta(df_labels['duration'], unit='s')
    
    # 调整并只保留我们关心的列
    final_df = df_labels[['onset', 'offset', 'description']]
    
    # 保存为 CSV
    final_df.to_csv("mne_emg_labels.csv", index=False)
    print("\n太棒了！已保存以下标签：")
    print(final_df)
else:
    print("\n你没有添加任何标签就关闭了窗口。")

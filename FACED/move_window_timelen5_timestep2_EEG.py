import numpy as np
import os
import time

# ================= 配置区域 =================
# 输入: 原始 EEG 特征文件
INPUT_FILE = r"D:\桌面\DAEST_FACED_npy\ext_fea\fea_r0\_r0_f110_fea_de.npy"

# 输出: 处理后的 EEG 保存目录
OUTPUT_DIR = r"D:\桌面\Data_Processing\data\FACED\EEG_timelen5_timestep2"
SAVE_NAME = "processed_eeg_features.npy"

# 参数
N_SUBS = 123
N_VIDS = 28
SECONDS = 30
FEAT_DIM = 160
WINDOW_SIZE = 5
STRIDE = 2


def process_eeg():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: 输入文件不存在 -> {INPUT_FILE}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"[{time.strftime('%H:%M:%S')}] 开始处理 EEG 数据...")

    # 1. 加载
    raw_data = np.load(INPUT_FILE)  # (Total, 160)

    # 2. Reshape: (Subject, Video, Seconds, Feat)
    try:
        reshaped_data = raw_data.reshape(N_SUBS, N_VIDS, SECONDS, FEAT_DIM)
    except ValueError:
        print(f"❌ Reshape 失败! 数据量 {raw_data.shape} 不匹配 {N_SUBS}x{N_VIDS}x{SECONDS}")
        return

    # 3. 滑动窗口
    segments = []
    # (30 - 5) // 2 + 1 = 13
    n_windows = (SECONDS - WINDOW_SIZE) // STRIDE + 1

    for i in range(n_windows):
        start = i * STRIDE
        end = start + WINDOW_SIZE
        # 切片: (123, 28, 5, 160)
        window = reshaped_data[:, :, start:end, :]
        segments.append(window)

    # 堆叠: (123, 28, 13, 5, 160)
    stacked = np.stack(segments, axis=2)

    # 4. 展平为最终 Dataset 格式: (Total_Samples, 5, 160)
    final_data = stacked.reshape(-1, WINDOW_SIZE, FEAT_DIM)

    # 5. 保存
    save_path = os.path.join(OUTPUT_DIR, SAVE_NAME)
    np.save(save_path, final_data)

    print(f"✅ EEG 处理完成! 保存至: {save_path}")
    print(f"最终形状: {final_data.shape} (预期: {123 * 28 * 13}, 5, 160)")


if __name__ == '__main__':
    process_eeg()
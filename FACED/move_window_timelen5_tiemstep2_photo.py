import os
import numpy as np

# ================= 配置区域 =================
# 输入: CLIP 提取好的视频特征 (.npy) - 注意里面包含子文件夹
# 🔧 请修改为你的图像特征目录
INPUT_DIR = r"./features/image_features_1024_output"

# 输出: 切片后的保存路径
OUTPUT_DIR = r"./features/photo_timelen5_timestep2_1024"

# 参数设置 (必须与 EEG 保持一致)
CLIP_SECONDS = 30  # 统一截取最后 30 秒 (如果不足30秒则补零)
WINDOW_SIZE = 5
STRIDE = 2
FEATURE_DIM = 1024  # 🚨 修正：将特征维度从 768 改为 1024

# 视频顺序列表 (必须固定，确保与 EEG 对应)
VIDEO_NAMES = [
    'neg_a_1', 'neg_a_2', 'neg_a_3', 'neg_d_1', 'neg_d_2', 'neg_d_3',
    'neg_f_1', 'neg_f_2', 'neg_f_3', 'neg_s_1', 'neg_s_2', 'neg_s_3',
    'neu_1', 'neu_2', 'neu_3', 'neu_4',
    'pos_a_1', 'pos_a_2', 'pos_a_3', 'pos_i_1', 'pos_i_2', 'pos_i_3',
    'pos_j_1', 'pos_j_2', 'pos_j_3', 'pos_t_1', 'pos_t_2', 'pos_t_3'
]


def process_video_features():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print(f"开始处理视频特征... 源路径: {INPUT_DIR}")

    for vid_name in VIDEO_NAMES:
        # 🚨 关键修改 1：加上了 vid_name 作为中间的子文件夹路径
        npy_path = os.path.join(INPUT_DIR, vid_name, f"{vid_name}_features.npy")

        # 🚨 关键修改 2：使用 1024 维初始化全零矩阵
        full_timeline = np.zeros((CLIP_SECONDS, FEATURE_DIM), dtype=np.float32)

        if os.path.exists(npy_path):
            # 加载特征 (T, 1024)
            feats = np.load(npy_path)

            # 截取逻辑: 取最后 30 秒
            # 如果特征长度 > 30: 取后 30
            # 如果特征长度 < 30: 放在最后，前面补零
            t_len = feats.shape[0]
            if t_len >= CLIP_SECONDS:
                full_timeline = feats[-CLIP_SECONDS:]
            else:
                full_timeline[-t_len:] = feats
        else:
            print(f"⚠️ Warning: Missing file {npy_path}, using zeros.")

        # 2. 滑动窗口切片
        segments = []
        # (30 - 5) // 2 + 1 = 13 个窗口
        for start_idx in range(0, CLIP_SECONDS - WINDOW_SIZE + 1, STRIDE):
            end_idx = start_idx + WINDOW_SIZE
            window = full_timeline[start_idx:end_idx, :]  # (5, 1024)
            segments.append(window)

        segments = np.array(segments)  # (13, 5, 1024)

        # 3. 保存
        save_path = os.path.join(OUTPUT_DIR, f"{vid_name}_features.npy")
        np.save(save_path, segments)

        # 打印一下进度
        print(f"✅ 成功处理并保存: {vid_name} -> Shape: {segments.shape}")

    print("🎉 所有视频特征滑动切片预处理完成！")


if __name__ == "__main__":
    process_video_features()
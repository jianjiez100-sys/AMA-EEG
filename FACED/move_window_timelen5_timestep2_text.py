import os
import numpy as np

# ================= 配置区域 =================
# 输入: CLIP 提取好的文本特征 (1024维)
# 🔧 请修改为你的文本特征目录
INPUT_DIR = r"./features/text_features_1024_output"

# 输出: 切片后的保存路径
OUTPUT_DIR = r"./features/text_timelen5_timestep2_1024"

# 参数设置
CLIP_SECONDS = 30
WINDOW_SIZE = 5
STRIDE = 2
FEATURE_DIM = 1024  # 已修改为 1024，匹配 Giant CLIP 模型特征维度

# 视频列表
VIDEO_NAMES = [
    'neg_a_1', 'neg_a_2', 'neg_a_3', 'neg_d_1', 'neg_d_2', 'neg_d_3',
    'neg_f_1', 'neg_f_2', 'neg_f_3', 'neg_s_1', 'neg_s_2', 'neg_s_3',
    'neu_1', 'neu_2', 'neu_3', 'neu_4',
    'pos_a_1', 'pos_a_2', 'pos_a_3', 'pos_i_1', 'pos_i_2', 'pos_i_3',
    'pos_j_1', 'pos_j_2', 'pos_j_3', 'pos_t_1', 'pos_t_2', 'pos_t_3'
]

def process_text_features():
    # 检查并创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    print(f"开始处理文本特征切片... 源路径: {INPUT_DIR}")

    for vid_name in VIDEO_NAMES:
        npy_path = os.path.join(INPUT_DIR, f"{vid_name}_features.npy")

        # 1. 初始化 (30, 1024) 全零矩阵
        full_timeline = np.zeros((CLIP_SECONDS, FEATURE_DIM), dtype=np.float32)

        if os.path.exists(npy_path):
            feats = np.load(npy_path)
            t_len = feats.shape[0]

            # 维度校验
            if feats.shape[1] != FEATURE_DIM:
                print(f"⚠️ Error: {vid_name} 维度不匹配! 实际: {feats.shape[1]}, 预期: {FEATURE_DIM}")
                continue

            # 对齐逻辑: 取最后 30 秒 (与视频特征对齐)
            if t_len >= CLIP_SECONDS:
                full_timeline = feats[-CLIP_SECONDS:]
            else:
                full_timeline[-t_len:] = feats
        else:
            print(f"⚠️ Warning: Missing file {vid_name}_features.npy, using zeros.")

        # 2. 滑动窗口切片逻辑
        segments = []
        # range(start, stop, step)
        # 0, 2, 4, ..., 24 (共13个点)
        for start_idx in range(0, CLIP_SECONDS - WINDOW_SIZE + 1, STRIDE):
            end_idx = start_idx + WINDOW_SIZE
            window = full_timeline[start_idx:end_idx, :]
            segments.append(window)

        segments = np.array(segments)  # 最终形状: (13, 5, 1024)

        # 3. 保存
        save_path = os.path.join(OUTPUT_DIR, f"{vid_name}_features.npy")
        np.save(save_path, segments)
        print(f"Processed {vid_name}: shape {segments.shape}")

    print("\n✅ 1024维文本特征切片预处理完成！")

if __name__ == "__main__":
    process_text_features()
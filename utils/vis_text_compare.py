"""
对比原始文本 vs 对齐后文本 vs 图像的特征空间分布
用法: python vis_text_compare.py
"""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# ==================== 配置 ====================
# 🔧 请修改为你的特征文件目录
TEXT_ORIG_DIR = "../features/text_timelen5_timestep2_1024_match_image"
TEXT_ALIGN_DIR = "../features/text_timelen5_timestep2_1024_match_image_aligned"
IMAGE_DIR = "../features/image_features_clip_vit_centercrop_timelen5_timestep2"
OUT_PATH = "./feature_space_text_compare.png"

EMOTION_MAP = {
    "neg_a": "Anger", "neg_d": "Disgust", "neg_f": "Fear", "neg_s": "Sadness",
    "neu": "Neutral",
    "pos_a": "Amusement", "pos_i": "Inspiration", "pos_j": "Joy", "pos_t": "Tenderness",
}

EMOTION_COLORS = {
    "Anger": "#E53935", "Disgust": "#8E24AA", "Fear": "#FB8C00",
    "Sadness": "#1E88E5", "Neutral": "#757575",
    "Amusement": "#43A047", "Inspiration": "#00ACC1", "Joy": "#FDD835", "Tenderness": "#EC407A",
}

N_SAMPLES_MAX = 2000


def load_features(feat_dir, is_text_5d=False):
    feats, labels = [], []
    for fname in sorted(os.listdir(feat_dir)):
        if not fname.endswith(".npy"):
            continue
        prefix = fname.split("_")[0] if fname.startswith("neu") else "_".join(fname.split("_")[:2])
        emotion = EMOTION_MAP.get(prefix, None)
        if emotion is None:
            continue

        data = np.load(os.path.join(feat_dir, fname))
        if is_text_5d and data.ndim == 3:
            data = data.mean(axis=1)
        data = data.reshape(-1, data.shape[-1])

        # L2 归一化
        data = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-8)

        feats.append(data)
        labels.extend([emotion] * len(data))

    feats = np.concatenate(feats, axis=0)
    if len(feats) > N_SAMPLES_MAX:
        idx = np.random.RandomState(7).choice(len(feats), N_SAMPLES_MAX, replace=False)
        feats = feats[idx]
        labels = [labels[i] for i in idx]
    return feats, labels


# ==================== 加载 ====================
print("Loading original text...")
t_orig, l_orig = load_features(TEXT_ORIG_DIR, is_text_5d=True)
print(f"  Original text: {t_orig.shape}")

print("Loading aligned text...")
t_align, l_align = load_features(TEXT_ALIGN_DIR)
print(f"  Aligned text:  {t_align.shape}")

print("Loading image...")
i_feat, l_img = load_features(IMAGE_DIR)
print(f"  Image:         {i_feat.shape}")

# ==================== t-SNE ====================
all_feats = np.concatenate([t_orig, t_align, i_feat], axis=0)
all_labels = l_orig + l_align + l_img
n_orig, n_align, n_img = len(t_orig), len(t_align), len(i_feat)
print(f"Running t-SNE on {all_feats.shape[0]} samples...")

tsne = TSNE(n_components=2, perplexity=50, random_state=7, n_jobs=-1, verbose=1)
all_2d = tsne.fit_transform(all_feats)

xy_orig = all_2d[:n_orig]
xy_align = all_2d[n_orig:n_orig + n_align]
xy_img = all_2d[n_orig + n_align:]
print("t-SNE done.")

# ==================== 绘图 ====================
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# ---- 图1: 原始文本 vs 图像 ----
ax = axes[0]
ax.scatter(xy_img[:, 0], xy_img[:, 1], c="#FF5722", label="Image", alpha=0.3, s=6, rasterized=True)
ax.scatter(xy_orig[:, 0], xy_orig[:, 1], c="#2196F3", label="Orig Text", alpha=0.4, s=8, rasterized=True)
ax.set_title("Original Text vs Image", fontsize=14, fontweight="bold")
ax.legend(markerscale=3, fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

# ---- 图2: 对齐文本 vs 图像 ----
ax = axes[1]
ax.scatter(xy_img[:, 0], xy_img[:, 1], c="#FF5722", label="Image", alpha=0.3, s=6, rasterized=True)
ax.scatter(xy_align[:, 0], xy_align[:, 1], c="#4CAF50", label="Aligned Text", alpha=0.4, s=8, rasterized=True)
ax.set_title("Aligned Text vs Image", fontsize=14, fontweight="bold")
ax.legend(markerscale=3, fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

# ---- 图3: 原始文本 vs 对齐文本（同图对比）----
ax = axes[2]
ax.scatter(xy_orig[:, 0], xy_orig[:, 1], c="#2196F3", label="Orig Text", alpha=0.3, s=6, rasterized=True)
ax.scatter(xy_align[:, 0], xy_align[:, 1], c="#4CAF50", label="Aligned Text", alpha=0.6, s=8, rasterized=True)
# 画箭头：从原始指向对齐（只画前100个点）
step = max(1, n_orig // 100)
for j in range(0, n_orig, step):
    ax.annotate("", xy=xy_align[j], xytext=xy_orig[j],
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.3, alpha=0.3))
ax.set_title("Shift: Original → Aligned Text", fontsize=14, fontweight="bold")
ax.legend(markerscale=3, fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
plt.close()

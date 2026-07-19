"""
可视化文本 vs 图像特征空间分布
用法: python vis_feature_space.py
"""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# ==================== 配置 ====================
# 🔧 请修改为你的特征文件目录
# TEXT_DIR = "../features/text_timelen5_timestep2_1024_match_image"
# TEXT_DIR = "../features/text_timelen5_timestep2_1024_objective"
TEXT_DIR = "../features/text_timelen5_timestep2_1024_match_image_aligned"
IMAGE_DIR = "../features/image_features_clip_vit_centercrop_timelen5_timestep2"

EMOTION_MAP = {
    "neg_a": "Anger", "neg_d": "Disgust", "neg_f": "Fear", "neg_s": "Sadness",
    "neu": "Neutral",
    "pos_a": "Amusement", "pos_i": "Inspiration", "pos_j": "Joy", "pos_t": "Tenderness",
}

EMOTION_COLORS = {
    "Anger":       "#E53935",
    "Disgust":     "#8E24AA",
    "Fear":        "#FB8C00",
    "Sadness":     "#1E88E5",
    "Neutral":     "#757575",
    "Amusement":   "#43A047",
    "Inspiration": "#00ACC1",
    "Joy":         "#FDD835",
    "Tenderness":  "#EC407A",
}

N_SAMPLES_MAX = 3000  # t-SNE 太慢，限制总样本数

# ==================== 加载数据 ====================
text_feats, image_feats = [], []
text_labels, image_labels = [], []

for fname in sorted(os.listdir(TEXT_DIR)):
    if not fname.endswith(".npy"):
        continue
    prefix = "_".join(fname.split("_")[:2]) if fname.startswith("neg") or fname.startswith("pos") else fname.split("_")[0]
    # 处理中立: neu → neu
    emotion = EMOTION_MAP.get(prefix, None)
    if emotion is None:
        continue

    # 文本: (N, 5, 1024) → 取时间维平均 → (N, 1024)
    t = np.load(os.path.join(TEXT_DIR, fname))
    if t.ndim == 3:
        t = t.mean(axis=1)  # (N, 5, 1024) → (N, 1024)

    # 图像: (N, 1024)
    i = np.load(os.path.join(IMAGE_DIR, fname))
    if i.ndim == 3:
        i = i.mean(axis=1)
    i = i.reshape(-1, i.shape[-1])

    # L2 归一化（投影到超球面，t-SNE 对余弦距离更友好）
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)
    i = i / (np.linalg.norm(i, axis=1, keepdims=True) + 1e-8)

    text_feats.append(t)
    image_feats.append(i)
    text_labels.extend([emotion] * len(t))
    image_labels.extend([emotion] * len(i))

text_all = np.concatenate(text_feats, axis=0)
image_all = np.concatenate(image_feats, axis=0)
print(f"Text: {text_all.shape}, Image: {image_all.shape}")

# 统一降采样
if len(text_all) > N_SAMPLES_MAX:
    idx = np.random.choice(len(text_all), N_SAMPLES_MAX, replace=False)
    text_all = text_all[idx]
    text_labels = [text_labels[i] for i in idx]
if len(image_all) > N_SAMPLES_MAX:
    idx = np.random.choice(len(image_all), N_SAMPLES_MAX, replace=False)
    image_all = image_all[idx]
    image_labels = [image_labels[i] for i in idx]

# ==================== t-SNE ====================
all_feats = np.concatenate([text_all, image_all], axis=0)
modality = ["Text"] * len(text_all) + ["Image"] * len(image_all)
all_labels = text_labels + image_labels

print(f"Running t-SNE on {all_feats.shape[0]} samples...")
tsne = TSNE(n_components=2, perplexity=50, random_state=7, n_jobs=-1, verbose=1)
all_2d = tsne.fit_transform(all_feats)

text_2d = all_2d[:len(text_all)]
image_2d = all_2d[len(text_all):]

# ==================== 绘图 ====================
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# ---- 图1: 按模态着色 (Text vs Image) ----
ax = axes[0]
ax.scatter(text_2d[:, 0], text_2d[:, 1], c="#2196F3", label="Text", alpha=0.4, s=8, rasterized=True)
ax.scatter(image_2d[:, 0], image_2d[:, 1], c="#FF5722", label="Image", alpha=0.4, s=8, rasterized=True)
ax.set_title("Feature Space by Modality", fontsize=14, fontweight="bold")
ax.legend(markerscale=3, fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

# ---- 图2: 按情感类别着色 (Text) ----
ax = axes[1]
for emo in sorted(EMOTION_COLORS.keys()):
    mask = [l == emo for l in text_labels]
    if sum(mask) > 0:
        ax.scatter(text_2d[mask, 0], text_2d[mask, 1], c=EMOTION_COLORS[emo],
                   label=emo, alpha=0.5, s=8, rasterized=True)
ax.set_title("Text Feature Space (colored by emotion)", fontsize=14, fontweight="bold")
ax.legend(markerscale=3, fontsize=8, ncol=2)
ax.set_xticks([]); ax.set_yticks([])

# ---- 图3: 按情感类别着色 (Image) ----
ax = axes[2]
for emo in sorted(EMOTION_COLORS.keys()):
    mask = [l == emo for l in image_labels]
    if sum(mask) > 0:
        ax.scatter(image_2d[mask, 0], image_2d[mask, 1], c=EMOTION_COLORS[emo],
                   label=emo, alpha=0.5, s=8, rasterized=True)
ax.set_title("Image Feature Space (colored by emotion)", fontsize=14, fontweight="bold")
ax.legend(markerscale=3, fontsize=8, ncol=2)
ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
out_path = "./feature_space_vis.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()

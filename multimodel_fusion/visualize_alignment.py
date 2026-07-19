"""
对比 Text-Image 对齐前后的特征空间分布
========================================
对原始 CLIP 特征和 multimodel_fusion 投影后的特征分别做 t-SNE，
生成 5 张高质量对比图 + 定量指标报告。

用法:
    python multimodel_fusion/visualize_alignment.py
"""

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import sys

# ==================== 配置 (按需修改) ====================
# 🔧 请修改为你的特征文件路径
TEXT_DIR = "../../features/text_timelen5_timestep2_1024_match_image"
IMAGE_DIR = "../../features/image_features_clip_vit_centercrop_timelen5_timestep2"

# 投影后的特征目录 (同目录下的 projected_text, projected_image)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_TEXT_DIR = os.path.join(BASE_DIR, "projected_text")
PROJ_IMAGE_DIR = os.path.join(BASE_DIR, "projected_image")
OUT_DIR = os.path.join(BASE_DIR, "figures")

N_SAMPLES_MAX = 2000  # t-SNE 最大采样数
PERPLEXITY = 50
RANDOM_SEED = 7

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

os.makedirs(OUT_DIR, exist_ok=True)


# ==================== 工具函数 ====================

def get_emotion(fname):
    """从文件名提取情绪类别"""
    if fname.startswith("neu"):
        prefix = "neu"
    else:
        prefix = "_".join(fname.split("_")[:2])
    return EMOTION_MAP.get(prefix, None)


def load_and_prepare(feat_dir, max_samples=N_SAMPLES_MAX):
    """加载目录下所有 .npy 文件, 返回 (features, labels)"""
    feats, labels = [], []
    for fname in sorted(os.listdir(feat_dir)):
        if not fname.endswith(".npy"):
            continue
        emotion = get_emotion(fname)
        if emotion is None:
            continue
        data = np.load(os.path.join(feat_dir, fname))
        if data.ndim == 3:
            data = data.mean(axis=1)          # (n_samples, seq_len, dim) → (n_samples, dim)
        data = data.reshape(-1, data.shape[-1])
        # L2 归一化, 使余弦相似度等同于内积
        data = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-8)
        feats.append(data)
        labels.extend([emotion] * len(data))

    feats = np.concatenate(feats, axis=0)
    if len(feats) > max_samples:
        idx = np.random.RandomState(RANDOM_SEED).choice(len(feats), max_samples, replace=False)
        feats = feats[idx]
        labels = [labels[i] for i in idx]
    return feats, labels


def safe_silhouette(feats, labels, max_n=2000):
    """安全计算 Silhouette Score (防止内存溢出)"""
    if len(np.unique(labels)) < 2:
        return 0.0
    feats_np = np.array(feats)
    if len(feats_np) > max_n:
        idx = np.random.RandomState(RANDOM_SEED).choice(len(feats_np), max_n, replace=False)
        feats_np = feats_np[idx]
        labels = [labels[i] for i in idx]
    return silhouette_score(feats_np, labels)


def cross_modal_gap(feats_a, labels_a, feats_b, labels_b, max_n=500):
    """
    跨模态重叠度: 计算同类跨模态余弦相似度 vs 异类跨模态余弦相似度。
    返回 (同类均值, 异类均值, gap=同类-异类)。
    gap 越大说明模态间按类别对齐越好。
    """
    if len(feats_a) > max_n:
        idx = np.random.RandomState(RANDOM_SEED).choice(len(feats_a), max_n, replace=False)
        feats_a = feats_a[idx]; labels_a = [labels_a[i] for i in idx]
    if len(feats_b) > max_n:
        idx = np.random.RandomState(RANDOM_SEED).choice(len(feats_b), max_n, replace=False)
        feats_b = feats_b[idx]; labels_b = [labels_b[i] for i in idx]
    feats_a = feats_a / (np.linalg.norm(feats_a, axis=1, keepdims=True) + 1e-8)
    feats_b = feats_b / (np.linalg.norm(feats_b, axis=1, keepdims=True) + 1e-8)
    sim = feats_a @ feats_b.T
    labels_a = np.array(labels_a); labels_b = np.array(labels_b)
    same_mask = labels_a[:, None] == labels_b[None, :]
    diff_mask = ~same_mask
    same_sim = sim[same_mask].mean() if same_mask.any() else 0.0
    diff_sim = sim[diff_mask].mean() if diff_mask.any() else 0.0
    return same_sim, diff_sim, same_sim - diff_sim


def save_fig(fig, name, dpi=250):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor='white', edgecolor='none')
    print(f"  ✓ {name}")
    plt.close(fig)


# ==================== 全局样式 ====================
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})

EMO_ORDER = sorted(EMOTION_COLORS.keys())


# ==================== 加载数据 ====================
print("=" * 60)
print("📂 Loading features...")
print(f"   Original text:  {TEXT_DIR}")
print(f"   Original image: {IMAGE_DIR}")
print(f"   Projected text: {PROJ_TEXT_DIR}")
print(f"   Projected image:{PROJ_IMAGE_DIR}")

# 检查投影后目录是否存在
if not os.path.isdir(PROJ_TEXT_DIR) or not os.path.isdir(PROJ_IMAGE_DIR):
    print("\n⚠️  Projected directories not found. Run train_text_image_alignment.py first!")
    print(f"   Expected: {PROJ_TEXT_DIR}")
    print(f"   Expected: {PROJ_IMAGE_DIR}")
    sys.exit(1)

t_orig, l_torig = load_and_prepare(TEXT_DIR)
i_orig, l_iorig = load_and_prepare(IMAGE_DIR)
t_proj, l_tproj = load_and_prepare(PROJ_TEXT_DIR)
i_proj, l_iproj = load_and_prepare(PROJ_IMAGE_DIR)

print(f"\n   Original text:   {t_orig.shape[0]} samples")
print(f"   Original image:  {i_orig.shape[0]} samples")
print(f"   Projected text:  {t_proj.shape[0]} samples")
print(f"   Projected image: {i_proj.shape[0]} samples")

# ==================== 联合 t-SNE ====================
# 将所有 4 组特征拼在一起做 t-SNE，保证坐标可直接比较
all_feats = np.concatenate([t_orig, i_orig, t_proj, i_proj], axis=0)
n_to, n_io = len(t_orig), len(i_orig)
n_tp, n_ip = len(t_proj), len(i_proj)

print(f"\n🎨 Running t-SNE on {all_feats.shape[0]} samples (perplexity={PERPLEXITY})...")
tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=RANDOM_SEED, n_jobs=-1, verbose=1)
all_2d = tsne.fit_transform(all_feats)

# 切分回各组
xy_to = all_2d[:n_to]
xy_io = all_2d[n_to:n_to + n_io]
xy_tp = all_2d[n_to + n_io:n_to + n_io + n_tp]
xy_ip = all_2d[n_to + n_io + n_tp:]

# ==================== 定量指标 ====================
print("\n📊 Computing metrics...")
sil_t_orig = safe_silhouette(t_orig, l_torig)
sil_t_proj = safe_silhouette(t_proj, l_tproj)
sil_i_orig = safe_silhouette(i_orig, l_iorig)
sil_i_proj = safe_silhouette(i_proj, l_iproj)

same_before, diff_before, gap_before = cross_modal_gap(t_orig, l_torig, i_orig, l_iorig)
same_after,  diff_after,  gap_after  = cross_modal_gap(t_proj, l_tproj, i_proj, l_iproj)

# Text-Image 直接模态重叠度 (同类配对相似度)
text_image_sim_before = np.diag(
    (t_orig / (np.linalg.norm(t_orig, axis=1, keepdims=True) + 1e-8)) @
    (i_orig / (np.linalg.norm(i_orig, axis=1, keepdims=True) + 1e-8)).T
).mean()
text_image_sim_after = np.diag(
    (t_proj / (np.linalg.norm(t_proj, axis=1, keepdims=True) + 1e-8)) @
    (i_proj / (np.linalg.norm(i_proj, axis=1, keepdims=True) + 1e-8)).T
).mean()

print(f"\n{'='*55}")
print(f"📋 定量指标汇总")
print(f"{'─'*55}")
print(f"  Silhouette (类内紧凑/类间分散):")
print(f"    Text:   {sil_t_orig:.4f} → {sil_t_proj:.4f}  ({'↑' if sil_t_proj > sil_t_orig else '↓'}{abs(sil_t_proj - sil_t_orig):.4f})")
print(f"    Image:  {sil_i_orig:.4f} → {sil_i_proj:.4f}  ({'↑' if sil_i_proj > sil_i_orig else '↓'}{abs(sil_i_proj - sil_i_orig):.4f})")
print(f"  Cross-modal Gap (同类-异类余弦相似度, 越大越好):")
print(f"    Before: {gap_before:.4f}  (same={same_before:.4f}, diff={diff_before:.4f})")
print(f"    After:  {gap_after:.4f}  (same={same_after:.4f}, diff={diff_after:.4f})")
print(f"  Diagonal Cosine (配对 text_i↔image_i 直接相似度):")
print(f"    Before: {text_image_sim_before:.4f}")
print(f"    After:  {text_image_sim_after:.4f}")
print(f"{'='*55}")


# ================================================================
#  图1 — Before: 按模态着色 (Text vs Image)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(xy_io[:, 0], xy_io[:, 1], c="#FF5722", label="Image", alpha=0.22, s=6, rasterized=True)
ax.scatter(xy_to[:, 0], xy_to[:, 1], c="#2196F3", label="Text",  alpha=0.32, s=8, rasterized=True)
ax.set_title("Before Alignment — by Modality")
ax.legend(loc="upper right", markerscale=4, framealpha=0.85)
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
ax.grid(True, alpha=0.12, linestyle='--')
ax.text(0.5, -0.04,
        f"Sil: Text={sil_t_orig:.3f}  Image={sil_i_orig:.3f}  "
        f"Diagonal Cos={text_image_sim_before:.3f}  Gap={gap_before:.3f}",
        transform=ax.transAxes, ha='center', fontsize=10, color='#555')
save_fig(fig, "01_before_by_modality.png")

# ================================================================
#  图2 — After: 按模态着色 (Proj Text vs Proj Image)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(xy_ip[:, 0], xy_ip[:, 1], c="#FF5722", label="Proj Image", alpha=0.22, s=6, rasterized=True)
ax.scatter(xy_tp[:, 0], xy_tp[:, 1], c="#4CAF50", label="Proj Text",  alpha=0.32, s=8, rasterized=True)
ax.set_title("After Alignment — by Modality")
ax.legend(loc="upper right", markerscale=4, framealpha=0.85)
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
ax.grid(True, alpha=0.12, linestyle='--')
ax.text(0.5, -0.04,
        f"Sil: Text={sil_t_proj:.3f}  Image={sil_i_proj:.3f}  "
        f"Diagonal Cos={text_image_sim_after:.3f}  Gap={gap_after:.3f}",
        transform=ax.transAxes, ha='center', fontsize=10, color='#555')
save_fig(fig, "02_after_by_modality.png")

# ================================================================
#  图3 — Text 位移: Original → Projected (带箭头)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(xy_to[:, 0], xy_to[:, 1], c="#90CAF9", label="Orig Text", alpha=0.20, s=5, rasterized=True)
ax.scatter(xy_tp[:, 0], xy_tp[:, 1], c="#1565C0", label="Proj Text", alpha=0.40, s=9, rasterized=True)
n_arrows = min(120, n_to)
step = max(1, n_to // n_arrows)
for j in range(0, n_to, step):
    ax.annotate("", xy=xy_tp[j], xytext=xy_to[j],
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.35, alpha=0.22))
shifts_t = np.linalg.norm(xy_tp - xy_to, axis=1)
ax.set_title("Text Shift: Original → Projected")
ax.legend(loc="upper right", markerscale=4, framealpha=0.85)
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
ax.grid(True, alpha=0.12, linestyle='--')
ax.text(0.5, -0.04,
        f"Mean shift: {shifts_t.mean():.2f}  Sil: {sil_t_orig:.3f}→{sil_t_proj:.3f}",
        transform=ax.transAxes, ha='center', fontsize=10, color='#555')
save_fig(fig, "03_text_shift.png")

# ================================================================
#  图4 — Before: 按情绪类别着色 (○=Text, ▲=Image)
# ================================================================
fig, ax = plt.subplots(figsize=(14, 10))
for emo in EMO_ORDER:
    c = EMOTION_COLORS[emo]
    mt = [l == emo for l in l_torig]
    if sum(mt) > 0:
        ax.scatter(xy_to[mt, 0], xy_to[mt, 1], c=c, marker='o', s=14,
                   alpha=0.55, edgecolors='white', linewidth=0.3, rasterized=True)
    mi = [l == emo for l in l_iorig]
    if sum(mi) > 0:
        ax.scatter(xy_io[mi, 0], xy_io[mi, 1], c=c, marker='^', s=16,
                   alpha=0.60, edgecolors='white', linewidth=0.3, rasterized=True)

legend_elements = (
    [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='● Text'),
     Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='▲ Image')]
    + [Line2D([0], [0], marker='o', color='w', markerfacecolor=EMOTION_COLORS[e], markersize=8, label=e)
       for e in EMO_ORDER]
)
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
          fontsize=9, framealpha=0.9, ncol=1)
ax.set_title("Before Alignment — by Emotion (●=Text, ▲=Image)")
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
ax.grid(True, alpha=0.12, linestyle='--')
save_fig(fig, "04_before_by_emotion.png")

# ================================================================
#  图5 — After: 按情绪类别着色 (○=Proj Text, ▲=Proj Image)
# ================================================================
fig, ax = plt.subplots(figsize=(14, 10))
for emo in EMO_ORDER:
    c = EMOTION_COLORS[emo]
    mt = [l == emo for l in l_tproj]
    if sum(mt) > 0:
        ax.scatter(xy_tp[mt, 0], xy_tp[mt, 1], c=c, marker='o', s=14,
                   alpha=0.55, edgecolors='white', linewidth=0.3, rasterized=True)
    mi = [l == emo for l in l_iproj]
    if sum(mi) > 0:
        ax.scatter(xy_ip[mi, 0], xy_ip[mi, 1], c=c, marker='^', s=16,
                   alpha=0.60, edgecolors='white', linewidth=0.3, rasterized=True)

legend_elements = (
    [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='● Proj Text'),
     Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='▲ Proj Image')]
    + [Line2D([0], [0], marker='o', color='w', markerfacecolor=EMOTION_COLORS[e], markersize=8, label=e)
       for e in EMO_ORDER]
)
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
          fontsize=9, framealpha=0.9, ncol=1)
ax.set_title("After Alignment — by Emotion (●=Proj Text, ▲=Proj Image)")
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
ax.grid(True, alpha=0.12, linestyle='--')
save_fig(fig, "05_after_by_emotion.png")


# ================================================================
#  汇总
# ================================================================
print(f"\n{'='*55}")
print(f"  5 张图片已保存到: {OUT_DIR}/")
print(f"  dpi=250, 白底, 风格与 text_image_alignment/visualize_gap.py 一致")
print(f"{'='*55}")
print(f"\n  图1: 01_before_by_modality.png   — 对齐前 Text/Image 模态分布")
print(f"  图2: 02_after_by_modality.png    — 对齐后 Text/Image 模态分布")
print(f"  图3: 03_text_shift.png           — Text 位移轨迹")
print(f"  图4: 04_before_by_emotion.png    — 对齐前 按 9 类情绪着色")
print(f"  图5: 05_after_by_emotion.png     — 对齐后 按 9 类情绪着色")

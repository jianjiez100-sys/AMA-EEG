"""
Text-Image 共享空间对齐 — 纯 InfoNCE + Residual 投影头 (SEED 版)
==============================================================
损失: 仅对称双向 InfoNCE (τ=0.07), 无其他辅助 loss
投影头: 1024→1024, ResidualAdd + LayerNorm (与骨干网络一致)
全量训练, 不划分验证集

数据集: SEED (3类: negative/neutral/positive, 15个视频)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from model.models import ResidualAdd

TEXT_DIR = os.path.join(PROJECT_ROOT, "features", "SEED", "text_timelen5_timestep2_1024")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "features", "SEED", "image_features_clip_vit_centercrop_timelen5_timestep2")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
LR = 0.001
WD = 0.0001
MAX_EPOCHS = 300
PATIENCE = 50
TEMPERATURE = 0.07
SYMMETRIC = True


class ResidualProjector(nn.Module):
    """1024 → 1024, 残差连接 + LayerNorm"""
    def __init__(self, in_dim=1024, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(in_dim, in_dim),
                nn.Dropout(dropout),
            )),
            nn.LayerNorm(in_dim),
        )
    def forward(self, x):
        return self.net(x)


def info_nce_loss(text_proj, image_proj, temperature, symmetric):
    """纯对称双向 InfoNCE"""
    t = F.normalize(text_proj, dim=1)
    i = F.normalize(image_proj, dim=1)
    B = t.shape[0]
    logits_t2i = torch.matmul(t, i.T) / temperature
    labels = torch.arange(B, dtype=torch.long, device=t.device)
    loss_t2i = F.cross_entropy(logits_t2i, labels)
    if symmetric:
        logits_i2t = torch.matmul(i, t.T) / temperature
        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss = (loss_t2i + loss_i2t) / 2.0
    else:
        loss = loss_t2i
    with torch.no_grad():
        acc = (logits_t2i.argmax(dim=1) == labels).float().mean()
    return loss, acc


# ==================== SEED 文件列表 ====================
# SEED: 3个情感子文件夹, 每个子文件夹下直接是 .npy 文件
SEED_EMOTION_DIRS = {
    "negative": ["1942_1_features.npy", "1942_2_features.npy", "1942_3_features.npy",
                 "Tangshan_Earthquake_1_features.npy", "Tangshan_Earthquake_2_features.npy"],
    "neutral":  ["Huangshan_1_features.npy", "Huangshan_2_features.npy",
                 "Lijiang_1_features.npy", "Suzhou_1_features.npy", "Suzhou_2_features.npy"],
    "positive": ["Flirting_Scholar_features.npy", "Just_Another_Pandoras_Box_1_features.npy",
                 "Just_Another_Pandoras_Box_2_features.npy",
                 "Lost_in_Thailand_1_features.npy", "Lost_in_Thailand_2_features.npy"],
}
FILE_NAMES = []
for emotion, vids in SEED_EMOTION_DIRS.items():
    for vid in vids:
        FILE_NAMES.append((emotion, vid))

print(f"📂 Text:  {TEXT_DIR}")
print(f"📂 Image: {IMAGE_DIR}")

text_list, image_list = [], []
n_aligned = 0
for emotion, fname in FILE_NAMES:
    t = np.load(os.path.join(TEXT_DIR, emotion, fname))
    i = np.load(os.path.join(IMAGE_DIR, emotion, fname))
    if t.ndim == 3: t = t.mean(axis=1)
    if i.ndim == 3: i = i.mean(axis=1)
    # 取最小长度对齐 (与 SEED 主数据集一致)
    min_len = min(len(t), len(i))
    if len(t) != len(i):
        print(f"  ⚠️  Aligning {emotion}/{fname}: text={len(t)} vs image={len(i)} → {min_len}")
    text_list.append(t[:min_len].astype(np.float32))
    image_list.append(i[:min_len].astype(np.float32))
    n_aligned += min_len

text_all  = np.concatenate(text_list, axis=0)
image_all = np.concatenate(image_list, axis=0)
n_total   = text_all.shape[0]
print(f"Loaded {len(FILE_NAMES)} files (15 SEED videos), {n_total} samples (全量训练)")

def T(arr): return torch.tensor(arr).to(DEVICE)
t_all, i_all = T(text_all), T(image_all)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(t_all, i_all),
    batch_size=BATCH_SIZE, shuffle=True)

text_proj  = ResidualProjector().to(DEVICE)
image_proj = ResidualProjector().to(DEVICE)
opt = torch.optim.Adam(list(text_proj.parameters()) + list(image_proj.parameters()), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

print(f"\n🔥 ResidualProjector (1024→1024, ResidualAdd + LayerNorm)")
print(f"   Loss: 纯双向 InfoNCE (τ={TEMPERATURE})  全量训练 ({n_total} samples)")
print(f"{'='*60}")

best_loss = float('inf')
patience_cnt = 0

for epoch in range(MAX_EPOCHS):
    text_proj.train(); image_proj.train()
    tr_loss, tr_acc = 0.0, 0.0
    for t_b, i_b in loader:
        loss, acc = info_nce_loss(text_proj(t_b), image_proj(i_b), TEMPERATURE, SYMMETRIC)
        opt.zero_grad(); loss.backward(); opt.step()
        tr_loss += loss.item() * t_b.size(0); tr_acc += acc.item() * t_b.size(0)
    tr_loss /= n_total; tr_acc /= n_total; scheduler.step()

    improved = tr_loss < best_loss
    flag = " ★" if improved else ""
    if improved:
        best_loss = tr_loss; patience_cnt = 0
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save(text_proj.state_dict(), os.path.join(SAVE_DIR, "projector_text.pt"))
        torch.save(image_proj.state_dict(), os.path.join(SAVE_DIR, "projector_image.pt"))
    else:
        patience_cnt += 1

    print(f"Epoch {epoch+1:3d}{flag}  Loss={tr_loss:.4f}  Acc={tr_acc:.4f}")
    if patience_cnt >= PATIENCE: print(f"Early stopping at {epoch+1}"); break

print(f"\n📦 Projecting all features...")
text_proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "projector_text.pt"), weights_only=True))
image_proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "projector_image.pt"), weights_only=True))
text_proj.eval(); image_proj.eval()

pt_dir = os.path.join(SAVE_DIR, "projected_text")
pi_dir = os.path.join(SAVE_DIR, "projected_image")
os.makedirs(pt_dir, exist_ok=True); os.makedirs(pi_dir, exist_ok=True)

with torch.no_grad():
    for emotion, fname in FILE_NAMES:
        # 文本投影
        orig_t = np.load(os.path.join(TEXT_DIR, emotion, fname)).astype(np.float32)
        shp = orig_t.shape
        flat = orig_t.reshape(-1, shp[-1]) if orig_t.ndim == 3 else orig_t
        out_t = text_proj(torch.tensor(flat).to(DEVICE)).cpu().numpy()
        if orig_t.ndim == 3: out_t = out_t.reshape(shp[0], shp[1], -1)
        # 保持子目录结构
        os.makedirs(os.path.join(pt_dir, emotion), exist_ok=True)
        np.save(os.path.join(pt_dir, emotion, fname), out_t)

        # 图像投影
        orig_i = np.load(os.path.join(IMAGE_DIR, emotion, fname)).astype(np.float32)
        shp = orig_i.shape
        flat = orig_i.reshape(-1, shp[-1]) if orig_i.ndim == 3 else orig_i
        out_i = image_proj(torch.tensor(flat).to(DEVICE)).cpu().numpy()
        if orig_i.ndim == 3: out_i = out_i.reshape(shp[0], shp[1], -1)
        os.makedirs(os.path.join(pi_dir, emotion), exist_ok=True)
        np.save(os.path.join(pi_dir, emotion, fname), out_i)

print(f"→ {pt_dir}/  ({len(FILE_NAMES)} files, 3 emotion subdirs)")
print(f"→ {pi_dir}/  ({len(FILE_NAMES)} files, 3 emotion subdirs)")

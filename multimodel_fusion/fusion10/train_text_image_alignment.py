"""
Text ↔ Image 双向对比学习对齐 (双方各一个投影头)
=====================================================
损失: 双向 InfoNCE (text→image + image→text)
投影头: 1024→1024, ResidualAdd + LayerNorm (文本+图像各一个)
100% 数据用于训练, 无验证集.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.models import ResidualAdd

# 🔧 请修改为你的特征文件路径
TEXT_DIR = "../../features/text_timelen5_timestep2_1024_objective"
IMAGE_DIR = "../../features/image_features_clip_vit_centercrop_timelen5_timestep2"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
LR = 0.001
WD = 0.0001
MAX_EPOCHS = 300
PATIENCE = 30
TRAIN_RATIO = 1.0       # 100% 数据训练
RANDOM_SEED = 7
TEMPERATURE = 0.07
SYMMETRIC = True         # 双向: text→image + image→text


class ResidualProjector(nn.Module):
    """1024 → 1024, 残差连接 + LayerNorm (仅文本侧)"""
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


def info_nce_loss(text_proj, image_proj, temperature, symmetric=True):
    """双向 InfoNCE: text→image + image→text"""
    t = F.normalize(text_proj, dim=1)
    i = F.normalize(image_proj, dim=1)
    B = t.shape[0]
    labels = torch.arange(B, dtype=torch.long, device=t.device)

    logits_t2i = torch.matmul(t, i.T) / temperature
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


FILE_NAMES = [
    f"neg_a_{i}_features.npy" for i in range(1, 4)
] + [f"neg_d_{i}_features.npy" for i in range(1, 4)
] + [f"neg_f_{i}_features.npy" for i in range(1, 4)
] + [f"neg_s_{i}_features.npy" for i in range(1, 4)
] + [f"neu_{i}_features.npy" for i in range(1, 5)
] + [f"pos_a_{i}_features.npy" for i in range(1, 4)
] + [f"pos_i_{i}_features.npy" for i in range(1, 4)
] + [f"pos_j_{i}_features.npy" for i in range(1, 4)
] + [f"pos_t_{i}_features.npy" for i in range(1, 4)]

print(f"📂 Text:  {TEXT_DIR}")
print(f"📂 Image: {IMAGE_DIR}")

text_list, image_list = [], []
for fname in FILE_NAMES:
    t = np.load(os.path.join(TEXT_DIR, fname))
    i = np.load(os.path.join(IMAGE_DIR, fname))
    if t.ndim == 3: t = t.mean(axis=1)
    if i.ndim == 3: i = i.mean(axis=1)
    text_list.append(t.astype(np.float32))
    image_list.append(i.astype(np.float32))

text_all  = np.concatenate(text_list, axis=0)
image_all = np.concatenate(image_list, axis=0)
n_total   = text_all.shape[0]
print(f"Loaded {len(FILE_NAMES)} files, {n_total} samples")

# 100% 数据训练, 不打乱划分
t_all = torch.tensor(text_all).to(DEVICE)
i_all = torch.tensor(image_all).to(DEVICE)
ds = torch.utils.data.TensorDataset(t_all, i_all)
loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
print(f"Train: {n_total} samples (100%)")

# 双方各一个投影头
text_proj = ResidualProjector().to(DEVICE)
image_proj = ResidualProjector().to(DEVICE)
params = list(text_proj.parameters()) + list(image_proj.parameters())
opt = torch.optim.Adam(params, lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

print(f"\n🔥 Text ↔ Image 双向对比学习对齐")
print(f"   Text:  ResidualProjector (1024→1024, 2.1M)")
print(f"   Image: ResidualProjector (1024→1024, 2.1M)")
print(f"   Loss:  双向 InfoNCE (τ={TEMPERATURE})")
print(f"{'='*60}")

best_loss = float('inf')
patience_cnt = 0

for epoch in range(MAX_EPOCHS):
    text_proj.train(); image_proj.train()
    epoch_loss, epoch_acc = 0.0, 0.0
    for t_b, i_b in loader:
        loss, acc = info_nce_loss(text_proj(t_b), image_proj(i_b), TEMPERATURE, SYMMETRIC)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item() * t_b.size(0); epoch_acc += acc.item() * t_b.size(0)
    epoch_loss /= n_total; epoch_acc /= n_total; scheduler.step()

    improved = epoch_loss < best_loss
    flag = " ★" if improved else ""
    if improved:
        best_loss = epoch_loss; patience_cnt = 0
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save(text_proj.state_dict(), os.path.join(SAVE_DIR, "projector_text.pt"))
        torch.save(image_proj.state_dict(), os.path.join(SAVE_DIR, "projector_image.pt"))
    else:
        patience_cnt += 1

    print(f"Epoch {epoch+1:3d}{flag}  Loss={epoch_loss:.4f} Acc={epoch_acc:.4f}")
    if patience_cnt >= PATIENCE: print(f"Early stopping at {epoch+1}"); break

# 投影存盘
print(f"\n📦 Projecting all features...")
text_proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "projector_text.pt"), weights_only=True))
image_proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "projector_image.pt"), weights_only=True))
text_proj.eval(); image_proj.eval()

pt_dir = os.path.join(SAVE_DIR, "projected_text")
pi_dir = os.path.join(SAVE_DIR, "projected_image")
os.makedirs(pt_dir, exist_ok=True); os.makedirs(pi_dir, exist_ok=True)

with torch.no_grad():
    for fname in FILE_NAMES:
        t = np.load(os.path.join(TEXT_DIR, fname)).astype(np.float32)
        i = np.load(os.path.join(IMAGE_DIR, fname)).astype(np.float32)
        if t.ndim == 3: t = t.mean(axis=1)
        if i.ndim == 3: i = i.mean(axis=1)
        shp_t, shp_i = t.shape, i.shape
        t_flat = t.reshape(-1, shp_t[-1]) if t.ndim > 2 else t
        i_flat = i.reshape(-1, shp_i[-1]) if i.ndim > 2 else i

        t_out = text_proj(torch.tensor(t_flat).to(DEVICE)).cpu().numpy()
        i_out = image_proj(torch.tensor(i_flat).to(DEVICE)).cpu().numpy()
        if t.ndim == 3: t_out = t_out.reshape(shp_t[0], shp_t[1], -1)
        if i.ndim == 3: i_out = i_out.reshape(shp_i[0], shp_i[1], -1)
        np.save(os.path.join(pt_dir, fname), t_out)
        np.save(os.path.join(pi_dir, fname), i_out)

print(f"→ {pt_dir}/  ({len(FILE_NAMES)} files) — 投影文本")
print(f"→ {pi_dir}/  ({len(FILE_NAMES)} files) — 投影图像")
print(f"→ {os.path.join(SAVE_DIR, 'projector_text.pt')}")
print(f"→ {os.path.join(SAVE_DIR, 'projector_image.pt')}")

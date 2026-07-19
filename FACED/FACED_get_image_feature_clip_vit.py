import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import CLIPModel, CLIPProcessor

# ================= 配置区域 =================
# 🔧 请修改为你的帧图像目录和输出路径
INPUT_ROOT = r"./FACED_frames"
OUTPUT_ROOT = r"./features/image_features_clip_vit_centercrop_timelen5_timestep2"
MODEL_CACHE_DIR = r"./huggingface_cache"

MODEL_ID = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FRAMES_PER_SECOND = 3
TARGET_SECONDS = 30
TARGET_FRAMES = TARGET_SECONDS * FRAMES_PER_SECOND  # 90
BATCH_SIZE = 64

CENTER_CROP_RATIO = 0.6
WINDOW_SIZE = 5
STRIDE = 2
TARGET_FEAT_DIM = 1024

VIDEO_NAMES = [
    'neg_a_1', 'neg_a_2', 'neg_a_3', 'neg_d_1', 'neg_d_2', 'neg_d_3',
    'neg_f_1', 'neg_f_2', 'neg_f_3', 'neg_s_1', 'neg_s_2', 'neg_s_3',
    'neu_1', 'neu_2', 'neu_3', 'neu_4',
    'pos_a_1', 'pos_a_2', 'pos_a_3', 'pos_i_1', 'pos_i_2', 'pos_i_3',
    'pos_j_1', 'pos_j_2', 'pos_j_3', 'pos_t_1', 'pos_t_2', 'pos_t_3'
]


def center_crop(img, ratio=CENTER_CROP_RATIO):
    w, h = img.size
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def load_model():
    print(f"Loading CLIP model: {MODEL_ID}")
    model = CLIPModel.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR).to(DEVICE).half()
    processor = CLIPProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    model.eval()
    return model, processor


def extract_clip_vit_features_batch(model, processor, image_paths):
    """
    CLIP ViT 隐藏层编码:
    center crop → CLIP preprocess → vision_model → mean pool patches → L2 norm
    返回: (batch, hidden_dim) 其中 hidden_dim=1664 for ViT-g
    """
    images = []
    valid_indices = []
    for i, path in enumerate(image_paths):
        try:
            raw_img = Image.open(path).convert("RGB")
            cropped = center_crop(raw_img)
            images.append(cropped)
            valid_indices.append(i)
        except Exception as e:
            print(f"  Error loading {path}: {e}")
            continue

    if not images:
        return None, valid_indices

    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    if DEVICE == "cuda":
        inputs = {k: v.half() for k, v in inputs.items()}

    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
        # last_hidden_state: (B, n_patches+1, 1664) including CLS token
        # mean pool over all tokens → (B, 1664)
        hidden = vision_outputs.last_hidden_state
        feats = hidden.mean(dim=1)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

    return feats.cpu().float().numpy(), valid_indices


def apply_sliding_window(sec_features):
    """
    输入: (30, D) 每秒一个特征
    策略: 取 5 秒窗口的中间秒（第 3 秒）→ 输出 (13, D)
    """
    segments = []
    for start_idx in range(0, TARGET_SECONDS - WINDOW_SIZE + 1, STRIDE):
        mid_idx = start_idx + WINDOW_SIZE // 2
        segments.append(sec_features[mid_idx:mid_idx + 1, :])
    result = np.concatenate(segments, axis=0)
    return result.astype(np.float32)


def process_single_video(model, processor, folder_path):
    """处理单个视频目录，返回秒级特征 (30, D) 和帧级特征"""
    folder_name = os.path.basename(folder_path)
    image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))

    if len(image_files) >= TARGET_FRAMES:
        use_frames = image_files[-TARGET_FRAMES:]
    else:
        num_valid = (len(image_files) // FRAMES_PER_SECOND) * FRAMES_PER_SECOND
        use_frames = image_files[-num_valid:]
        print(f"  Warning: {folder_name} only has {len(use_frames)} usable frames")

    if len(use_frames) < FRAMES_PER_SECOND:
        print(f"  Error: {folder_name} has fewer than 3 frames, skipping")
        return None, None

    # 批量 CLIP ViT 编码所有帧
    all_frame_features = []
    for i in range(0, len(use_frames), BATCH_SIZE):
        batch_paths = use_frames[i: i + BATCH_SIZE]
        feats, _ = extract_clip_vit_features_batch(model, processor, batch_paths)
        if feats is not None:
            all_frame_features.append(feats)

    if not all_frame_features:
        print(f"  Error: No features extracted for {folder_name}")
        return None, None

    all_frame_features = np.vstack(all_frame_features)

    # 每秒 max pooling（3 帧 → 1）
    num_seconds = all_frame_features.shape[0] // FRAMES_PER_SECOND
    reshaped = all_frame_features[:num_seconds * FRAMES_PER_SECOND].reshape(
        num_seconds, FRAMES_PER_SECOND, -1)
    sec_features = np.max(reshaped, axis=1)

    # 补齐到 30 秒（不足则前面补零）
    if sec_features.shape[0] < TARGET_SECONDS:
        padded = np.zeros((TARGET_SECONDS, sec_features.shape[1]), dtype=np.float32)
        padded[-sec_features.shape[0]:] = sec_features
        sec_features = padded
    elif sec_features.shape[0] > TARGET_SECONDS:
        sec_features = sec_features[-TARGET_SECONDS:]

    return sec_features, all_frame_features


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    model, processor = load_model()

    # 检测 ViT 输出维度
    print("Detecting CLIP ViT output dimension...")
    dummy_img = Image.new("RGB", (224, 224))
    dummy_inputs = processor(images=[dummy_img], return_tensors="pt").to(DEVICE)
    if DEVICE == "cuda":
        dummy_inputs = {k: v.half() for k, v in dummy_inputs.items()}
    with torch.no_grad():
        dummy_out = model.vision_model(**dummy_inputs)
    clip_dim = dummy_out.last_hidden_state.shape[-1]
    print(f"CLIP ViT hidden dim: {clip_dim}")
    del dummy_out
    torch.cuda.empty_cache()

    video_folders = sorted([f.path for f in os.scandir(INPUT_ROOT) if f.is_dir()])

    # ================= Phase 1: 提取所有秒级特征 =================
    print(f"\n{'='*60}")
    print("Phase 1: Extracting per-second features from all videos")
    print(f"{'='*60}")

    video_sec_features = {}
    all_frame_features_pca = []

    for folder in tqdm(video_folders, desc="Phase 1"):
        folder_name = os.path.basename(folder)
        if folder_name not in VIDEO_NAMES:
            continue

        sec_features, frame_feats = process_single_video(model, processor, folder)

        if sec_features is None:
            print(f"  Skipping {folder_name}")
            continue

        video_sec_features[folder_name] = sec_features  # (30, clip_dim)
        all_frame_features_pca.append(frame_feats)

    # ================= Phase 2: PCA 降维 =================
    print(f"\n{'='*60}")
    print(f"Phase 2: Fitting PCA {clip_dim} → {TARGET_FEAT_DIM}")
    print(f"{'='*60}")

    all_frames_stacked = np.vstack(all_frame_features_pca)
    print(f"PCA fitting data (frame-level): {all_frames_stacked.shape}")

    pca = PCA(n_components=TARGET_FEAT_DIM)
    pca.fit(all_frames_stacked)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance ({TARGET_FEAT_DIM}/{clip_dim}): {explained:.4f} ({explained*100:.1f}%)")

    # ================= Phase 3: 变换 + 滑动窗口 + 保存 =================
    print(f"\n{'='*60}")
    print("Phase 3: PCA transform + sliding window + save")
    print(f"{'='*60}")

    for folder_name in tqdm(video_sec_features, desc="Phase 3"):
        sec_features = video_sec_features[folder_name]  # (30, clip_dim)
        sec_features = pca.transform(sec_features).astype(np.float32)
        windowed = apply_sliding_window(sec_features)  # (13, 1024)

        save_path = os.path.join(OUTPUT_ROOT, f"{folder_name}_features.npy")
        np.save(save_path, windowed)

    print(f"\nDone! Features saved to: {OUTPUT_ROOT}")
    print(f"Output: {len(video_sec_features)} files, each (13, {TARGET_FEAT_DIM})")


if __name__ == "__main__":
    main()

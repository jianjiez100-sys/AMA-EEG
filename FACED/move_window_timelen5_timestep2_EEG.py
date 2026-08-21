"""Slice FACED EEG features into fixed-length overlapping windows."""

import argparse
import time
from pathlib import Path

import numpy as np


def process_eeg(
    input_file,
    output_dir,
    save_name="processed_eeg_features.npy",
    n_subs=123,
    n_vids=28,
    seconds=30,
    feat_dim=160,
    window_size=5,
    stride=2,
):
    """Create sliding windows and return the generated ``.npy`` path."""
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    if window_size <= 0 or stride <= 0 or window_size > seconds:
        raise ValueError("Require 0 < window_size <= seconds and stride > 0")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] 开始处理 EEG 数据...")

    # 1. 加载
    raw_data = np.load(input_file)  # (Total, feat_dim)

    # 2. Reshape: (Subject, Video, Seconds, Feat)
    try:
        reshaped_data = raw_data.reshape(n_subs, n_vids, seconds, feat_dim)
    except ValueError as exc:
        expected = n_subs * n_vids * seconds * feat_dim
        raise ValueError(
            f"Input shape {raw_data.shape} has {raw_data.size} values; "
            f"expected {expected} for {n_subs}x{n_vids}x{seconds}x{feat_dim}"
        ) from exc

    # 3. 滑动窗口
    segments = []
    # (30 - 5) // 2 + 1 = 13
    n_windows = (seconds - window_size) // stride + 1

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        # 切片: (123, 28, 5, 160)
        window = reshaped_data[:, :, start:end, :]
        segments.append(window)

    # 堆叠: (123, 28, 13, 5, 160)
    stacked = np.stack(segments, axis=2)

    # 4. 展平为最终 Dataset 格式: (Total_Samples, 5, 160)
    final_data = stacked.reshape(-1, window_size, feat_dim)

    # 5. 保存
    save_path = output_dir / save_name
    np.save(save_path, final_data)

    print(f"✅ EEG 处理完成! 保存至: {save_path}")
    expected_rows = n_subs * n_vids * n_windows
    print(f"最终形状: {final_data.shape} (预期: {expected_rows}, {window_size}, {feat_dim})")
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--save-name", default="processed_eeg_features.npy")
    parser.add_argument("--n-subs", type=int, default=123)
    parser.add_argument("--n-vids", type=int, default=28)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--feat-dim", type=int, default=160)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_eeg(
        input_file=args.input_file,
        output_dir=args.output_dir,
        save_name=args.save_name,
        n_subs=args.n_subs,
        n_vids=args.n_vids,
        seconds=args.seconds,
        feat_dim=args.feat_dim,
        window_size=args.window_size,
        stride=args.stride,
    )

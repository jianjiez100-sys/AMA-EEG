import numpy as np

# 🔧 请修改为你要检查的 .npy 文件路径
file_path = r"./features/your_features.npy"

try:
    # 1. 加载数据
    data = np.load(file_path)

    # 2. 打印维度信息
    print("=" * 30)
    print(f"文件路径: {file_path}")
    print(f"数据维度 (Shape): {data.shape}")

    # 3. 解释维度含义
    if len(data.shape) == 2:
        num_seconds, feature_dim = data.shape
        print(f"数据量 (秒数/样本数): {num_seconds}")
        print(f"特征维度: {feature_dim}")

    # 4. 查看具体的特征值
    print("-" * 30)
    print("前 2 个样本的特征值片段 (前 5 维):")
    # 只打印前两个样本的前5个数值，避免刷屏
    print(data[:2, :5])

    print("-" * 30)
    print(f"数值范围: Max={data.max():.4f}, Min={data.min():.4f}")
    print(f"数据类型: {data.dtype}")
    print("=" * 30)

except Exception as e:
    print(f"读取失败，请检查路径是否正确: {e}")
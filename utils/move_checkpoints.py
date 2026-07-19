import os
import shutil

# ================= 路径配置 =================
# 🔧 请修改为你的 checkpoint 目录
src_dir = "./checkpoints_source"
dst_dir = "./checkpoints_target"

# 如果目标文件夹不存在，自动创建
os.makedirs(dst_dir, exist_ok=True)

# ================= 精确的文件名名单 =================
# 这是从你的 Image Mode 日志中精准提取的 10 个 Best Checkpoint
image_ckpts = [
    "fold0_epoch=15.ckpt",
    "fold1_epoch=26.ckpt",
    "fold2_epoch=21.ckpt",
    "fold3_epoch=04.ckpt",
    "fold4_epoch=15.ckpt",
    "fold5_epoch=21.ckpt",
    "fold6_epoch=03.ckpt",
    "fold7_epoch=12.ckpt",
    "fold8_epoch=26.ckpt",
    "fold9_epoch=06.ckpt"
]

print("🔍 开始执行精确文件迁移...\n")

moved_count = 0
missing_files = []

for filename in image_ckpts:
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    # 检查文件是否存在
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"✅ 成功移动: {filename}")
        moved_count += 1
    else:
        print(f"❌ 找不到文件: {filename}")
        missing_files.append(filename)

print("\n=========================================")
print(f"🎉 迁移完成！成功移走 {moved_count} 个图像模式权重。")
if missing_files:
    print(f"⚠️ 以下文件未找到: {missing_files}")
print("=========================================")
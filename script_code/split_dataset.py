import os
import re
import shutil
import random

# ===========================
# 参数配置
# ===========================

# 源目录（output下的子文件夹）
OUTPUT_DIR = "D:\\YOLO_Project\\YOLO-Resnet-test\\dataset\\output"

# 目标根目录（train/val已在里面建好cap/和no-cap/子文件夹）
TARGET_DIR = "D:\\YOLO_Project\\YOLO-Resnet-test\\dataset"

# 训练集:验证集 比例
TRAIN_RATIO = 0.7

# 连续帧区块大小（同一区块内的帧分配到同一集合，避免相邻帧分散）
CHUNK_SIZE = 5

# 随机种子（保证可复现）
random.seed(42)

# ===========================
# 辅助函数
# ===========================

def extract_number(filename):
    """从文件名中提取第一个数字序列，用于排序"""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0


def get_label(folder_name):
    """根据输出子文件夹名判断类别：cap 或 no-cap"""
    if "no_cap" in folder_name.lower() or "no-cap" in folder_name.lower():
        return "no-cap"
    return "cap"


# ===========================
# 主逻辑
# ===========================

# 收集所有输出子目录
sub_dirs = [
    d for d in os.listdir(OUTPUT_DIR)
    if os.path.isdir(os.path.join(OUTPUT_DIR, d))
]

print(f"{'='*60}")
print(f"数据集划分脚本")
print(f"源目录: {OUTPUT_DIR}")
print(f"目标目录: {TARGET_DIR}")
print(f"训练:验证 = {TRAIN_RATIO}:{1-TRAIN_RATIO}")
print(f"连续帧区块大小: {CHUNK_SIZE}")
print(f"{'='*60}\n")

total_train = 0
total_val = 0

for sub_dir in sub_dirs:
    sub_path = os.path.join(OUTPUT_DIR, sub_dir)
    label = get_label(sub_dir)

    # 收集图片文件
    files = [
        f for f in os.listdir(sub_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]

    if not files:
        print(f"[{sub_dir}] 无图片文件，跳过")
        continue

    # 按文件名中的数字排序
    files.sort(key=extract_number)

    # 按区块划分（每 CHUNK_SIZE 帧为一个区块）
    chunks = [files[i:i + CHUNK_SIZE] for i in range(0, len(files), CHUNK_SIZE)]

    # 随机打乱区块顺序
    random.shuffle(chunks)

    # 按比例分配区块
    train_chunk_count = max(1, round(len(chunks) * TRAIN_RATIO))
    train_chunks = chunks[:train_chunk_count]
    val_chunks = chunks[train_chunk_count:]

    train_count = 0
    val_count = 0

    print(f"\n--- [{sub_dir}] ({label}) 共 {len(files)} 张 ---")

    # 复制到训练集
    train_dst = os.path.join(TARGET_DIR, "train", label)
    os.makedirs(train_dst, exist_ok=True)
    for chunk in train_chunks:
        for fname in chunk:
            src = os.path.join(sub_path, fname)
            dst = os.path.join(train_dst, fname)
            shutil.copy2(src, dst)
            train_count += 1
            print(f"  TRAIN ← {fname}")

    # 复制到验证集
    val_dst = os.path.join(TARGET_DIR, "val", label)
    os.makedirs(val_dst, exist_ok=True)
    for chunk in val_chunks:
        for fname in chunk:
            src = os.path.join(sub_path, fname)
            dst = os.path.join(val_dst, fname)
            shutil.copy2(src, dst)
            val_count += 1
            print(f"  VAL   ← {fname}")

    total_train += train_count
    total_val += val_count
    print(f"  [{sub_dir}] 训练 {train_count} | 验证 {val_count}")

# ===========================
# 汇总
# ===========================
print(f"\n{'='*60}")
print(f"划分完成！")
print(f"训练集总计: {total_train} 张")
print(f"验证集总计: {total_val} 张")
print(f"实际比例: {total_train/(total_train+total_val):.2f} : {total_val/(total_train+total_val):.2f}" if (total_train+total_val) > 0 else "")
print(f"{'='*60}")

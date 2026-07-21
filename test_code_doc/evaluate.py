# ===========================
# 测试脚本 — 对新图片做详细评估
# ===========================

import torch
import os
from PIL import Image as PILImage

from model import create_model
from dataset import val_transform
from utils import compute_metrics, print_metrics


def load_image(path):
    """读取图片并转为 RGB"""
    return PILImage.open(path).convert("RGB")


# ===========================
# 配置
# ===========================
MODEL_PATH = "../models/run_002/bottle_cap_resnet18.pth"
TEST_DIR   = "../dataset/test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================
# 加载模型
# ===========================
model = create_model(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print(f"设备: {DEVICE}")
print(f"模型: {MODEL_PATH}")

# ===========================
# 按类别逐张预测
# ===========================
class_names = sorted(os.listdir(TEST_DIR))  # ImageFolder 按字母序：cap=0, no-cop=1
print(f"类别映射: {dict(enumerate(class_names))}")
print(f"{'=' * 60}")

all_labels = []
all_preds  = []
misclassified = []       # 记录分错的图片
correct_by_class = {}    # 各类正确数
total_by_class   = {}    # 各类总数

for label_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(TEST_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    files = sorted(os.listdir(class_dir))
    correct_by_class[class_name] = 0
    total_by_class[class_name]   = 0

    print(f"\n--- {class_name} ({len(files)} 张) ---")

    for fname in files:
        filepath = os.path.join(class_dir, fname)
        try:
            img = load_image(filepath)
        except:
            continue

        img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            probs  = torch.softmax(output, dim=1)
            pred   = torch.argmax(probs, dim=1).item()
            conf   = probs[0, pred].item()

        all_labels.append(label_idx)
        all_preds.append(pred)
        total_by_class[class_name] += 1

        if pred == label_idx:
            correct_by_class[class_name] += 1
        else:
            misclassified.append({
                "file": filepath,
                "true": class_name,
                "pred": class_names[pred],
                "conf": conf,
            })
            print(f"  ✗ {fname:<40s}  真实:{class_name:<6s}  预测:{class_names[pred]:<6s}  conf={conf:.4f}")

# ===========================
# 总体指标
# ===========================
print(f"\n{'=' * 60}")
metrics = compute_metrics(all_labels, all_preds)
print_metrics(metrics, prefix="测试集总体指标")

# ===========================
# 各类别准确率
# ===========================
print(f"\n各类别准确率:")
for cls_name in class_names:
    total = total_by_class[cls_name]
    correct = correct_by_class[cls_name]
    acc = correct / total if total > 0 else 0
    print(f"  {cls_name:<8s}: {correct}/{total} = {acc:.4f}")

# ===========================
# 分错样本汇总
# ===========================
if misclassified:
    print(f"\n{'─' * 60}")
    print(f"分错样本共 {len(misclassified)} 张:")
    for item in misclassified:
        print(f"  {item['file']}")
        print(f"    真实: {item['true']}  →  预测: {item['pred']}  (conf={item['conf']:.4f})")
else:
    print(f"\n✓ 全部预测正确！")

print(f"\n{'=' * 60}")
print(f"测试完成 | 总样本: {len(all_labels)} | 错误: {len(misclassified)}")

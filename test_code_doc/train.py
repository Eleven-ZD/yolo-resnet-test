# ===========================
# 训练脚本 — cap / no_cap 二分类
# ===========================

import torch
import torch.nn as nn
import torch.optim as optim
import os
import re

from model import create_model
from dataset import get_loader
from utils import compute_metrics, print_metrics

# ===========================
# 超参数
# ===========================
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0001
TRAIN_DIR = "../dataset/train"       # 其下应有 cap/ 和 no-cap/
VAL_DIR   = "../dataset/val"         # 同上

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================
# 自动编号 — 每次训练新建一个 run 文件夹
# ===========================
MODELS_ROOT = "../models"
os.makedirs(MODELS_ROOT, exist_ok=True)

# 扫描已有 run_xxx，取最大编号 +1
existing = [
    int(re.search(r"run_(\d+)", d).group(1))
    for d in os.listdir(MODELS_ROOT)
    if os.path.isdir(os.path.join(MODELS_ROOT, d)) and re.search(r"run_(\d+)", d)
]
run_id = max(existing) + 1 if existing else 1
RUN_DIR = os.path.join(MODELS_ROOT, f"run_{run_id:03d}")
os.makedirs(RUN_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(RUN_DIR, "bottle_cap_resnet18.pth")

# ===========================
# 数据加载
# ===========================
train_loader = get_loader(TRAIN_DIR, BATCH_SIZE, train=True)
val_loader   = get_loader(VAL_DIR,   BATCH_SIZE, train=False)

# ===========================
# 模型、损失、优化器
# ===========================
model = create_model(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ===========================
# 训练循环
# ===========================
best_acc = 0.0

print(f"设备: {DEVICE} | 运行编号: run_{run_id:03d}")
print(f"模型保存: {MODEL_SAVE_PATH}")
print(f"训练集: {len(train_loader.dataset)} 张 | 验证集: {len(val_loader.dataset)} 张")
print(f"Batch: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LR}")
print(f"{'=' * 50}")

for epoch in range(EPOCHS):
    # ---- 训练 ----
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # ---- 验证 ----
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    acc = metrics["accuracy"]

    print(f"\nEpoch {epoch + 1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
    print_metrics(metrics, prefix="验证集指标")

    # ---- 保存最优模型 ----
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ★ 最优模型已保存 (Acc={best_acc:.4f})")

# ===========================
# 训练结束
# ===========================
print(f"\n{'=' * 50}")
print(f"训练完成！最优验证准确率: {best_acc:.4f}")
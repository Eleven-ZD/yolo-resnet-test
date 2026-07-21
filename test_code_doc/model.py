# ===========================
# 模型定义 — ResNet18 二分类 (cap / no_cap)
# ===========================

import torch.nn as nn
import torchvision.models as models


def create_model(num_classes=2, pretrained=True):
    """创建 ResNet18 分类模型

    Args:
        num_classes: 分类数（cap / no_cap = 2）
        pretrained: 是否加载 ImageNet 预训练权重
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.resnet18(weights=weights)

    # 替换最后一层 FC：512 → num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

# ===========================
# 数据加载 — ImageFolder + 增强
# ===========================

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ImageNet 标准化参数
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ===========================
# 训练数据增强（强增强，提升泛化能力）
# ===========================
train_transform = transforms.Compose([
    # 随机裁剪 + 缩放（比固定 Resize 更强）
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33)),

    # 几何增强
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),                # 极小概率上下翻转
    transforms.RandomRotation(degrees=15),                 # ±15° 旋转
    transforms.RandomPerspective(distortion_scale=0.15, p=0.3),  # 透视变换，模拟不同角度

    # 颜色增强
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
    ),

    # 灰度化（模拟黑白画面）
    transforms.RandomGrayscale(p=0.05),

    # 模糊（模拟运动模糊 / 失焦）
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),

    # 转张量
    transforms.ToTensor(),

    # 随机遮挡（模拟瓶子被部分遮挡）
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),

    # 标准化
    transforms.Normalize(mean=MEAN, std=STD),
])

# ===========================
# 验证 / 测试
# ===========================
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def get_loader(path, batch_size, train=True):
    """返回 DataLoader

    Args:
        path: 数据集根目录（其下应有 cap/ 和 no_cap/ 子文件夹）
        batch_size: 批次大小
        train: True 则用训练增强 + shuffle
    """
    transform = train_transform if train else val_transform
    dataset = ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader

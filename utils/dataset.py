from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader



# 训练数据增强

train_transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),


    # 随机旋转
    transforms.RandomRotation(
        10
    ),


    # 随机亮度
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),


    # 水平翻转
    transforms.RandomHorizontalFlip(),


    transforms.ToTensor(),


    # ImageNet标准化

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])




# 验证/测试

val_transform = transforms.Compose([


    transforms.Resize(
        (224,224)
    ),


    transforms.ToTensor(),


    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])




def get_loader(
        path,
        batch_size,
        train=True
):


    if train:

        dataset = ImageFolder(
            path,
            transform=train_transform
        )

    else:

        dataset = ImageFolder(
            path,
            transform=val_transform
        )



    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=train

    )


    return loader
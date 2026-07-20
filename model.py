import torch.nn as nn
import torchvision.models as models


def create_model():

    # 加载ImageNet预训练ResNet18

    model = models.resnet18(
        weights="IMAGENET1K_V1"
    )


    # 原来:
    #
    # fc:
    # 512 -> 1000
    #
    # 修改:
    #
    # 512 -> 2


    model.fc = nn.Linear(
        512,
        2
    )


    return model
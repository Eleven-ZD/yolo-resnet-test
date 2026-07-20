import torch
import torch.nn as nn
import torch.optim as optim
from model import create_model
from utils.dataset import get_loader
# ======================
# 参数
# ======================
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0001

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

train_loader = get_loader(
    "dataset/train",
    BATCH_SIZE,
    True

)

val_loader = get_loader(
    "dataset/val",
    BATCH_SIZE,
    False
)
# ======================
# 模型
# ======================
model=create_model()

model=model.to(
    DEVICE
)

# ======================
# 损失
# ======================
criterion = nn.CrossEntropyLoss()


# ======================
# 优化器
# ======================
optimizer = optim.AdamW(

    model.parameters(),

    lr=LR

)

# ======================
# 开始训练
# ======================
best_acc=0

for epoch in range(EPOCHS):


    model.train()

    total_loss=0

    for images,labels in train_loader:
        images=images.to(
            DEVICE
        )
        labels=labels.to(
            DEVICE
        )
        # 前向
        outputs=model(
            images
        )


        loss=criterion(

            outputs,

            labels

        )


        optimizer.zero_grad()


        loss.backward()


        optimizer.step()


        total_loss += loss.item()



    # =====验证=====


    model.eval()


    correct=0

    total=0



    with torch.no_grad():


        for images,labels in val_loader:


            images=images.to(
                DEVICE
            )

            labels=labels.to(
                DEVICE
            )


            outputs=model(
                images
            )


            _,pred=torch.max(

                outputs,

                1

            )


            total += labels.size(0)


            correct += (

                pred==labels

            ).sum().item()



    acc = correct / total



    print(
        f"Epoch {epoch+1}/{EPOCHS}",
        f"Loss:{total_loss:.4f}",
        f"Val Acc:{acc:.4f}"
    )



    # 保存最好模型

    if acc > best_acc:


        best_acc=acc


        torch.save(

            model.state_dict(),

            "models/bottle_cap_resnet18.pth"

        )


        print(
            "模型已保存"
        )



print(
    "训练完成"
)
import torch

from PIL import Image

from model import create_model

from utils.dataset import val_transform



DEVICE="cuda"



model=create_model()


model.load_state_dict(

    torch.load(

        "models/bottle_cap_resnet18.pth"

    )

)


model.to(
    DEVICE
)


model.eval()



img=Image.open(
    "test.jpg"
)



img=val_transform(
    img
)


img=img.unsqueeze(0)


img=img.to(
    DEVICE
)



output=model(img)



prob=torch.softmax(
    output,
    dim=1
)



cls=torch.argmax(
    prob
)



classes=[

    "cap",

    "no_cap"

]



print(

    "结果:",

    classes[cls],

    "概率:",

    prob

)
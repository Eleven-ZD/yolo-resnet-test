import torch


from model import create_model

from utils.dataset import get_loader

from utils.metrics import evaluate_metric



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



test_loader=get_loader(

    "dataset/test",

    32,

    False

)



labels=[]

preds=[]



with torch.no_grad():


    for images,y in test_loader:


        images=images.to(
            DEVICE
        )


        output=model(
            images
        )


        _,pred=torch.max(

            output,

            1

        )


        labels.extend(

            y.numpy()

        )


        preds.extend(

            pred.cpu().numpy()

        )



evaluate_metric(

    labels,

    preds

)
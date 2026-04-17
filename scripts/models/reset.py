import torch
from torchvision import models


def get_resnet50(device="cpu"):
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    except Exception:
        model = models.resnet50(pretrained=True)
    model.eval()
    return model.to(device)


@torch.no_grad()
def predict_class(model, input_tensor):
    logits = model(input_tensor)
    return int(logits.argmax(dim=1).item())

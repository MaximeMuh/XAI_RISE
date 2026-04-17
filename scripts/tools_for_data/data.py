from pathlib import Path

import json
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T


def default_preprocess(image_size = 224):
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image(path):
    return Image.open(path).convert("RGB")


def image_to_tensor(path, image_size, device = "cpu"):
    img = load_image(path)
    preprocess = default_preprocess(image_size=image_size)
    return preprocess(img).unsqueeze(0).to(device)


def load_imagenet_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


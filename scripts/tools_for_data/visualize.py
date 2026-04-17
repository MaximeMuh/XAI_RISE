from pathlib import Path

import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image


def overlay_cam(rgb_img_0_1, saliency_hw):
    return show_cam_on_image(rgb_img_0_1, saliency_hw)


def save_overlay_from_pil(image, saliency_hw, out_path, image_size = 224):
    rgb = np.array(image.resize((image_size, image_size))).astype(np.float32) / 255.0
    vis = overlay_cam(rgb, saliency_hw)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis).save(out_path)


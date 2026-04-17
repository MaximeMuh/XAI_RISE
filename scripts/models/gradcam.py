"""
Ce fichier contient l'implémentation de Grad-CAM.
Comme le projet est centré sur RISE, on s'est appuyé sur les implémentations existantes.
La méthode s'appuie sur les gradients qui remontent jusqu'à la dernière couche de convolution
pour calculer l'importance de chaque petite région de l'image.
Cela va nous servir de méthode de référence face à RISE !
"""
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def compute_gradcam(model, input_tensor, target_layers, target_class = None):
    if target_class is None:
        with torch.no_grad():
            target_class = int(model(input_tensor).argmax(dim=1).item())

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])[0]
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
    return grayscale_cam.astype(np.float32)


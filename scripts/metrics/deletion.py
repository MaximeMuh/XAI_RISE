"""
On code ici la métrique de délétion utile pour les expériences. 
On efface petit à petit les pixels considérés importants par la méthode utilisée (Grad-CAM ou RISE), 
et on regarde comment la confiance du modèle chute. Plus elle s'effondre vite, mieux c'est !
"""
import numpy as np
import torch


def deletion_curve(model, image_chw, saliency_hw, pred_class,steps = 20, device = "cpu"):
    if isinstance(saliency_hw, torch.Tensor):
        saliency_hw = saliency_hw.detach().cpu().numpy()

    ranking = np.asarray(saliency_hw, dtype=np.float32).reshape(-1).argsort()[::-1].copy()
    _, h, w = image_chw.shape
    total_pixels = h * w
    probs: list[float] = []

    model.eval()
    with torch.no_grad():
        for k in range(steps):
            masked = image_chw.clone()
            n_pixels = (k * total_pixels) // steps
            idx = ranking[:n_pixels]
            rows = idx // w
            cols = idx % w
            masked[:, rows, cols] = 0
            out = model(masked.unsqueeze(0).to(device))
            probs.append(float(torch.softmax(out, dim=1)[0, pred_class].item()))

    return np.array(probs, dtype=np.float32)


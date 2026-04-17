"""
Ce script contient la métrique d'insertion pour évaluer les cartes de saillance.
Le principe est de partir d'une image masquée (soit noire, soit floutée) et de réinsérer
progressivement les pixels les plus importants trouvés par la méthode XAI (Grad-CAM ou RISE).
Si la probabilité de la bonne classe remonte très vite, c'est que l'explication est pertinente.
"""
import numpy as np
import torch


def insertion_curve(model, image_chw, saliency_hw, pred_class, steps = 20, device = "cpu", baseline = "black"):
    if isinstance(saliency_hw, torch.Tensor):
        saliency_hw = saliency_hw.detach().cpu().numpy()

    ranking = np.asarray(saliency_hw, dtype=np.float32).reshape(-1).argsort()[::-1].copy()
    _, h, w = image_chw.shape
    total_pixels = h * w

    if baseline == "black":
        start = torch.zeros_like(image_chw)
    elif baseline == "blur":
        start = torch.nn.functional.avg_pool2d(
            image_chw.unsqueeze(0), kernel_size=11, stride=1, padding=5
        )[0]
    else:
        raise ValueError("baseline must be 'black' or 'blur'")

    probs: list[float] = []
    model.eval()
    with torch.no_grad():
        for k in range(steps):
            img = start.clone()
            n_pixels = (k * total_pixels) // steps
            idx = ranking[:n_pixels]
            rows = idx // w
            cols = idx % w
            img[:, rows, cols] = image_chw[:, rows, cols]
            out = model(img.unsqueeze(0).to(device))
            probs.append(float(torch.softmax(out, dim=1)[0, pred_class].item()))

    return np.array(probs, dtype=np.float32)


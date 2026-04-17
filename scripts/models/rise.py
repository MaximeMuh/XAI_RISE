"""
Voici le coeur de notre implémentation de RISE pour les images.
On génère d'abord une grande série de masques aléatoires.
Ensuite, on regarde les prédictions du modèle sur ces versions masquées de l'image.
La carte finale est simplement la somme des masques, pondérée par leurs scores de prédiction.
"""
import numpy as np
import torch
import tqdm as tqdm


def generate_masks(N, s, p1, H, W, device="cpu"):
    """
    Genere N masques aleatoires upsampled en HxW.
    """
    masks = (torch.rand((N, s, s), device=device) < p1).float()
    masks = masks.unsqueeze(1)  # (N,1,s,s)
    masks = torch.nn.functional.interpolate(
        masks,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    return masks.squeeze(1)  # (N,H,W)


def compute_rise_saliency(
    model,
    input_tensor,
    pred_class,
    num_masks=300,
    s=8,
    p1=0.5,
    device="cpu",
    batch_size=50,
    N=None,
):
    if N is not None:
        num_masks = N
    if batch_size is None or batch_size <= 0:
        batch_size = num_masks

    input_tensor = input_tensor.to(device)
    _, _, H, W = input_tensor.shape

    masks = generate_masks(num_masks, s, p1, H, W, device=device)
    saliency = torch.zeros(H, W, device=device)

    model.eval()

    with torch.no_grad():
        for i in tqdm.tqdm(range(0, num_masks, batch_size), desc="RISE"):
            batch_masks = masks[i:i + batch_size]  # (B,H,W)
            masked_input = input_tensor * batch_masks.unsqueeze(1)  # (B,3,H,W)
            output = model(masked_input)
            probs = torch.softmax(output, dim=1)[:, pred_class]  # (B,)
            saliency += (probs[:, None, None] * batch_masks).sum(dim=0)

    saliency = saliency / num_masks
    saliency = saliency.cpu().numpy()
    saliency = (
        (saliency - saliency.min())
        / (saliency.max() - saliency.min() + 1e-8)
    )

    return saliency.astype(np.float32)

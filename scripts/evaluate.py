"""
C'est le script principal pour lancer l'évaluation finale et comparer nos méthodes XAI !
On prend une image, on génère les cartes Grad-CAM et RISE, puis on trace les courbes de 
délétion et d'insertion pour voir quelle méthode est la meilleure en regardant notamment l'AUC.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

from scripts.tools_for_data.data import image_to_tensor
from scripts.models.gradcam import compute_gradcam
from scripts.models.rise import compute_rise_saliency
from scripts.metrics.deletion import deletion_curve
from scripts.metrics.insertion import insertion_curve
from scripts.models.reset import get_resnet50, predict_class


def main():
    parser = argparse.ArgumentParser(description="Compare Grad-CAM vs RISE metrics on one image.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--out", default="outputs/eval", help="Output directory.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--steps", type=int, default=20, help="Number of deletion/insertion steps.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_resnet50(device=args.device)
    inp = image_to_tensor(args.image, image_size=224, device=args.device)
    pred_class = predict_class(model, inp)
    image_chw = inp[0].detach().cpu()

    grad_map = compute_gradcam(model, inp, target_layers=[model.layer4[-1]], target_class=pred_class)
    rise_map = compute_rise_saliency(
        model,
        inp,
        pred_class=pred_class,
        num_masks=300,
        device=args.device,
        batch_size=50,
    )

    del_grad = deletion_curve(model, image_chw, grad_map, pred_class, steps=args.steps, device=args.device)
    del_rise = deletion_curve(model, image_chw, rise_map, pred_class, steps=args.steps, device=args.device)
    ins_grad = insertion_curve(model, image_chw, grad_map, pred_class, steps=args.steps, device=args.device)
    ins_rise = insertion_curve(model, image_chw, rise_map, pred_class, steps=args.steps, device=args.device)

    x = np.linspace(0, 1, args.steps)
    auc_del_grad = auc(x, del_grad)
    auc_del_rise = auc(x, del_rise)
    auc_ins_grad = auc(x, ins_grad)
    auc_ins_rise = auc(x, ins_rise)

    plt.figure(figsize=(7, 5))
    plt.plot(x, del_grad, label="Deletion Grad-CAM")
    plt.plot(x, del_rise, label="Deletion RISE")
    plt.plot(x, ins_grad, label="Insertion Grad-CAM")
    plt.plot(x, ins_rise, label="Insertion RISE")
    plt.xlabel("Fraction of pixels")
    plt.ylabel("Target class probability")
    plt.title("XAI metric comparison")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_curves.png", dpi=140)

    print(f"Saved evaluation plot to: {out_dir / 'comparison_curves.png'}")
    print(f"AUC deletion Grad-CAM: {auc_del_grad:.4f}")
    print(f"AUC deletion RISE: {auc_del_rise:.4f}")
    print(f"AUC insertion Grad-CAM: {auc_ins_grad:.4f}")
    print(f"AUC insertion RISE: {auc_ins_rise:.4f}")


if __name__ == "__main__":
    main()

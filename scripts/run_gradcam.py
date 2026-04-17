"""
Ici, on lance une petite démo pour Grad-CAM, qui sera un peu notre baseline.
On regarde surtout la dernière couche de notre ResNet50 pour récupérer ses activations.
"""
import argparse
from pathlib import Path

from PIL import Image

from scripts.tools_for_data.data import image_to_tensor, load_image
from scripts.models.gradcam import compute_gradcam
from scripts.models.reset import get_resnet50, predict_class
from scripts.tools_for_data.visualize import save_overlay_from_pil


def main():
    parser = argparse.ArgumentParser(description="Run Grad-CAM on one image.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--out", default="outputs/gradcam", help="Output directory.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_resnet50(device=args.device)
    input_tensor = image_to_tensor(args.image, image_size=224, device=args.device)
    pred_class = predict_class(model, input_tensor)
    target_layers = [model.layer4[-1]]

    cam = compute_gradcam(model, input_tensor, target_layers=target_layers, target_class=pred_class)

    img = load_image(args.image)
    save_overlay_from_pil(img, cam, out_dir / "gradcam_overlay.jpg")
    img.resize((224, 224)).save(out_dir / "input.jpg")
    Image.fromarray((cam * 255).astype("uint8")).save(out_dir / "gradcam_map.jpg")

    print(f"Saved Grad-CAM outputs to: {out_dir}")
    print(f"Predicted class index: {pred_class}")


if __name__ == "__main__":
    main()

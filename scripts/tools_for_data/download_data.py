import argparse
import os
import json
import urllib.request
import tarfile


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"


def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Saved to {dest_path}")
    else:
        print(f"File already exists: {dest_path}")


def extract_tgz(tgz_path, extract_to):
    print(f"Extracting {tgz_path} ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted to {extract_to}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Imagenette 320 dataset + imagenet_labels.json"
    )
    parser.add_argument(
        "--root",
        default="data",
        help="Root directory where data will be stored",
    )
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)

    # Imagenette
    tgz_path = os.path.join(args.root, "imagenette2-320.tgz")
    download_file(IMAGENETTE_URL, tgz_path)

    extract_dir = os.path.join(args.root, "imagenette2-320")
    if not os.path.exists(extract_dir):
        extract_tgz(tgz_path, args.root)
    else:
        print(f"Imagenette already extracted: {extract_dir}")

    # Labels JSON 
    labels_path = os.path.join(args.root, "imagenet_labels.json")
    download_file(LABELS_URL, labels_path)

    with open(labels_path, "r") as f:
        labels = json.load(f)
    print(f"Loaded {len(labels)} labels")

    print(f"\nAll files downloaded in: {args.root}")


if __name__ == "__main__":
    main()

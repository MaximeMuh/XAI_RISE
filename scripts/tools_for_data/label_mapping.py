from __future__ import annotations

import re
from pathlib import Path

from transformers import AutoConfig


def _looks_generic(id2label: dict) -> bool:
    if not id2label:
        return True
    pattern = re.compile(r"^LABEL_\\d+$")
    return all(pattern.match(str(v)) for v in id2label.values())


def load_id2label(model_id_or_path: str) -> dict[int, str]:
    """
    Load a robust id->label mapping.
    """
    id2label: dict[int, str] = {}

    try:
        config = AutoConfig.from_pretrained(model_id_or_path)
        if getattr(config, "id2label", None):
            id2label = {int(k): str(v) for k, v in config.id2label.items()}
    except Exception:
        pass

    if not _looks_generic(id2label):
        return id2label

    try:
        from huggingface_hub import hf_hub_download
        import pickle

        if Path(model_id_or_path).exists():
            le_path = Path(model_id_or_path) / "label_encoder.pkl"
        else:
            le_path = hf_hub_download(repo_id=model_id_or_path, filename="label_encoder.pkl")

        with open(le_path, "rb") as f:
            label_encoder = pickle.load(f)
        classes = list(getattr(label_encoder, "classes_", []))
        if classes:
            return {int(i): str(c) for i, c in enumerate(classes)}
    except Exception:
        pass

    return id2label

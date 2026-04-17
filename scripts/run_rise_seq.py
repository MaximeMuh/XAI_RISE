"""
Ici, on adapte la méthode RISE pour les séquences de protéines.
Au lieu de masquer des pixels, on va cacher des tokens directement dans la séquence.
L'objectif reste le même : voir quelles parties de la protéine comptent le plus pour la prédiction du modèle.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.models.rise_seq import compute_rise_saliency_seq
from scripts.tools_for_data.seq_data import read_fasta


def _load_sequence(sequence, fasta_path):
    if sequence:
        return "query", sequence.strip().replace(" ", "")
    if fasta_path:
        entries = read_fasta(fasta_path)
        if not entries:
            raise ValueError("No sequences found in FASTA.")
        if len(entries) > 1:
            print("FASTA contains multiple sequences. Using the first one.")
        return entries[0]
    raise ValueError("Provide --sequence or --fasta.")


def main():
    parser = argparse.ArgumentParser(description="Run RISE on a protein sequence.")
    parser.add_argument("--sequence", help="Protein sequence string.")
    parser.add_argument("--fasta", help="Path to FASTA file.")
    parser.add_argument("--model", required=True, help="HF model id or local path.")
    parser.add_argument("--out", default="outputs/rise_seq", help="Output directory.")
    parser.add_argument("--num-masks", type=int, default=300, help="Number of random masks.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for RISE masks.")
    parser.add_argument("--p1", type=float, default=0.5, help="Mask keep probability.")
    parser.add_argument("--s", type=int, default=None, help="Coarse mask size for smoothing (optional).")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--target-class", type=int, default=None, help="Class index to explain.")
    parser.add_argument("--multi-label", action="store_true", help="Use sigmoid for multi-label models.")
    parser.add_argument(
        "--target-labels",
        default=None,
        help="Comma-separated list of class indices to explain (multi-label).",
    )
    parser.add_argument("--max-length", type=int, default=1024, help="Tokenizer max length.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    name, seq = _load_sequence(args.sequence, args.fasta)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no mask token; cannot run RISE masking.")
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(args.device)
    model.eval()

    inputs = tokenizer(
        seq,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    with torch.no_grad():
        logits = model(
            input_ids=input_ids.to(args.device),
            attention_mask=attention_mask.to(args.device),
        ).logits
        if args.multi_label:
            pred_class = int(torch.sigmoid(logits)[0].argmax().item())
        else:
            pred_class = int(logits.argmax(dim=1).item())

    target_class = pred_class if args.target_class is None else args.target_class
    if args.target_labels:
        target_class = int(args.target_labels.split(",")[0])

    saliency = compute_rise_saliency_seq(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pred_class=target_class,
        mask_token_id=tokenizer.mask_token_id,
        num_masks=args.num_masks,
        p1=args.p1,
        device=args.device,
        batch_size=args.batch_size,
        special_token_ids=tokenizer.all_special_ids,
        multi_label=args.multi_label,
        s=args.s,
    )

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    saliency_out = [
        {"position": i, "token": tok, "saliency": float(saliency[i])}
        for i, tok in enumerate(tokens)
    ]

    np.save(out_dir / "saliency.npy", np.asarray(saliency, dtype=np.float32))
    with (out_dir / "saliency.csv").open("w", encoding="utf-8") as f:
        f.write("position,token,saliency\n")
        for row in saliency_out:
            f.write(f"{row['position']},{row['token']},{row['saliency']:.6f}\n")

    print(f"Saved RISE sequence saliency to: {out_dir}")
    print(f"Sequence name: {name}")
    print(f"Predicted class index: {pred_class}")
    print(f"Explained class index: {target_class}")


if __name__ == "__main__":
    main()

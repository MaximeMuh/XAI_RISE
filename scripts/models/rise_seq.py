import numpy as np
import torch
import tqdm as tqdm


def _build_mask(
    num_masks,
    length,
    p1,
    device,
    s=None,
    attention_mask=None,
    input_ids=None,
    special_token_ids=None,
):
    if s is not None and 0 < s < length:
        coarse = (torch.rand((num_masks, s), device=device) < p1).float()
        coarse = coarse.unsqueeze(1)  # (N,1,s)
        masks = torch.nn.functional.interpolate(
            coarse, size=length, mode="linear", align_corners=False
        ).squeeze(1)  # (N,L) in [0,1]
    else:
        masks = (torch.rand((num_masks, length), device=device) < p1).float()

    if attention_mask is not None:
        masks = masks * attention_mask.float()

    if input_ids is not None and special_token_ids:
        special = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in special_token_ids:
            special = special | (input_ids == token_id)
        masks = masks.masked_fill(special.expand_as(masks), 1.0)

    return masks


def compute_rise_saliency_seq(
    model,
    input_ids,
    attention_mask,
    pred_class,
    mask_token_id,
    num_masks=300,
    p1=0.5,
    device="cpu",
    batch_size=32,
    special_token_ids=None,
    multi_label=False,
    s=None,
):
    if batch_size is None or batch_size <= 0:
        batch_size = num_masks

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    length = input_ids.shape[1]

    masks = _build_mask(
        num_masks=num_masks,
        length=length,
        p1=p1,
        device=device,
        s=s,
        attention_mask=attention_mask,
        input_ids=input_ids,
        special_token_ids=special_token_ids,
    )

    saliency = torch.zeros(length, device=device)
    model.eval()

    with torch.no_grad():
        for i in tqdm.tqdm(range(0, num_masks, batch_size), desc="RISE-SEQ"):
            batch_masks = masks[i:i + batch_size]  # (B, L)
            masked_input_ids = input_ids.repeat(batch_masks.shape[0], 1)
            to_mask = batch_masks < 0.5
            masked_input_ids[to_mask] = mask_token_id

            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask.repeat(batch_masks.shape[0], 1),
            )
            logits = outputs.logits
            num_labels = logits.shape[1]
            if pred_class >= num_labels:
                raise ValueError(
                    f"pred_class {pred_class} is out of bounds for num_labels {num_labels}. "
                    "Use a label index compatible with the model head."
                )
            if multi_label:
                probs = torch.sigmoid(logits)[:, pred_class]  # (B,)
            else:
                probs = torch.softmax(logits, dim=1)[:, pred_class]  # (B,)
            saliency += (probs[:, None] * batch_masks).sum(dim=0)

    saliency = saliency / num_masks
    saliency = saliency.cpu().numpy()
    saliency = (
        (saliency - saliency.min())
        / (saliency.max() - saliency.min() + 1e-8)
    )
    return saliency.astype(np.float32)

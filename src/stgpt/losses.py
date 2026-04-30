from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .models import ImageGeneSTGPTOutput


def masked_mse(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    active = mask.bool()
    if not active.any():
        return input.sum() * 0.0
    return F.mse_loss(input[active], target[active])


def image_gene_contrastive_loss(region_emb: Tensor, image_emb: Tensor, temperature: float = 0.07) -> Tensor:
    if region_emb.shape[0] < 2:
        return region_emb.sum() * 0.0
    logits = region_emb @ image_emb.T / temperature
    labels = torch.arange(region_emb.shape[0], device=region_emb.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def compute_losses(
    output: ImageGeneSTGPTOutput,
    batch: dict[str, Tensor],
    *,
    image_gene_weight: float,
    neighborhood_weight: float,
    structure_weight: float,
) -> dict[str, Tensor]:
    mask = batch["mask"] & ~batch["gene_padding_mask"]
    gene = masked_mse(output.gene_pred, batch["target_values"], mask)
    neighbor = masked_mse(output.neighbor_pred, batch["neighbor_values"], ~batch["gene_padding_mask"])
    contrastive = image_gene_contrastive_loss(output.region_emb, output.image_emb)
    if output.structure_logits is not None and "structure_labels" in batch:
        structure = F.cross_entropy(output.structure_logits, batch["structure_labels"].long())
    else:
        structure = gene.sum() * 0.0
    total = gene + neighborhood_weight * neighbor + image_gene_weight * contrastive + structure_weight * structure
    return {
        "loss": total,
        "gene_loss": gene.detach(),
        "neighbor_loss": neighbor.detach(),
        "image_gene_loss": contrastive.detach(),
        "structure_loss": structure.detach(),
    }

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ImageGeneSTGPTOutput:
    gene_pred: Tensor
    neighbor_pred: Tensor
    cell_emb: Tensor
    image_emb: Tensor
    structure_logits: Tensor | None


class PatchEncoder(nn.Module):
    def __init__(self, image_channels: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image)


class ImageGeneSTGPT(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        n_structures: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
        n_expression_bins: int = 51,
        image_channels: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_genes = int(n_genes)
        self.n_structures = int(max(1, n_structures))
        self.d_model = int(d_model)
        self.gene_embedding = nn.Embedding(self.n_genes + 1, d_model, padding_idx=0)
        self.expression_value = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.expression_bin = nn.Embedding(n_expression_bins, d_model)
        self.patch_encoder = PatchEncoder(image_channels, d_model)
        self.spatial_encoder = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.context_embedding = nn.Embedding(self.n_structures + 1, d_model, padding_idx=0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward or d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.gene_decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.neighbor_decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.structure_head = nn.Linear(d_model, self.n_structures) if self.n_structures > 1 else None
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        *,
        gene_ids: Tensor,
        expr_values: Tensor,
        expr_bins: Tensor,
        image: Tensor,
        spatial: Tensor,
        context_ids: Tensor | None = None,
        gene_padding_mask: Tensor | None = None,
    ) -> ImageGeneSTGPTOutput:
        batch_size, seq_len = gene_ids.shape
        if gene_padding_mask is None:
            gene_padding_mask = gene_ids.eq(0)
        if context_ids is None:
            context_ids = torch.zeros(batch_size, dtype=torch.long, device=gene_ids.device)

        gene_tok = self.gene_embedding(gene_ids)
        value_tok = self.expression_value(expr_values.unsqueeze(-1))
        bin_tok = self.expression_bin(expr_bins.clamp_min(0))
        gene_tokens = gene_tok + value_tok + bin_tok

        image_emb = self.patch_encoder(image)
        spatial_emb = self.spatial_encoder(spatial.float())
        context_emb = self.context_embedding(context_ids.clamp(min=0, max=self.n_structures))
        cls = self.cls_token.expand(batch_size, -1, -1)
        prefix = torch.stack([image_emb, spatial_emb, context_emb], dim=1)
        tokens = torch.cat([cls, prefix, gene_tokens], dim=1)

        prefix_mask = torch.zeros(batch_size, 4, dtype=torch.bool, device=gene_ids.device)
        padding_mask = torch.cat([prefix_mask, gene_padding_mask], dim=1)
        encoded = self.final_norm(self.transformer(tokens, src_key_padding_mask=padding_mask))
        cell_emb = encoded[:, 0, :]
        gene_out = encoded[:, 4 : 4 + seq_len, :]
        gene_pred = self.gene_decoder(gene_out).squeeze(-1)
        neighbor_pred = self.neighbor_decoder(gene_out).squeeze(-1)
        structure_logits = self.structure_head(cell_emb) if self.structure_head is not None else None
        return ImageGeneSTGPTOutput(
            gene_pred=gene_pred,
            neighbor_pred=neighbor_pred,
            cell_emb=F.normalize(cell_emb, dim=1),
            image_emb=F.normalize(image_emb, dim=1),
            structure_logits=structure_logits,
        )

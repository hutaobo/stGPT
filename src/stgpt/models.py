from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    def __init__(self, image_channels: int, d_model: int, *, scales: list[int] | tuple[int, ...] = (1,)) -> None:
        super().__init__()
        self.scales = tuple(sorted({int(scale) for scale in scales if int(scale) >= 1})) or (1,)
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
        self.fusion = (
            nn.Sequential(nn.Linear(d_model * len(self.scales), d_model), nn.GELU(), nn.LayerNorm(d_model))
            if len(self.scales) > 1
            else nn.Identity()
        )

    def forward(self, image: Tensor) -> Tensor:
        features = []
        for scale in self.scales:
            if scale == 1:
                scaled = image
            else:
                scaled = F.avg_pool2d(image, kernel_size=scale, stride=scale, ceil_mode=True)
            features.append(self.net(scaled))
        return self.fusion(torch.cat(features, dim=1) if len(features) > 1 else features[0])


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
        patch_scales: list[int] | tuple[int, ...] = (1,),
        use_expression_values: bool = True,
        use_image_context: bool = True,
        use_spatial_context: bool = True,
        use_structure_context: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_genes = int(n_genes)
        self.n_structures = int(max(1, n_structures))
        self.d_model = int(d_model)
        self.use_expression_values = bool(use_expression_values)
        self.use_image_context = bool(use_image_context)
        self.use_spatial_context = bool(use_spatial_context)
        self.use_structure_context = bool(use_structure_context)
        self.gene_embedding = nn.Embedding(self.n_genes + 1, d_model, padding_idx=0)
        self.expression_value = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.expression_bin = nn.Embedding(n_expression_bins, d_model)
        self.patch_encoder = PatchEncoder(image_channels, d_model, scales=patch_scales)
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

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str = "auto",
    ) -> ImageGeneSTGPT:
        """Load a pretrained model from a checkpoint produced by ``stgpt train``.

        The model is loaded in evaluation mode on the requested device and is ready
        for inference.  Vocabulary, model architecture, and structure count are all
        recovered from the checkpoint so no additional config is required.

        Args:
            checkpoint: Path to a ``*.pt`` checkpoint file written by
                :func:`stgpt.training.train`.
            device: PyTorch device string: ``"auto"``, ``"cpu"``, or ``"cuda"``.
                ``"auto"`` selects CUDA when available, otherwise CPU.

        Returns:
            :class:`ImageGeneSTGPT` in eval mode on the requested device.

        Example::

            from stgpt.models import ImageGeneSTGPT

            model = ImageGeneSTGPT.from_pretrained("outputs/train/checkpoints/last.pt")
            model.eval()
        """
        from .config import StGPTConfig  # local import keeps models.py self-contained

        payload: dict[str, Any] = torch.load(checkpoint, map_location="cpu")
        cfg = StGPTConfig.model_validate(payload["config"])
        genes: list[str] = payload.get("vocab", {}).get("genes", [])
        n_genes = len(genes) if genes else cfg.data.n_genes
        n_structures = int(payload.get("n_structures", 1))

        model = cls(
            n_genes=n_genes,
            n_structures=n_structures,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.n_layers,
            dim_feedforward=cfg.model.dim_feedforward,
            n_expression_bins=cfg.model.n_expression_bins,
            image_channels=cfg.model.image_channels,
            dropout=cfg.model.dropout,
        )
        model.load_state_dict(payload["model_state"], strict=False)
        target = _resolve_device(device)
        model.to(target)
        model.eval()
        return model

    @staticmethod
    def load_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
        """Load a raw checkpoint dict from disk.

        Returns the full checkpoint payload (model state dict, config, vocab,
        structure metadata, and training metrics) as a plain dictionary.  This is
        intended for inspection and advanced usage; for standard inference use
        :meth:`from_pretrained` instead.

        Args:
            checkpoint: Path to a ``*.pt`` checkpoint file.

        Returns:
            The raw ``dict`` payload stored in the checkpoint.
        """
        return dict(torch.load(checkpoint, map_location="cpu"))

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
        if self.use_expression_values:
            value_tok = self.expression_value(expr_values.unsqueeze(-1))
            bin_tok = self.expression_bin(expr_bins.clamp_min(0))
        else:
            value_tok = torch.zeros_like(gene_tok)
            bin_tok = torch.zeros_like(gene_tok)
        gene_tokens = gene_tok + value_tok + bin_tok

        image_emb = self.patch_encoder(image)
        if not self.use_image_context:
            image_emb = torch.zeros_like(image_emb)
        spatial_emb = self.spatial_encoder(spatial.float())
        if not self.use_spatial_context:
            spatial_emb = torch.zeros_like(spatial_emb)
        context_emb = self.context_embedding(context_ids.clamp(min=0, max=self.n_structures))
        if not self.use_structure_context:
            context_emb = torch.zeros_like(context_emb)
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


def _resolve_device(name: str) -> torch.device:
    normalized = str(name).lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)

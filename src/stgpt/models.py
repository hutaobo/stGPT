from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ImageGeneSTGPTOutput:
    gene_pred: Tensor
    neighbor_pred: Tensor
    region_emb: Tensor
    image_emb: Tensor
    structure_logits: Tensor | None
    prototype_logits: Tensor | None = None
    prototype_probs: Tensor | None = None
    prototype_ids: Tensor | None = None
    prototype_confidence: Tensor | None = None

    @property
    def cell_emb(self) -> Tensor:
        """Compatibility alias; stGPT now trains region embeddings."""
        return self.region_emb


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


ImageEncoderBackend = Literal["cnn", "timm", "hf", "precomputed"]
ImageEncoderPreset = Literal["virchow", "virchow2"]

VIRCHOW_MODEL_IDS: dict[str, str] = {
    "virchow": "hf-hub:paige-ai/Virchow",
    "virchow2": "hf-hub:paige-ai/Virchow2",
}

VIRCHOW_PATCH_TOKEN_START: dict[str, int] = {
    "virchow": 1,
    "virchow2": 5,
}


@dataclass(frozen=True)
class ImageEncoderResolvedSpec:
    backend: str
    preset: str | None
    name: str | None
    image_size: int | None
    input_mode: str
    normalization_source: str
    embedding_strategy: str
    gated_access: bool


def resolve_image_encoder_spec(
    *,
    backend: str,
    name: str | None = None,
    preset: str | None = None,
) -> ImageEncoderResolvedSpec:
    normalized_backend = str(backend)
    normalized_preset = str(preset).lower() if preset else None
    if normalized_preset in VIRCHOW_MODEL_IDS:
        return ImageEncoderResolvedSpec(
            backend="timm",
            preset=normalized_preset,
            name=VIRCHOW_MODEL_IDS[normalized_preset],
            image_size=224,
            input_mode="RGB",
            normalization_source="timm.resolve_data_config(pretrained_cfg)",
            embedding_strategy="class_token_plus_mean_patch_tokens",
            gated_access=True,
        )
    if normalized_preset is not None:
        raise ValueError("model.image_encoder_preset must be one of: virchow, virchow2")
    return ImageEncoderResolvedSpec(
        backend=normalized_backend,
        preset=None,
        name=name,
        image_size=None,
        input_mode="RGB",
        normalization_source="stgpt.load_image_tensor_0_1",
        embedding_strategy="global_pool_or_mean_tokens",
        gated_access=False,
    )


class TimmImageEncoder(nn.Module):
    """Frozen timm feature extractor plus a trainable projection into stGPT space."""

    def __init__(self, name: str, d_model: int, *, frozen: bool = True, preset: str | None = None) -> None:
        super().__init__()
        try:
            import timm  # type: ignore[import-untyped]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install timm to use model.image_encoder_backend='timm'.") from exc
        spec = resolve_image_encoder_spec(backend="timm", name=name, preset=preset)
        self.name = str(spec.name)
        self.preset = spec.preset
        self.expected_image_size = spec.image_size
        self.frozen = bool(frozen)
        try:
            create_kwargs = self._create_kwargs()
            self.encoder = timm.create_model(self.name, pretrained=True, **create_kwargs)
        except Exception as exc:  # pragma: no cover - depends on optional remote model access
            hint = (
                " Virchow/Virchow2 are gated Hugging Face models; run `huggingface-cli login` "
                "and accept the model terms before using this preset."
                if self.preset in VIRCHOW_MODEL_IDS
                else ""
            )
            raise RuntimeError(f"Could not create timm image encoder {self.name!r}.{hint}") from exc
        feature_dim = int(getattr(self.encoder, "num_features", 0) or 0)
        if self.preset in VIRCHOW_MODEL_IDS and feature_dim > 0:
            feature_dim *= 2
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, d_model) if feature_dim > 0 else nn.LazyLinear(d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        if self.frozen:
            self.encoder.eval()
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        self._register_preprocessing_buffers()

    def forward(self, image: Tensor) -> Tensor:
        image = self._preprocess_image(image)
        if self.frozen:
            self.encoder.eval()
            with torch.no_grad():
                features = self.encoder(image)
        else:
            features = self.encoder(image)
        if isinstance(features, (list, tuple)):
            features = features[0]
        if self.preset in VIRCHOW_MODEL_IDS and features.ndim == 3:
            patch_start = VIRCHOW_PATCH_TOKEN_START[str(self.preset)]
            class_token = features[:, 0]
            patch_tokens = features[:, patch_start:]
            features = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=1)
        elif features.ndim == 3:
            features = features.mean(dim=1)
        if features.ndim > 2:
            features = features.flatten(start_dim=2).mean(dim=-1)
        return self.projection(features.float())

    def _create_kwargs(self) -> dict[str, Any]:
        if self.preset in VIRCHOW_MODEL_IDS:
            try:
                from timm.layers import SwiGLUPacked  # type: ignore[import-untyped]
            except Exception as exc:  # pragma: no cover - optional dependency version issue
                raise RuntimeError("Virchow presets require timm.layers.SwiGLUPacked.") from exc
            return {"mlp_layer": SwiGLUPacked, "act_layer": torch.nn.SiLU}
        return {"num_classes": 0, "global_pool": "avg"}

    def _register_preprocessing_buffers(self) -> None:
        cfg = getattr(self.encoder, "pretrained_cfg", {}) or {}
        mean = cfg.get("mean", (0.0, 0.0, 0.0))
        std = cfg.get("std", (1.0, 1.0, 1.0))
        self.register_buffer("input_mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)

    def _preprocess_image(self, image: Tensor) -> Tensor:
        values = image.float()
        if self.expected_image_size and tuple(values.shape[-2:]) != (self.expected_image_size, self.expected_image_size):
            values = F.interpolate(
                values,
                size=(self.expected_image_size, self.expected_image_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.input_mean.shape[1] == values.shape[1]:
            values = (values - self.input_mean.to(values.device, values.dtype)) / self.input_std.to(values.device, values.dtype)
        return values


class HFImageEncoder(nn.Module):
    """Frozen Hugging Face vision backbone plus a trainable projection."""

    def __init__(self, name: str, d_model: int, *, frozen: bool = True) -> None:
        super().__init__()
        try:
            from transformers import AutoModel  # type: ignore[import-untyped]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install transformers to use model.image_encoder_backend='hf'.") from exc
        self.name = str(name)
        self.frozen = bool(frozen)
        self.encoder = AutoModel.from_pretrained(self.name)
        feature_dim = int(getattr(getattr(self.encoder, "config", None), "hidden_size", 0) or 0)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, d_model) if feature_dim > 0 else nn.LazyLinear(d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        if self.frozen:
            self.encoder.eval()
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def forward(self, image: Tensor) -> Tensor:
        if self.frozen:
            self.encoder.eval()
            with torch.no_grad():
                output = self.encoder(pixel_values=image)
        else:
            output = self.encoder(pixel_values=image)
        features = getattr(output, "pooler_output", None)
        if features is None:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, (list, tuple)) and output:
                hidden = output[0]
            if hidden is None:
                raise RuntimeError("Hugging Face image encoder did not return pooler_output or last_hidden_state.")
            features = hidden[:, 0] if hidden.ndim == 3 else hidden
        return self.projection(features.float())


def build_image_encoder(
    *,
    backend: ImageEncoderBackend,
    image_channels: int,
    d_model: int,
    scales: list[int] | tuple[int, ...] = (1,),
    name: str | None = None,
    frozen: bool = True,
    preset: str | None = None,
) -> nn.Module:
    if backend == "cnn":
        return PatchEncoder(image_channels, d_model, scales=scales)
    if backend == "timm":
        spec = resolve_image_encoder_spec(backend=backend, name=name, preset=preset)
        if not spec.name:
            raise ValueError("model.image_encoder_name is required for image_encoder_backend='timm'.")
        return TimmImageEncoder(spec.name, d_model, frozen=frozen, preset=preset)
    if backend == "hf":
        if not name:
            raise ValueError("model.image_encoder_name is required for image_encoder_backend='hf'.")
        return HFImageEncoder(name, d_model, frozen=frozen)
    raise ValueError("build_image_encoder does not build the precomputed backend.")


def image_encoder_provenance(config: Any) -> dict[str, Any]:
    model = getattr(config, "model", config)
    data = getattr(config, "data", None)
    backend = str(getattr(model, "image_encoder_backend", "cnn"))
    if data is not None and getattr(data, "image_embedding_store", None):
        backend = "precomputed"
    spec = resolve_image_encoder_spec(
        backend=backend,
        name=getattr(model, "image_encoder_name", None),
        preset=getattr(model, "image_encoder_preset", None),
    )
    return {
        "backend": backend,
        "preset": spec.preset,
        "name": spec.name,
        "frozen": bool(getattr(model, "image_encoder_frozen", True)),
        "image_embedding_dim": getattr(model, "image_embedding_dim", None),
        "image_size": spec.image_size or getattr(model, "image_size", None),
        "image_channels": getattr(model, "image_channels", None),
        "patch_scales": list(getattr(model, "patch_scales", [1])),
        "input_mode": spec.input_mode,
        "normalization_source": spec.normalization_source,
        "embedding_strategy": spec.embedding_strategy,
        "gated_access": spec.gated_access,
        "image_embedding_store": getattr(data, "image_embedding_store", None) if data is not None else None,
        "stain_normalization": getattr(data, "image_stain_normalization", None) if data is not None else None,
    }


class ContourEvidenceEncoder(nn.Module):
    def __init__(
        self,
        image_channels: int,
        d_model: int,
        *,
        scales: list[int] | tuple[int, ...] = (1,),
        image_encoder_backend: ImageEncoderBackend = "cnn",
        image_encoder_preset: ImageEncoderPreset | None = None,
        image_encoder_name: str | None = None,
        image_encoder_frozen: bool = True,
        image_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder_backend = str(image_encoder_backend)
        self.image_embedding_dim = image_embedding_dim
        if image_encoder_backend == "precomputed":
            self.object_encoder = None
            self.context_encoder = None
        else:
            self.object_encoder = build_image_encoder(
                backend=image_encoder_backend,
                image_channels=image_channels,
                d_model=d_model,
                scales=scales,
                name=image_encoder_name,
                frozen=image_encoder_frozen,
                preset=image_encoder_preset,
            )
            self.context_encoder = build_image_encoder(
                backend=image_encoder_backend,
                image_channels=image_channels,
                d_model=d_model,
                scales=scales,
                name=image_encoder_name,
                frozen=image_encoder_frozen,
                preset=image_encoder_preset,
            )
        self.precomputed_projection = (
            self._precomputed_projection(image_embedding_dim, d_model)
            if image_encoder_backend == "precomputed"
            else nn.Identity()
        )
        self.shape_encoder = nn.Sequential(nn.LazyLinear(d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.token_norm = nn.LayerNorm(d_model)
        self.summary = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(
        self,
        *,
        object_image: Tensor,
        context_image: Tensor | None = None,
        contour_mask: Tensor | None = None,
        contour_geometry: Tensor | None = None,
        precomputed_image_embedding: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if precomputed_image_embedding is not None and precomputed_image_embedding.numel() > 0:
            image_token = self.precomputed_projection(precomputed_image_embedding.float())
            shape_token = self._shape_token(object_image, contour_geometry)
            tokens = self.token_norm(torch.stack([image_token, image_token, shape_token], dim=1))
            return tokens, self.summary(tokens.flatten(start_dim=1))
        if self.object_encoder is None or self.context_encoder is None:
            raise ValueError("precomputed image encoder backend requires precomputed_image_embedding tensors.")
        context_image = object_image if context_image is None else context_image
        masked_object = self._masked_object(object_image, contour_mask)
        object_token = self.object_encoder(masked_object)
        context_token = self.context_encoder(context_image)
        shape_token = self._shape_token(object_image, contour_geometry)
        tokens = self.token_norm(torch.stack([object_token, context_token, shape_token], dim=1))
        image_emb = self.summary(tokens.flatten(start_dim=1))
        return tokens, image_emb

    @staticmethod
    def _precomputed_projection(image_embedding_dim: int | None, d_model: int) -> nn.Module:
        if image_embedding_dim == d_model:
            return nn.Identity()
        return nn.Sequential(
            nn.Linear(image_embedding_dim, d_model) if image_embedding_dim else nn.LazyLinear(d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _masked_object(object_image: Tensor, contour_mask: Tensor | None) -> Tensor:
        if contour_mask is None or contour_mask.numel() == 0:
            return object_image
        mask = contour_mask.to(device=object_image.device, dtype=object_image.dtype)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != object_image.shape[-2:]:
            mask = F.interpolate(mask, size=object_image.shape[-2:], mode="nearest")
        if mask.shape[1] == 1 and object_image.shape[1] > 1:
            mask = mask.expand(-1, object_image.shape[1], -1, -1)
        fill = object_image.mean(dim=(-2, -1), keepdim=True)
        return object_image * mask + fill * (1.0 - mask)

    def _shape_token(self, object_image: Tensor, contour_geometry: Tensor | None) -> Tensor:
        batch_size = int(object_image.shape[0])
        if contour_geometry is None or contour_geometry.numel() == 0:
            contour_geometry = torch.zeros(batch_size, 1, dtype=object_image.dtype, device=object_image.device)
        else:
            contour_geometry = contour_geometry.to(device=object_image.device, dtype=object_image.dtype)
            if contour_geometry.ndim == 1:
                contour_geometry = contour_geometry.unsqueeze(1)
        return self.shape_encoder(contour_geometry)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = dim_feedforward or d_model * 4
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ffn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, query_tokens: Tensor, evidence_tokens: Tensor) -> Tensor:
        if evidence_tokens.numel() == 0:
            return query_tokens
        attn_out, _ = self.cross_attn(
            query=self.norm1(query_tokens),
            key=evidence_tokens,
            value=evidence_tokens,
            need_weights=False,
        )
        hidden = query_tokens + torch.tanh(self.attn_gate) * self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(hidden))
        return hidden + torch.tanh(self.ffn_gate) * self.dropout(ffn_out)


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
        image_encoder_backend: ImageEncoderBackend = "cnn",
        image_encoder_preset: ImageEncoderPreset | None = None,
        image_encoder_name: str | None = None,
        image_encoder_frozen: bool = True,
        image_embedding_dim: int | None = None,
        n_prototypes: int = 0,
        prototype_temperature: float = 0.1,
        use_expression_values: bool = True,
        use_image_context: bool = True,
        use_spatial_context: bool = True,
        use_structure_context: bool = True,
        use_cell_context: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_genes = int(n_genes)
        self.n_structures = int(max(1, n_structures))
        self.n_prototypes = int(max(0, n_prototypes))
        self.prototype_temperature = float(prototype_temperature)
        self.d_model = int(d_model)
        self.use_expression_values = bool(use_expression_values)
        self.use_image_context = bool(use_image_context)
        self.use_spatial_context = bool(use_spatial_context)
        self.use_structure_context = bool(use_structure_context)
        self.use_cell_context = bool(use_cell_context)
        self.image_encoder_backend = str(image_encoder_backend)
        self.image_encoder_preset = image_encoder_preset
        self.image_encoder_name = image_encoder_name
        self.image_encoder_frozen = bool(image_encoder_frozen)
        self.image_embedding_dim = image_embedding_dim
        self.gene_embedding = nn.Embedding(self.n_genes + 1, d_model, padding_idx=0)
        self.expression_value = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.expression_bin = nn.Embedding(n_expression_bins, d_model)
        self.patch_encoder = PatchEncoder(image_channels, d_model, scales=patch_scales)
        self.contour_encoder = ContourEvidenceEncoder(
            image_channels,
            d_model,
            scales=patch_scales,
            image_encoder_backend=image_encoder_backend,
            image_encoder_preset=image_encoder_preset,
            image_encoder_name=image_encoder_name,
            image_encoder_frozen=image_encoder_frozen,
            image_embedding_dim=image_embedding_dim,
        )
        self.spatial_encoder = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.context_embedding = nn.Embedding(self.n_structures + 1, d_model, padding_idx=0)
        self.cell_context_norm = nn.LayerNorm(d_model)
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
        self.gated_fusion = GatedCrossAttentionBlock(
            d_model,
            n_heads,
            dim_feedforward=dim_feedforward or d_model * 4,
            dropout=dropout,
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.gene_decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.neighbor_decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.structure_head = nn.Linear(d_model, self.n_structures) if self.n_structures > 1 else None
        self.prototype_head = nn.Linear(d_model, self.n_prototypes, bias=False) if self.n_prototypes > 0 else None
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
            patch_scales=cfg.model.patch_scales,
            image_encoder_backend="precomputed" if cfg.data.image_embedding_store else cfg.model.image_encoder_backend,
            image_encoder_preset=cfg.model.image_encoder_preset,
            image_encoder_name=cfg.model.image_encoder_name,
            image_encoder_frozen=cfg.model.image_encoder_frozen,
            image_embedding_dim=cfg.model.image_embedding_dim,
            n_prototypes=cfg.model.n_prototypes,
            prototype_temperature=cfg.model.prototype_temperature,
            use_expression_values=cfg.model.use_expression_values,
            use_image_context=cfg.model.use_image_context,
            use_spatial_context=cfg.model.use_spatial_context,
            use_structure_context=cfg.model.use_structure_context,
            use_cell_context=cfg.model.use_cell_context,
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
        cell_expr_values: Tensor | None = None,
        cell_token_mask: Tensor | None = None,
        object_image: Tensor | None = None,
        context_image: Tensor | None = None,
        contour_mask: Tensor | None = None,
        contour_geometry: Tensor | None = None,
        precomputed_image_embedding: Tensor | None = None,
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

        contour_tokens, image_emb = self._contour_tokens(
            image=image,
            object_image=object_image,
            context_image=context_image,
            contour_mask=contour_mask,
            contour_geometry=contour_geometry,
            precomputed_image_embedding=precomputed_image_embedding,
        )
        if not self.use_image_context:
            image_emb = torch.zeros_like(image_emb)
            contour_tokens = torch.zeros_like(contour_tokens)
        spatial_emb = self.spatial_encoder(spatial.float())
        if not self.use_spatial_context:
            spatial_emb = torch.zeros_like(spatial_emb)
        context_emb = self.context_embedding(context_ids.clamp(min=0, max=self.n_structures))
        if not self.use_structure_context:
            context_emb = torch.zeros_like(context_emb)
        evidence_tokens = torch.cat([contour_tokens, torch.stack([spatial_emb, context_emb], dim=1)], dim=1)
        cell_tokens, cell_padding_mask = self._cell_context_tokens(gene_tok, cell_expr_values, cell_token_mask)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, cell_tokens, gene_tokens], dim=1)

        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=gene_ids.device)
        padding_mask = torch.cat([cls_mask, cell_padding_mask, gene_padding_mask], dim=1)
        encoded = self.final_norm(self._run_gated_midfusion(tokens, evidence_tokens, padding_mask))
        region_emb = encoded[:, 0, :]
        normalized_region_emb = F.normalize(region_emb, dim=1)
        gene_start = 1 + cell_tokens.shape[1]
        gene_out = encoded[:, gene_start : gene_start + seq_len, :]
        gene_pred = self.gene_decoder(gene_out).squeeze(-1)
        neighbor_pred = self.neighbor_decoder(gene_out).squeeze(-1)
        structure_logits = self.structure_head(region_emb) if self.structure_head is not None else None
        prototype_logits, prototype_probs, prototype_ids, prototype_confidence = self._prototype_outputs(normalized_region_emb)
        return ImageGeneSTGPTOutput(
            gene_pred=gene_pred,
            neighbor_pred=neighbor_pred,
            region_emb=normalized_region_emb,
            image_emb=F.normalize(image_emb, dim=1),
            structure_logits=structure_logits,
            prototype_logits=prototype_logits,
            prototype_probs=prototype_probs,
            prototype_ids=prototype_ids,
            prototype_confidence=prototype_confidence,
        )

    def _prototype_outputs(self, region_emb: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        if self.prototype_head is None:
            return None, None, None, None
        prototype_weight = F.normalize(self.prototype_head.weight, dim=1)
        logits = F.linear(region_emb, prototype_weight)
        probs = torch.softmax(logits / max(self.prototype_temperature, 1e-6), dim=1)
        confidence, ids = probs.max(dim=1)
        return logits, probs, ids, confidence

    def _run_gated_midfusion(self, tokens: Tensor, evidence_tokens: Tensor, padding_mask: Tensor) -> Tensor:
        layers = list(self.transformer.layers)
        if not layers:
            return self.gated_fusion(tokens, evidence_tokens)
        fusion_after = max(0, len(layers) // 2 - 1)
        hidden = tokens
        for layer_idx, layer in enumerate(layers):
            hidden = layer(hidden, src_key_padding_mask=padding_mask)
            if layer_idx == fusion_after:
                hidden = self.gated_fusion(hidden, evidence_tokens)
        return hidden

    def _contour_tokens(
        self,
        *,
        image: Tensor,
        object_image: Tensor | None,
        context_image: Tensor | None,
        contour_mask: Tensor | None,
        contour_geometry: Tensor | None,
        precomputed_image_embedding: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        object_input = image if object_image is None else object_image
        tokens, image_emb = self.contour_encoder(
            object_image=object_input,
            context_image=context_image,
            contour_mask=contour_mask,
            contour_geometry=contour_geometry,
            precomputed_image_embedding=precomputed_image_embedding,
        )
        return tokens, image_emb

    def _cell_context_tokens(
        self,
        gene_tok: Tensor,
        cell_expr_values: Tensor | None,
        cell_token_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, d_model = gene_tok.shape
        device = gene_tok.device
        if cell_expr_values is None or cell_expr_values.numel() == 0 or not self.use_cell_context:
            return torch.zeros(batch_size, 0, d_model, device=device), torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        cell_expr_values = cell_expr_values.to(device)
        if cell_token_mask is None:
            cell_token_mask = torch.zeros(cell_expr_values.shape[:2], dtype=torch.bool, device=device)
        else:
            cell_token_mask = cell_token_mask.to(device)
        cell_values = self.expression_value(cell_expr_values.unsqueeze(-1))
        cell_gene_tokens = gene_tok.unsqueeze(1) + cell_values
        gene_mask = gene_tok.abs().sum(dim=-1).gt(0).float()
        denom = gene_mask.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
        tokens = (cell_gene_tokens * gene_mask[:, None, :, None]).sum(dim=2) / denom
        tokens = self.cell_context_norm(tokens)
        tokens = torch.where(cell_token_mask.unsqueeze(-1), torch.zeros_like(tokens), tokens)
        return tokens, cell_token_mask


def _resolve_device(name: str) -> torch.device:
    normalized = str(name).lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)

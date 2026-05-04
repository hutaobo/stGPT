from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from .models import ImageGeneSTGPTOutput


@dataclass(frozen=True)
class SinkhornResult:
    assignments: Tensor
    row_residual: Tensor
    col_residual: Tensor
    nonfinite_count: Tensor


class SinkhornModule(nn.Module):
    """FP32 numerical island for balanced prototype pseudo-label assignment."""

    def __init__(self, *, iterations: int = 3, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.iterations = int(max(1, iterations))
        self.epsilon = float(epsilon)

    def forward(self, logits: Tensor, *, temperature: float) -> SinkhornResult:
        if logits.ndim != 2:
            raise ValueError("SinkhornModule expects logits with shape [n_samples, n_prototypes].")
        if logits.shape[0] == 0 or logits.shape[1] == 0:
            raise ValueError("SinkhornModule requires at least one sample and one prototype.")
        with torch.no_grad(), _autocast_disabled(logits.device):
            scores = logits.detach().float() / max(float(temperature), self.epsilon)
            nonfinite_count = (~torch.isfinite(scores)).sum().float()
            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            scores = scores - scores.max(dim=1, keepdim=True).values
            q = torch.exp(scores).t()
            n_prototypes, local_n = q.shape
            global_n = _distributed_sum_scalar(torch.tensor(float(local_n), device=q.device))

            total = _distributed_sum_scalar(q.sum()).clamp_min(self.epsilon)
            q = q / total
            for _ in range(self.iterations):
                row_sum = _distributed_sum_tensor(q.sum(dim=1, keepdim=True)).clamp_min(self.epsilon)
                q = q / row_sum
                q = q / n_prototypes

                col_sum = q.sum(dim=0, keepdim=True).clamp_min(self.epsilon)
                q = q / col_sum
                q = q / global_n.clamp_min(1.0)

            q = q * global_n.clamp_min(1.0)
            assignments = q.t().contiguous()
            row_residual = assignments.sum(dim=1).sub(1.0).abs().max()
            col_sum = _distributed_sum_tensor(assignments.sum(dim=0))
            col_target = global_n / max(1, n_prototypes)
            col_residual = (col_sum - col_target).abs().max() / col_target.clamp_min(self.epsilon)
            return SinkhornResult(
                assignments=assignments,
                row_residual=row_residual,
                col_residual=col_residual,
                nonfinite_count=nonfinite_count,
            )


class ContourMemoryQueue(nn.Module):
    """Preallocated FIFO queue for detached contour/region embeddings."""

    def __init__(self, *, embedding_dim: int, queue_size: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.queue_size = int(max(0, queue_size))
        self.register_buffer("queue", torch.zeros(self.queue_size, self.embedding_dim, dtype=torch.float32))
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("queue_filled", torch.zeros((), dtype=torch.long))

    @property
    def filled(self) -> int:
        return int(self.queue_filled.item())

    def features(self) -> Tensor:
        filled = self.filled
        if filled <= 0:
            return self.queue[:0]
        return self.queue[:filled].detach()

    @torch.no_grad()
    def enqueue(self, features: Tensor) -> None:
        if self.queue_size <= 0 or features.numel() == 0:
            return
        incoming = F.normalize(features.detach().float(), dim=1)
        if incoming.shape[0] > self.queue_size:
            incoming = incoming[-self.queue_size :]
        count = int(incoming.shape[0])
        ptr = int(self.queue_ptr.item())
        end = ptr + count
        if end <= self.queue_size:
            self.queue[ptr:end].copy_(incoming)
        else:
            first = self.queue_size - ptr
            self.queue[ptr:].copy_(incoming[:first])
            self.queue[: end % self.queue_size].copy_(incoming[first:])
        self.queue_ptr.fill_(end % self.queue_size)
        self.queue_filled.fill_(min(self.queue_size, self.filled + count))


def compute_prototype_loss(
    output: ImageGeneSTGPTOutput,
    *,
    prototype_weight: Tensor | None,
    queue: ContourMemoryQueue | None,
    sinkhorn: SinkhornModule,
    temperature: float,
    step: int,
    queue_start_steps: int,
) -> dict[str, Tensor]:
    if output.prototype_logits is None or prototype_weight is None:
        zero = output.region_emb.sum() * 0.0
        return {"prototype_loss": zero}

    batch_logits = output.prototype_logits
    pool_logits = batch_logits.detach()
    queue_filled = torch.zeros((), dtype=torch.float32, device=batch_logits.device)
    if queue is not None:
        queue_filled = queue.queue_filled.to(device=batch_logits.device, dtype=torch.float32)
        queued = queue.features().to(batch_logits.device)
        if int(step) >= int(queue_start_steps) and queued.numel() > 0:
            normalized_weight = F.normalize(prototype_weight.detach().float(), dim=1)
            queue_logits = F.linear(queued.float(), normalized_weight)
            pool_logits = torch.cat([pool_logits, queue_logits.detach()], dim=0)

    sinkhorn_result = sinkhorn(pool_logits, temperature=temperature)
    q_batch = sinkhorn_result.assignments[: batch_logits.shape[0]].detach()
    log_probs = F.log_softmax(batch_logits.float() / max(float(temperature), 1e-6), dim=1)
    loss = -(q_batch * log_probs).sum(dim=1).mean()
    metrics = _assignment_metrics(q_batch)

    if queue is not None:
        queue.enqueue(output.region_emb)

    return {
        "prototype_loss": loss,
        "prototype_entropy": metrics["entropy"],
        "prototype_entropy_normalized": metrics["entropy_normalized"],
        "prototype_usage_count": metrics["usage_count"],
        "prototype_dead_codes": metrics["dead_codes"],
        "prototype_assignment_pool_size": torch.tensor(float(pool_logits.shape[0]), device=batch_logits.device),
        "prototype_queue_filled": queue_filled,
        "sinkhorn_row_residual": sinkhorn_result.row_residual.to(batch_logits.device),
        "sinkhorn_col_residual": sinkhorn_result.col_residual.to(batch_logits.device),
        "sinkhorn_nonfinite_count": sinkhorn_result.nonfinite_count.to(batch_logits.device),
    }


def _assignment_metrics(assignments: Tensor) -> dict[str, Tensor]:
    n_prototypes = assignments.shape[1]
    avg = assignments.float().mean(dim=0)
    entropy = -(avg * torch.log(avg.clamp_min(1e-8))).sum()
    entropy_max = torch.log(torch.tensor(float(max(1, n_prototypes)), device=assignments.device))
    hard = assignments.argmax(dim=1)
    usage_count = torch.bincount(hard, minlength=n_prototypes).gt(0).sum().float()
    return {
        "entropy": entropy,
        "entropy_normalized": entropy / entropy_max.clamp_min(1e-8),
        "usage_count": usage_count,
        "dead_codes": torch.tensor(float(n_prototypes), device=assignments.device) - usage_count,
    }


def _autocast_disabled(device: torch.device):
    if device.type in {"cpu", "cuda"}:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


def _distributed_sum_scalar(value: Tensor) -> Tensor:
    out = value.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def _distributed_sum_tensor(value: Tensor) -> Tensor:
    out = value.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out

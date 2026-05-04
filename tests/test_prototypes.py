from __future__ import annotations

import torch

from stgpt.prototypes import ContourMemoryQueue, SinkhornModule


def test_sinkhorn_module_is_stable_for_large_half_precision_logits() -> None:
    logits = torch.tensor(
        [[1000.0, -1000.0, 25.0], [500.0, 500.0, -500.0], [-800.0, 1200.0, 0.0], [1.0, 2.0, 3.0]],
        dtype=torch.float16,
    )
    sinkhorn = SinkhornModule(iterations=5)

    result = sinkhorn(logits, temperature=0.05)

    assert result.assignments.dtype == torch.float32
    assert torch.isfinite(result.assignments).all()
    assert torch.allclose(result.assignments.sum(dim=1), torch.ones(logits.shape[0]), atol=1e-4)
    assert torch.isfinite(result.row_residual)
    assert torch.isfinite(result.col_residual)


def test_contour_memory_queue_wraps_without_resizing() -> None:
    queue = ContourMemoryQueue(embedding_dim=3, queue_size=5)
    first = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    second = torch.arange(12, 21, dtype=torch.float32).reshape(3, 3)

    initial_storage = queue.queue.data_ptr()
    queue.enqueue(first)
    queue.enqueue(second)

    assert queue.queue.data_ptr() == initial_storage
    assert queue.filled == 5
    assert int(queue.queue_ptr.item()) == 2
    assert queue.features().shape == (5, 3)
    norms = torch.linalg.vector_norm(queue.features(), dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

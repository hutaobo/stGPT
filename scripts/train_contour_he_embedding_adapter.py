"""Train a neural naming adapter on contour H&E foundation embeddings."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class AdapterConfig:
    embeddings: str
    manifest: str
    output: str
    hidden_dim: int
    depth: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    seed: int
    device: str


class EmbeddingDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, x: np.ndarray, fine: np.ndarray, parent: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.fine = torch.from_numpy(fine.astype(np.int64, copy=False))
        self.parent = torch.from_numpy(parent.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "fine": self.fine[index], "parent": self.parent[index]}


class ContourHEAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, depth: int, dropout: float, n_fine: int, n_parent: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_dim
        for _ in range(max(1, depth)):
            layers.extend(
                [
                    nn.Linear(current, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current = hidden_dim
        self.adapter = nn.Sequential(*layers)
        self.fine_head = nn.Linear(current, n_fine)
        self.parent_head = nn.Linear(current, n_parent)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.adapter(x)
        return {"fine": self.fine_head(z), "parent": self.parent_head(z)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = AdapterConfig(
        embeddings=str(args.embeddings),
        manifest=str(args.manifest),
        output=str(args.output),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed_everything(args.seed)

    manifest = pd.read_csv(args.manifest)
    embeddings = np.load(args.embeddings)
    if len(manifest) != embeddings.shape[0]:
        raise ValueError(f"manifest rows ({len(manifest)}) != embeddings rows ({embeddings.shape[0]})")
    required = {"slide", "trainable_standard_name", "standard_parent_class"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")

    splits = slide_heldout_splits(manifest, seed=args.seed)
    train_idx = splits["train"]
    val_idx = filter_seen_labels(manifest, splits["val"], train_idx)
    test_idx = filter_seen_labels(manifest, splits["test"], train_idx)
    excluded = {
        "val_unseen_labels": int(len(splits["val"]) - len(val_idx)),
        "test_unseen_labels": int(len(splits["test"]) - len(test_idx)),
    }

    fine_encoder = LabelEncoder().fit(manifest.iloc[train_idx]["trainable_standard_name"].astype(str))
    parent_encoder = LabelEncoder().fit(manifest.iloc[train_idx]["standard_parent_class"].astype(str))
    fine_ids = encode_with_train_classes(manifest["trainable_standard_name"].astype(str), fine_encoder)
    parent_ids = encode_with_train_classes(manifest["standard_parent_class"].astype(str), parent_encoder)

    mean = embeddings[train_idx].mean(axis=0, keepdims=True)
    std = embeddings[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x = (embeddings - mean) / std

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = ContourHEAdapter(
        in_dim=x.shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        n_fine=len(fine_encoder.classes_),
        n_parent=len(parent_encoder.classes_),
    ).to(device)

    train_loader = make_loader(x, fine_ids, parent_ids, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(x, fine_ids, parent_ids, val_idx, args.batch_size, shuffle=False)
    fine_weight = class_weight(fine_ids[train_idx], len(fine_encoder.classes_)).to(device)
    parent_weight = class_weight(parent_ids[train_idx], len(parent_encoder.classes_)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    fine_loss_fn = nn.CrossEntropyLoss(weight=fine_weight)
    parent_loss_fn = nn.CrossEntropyLoss(weight=parent_weight)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, fine_loss_fn, parent_loss_fn, device)
        val_metrics = evaluate(model, val_loader, fine_encoder, parent_encoder, device)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        score = float(val_metrics["fine_macro_f1"] + 0.5 * val_metrics["parent_macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    train_metrics = evaluate(model, make_loader(x, fine_ids, parent_ids, train_idx, args.batch_size, False), fine_encoder, parent_encoder, device)
    val_metrics = evaluate(model, val_loader, fine_encoder, parent_encoder, device)
    test_loader = make_loader(x, fine_ids, parent_ids, test_idx, args.batch_size, False)
    test_metrics, predictions = evaluate_with_predictions(
        model,
        test_loader,
        manifest.iloc[test_idx].reset_index(drop=True),
        fine_encoder,
        parent_encoder,
        device,
    )
    predictions.to_csv(args.output / "test_predictions.csv", index=False)

    checkpoint = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": asdict(cfg),
        "input_dim": int(x.shape[1]),
        "fine_classes": fine_encoder.classes_.tolist(),
        "parent_classes": parent_encoder.classes_.tolist(),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, args.output / "adapter.pt")
    metrics = {
        "status": "pass",
        "config": asdict(cfg),
        "n_rows": int(len(manifest)),
        "embedding_shape": list(embeddings.shape),
        "split_counts": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx)), **excluded},
        "n_fine_classes_train": int(len(fine_encoder.classes_)),
        "n_parent_classes_train": int(len(parent_encoder.classes_)),
        "best_epoch": int(best_epoch),
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "runtime_seconds": round(time.time() - started, 3),
        "artifacts": {
            "adapter": str(args.output / "adapter.pt"),
            "history": str(args.output / "history.csv"),
            "test_predictions": str(args.output / "test_predictions.csv"),
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def slide_heldout_splits(frame: pd.DataFrame, *, seed: int) -> dict[str, np.ndarray]:
    indices = np.arange(len(frame))
    groups = frame["slide"].astype(str).to_numpy()
    label = frame["trainable_standard_name"].astype(str).to_numpy()
    train_idx, temp_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed).split(indices, label, groups))
    temp_groups = groups[temp_idx]
    temp_labels = label[temp_idx]
    rel_val, rel_test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1).split(temp_idx, temp_labels, temp_groups)
    )
    return {"train": train_idx, "val": temp_idx[rel_val], "test": temp_idx[rel_test]}


def filter_seen_labels(frame: pd.DataFrame, candidate_idx: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    train_fine = set(frame.iloc[train_idx]["trainable_standard_name"].astype(str))
    train_parent = set(frame.iloc[train_idx]["standard_parent_class"].astype(str))
    sub = frame.iloc[candidate_idx]
    seen = sub["trainable_standard_name"].astype(str).isin(train_fine) & sub["standard_parent_class"].astype(str).isin(train_parent)
    return candidate_idx[seen.to_numpy()]


def encode_with_train_classes(values: pd.Series, encoder: LabelEncoder) -> np.ndarray:
    lookup = {label: idx for idx, label in enumerate(encoder.classes_)}
    return values.map(lookup).fillna(-1).astype(np.int64).to_numpy()


def make_loader(
    x: np.ndarray,
    fine_ids: np.ndarray,
    parent_ids: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    dataset = EmbeddingDataset(x[indices], fine_ids[indices], parent_ids[indices])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def class_weight(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels[labels >= 0], minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weight = counts.sum() / (n_classes * counts)
    return torch.from_numpy(weight.astype(np.float32))


def train_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    fine_loss_fn: nn.Module,
    parent_loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        fine = batch["fine"].to(device)
        parent = batch["parent"].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = fine_loss_fn(out["fine"], fine) + 0.5 * parent_loss_fn(out["parent"], parent)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * int(x.shape[0])
        n += int(x.shape[0])
    return total / max(1, n)


def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    fine_encoder: LabelEncoder,
    parent_encoder: LabelEncoder,
    device: torch.device,
) -> dict[str, float]:
    metrics, _ = evaluate_with_predictions(model, loader, None, fine_encoder, parent_encoder, device)
    return metrics


def evaluate_with_predictions(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    frame: pd.DataFrame | None,
    fine_encoder: LabelEncoder,
    parent_encoder: LabelEncoder,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    fine_true: list[np.ndarray] = []
    parent_true: list[np.ndarray] = []
    fine_prob: list[np.ndarray] = []
    parent_prob: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"].to(device))
            fine_true.append(batch["fine"].numpy())
            parent_true.append(batch["parent"].numpy())
            fine_prob.append(torch.softmax(out["fine"], dim=1).cpu().numpy())
            parent_prob.append(torch.softmax(out["parent"], dim=1).cpu().numpy())
    y_fine = np.concatenate(fine_true)
    y_parent = np.concatenate(parent_true)
    p_fine = np.concatenate(fine_prob)
    p_parent = np.concatenate(parent_prob)
    fine_pred = p_fine.argmax(axis=1)
    parent_pred = p_parent.argmax(axis=1)
    metrics = {
        **prefix_metrics("fine", y_fine, fine_pred, p_fine, len(fine_encoder.classes_)),
        **prefix_metrics("parent", y_parent, parent_pred, p_parent, len(parent_encoder.classes_)),
    }
    if frame is None:
        return metrics, pd.DataFrame()
    pred_frame = frame.copy()
    pred_frame["predicted_trainable_standard_name"] = fine_encoder.classes_[fine_pred]
    pred_frame["trainable_standard_name_confidence"] = p_fine.max(axis=1)
    pred_frame["trainable_standard_name_correct"] = pred_frame["trainable_standard_name"].astype(str).to_numpy() == pred_frame[
        "predicted_trainable_standard_name"
    ].to_numpy()
    pred_frame["predicted_standard_parent_class"] = parent_encoder.classes_[parent_pred]
    pred_frame["standard_parent_class_confidence"] = p_parent.max(axis=1)
    pred_frame["standard_parent_class_correct"] = pred_frame["standard_parent_class"].astype(str).to_numpy() == pred_frame[
        "predicted_standard_parent_class"
    ].to_numpy()
    return metrics, pred_frame


def prefix_metrics(prefix: str, y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray, n_classes: int) -> dict[str, float]:
    k = min(5, n_classes)
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        f"{prefix}_top{k}_accuracy": float(top_k_accuracy_score(y_true, prob, k=k, labels=np.arange(n_classes))),
    }


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())

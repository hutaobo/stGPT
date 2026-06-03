"""Evaluate contour H&E tiles with a public Hugging Face pathology model.

This is a lightweight "can public H&E foundation features improve naming?"
analysis. It uses the exported contour-centered H&E fields in ``broute_tiles``,
joins curated labels from ``structure_assignments_v2_name.csv``, extracts frozen
image embeddings, and fits slide-held-out linear probes for parent and fine
structure names.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

IGNORE_LABEL = "__ignore_review_needed__"
USABLE_CONFIDENCE_TIERS = {"curated", "high", "medium"}


@dataclass
class ProbeConfig:
    slides_root: str
    tiles_root: str
    output: str
    model_id: str
    batch_size: int
    num_workers: int
    device: str
    seed: int
    max_tiles: int
    max_per_class: int
    min_class_count: int


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides-root", type=Path, required=True)
    parser.add_argument("--tiles-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="owkin/phikon-v2")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = ProbeConfig(
        slides_root=str(args.slides_root),
        tiles_root=str(args.tiles_root),
        output=str(args.output),
        model_id=args.model_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        max_tiles=args.max_tiles,
        max_per_class=args.max_per_class,
        min_class_count=args.min_class_count,
    )

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    manifest = build_tile_manifest(
        args.slides_root,
        args.tiles_root,
        seed=args.seed,
        max_tiles=args.max_tiles,
        max_per_class=args.max_per_class,
        min_class_count=args.min_class_count,
    )
    manifest_path = out / "contour_he_tiles_with_curated_labels.csv"
    manifest.to_csv(manifest_path, index=False)
    inventory = {
        "n_tiles": int(len(manifest)),
        "n_slides": int(manifest["slide"].nunique()) if not manifest.empty else 0,
        "n_parent_labels": int(manifest["standard_parent_class"].nunique()) if not manifest.empty else 0,
        "n_fine_labels": int(manifest["trainable_standard_name"].nunique()) if not manifest.empty else 0,
        "top_fine_labels": manifest["trainable_standard_name"].value_counts().head(20).to_dict(),
        "top_parent_labels": manifest["standard_parent_class"].value_counts().head(20).to_dict(),
    }
    (out / "inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "manifest": str(manifest_path), **inventory}, indent=2))
        return 0

    embeddings = extract_embeddings(manifest, cfg)
    np.save(out / "embeddings.npy", embeddings.astype(np.float32))

    fine_result = run_linear_probe(
        embeddings,
        manifest,
        label_column="trainable_standard_name",
        seed=args.seed,
        min_class_count=args.min_class_count,
    )
    parent_result = run_linear_probe(
        embeddings,
        manifest,
        label_column="standard_parent_class",
        seed=args.seed,
        min_class_count=args.min_class_count,
    )
    fine_result["predictions"].to_csv(out / "fine_structure_predictions.csv", index=False)
    parent_result["predictions"].to_csv(out / "parent_structure_predictions.csv", index=False)
    pd.DataFrame(fine_result["per_class"]).to_csv(out / "fine_structure_per_class.csv", index=False)
    pd.DataFrame(parent_result["per_class"]).to_csv(out / "parent_structure_per_class.csv", index=False)

    metrics = {
        "status": "pass",
        "config": asdict(cfg),
        "inventory": inventory,
        "embedding_shape": list(embeddings.shape),
        "fine_structure": fine_result["metrics"],
        "parent_structure": parent_result["metrics"],
        "runtime_seconds": round(time.time() - started, 3),
        "artifacts": {
            "manifest": str(manifest_path),
            "embeddings": str(out / "embeddings.npy"),
            "fine_predictions": str(out / "fine_structure_predictions.csv"),
            "parent_predictions": str(out / "parent_structure_predictions.csv"),
            "fine_per_class": str(out / "fine_structure_per_class.csv"),
            "parent_per_class": str(out / "parent_structure_per_class.csv"),
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def build_tile_manifest(
    slides_root: Path,
    tiles_root: Path,
    *,
    seed: int,
    max_tiles: int,
    max_per_class: int,
    min_class_count: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for tile_manifest in sorted(tiles_root.glob("*/tiles_manifest.csv")):
        slide = tile_manifest.parent.name
        label_path = slides_root / slide / "structure_assignments_v2_name.csv"
        if not label_path.exists():
            continue
        tiles = pd.read_csv(tile_manifest)
        labels = pd.read_csv(label_path)
        if "contour_id" not in tiles or "contour_id" not in labels:
            continue
        usable = usable_curated_labels(labels)
        labels = labels.loc[usable].copy()
        keep_cols = [
            "contour_id",
            "standard_parent_class",
            "trainable_standard_name",
            "standard_confidence_tier",
            "label_confidence",
        ]
        keep_cols = [col for col in keep_cols if col in labels.columns]
        joined = tiles.merge(labels[keep_cols], on="contour_id", how="inner")
        if joined.empty:
            continue
        joined["slide"] = slide
        joined["image_path"] = joined["tile_file"].map(lambda name, base=tile_manifest.parent: str((base / str(name)).resolve()))
        rows.append(joined)
    if not rows:
        raise ValueError(f"no labeled contour H&E tiles found under {tiles_root}")
    frame = pd.concat(rows, ignore_index=True)
    frame = frame[frame["image_path"].map(lambda value: Path(str(value)).exists())].copy()
    frame["standard_parent_class"] = frame["standard_parent_class"].fillna("unknown").astype(str)
    frame["trainable_standard_name"] = frame["trainable_standard_name"].fillna("unknown").astype(str)
    frame = frame[frame["trainable_standard_name"] != IGNORE_LABEL].copy()
    counts = frame["trainable_standard_name"].value_counts()
    frame = frame[frame["trainable_standard_name"].isin(counts[counts >= min_class_count].index)].copy()
    if max_per_class > 0:
        frame = (
            frame.groupby("trainable_standard_name", group_keys=False)
            .apply(lambda group: group.sample(n=min(len(group), max_per_class), random_state=seed))
            .reset_index(drop=True)
        )
    if max_tiles > 0 and len(frame) > max_tiles:
        frame = frame.sample(n=max_tiles, random_state=seed).reset_index(drop=True)
    return frame.sort_values(["slide", "trainable_standard_name", "contour_id"]).reset_index(drop=True)


def usable_curated_labels(labels: pd.DataFrame) -> pd.Series:
    index = labels.index
    trainable = labels["trainable_standard_name"].notna() & (labels["trainable_standard_name"].astype(str) != IGNORE_LABEL)
    if "standard_confidence_tier" in labels:
        confidence = labels["standard_confidence_tier"].fillna("").astype(str).str.lower().isin(USABLE_CONFIDENCE_TIERS)
    else:
        confidence = pd.Series(True, index=index)
    if "standard_needs_review" in labels:
        review_values = labels["standard_needs_review"]
        if review_values.dtype == bool:
            no_review = ~review_values
        else:
            no_review = ~review_values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        no_review = pd.Series(True, index=index)
    return trainable & confidence & no_review


class TileDataset:
    def __init__(self, frame: pd.DataFrame, processor: Any) -> None:
        self.frame = frame.reset_index(drop=True)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        row = self.frame.iloc[index]
        image = Image.open(str(row["image_path"])).convert("RGB")
        processed = self.processor(images=image, return_tensors="pt")
        return {"pixel_values": processed["pixel_values"].squeeze(0), "index": index}


def extract_embeddings(frame: pd.DataFrame, cfg: ProbeConfig) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(cfg.model_id)
    model = AutoModel.from_pretrained(cfg.model_id, trust_remote_code=True)
    model.eval().to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    dataset = TileDataset(frame, processor)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            output = model(pixel_values=pixel_values)
            features = pool_model_output(output)
            features = torch.nn.functional.normalize(features.float(), dim=1)
            chunks.append(features.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def pool_model_output(output: Any) -> Any:
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        hidden = output.last_hidden_state
        if hidden.ndim == 3 and hidden.shape[1] > 0:
            return hidden[:, 0]
        return hidden.mean(dim=1)
    if isinstance(output, tuple):
        first = output[0]
        if first.ndim == 3:
            return first[:, 0]
        return first
    raise RuntimeError("could not pool model output")


def run_linear_probe(
    embeddings: np.ndarray,
    frame: pd.DataFrame,
    *,
    label_column: str,
    seed: int,
    min_class_count: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        f1_score,
        top_k_accuracy_score,
    )
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    labels = frame[label_column].astype(str).to_numpy()
    counts = pd.Series(labels).value_counts()
    eligible = np.asarray([counts[label] >= min_class_count for label in labels], dtype=bool)
    X = embeddings[eligible]
    y_text = labels[eligible]
    groups = frame.loc[eligible, "slide"].astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError(f"{label_column} has fewer than 2 eligible classes")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y_text, groups=groups))
    train_labels = set(y_text[train_idx])
    seen_test = np.asarray([label in train_labels for label in y_text[test_idx]], dtype=bool)
    heldout_unseen = int((~seen_test).sum())
    test_idx_seen = test_idx[seen_test]
    if len(test_idx_seen) == 0:
        raise ValueError(f"{label_column} slide-heldout split has no test labels seen during training")

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_text[train_idx])
    y_test = encoder.transform(y_text[test_idx_seen])
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", multi_class="auto"),
    )
    clf.fit(X[train_idx], y_train)
    pred = clf.predict(X[test_idx_seen])
    prob = clf.predict_proba(X[test_idx_seen])
    classes = encoder.classes_
    top_k = min(5, len(classes))
    metrics = {
        "label_column": label_column,
        "n_train": int(len(train_idx)),
        "n_test_seen": int(len(test_idx_seen)),
        "n_test_unseen_label": heldout_unseen,
        "n_classes_train": int(len(classes)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "top_k": int(top_k),
        "top_k_accuracy": float(top_k_accuracy_score(y_test, prob, k=top_k, labels=np.arange(len(classes)))),
    }
    report = classification_report(y_test, pred, labels=np.arange(len(classes)), target_names=classes, output_dict=True, zero_division=0)
    per_class = [
        {"label": label, **{key: _json_float(value) for key, value in scores.items()}}
        for label, scores in report.items()
        if isinstance(scores, dict)
    ]
    pred_labels = classes[pred]
    confidence = prob.max(axis=1)
    prediction_frame = frame.loc[eligible].iloc[test_idx_seen].copy()
    prediction_frame[f"predicted_{label_column}"] = pred_labels
    prediction_frame[f"{label_column}_confidence"] = confidence
    prediction_frame[f"{label_column}_correct"] = prediction_frame[label_column].astype(str).to_numpy() == pred_labels
    return {"metrics": metrics, "per_class": per_class, "predictions": prediction_frame}


def _json_float(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())

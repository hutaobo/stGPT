from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Annotated

import anndata as ad
import torch
import typer

from . import __version__
from .config import StGPTConfig
from .data import build_training_manifest
from .evaluation import evaluate as evaluate_model
from .inference import embed_anndata, export_spatho_summaries, write_embeddings_table
from .qc import validate_data
from .training import train as train_model

app = typer.Typer(help="stGPT image-gene spatial transcriptomics prototype.")
DEFAULT_EMBED_OUTPUT = Path("outputs/stgpt_embeddings.parquet")


@app.command()
def doctor() -> None:
    payload = {
        "stgpt": __version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pyxenium_available": importlib.util.find_spec("pyXenium") is not None,
        "spatho_available": importlib.util.find_spec("spatho") is not None,
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("prepare-xenium")
def prepare_xenium(config: Annotated[Path, typer.Option("--config", "-c", exists=True)]) -> None:
    cfg = StGPTConfig.from_file(config)
    manifest = build_training_manifest(cfg)
    typer.echo(json.dumps(manifest, indent=2))


@app.command("validate-data")
def validate_data_command(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    cfg = StGPTConfig.from_file(config)
    result = validate_data(cfg, output_dir=output)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    preset: Annotated[str | None, typer.Option("--preset")] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps")] = None,
) -> None:
    result = train_model(config, preset=preset, max_steps=max_steps)
    printable = {key: value for key, value in result.items() if key != "metrics"}
    if result.get("metrics"):
        printable["last_metrics"] = result["metrics"][-1]
    typer.echo(json.dumps(printable, indent=2))


@app.command()
def evaluate(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-k", exists=True)],
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    splits: Annotated[Path, typer.Option("--splits", "-s", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    batch_size: Annotated[int, typer.Option("--batch-size")] = 32,
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    result = evaluate_model(
        checkpoint=checkpoint,
        config=config,
        splits=splits,
        output_dir=output,
        batch_size=batch_size,
        device=device,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command()
def embed(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-k", exists=True)],
    input: Annotated[Path, typer.Option("--input", "-i", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")] = DEFAULT_EMBED_OUTPUT,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 32,
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    adata = ad.read_h5ad(input)
    embedded = embed_anndata(adata, checkpoint=checkpoint, batch_size=batch_size, device=device)
    path = write_embeddings_table(embedded, output)
    typer.echo(json.dumps({"embeddings": str(path)}, indent=2))


@app.command("export-spatho")
def export_spatho(
    embeddings: Annotated[Path, typer.Option("--embeddings", "-e", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    outputs = export_spatho_summaries(embeddings, output)
    typer.echo(json.dumps(outputs, indent=2))

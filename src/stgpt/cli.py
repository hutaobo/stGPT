from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Annotated

import anndata as ad
import torch
import typer

from . import __version__
from .annotation import annotate_regions as annotate_regions_backend
from .config import StGPTConfig
from .contour_store import pack_contour_patches
from .curated_spatial import audit_curated_structures, predict_curated_spatial_prior, train_curated_spatial_prior
from .data import build_training_manifest
from .evaluation import evaluate as evaluate_model
from .evidence import (
    build_failure_gallery,
    build_latent_manifold,
    check_artifact_contract,
    generate_watchtower_report,
    run_contour_ablation,
    summarize_evidence_suite,
)
from .foundation import package_model as package_model_backend
from .image_qc import inspect_images as inspect_images_backend
from .image_qc import precompute_image_embeddings
from .inference import embed_anndata, write_embeddings_table
from .inspection import inspect_registry as inspect_registry_backend
from .pseudo_spatial import predict_pseudo_spatial, train_pseudo_spatial_prior
from .qc import validate_data
from .spatho import run_spatho_export
from .training import initialize_random_checkpoint
from .training import train as train_model
from .visual import build_contour_panel

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


@app.command("inspect-registry")
def inspect_registry_command(
    registry: Annotated[Path, typer.Option("--registry", "-r", exists=True)],
    root: Annotated[Path | None, typer.Option("--root")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    sample_images: Annotated[int, typer.Option("--sample-images", min=0)] = 50,
) -> None:
    result = inspect_registry_backend(registry, root=root, output=output, sample_images=sample_images)
    typer.echo(json.dumps(result["summary"], indent=2))


@app.command("pack-contour-patches")
def pack_contour_patches_command(
    patch_manifest: Annotated[Path, typer.Option("--patch-manifest", exists=True)],
    store: Annotated[Path, typer.Option("--store")],
    manifest: Annotated[Path, typer.Option("--manifest")],
    slide_id: Annotated[str, typer.Option("--slide-id")],
    image_size: Annotated[int, typer.Option("--image-size", min=16)] = 224,
    max_neighbors: Annotated[int, typer.Option("--max-neighbors", min=1)] = 16,
    chunk_size: Annotated[int, typer.Option("--chunk-size", min=1)] = 1024,
) -> None:
    result = pack_contour_patches(
        patch_manifest,
        store,
        manifest,
        slide_id=slide_id,
        image_size=image_size,
        max_neighbors=max_neighbors,
        chunk_size=chunk_size,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("inspect-images")
def inspect_images_command(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    result = inspect_images_backend(config, output_dir=output)
    typer.echo(json.dumps(result["summary"], indent=2))


@app.command("precompute-images")
def precompute_images_command(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    encoder: Annotated[str | None, typer.Option("--encoder")] = None,
    encoder_backend: Annotated[str | None, typer.Option("--encoder-backend")] = None,
    encoder_preset: Annotated[str | None, typer.Option("--encoder-preset")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 32,
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    result = precompute_image_embeddings(
        config,
        output=output,
        encoder_backend=encoder_backend,
        encoder_preset=encoder_preset,
        encoder_name=encoder,
        batch_size=batch_size,
        device=device,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    preset: Annotated[str | None, typer.Option("--preset")] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps")] = None,
    ablation: Annotated[str | None, typer.Option("--ablation")] = None,
    resume: Annotated[Path | None, typer.Option("--resume", exists=True)] = None,
) -> None:
    result = train_model(config, preset=preset, max_steps=max_steps, ablation=ablation, resume=resume)
    printable = {key: value for key, value in result.items() if key != "metrics"}
    if result.get("metrics"):
        printable["last_metrics"] = result["metrics"][-1]
    typer.echo(json.dumps(printable, indent=2))


@app.command("train-pseudo-spatial")
def train_pseudo_spatial_command(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    preset: Annotated[str | None, typer.Option("--preset")] = None,
    max_steps: Annotated[int, typer.Option("--max-steps", min=1)] = 2000,
    n_spatial_bins: Annotated[int, typer.Option("--n-spatial-bins", min=2)] = 32,
    n_niches: Annotated[int, typer.Option("--n-niches", min=1)] = 32,
    max_genes: Annotated[int, typer.Option("--max-genes", min=1)] = 512,
    d_model: Annotated[int, typer.Option("--d-model", min=16)] = 256,
    hidden_layers: Annotated[int, typer.Option("--hidden-layers", min=1)] = 2,
    dropout: Annotated[float, typer.Option("--dropout", min=0.0, max=0.9)] = 0.1,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 512,
    learning_rate: Annotated[float, typer.Option("--learning-rate", min=1e-8)] = 3e-4,
    weight_decay: Annotated[float, typer.Option("--weight-decay", min=0.0)] = 0.01,
    device: Annotated[str, typer.Option("--device")] = "auto",
    num_workers: Annotated[int, typer.Option("--num-workers", min=0)] = 0,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    data_parallel: Annotated[bool, typer.Option("--data-parallel/--single-gpu")] = True,
) -> None:
    result = train_pseudo_spatial_prior(
        config,
        output_dir=output,
        preset=preset,
        max_steps=max_steps,
        n_spatial_bins=n_spatial_bins,
        n_niches=n_niches,
        max_genes=max_genes,
        d_model=d_model,
        hidden_layers=hidden_layers,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        num_workers=num_workers,
        seed=seed,
        data_parallel=data_parallel,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("predict-pseudo-spatial")
def predict_pseudo_spatial_command(
    model: Annotated[Path, typer.Option("--model", "-m", exists=True)],
    input: Annotated[Path, typer.Option("--input", "-i", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    reference_regions: Annotated[Path | None, typer.Option("--reference-regions", exists=True)] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 1024,
    device: Annotated[str, typer.Option("--device")] = "auto",
    full_probabilities: Annotated[bool, typer.Option("--full-probabilities/--top1-only")] = True,
) -> None:
    result = predict_pseudo_spatial(
        model,
        input,
        output=output,
        reference_regions=reference_regions,
        batch_size=batch_size,
        device=device,
        full_probabilities=full_probabilities,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("audit-curated-structures")
def audit_curated_structures_command(
    manifest: Annotated[Path, typer.Option("--manifest", "-m", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    case_column: Annotated[str, typer.Option("--case-column")] = "case_leaf",
    root_base: Annotated[Path | None, typer.Option("--root-base")] = None,
    exclude_case_leaf: Annotated[list[str] | None, typer.Option("--exclude-case-leaf")] = None,
) -> None:
    result = audit_curated_structures(
        manifest,
        output=output,
        case_column=case_column,
        root_base=root_base,
        excluded_case_leaves=exclude_case_leaf,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("train-curated-spatial-prior")
def train_curated_spatial_prior_command(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    preset: Annotated[str | None, typer.Option("--preset")] = None,
    max_steps: Annotated[int, typer.Option("--max-steps", min=1)] = 2000,
    n_spatial_bins: Annotated[int, typer.Option("--n-spatial-bins", min=2)] = 32,
    max_genes: Annotated[int, typer.Option("--max-genes", min=1)] = 512,
    d_model: Annotated[int, typer.Option("--d-model", min=16)] = 256,
    hidden_layers: Annotated[int, typer.Option("--hidden-layers", min=1)] = 2,
    dropout: Annotated[float, typer.Option("--dropout", min=0.0, max=0.9)] = 0.1,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 512,
    learning_rate: Annotated[float, typer.Option("--learning-rate", min=1e-8)] = 3e-4,
    weight_decay: Annotated[float, typer.Option("--weight-decay", min=0.0)] = 0.01,
    device: Annotated[str, typer.Option("--device")] = "auto",
    num_workers: Annotated[int, typer.Option("--num-workers", min=0)] = 0,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    data_parallel: Annotated[bool, typer.Option("--data-parallel/--single-gpu")] = True,
    exclude_case_leaf: Annotated[list[str] | None, typer.Option("--exclude-case-leaf")] = None,
) -> None:
    result = train_curated_spatial_prior(
        config,
        output_dir=output,
        preset=preset,
        max_steps=max_steps,
        n_spatial_bins=n_spatial_bins,
        max_genes=max_genes,
        d_model=d_model,
        hidden_layers=hidden_layers,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        num_workers=num_workers,
        seed=seed,
        data_parallel=data_parallel,
        excluded_case_leaves=exclude_case_leaf,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("predict-curated-spatial-prior")
def predict_curated_spatial_prior_command(
    model: Annotated[Path, typer.Option("--model", "-m", exists=True)],
    input: Annotated[Path, typer.Option("--input", "-i", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    reference_regions: Annotated[Path | None, typer.Option("--reference-regions", exists=True)] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 1024,
    device: Annotated[str, typer.Option("--device")] = "auto",
    full_probabilities: Annotated[bool, typer.Option("--full-probabilities/--top1-only")] = True,
) -> None:
    result = predict_curated_spatial_prior(
        model,
        input,
        output=output,
        reference_regions=reference_regions,
        batch_size=batch_size,
        device=device,
        full_probabilities=full_probabilities,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("init-random-checkpoint")
def init_random_checkpoint_command(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    preset: Annotated[str | None, typer.Option("--preset")] = None,
    ablation: Annotated[str | None, typer.Option("--ablation")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
) -> None:
    result = initialize_random_checkpoint(config, output, preset=preset, ablation=ablation, seed=seed)
    typer.echo(json.dumps(result, indent=2))


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


@app.command("evidence-summary")
def evidence_summary_command(
    suite: Annotated[Path, typer.Option("--suite", "-s", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    pointer_sample_size: Annotated[int, typer.Option("--pointer-sample-size", min=0)] = 50,
) -> None:
    result = summarize_evidence_suite(suite, output, pointer_sample_size=pointer_sample_size)
    typer.echo(json.dumps(result, indent=2))


@app.command("watchtower")
def watchtower_command(
    suite: Annotated[Path, typer.Option("--suite", "-s", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    watch: Annotated[bool, typer.Option("--watch")] = False,
    interval_seconds: Annotated[int, typer.Option("--interval-seconds", min=1)] = 3600,
    iterations: Annotated[int | None, typer.Option("--iterations", min=1)] = None,
) -> None:
    count = 0
    last_result: dict[str, object] | None = None
    while True:
        last_result = generate_watchtower_report(suite, output)
        count += 1
        if not watch or (iterations is not None and count >= iterations):
            break
        time.sleep(interval_seconds)
    typer.echo(json.dumps(last_result or {}, indent=2))


@app.command("contour-panel")
def contour_panel_command(
    evidence_chain: Annotated[Path, typer.Option("--evidence-chain", "-e", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    sample_size: Annotated[int, typer.Option("--sample-size", min=0)] = 12,
    sort_by: Annotated[str, typer.Option("--sort-by")] = "low_confidence",
    top_genes: Annotated[int, typer.Option("--top-genes", min=0)] = 8,
) -> None:
    result = build_contour_panel(evidence_chain, output, sample_size=sample_size, sort_by=sort_by, top_genes=top_genes)
    typer.echo(json.dumps(result, indent=2))


@app.command("failure-gallery")
def failure_gallery_command(
    run_dir: Annotated[Path, typer.Option("--run-dir", "-r", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    max_items: Annotated[int, typer.Option("--max-items", min=0)] = 24,
    top_genes: Annotated[int, typer.Option("--top-genes", min=0)] = 8,
    rare_prototype_fraction: Annotated[float, typer.Option("--rare-prototype-fraction", min=0.0)] = 0.02,
) -> None:
    result = build_failure_gallery(
        run_dir,
        output,
        max_items=max_items,
        top_genes=top_genes,
        rare_prototype_fraction=rare_prototype_fraction,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("ablate")
def ablate_command(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-k", exists=True)],
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    targets: Annotated[Path, typer.Option("--targets", "-t", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 32,
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    result = run_contour_ablation(
        checkpoint=checkpoint,
        config=config,
        targets=targets,
        output_dir=output,
        batch_size=batch_size,
        device=device,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("latent-manifold")
def latent_manifold_command(
    suite: Annotated[Path, typer.Option("--suite", "-s", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    reducer: Annotated[str, typer.Option("--reducer")] = "auto",
    max_points_per_run: Annotated[int, typer.Option("--max-points-per-run", min=0)] = 0,
    max_html_points: Annotated[int, typer.Option("--max-html-points", min=0)] = 5000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    result = build_latent_manifold(
        suite,
        output,
        reducer=reducer,  # type: ignore[arg-type]
        max_points_per_run=max_points_per_run,
        max_html_points=max_html_points,
        seed=seed,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("figure-manifold")
def figure_manifold_command(
    manifold: Annotated[Path, typer.Option("--manifold", "-m", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    name: Annotated[str, typer.Option("--name")] = "f1_cross_platform_manifold",
    batch_key: Annotated[str, typer.Option("--batch-key")] = "auto",
    structure_key: Annotated[str, typer.Option("--structure-key")] = "structure_label",
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    batch_mixing_csv: Annotated[Path | None, typer.Option("--batch-mixing-csv")] = None,
    embedding_qc_csv: Annotated[Path | None, typer.Option("--embedding-qc-csv")] = None,
    fmt: Annotated[str, typer.Option("--formats", help="comma-separated, e.g. pdf,png")] = "pdf,png",
    max_points: Annotated[int, typer.Option("--max-points", min=0)] = 50000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Render the F1 cross-platform manifold figure (requires the figures extra)."""
    try:
        from .figures import plot_cross_platform_manifold
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise typer.BadParameter(
            "figure-manifold needs the optional figures extra: pip install -e \".[figures]\""
        ) from exc
    result = plot_cross_platform_manifold(
        manifold,
        output,
        name=name,
        batch_key=batch_key,
        structure_key=structure_key,
        run_id=run_id,
        batch_mixing_csv=batch_mixing_csv,
        embedding_qc_csv=embedding_qc_csv,
        formats=tuple(part.strip() for part in fmt.split(",") if part.strip()),
        max_points=max_points,
        seed=seed,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("figure-ablation")
def figure_ablation_command(
    summary: Annotated[Path, typer.Option("--summary", "-s", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    name: Annotated[str, typer.Option("--name")] = "f2_ablation_comparison",
    condition_key: Annotated[str, typer.Option("--condition-key")] = "condition",
    group_key: Annotated[str, typer.Option("--group-key")] = "tissue",
    run_ids: Annotated[str | None, typer.Option("--run-ids", help="comma-separated run IDs to keep")] = None,
    fmt: Annotated[str, typer.Option("--formats", help="comma-separated, e.g. pdf,png")] = "pdf,png",
) -> None:
    """Render the F2 ablation comparison figure (requires the figures extra)."""
    try:
        from .figures import plot_ablation_comparison
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise typer.BadParameter(
            "figure-ablation needs the optional figures extra: pip install -e \".[figures]\""
        ) from exc
    result = plot_ablation_comparison(
        summary,
        output,
        name=name,
        condition_key=condition_key,
        group_key=group_key,
        run_ids=tuple(part.strip() for part in run_ids.split(",") if part.strip()) if run_ids else None,
        formats=tuple(part.strip() for part in fmt.split(",") if part.strip()),
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("figure-dynamics")
def figure_dynamics_command(
    learning_dynamics: Annotated[Path, typer.Option("--learning-dynamics", "-l", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    name: Annotated[str, typer.Option("--name")] = "f3_learning_dynamics",
    run_ids: Annotated[
        str,
        typer.Option("--run-ids", help="comma-separated run IDs to keep"),
    ] = "gene_spatial_contour_unit_20k,full_m6_contour_store_lambda_0_01_20k,structure_context_m6_20k",
    fmt: Annotated[str, typer.Option("--formats", help="comma-separated, e.g. pdf,png")] = "pdf,png",
) -> None:
    """Render the F3 43-case learning-dynamics figure (requires the figures extra)."""
    try:
        from .figures import plot_learning_dynamics
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise typer.BadParameter(
            "figure-dynamics needs the optional figures extra: pip install -e \".[figures]\""
        ) from exc
    result = plot_learning_dynamics(
        learning_dynamics,
        output,
        name=name,
        run_ids=tuple(part.strip() for part in run_ids.split(",") if part.strip()),
        formats=tuple(part.strip() for part in fmt.split(",") if part.strip()),
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("figure-structure-context")
def figure_structure_context_command(
    evidence_summary: Annotated[Path, typer.Option("--evidence-summary", "-s", exists=True)],
    run_dir: Annotated[Path, typer.Option("--run-dir", "-r", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    pointer_audit: Annotated[Path | None, typer.Option("--pointer-audit", exists=True)] = None,
    name: Annotated[str, typer.Option("--name")] = "f4_auditable_structure_context_evidence",
    run_ids: Annotated[
        str,
        typer.Option("--run-ids", help="comma-separated run IDs to keep"),
    ] = "gene_spatial_contour_unit_20k,full_m6_contour_store_lambda_0_01_20k,structure_context_m6_20k",
    structure_run_id: Annotated[str, typer.Option("--structure-run-id")] = "structure_context_m6_20k",
    fmt: Annotated[str, typer.Option("--formats", help="comma-separated, e.g. pdf,png")] = "pdf,png",
) -> None:
    """Render the F4 auditable structure-context evidence figure (requires the figures extra)."""
    try:
        from .figures import plot_structure_context_evidence
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise typer.BadParameter(
            "figure-structure-context needs the optional figures extra: pip install -e \".[figures]\""
        ) from exc
    result = plot_structure_context_evidence(
        evidence_summary,
        run_dir,
        output,
        pointer_audit=pointer_audit,
        name=name,
        run_ids=tuple(part.strip() for part in run_ids.split(",") if part.strip()),
        structure_run_id=structure_run_id,
        formats=tuple(part.strip() for part in fmt.split(",") if part.strip()),
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("check-contract")
def check_contract_command(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-k", exists=True)],
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    run_dir: Annotated[Path | None, typer.Option("--run-dir", "-r")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    result = check_artifact_contract(checkpoint=checkpoint, config=config, run_dir=run_dir, output=output)
    typer.echo(json.dumps(result, indent=2))


@app.command("package-model")
def package_model_command(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-k", exists=True)],
    evaluation: Annotated[Path, typer.Option("--eval", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    model_name: Annotated[str | None, typer.Option("--model-name")] = None,
) -> None:
    result = package_model_backend(checkpoint=checkpoint, evaluation=evaluation, output_dir=output, model_name=model_name)
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


@app.command("spatho-embed")
def spatho_embed_command(
    model: Annotated[Path, typer.Option("--model", "-m", exists=True)],
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    batch_size: Annotated[int, typer.Option("--batch-size")] = 32,
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    cfg = StGPTConfig.from_file(config)
    result = run_spatho_export(cfg, checkpoint=model, output_dir=output, batch_size=batch_size, device=device)
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command("embed-regions")
def embed_regions_command(
    model: Annotated[Path, typer.Option("--model", "-m", exists=True)],
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    batch_size: Annotated[int, typer.Option("--batch-size")] = 32,
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    cfg = StGPTConfig.from_file(config)
    result = run_spatho_export(cfg, checkpoint=model, output_dir=output, batch_size=batch_size, device=device)
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command("annotate-regions")
def annotate_regions_command(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-k", exists=True)],
    seed_labels: Annotated[Path, typer.Option("--seed-labels", "-s", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    region_ids: Annotated[Path | None, typer.Option("--region-ids", exists=True)] = None,
    include_no_image: Annotated[bool, typer.Option("--include-no-image/--require-image")] = False,
    classifier: Annotated[str, typer.Option("--classifier")] = "both",
    abstain_prob: Annotated[float, typer.Option("--abstain-prob", min=0.0, max=1.0)] = 0.5,
    write_probabilities: Annotated[bool, typer.Option("--write-probabilities/--no-probabilities")] = False,
    seed_folds: Annotated[int, typer.Option("--seed-folds", min=2)] = 5,
    rng_seed: Annotated[int, typer.Option("--rng-seed")] = 42,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 32,
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    result = annotate_regions_backend(
        config=config,
        checkpoint=checkpoint,
        seed_labels=seed_labels,
        output_dir=output,
        region_ids=region_ids,
        include_no_image=include_no_image,
        classifier=classifier,  # type: ignore[arg-type]
        abstain_prob=abstain_prob,
        write_probabilities=write_probabilities,
        seed_folds=seed_folds,
        rng_seed=rng_seed,
        batch_size=batch_size,
        device=device,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("export-spatho")
def export_spatho(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True)],
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-k", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    batch_size: Annotated[int, typer.Option("--batch-size")] = 32,
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    cfg = StGPTConfig.from_file(config)
    result = run_spatho_export(cfg, checkpoint=checkpoint, output_dir=output, batch_size=batch_size, device=device)
    typer.echo(json.dumps(result.to_dict(), indent=2))

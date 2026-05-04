from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .inspection import inspect_registry

app = typer.Typer(help="Inspect stGPT XeniumSlide registries and evidence-path contracts.")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    registry: Annotated[Path | None, typer.Option("--registry", "-r", exists=True, help="Registry CSV/JSON/Parquet to inspect.")] = None,
    root: Annotated[Path | None, typer.Option("--root", help="Expected normalized output root. Defaults to registry parent.")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Optional JSON report path.")] = None,
    sample_images: Annotated[int, typer.Option("--sample-images", min=0, help="Patch image rows to sample per case.")] = 50,
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if registry is None:
        raise typer.BadParameter("--registry is required when no subcommand is used.")
    result = inspect_registry(registry, root=root, output=output, sample_images=sample_images)
    typer.echo(json.dumps(result["summary"], indent=2))


@app.command("registry")
def registry_command(
    registry: Annotated[Path, typer.Argument(exists=True)],
    root: Annotated[Path | None, typer.Option("--root", help="Expected normalized output root. Defaults to registry parent.")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Optional JSON report path.")] = None,
    sample_images: Annotated[int, typer.Option("--sample-images", min=0)] = 50,
) -> None:
    result = inspect_registry(registry, root=root, output=output, sample_images=sample_images)
    typer.echo(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    app()

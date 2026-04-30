# stGPT Foundation Model Development Requirements

Status: Draft v0.1
Audience: stGPT and spatho developers
Primary goal: guide development from the current Xenium-first prototype toward a reproducible morpho-molecular foundation model and an agentic spatial pathology workbench.

Companion guidance: [`contour_region_foundation_model.md`](contour_region_foundation_model.md) defines the preferred final product shape where contour and region embeddings are the primary runtime output, while cell-level data remains the auditable molecular evidence layer.

## 1. Product Thesis

stGPT learns reusable contour/region morpho-molecular representations; spatho plans, validates, and turns them into auditable spatial pathology evidence.

The target system is a closed scientific loop:

```text
Raw Data -> stgpt.contracts -> stgpt.foundation -> stgpt.runtime -> spatho.workbench
               ^                                                    |
               |                                                    v
        stgpt.evidence <- Audit Trail and Evidence Graph <- spatho.reports
```

stGPT should not be positioned as only an H&E-to-expression predictor. It should learn reusable Xenium-centered tissue representations from real spatial transcriptomics cases through masked molecular modeling, image-gene alignment, spatial niche reconstruction, and panel-aware imputation. The preferred runtime representation is contour/region-level, with cell-level data retained as the traceable molecular evidence layer.

spatho should not be positioned as only a web wrapper. It should become an agentic spatial pathology workbench that calls stGPT, HistoSeg, QC tools, and evidence judges to produce traceable biological claims.

## 2. Definition of a Foundation Model

stGPT can be called a foundation model only when one checkpoint supports multiple downstream tasks on new slides without task-specific retraining.

Minimum qualifying capabilities:

- Generate useful contour/region embeddings for unseen Xenium slides, with cell, niche, and region evidence available for provenance.
- Support image-gene retrieval between H&E patches, cells, and spatial regions.
- Perform panel-aware reconstruction or imputation inside the measured Xenium panel.
- Improve niche discovery, structure annotation, or region comparison over simple baselines.
- Preserve useful behavior across held-out slides, cases, batches, and stains.
- Ship with a checkpoint card that documents training data, modalities, panel, split fingerprint, known failure modes, and intended use.

Non-goals for the first foundation checkpoint:

- Do not claim whole-transcriptome prediction unless paired matched whole-transcriptome data are available.
- Do not claim clinical diagnosis or treatment recommendation.
- Do not treat imputed values as measured molecular evidence.
- Do not publish random cell-level splits as evidence of generalization.

## 3. Real Xenium Data Learning Pipeline

Every real Xenium case must be converted into a canonical learning object before training.

Required per-case inputs:

- Cell-by-gene expression matrix.
- Cell centroids and, when available, cell boundaries.
- Spatial coordinates in physical micrometer units.
- Xenium panel metadata and gene identifiers.
- H&E or morphology image references.
- Image-to-Xenium registration transform, when image context is used.
- Optional clusters, structures, pathology regions, and spatho or HistoSeg contour outputs.
- Batch metadata: case, slide, donor, organ, stain, scanner, run, and platform when available.

Required validation before training:

- Spatial coordinate validity.
- Gene name and panel consistency.
- Duplicate gene reporting.
- Patch coverage and missing image detection.
- Registration provenance check.
- Deterministic split generation.
- Data and config fingerprints.

## 4. Canonical Data Contract

Create a contract layer before adding more model-specific adapters. The recommended package name is `stgpt.contracts`; `stgpt.modalities` is acceptable if the codebase prefers modality language.

This layer owns typed schemas and interchange rules for:

- `XeniumSlide`: one validated spatial transcriptomics case.
- `GenePanel`: measured genes, panel version, missing-gene sentinels, and panel compatibility rules.
- `SpatialTransform`: explicit transforms between physical micrometer space, image pixel space, and registered analysis space.
- `HEPatch`: patch image reference, source image, physical location, extraction parameters, and registration metadata.
- `CellEmbedding`: embedding vector plus checkpoint, data, and cell provenance.
- `NicheContext`: local neighborhood, graph, structure label, and context features.
- `SpatialRegion`: contour, bounding box, member cells, region-level summary, and evidence provenance.

Every higher layer should import these contracts instead of creating new ad hoc schema variants.

Required interchange support:

- AnnData input and output.
- Zarr or OME-NGFF image references where practical.
- JSON manifest export for reports and audit trails.
- Stable fingerprinting for input data, splits, configs, and checkpoint cards.

## 5. stGPT Foundation Requirements

The foundation layer owns representation learning.

Required model inputs:

- Gene identity tokens.
- Expression value or bin embeddings.
- H&E patch embeddings.
- Spatial coordinate embeddings.
- Optional structure or pathology context tokens.
- Optional neighborhood graph or region context.

Required pretraining objectives:

- Masked gene reconstruction to learn within-cell molecular structure.
- Neighborhood reconstruction to learn spatial co-localization and tissue context.
- Image-gene contrastive alignment between morphology and expression.
- Spatial context or niche prediction.
- Structure prediction from weak spatho or HistoSeg supervision when available.
- Panel-aware masked imputation inside the measured panel.

Required ablations:

- Gene-only.
- Image-only.
- Spatial-only.
- Image + gene.
- Image + gene + spatial.
- Full image + gene + spatial + structure/context.

Required checkpoint artifacts:

- Model weights.
- Training config.
- Gene vocabulary and panel descriptor.
- Split fingerprint.
- Metrics history.
- Checkpoint card.
- Known failure modes and intended-use notes.

## 6. stGPT Evidence Requirements

`stgpt.evidence` is both an offline benchmark suite and an inline evidence judge layer.

Offline requirements:

- Fixed-split benchmark suites.
- Cryptographic split fingerprints.
- Reconstruction metrics for masked genes and neighborhoods.
- Image-gene retrieval metrics.
- Embedding quality metrics for clusters, structures, and niches.
- Batch, stain, slide, and domain shift tests.
- Ablation tables generated from the same split.
- Failure analysis for patch coverage, missing images, registration, panel mismatch, and label availability.

Inline judge requirements:

- QC pass or fail verdict.
- Registration quality score.
- Patch and image coverage score.
- Panel compatibility verdict.
- Imputation flag propagation.
- Confidence score for generated structures or annotations.
- Guardrail verdicts before any biological conclusion is accepted.

Failure registry requirements:

- Versioned failure mode names.
- Rationale and reproduction recipe.
- Example affected datasets or synthetic fixtures.
- Expected guardrail behavior.
- Linked tests.

## 7. stGPT Runtime Requirements

`stgpt.runtime` exposes stable tool APIs for spatho, notebooks, CLI, web apps, and MCP clients.

Required Python tools:

- `validate_case`
- `embed_regions`
- `summarize_structures`
- `retrieve_regions`
- `impute_region_panel`
- `score_niche`
- `compare_regions`
- `explain_region`
- `export_spatho_artifacts`
- `embed_regions`
- `embed_cells` as a deprecated compatibility wrapper
- `evaluate_checkpoint`

Each tool must define:

- Typed input schema.
- Typed output schema.
- Required checkpoint or model reference.
- Required permissions.
- Guardrail context.
- Audit log event.
- Evidence identifiers for downstream reports.

Every runtime output must include evidence IDs, input fingerprints, config fingerprints, checkpoint or checkpoint-card fingerprints, QC verdicts, warnings, output fingerprints, and audit metadata. Runtime tools should prefer typed return objects and artifact manifests over free-form dictionaries once the contract stabilizes.

MCP support should wrap the stable Python API after the Python contract is reliable. MCP tools must be discoverable and schema-first.

Audit logging requirements:

- Every tool call emits an append-only event.
- Events include input fingerprint, checkpoint card fingerprint, tool version, parameters, output fingerprint, judge scores, and warnings.
- Audit logs feed `spatho.reports` and report reproduction.

## 8. spatho Workbench Requirements

`spatho.workbench` is the agentic orchestration layer. It should plan analyses, call stGPT tools, call HistoSeg or geometry tools, run evidence judges, and create evidence chains.

Required operating loop:

```text
Plan -> Tool Calls -> QC/Critic -> Evidence Graph -> Report -> Human Review -> Model Improvement
```

This loop is a development requirement, not only a product story. `spatho` should decide which analysis paths are valid, execute deterministic and model-backed tools, critique results with guardrails, build an evidence graph, and route low-confidence or conflict-heavy outputs to human review.

Design rule:

No biological conclusion may be emitted without a traceable evidence chain.

Every conclusion must link to:

- Tool call IDs.
- Input data slices.
- Checkpoint versions.
- QC verdicts.
- Judge scores.
- Imputation flags.
- Human review status when required.

The workbench must support:

- Data readiness checks.
- Automatic selection of valid analysis paths.
- Region and niche discovery workflows.
- Structure annotation workflows.
- Similar-region retrieval workflows.
- Panel-aware imputation workflows.
- Human escalation when guardrails fail.

## 9. spatho Reports Requirements

`spatho.reports` consumes audit logs and evidence chains. It should produce reports, but the core artifact is an evidence graph.

Required outputs:

- Reproducible HTML report.
- Optional PDF export.
- Machine-readable evidence manifest.
- Claim graph linking every claim to supporting tool calls and checkpoint versions.
- Review table for low-confidence or guardrail-flagged conclusions.

Required CLI:

```bash
spatho reports reproduce <report_id>
```

A report is valid only if it can be reproduced from:

- Input data fingerprint.
- Workflow config.
- Random seed.
- Checkpoint card.
- Tool versions.
- Audit log.

## 10. Guardrail Specification

Guardrails must be versioned, tested, and citeable. They should be implemented in code, but also documented as a specification.

Initial guardrails:

- `QC_BEFORE_ANNOTATION`: no cluster, niche, or structure annotation if required QC checks fail.
- `IMPUTATION_NOT_MEASURED`: imputed values must be flagged in every downstream artifact.
- `REGISTRATION_CONFIDENCE`: spatial image integration is blocked or escalated if registration quality is below threshold.
- `PANEL_COMPATIBILITY`: checkpoint use is blocked or warned when the case panel is incompatible.
- `HUMAN_ESCALATION`: pathologist or domain expert review is required for novel structure types or low-confidence conclusions.

Each guardrail must define:

- Name.
- Version.
- Rationale.
- Inputs.
- Thresholds.
- Failure behavior.
- Tests.
- Applicable checkpoint versions.

## 11. Reproducibility Contract

The architecture should make unreproducible reports difficult to produce.

Requirements:

- All stochastic operations use explicit seeded random number generators.
- Seeds are logged in audit events.
- Workflow configs are pinned to checkpoint cards.
- Split files are fingerprinted.
- Training and inference configs are serialized.
- Report manifests include tool versions and package versions.
- Report reproduction is part of CI once fixtures exist.

## 12. Development Milestones

### v0: Trustworthy Xenium Prototype

Scope:

- One organ or focused disease setting.
- One or a small number of Xenium panels.
- Canonical contract objects for slide, panel, patch, transform, embedding, and region.
- Current stGPT model trained on validated cases.
- QC, evaluation, failure analysis, and ablation outputs.

Acceptance criteria:

- Real cases pass `validate-data`.
- Training emits checkpoint card and split fingerprint.
- Evaluation emits reconstruction, retrieval, embedding QC, and failure analysis artifacts.
- Ablations run from the same split.
- Reports can cite checkpoint and tool-call provenance.

### v1: Multi-case Xenium Foundation Checkpoint

Scope:

- Multiple cases, slides, batches, and stains.
- Held-out slide and held-out case evaluation.
- Stronger retrieval, niche discovery, and structure annotation benchmarks.
- Early spatho workbench integration.

Acceptance criteria:

- Full model beats ablations on at least two downstream tasks.
- Held-out case performance is reported.
- Batch or stain shift is explicitly measured.
- Every generated claim in spatho has an evidence chain.

### v2: Multi-organ and Runtime Expansion

Scope:

- Multi-organ Xenium corpus.
- Optional Visium HD, CosMx, MERFISH, or other platform adapters.
- MCP runtime tools.
- Evidence graph reports with reproduction CLI.

Acceptance criteria:

- New organ or platform performance is evaluated with documented limitations.
- MCP tools are discoverable and schema-first.
- `spatho reports reproduce <report_id>` works on public fixtures.
- Failure registry is versioned and tested.

## 13. Near-term Implementation Order

1. Create `stgpt.contracts` with minimal typed objects and fingerprints.
2. Add checkpoint cards to stGPT training outputs.
3. Move current QC and evaluation logic toward `stgpt.evidence`.
4. Add split fingerprints and data fingerprints.
5. Add Python runtime APIs for `embed_regions`, `retrieve_regions`, and `impute_panel`; keep `embed_cells` only as a compatibility wrapper.
6. Add evidence chain IDs to runtime outputs.
7. Make spatho consume runtime outputs and produce claim graphs.
8. Add report reproduction command.
9. Wrap stable runtime APIs as MCP tools.

## 14. One-line Pitch

stGPT learns canonical morpho-molecular tissue representations; spatho converts them into signed, reproducible, human-reviewable evidence graphs for spatial pathology.

# stGPT Contour/Region Foundation Model Vision

Status: Draft v0.1
Audience: stGPT and spatho developers
Purpose: define the target product shape where contour and region embeddings are the primary foundation-model output, while cell-level data remains the auditable molecular evidence layer.

## 1. Final Product Shape

The final stGPT product should be a contour- and region-level morpho-molecular foundation model for spatial pathology.

The primary runtime output is not required to be one embedding per cell. The preferred product unit is:

```text
Contour or SpatialRegion embedding
  = registered H&E morphology for a contour or region
  + Xenium molecular evidence from cells inside and near that contour
  + spatial position, shape, boundary, and neighbor context
  + optional structure or pathology context
  + QC, registration, and provenance metadata
```

Cells remain essential, but they are not necessarily the final user-facing unit. They are the molecular evidence substrate that supports contour-level and region-level embeddings, explanations, retrieval, and reports.

The guiding product statement is:

```text
stGPT learns reusable contour/region morpho-molecular representations; spatho plans, validates, and turns them into auditable spatial pathology evidence.
```

## 2. Why Contour/Region-Level First

Contour-level embeddings better match how spatial pathology users reason about tissue:

- A pathologist usually reasons over structures, glands, tumor nests, stromal regions, immune niches, necrotic areas, ducts, vessels, and boundaries, not isolated transcriptomic cells.
- spatho already produces contour-like geometry and structure evidence that can anchor model outputs.
- Region embeddings can combine morphology, molecular signal, geometry, and neighborhood context in one inspectable object.
- Region-level outputs are easier to cite in reports than large unstructured sets of cell embeddings.

This does not remove cell-level modeling. A strong implementation should use cell-level tokens, cell membership, and neighborhood graphs internally, then pool or attend over them to produce the contour/region embedding.

Recommended stance:

```text
Primary runtime output: contour/region embeddings.
Required provenance layer: cell membership and molecular summaries.
Optional internal training output: cell embeddings and cell-token representations.
```

`cell_embeddings.parquet` should therefore be treated as a compatibility and provenance artifact, not the primary product object. The first-class downstream object is a `ContourRegion` or `SpatialRegion` embedding with enough measured molecular, image, spatial, and QC context for a workbench to cite it.

## 3. Foundation Model Definition

stGPT can claim a contour/region foundation model only when one checkpoint supports multiple tasks on unseen slides or cases without task-specific retraining.

Minimum qualifying capabilities:

- Embed unseen contours and regions into a reusable morpho-molecular space.
- Retrieve similar contours or regions across slides, cases, and batches.
- Compare regions such as tumor versus stroma, boundary versus core, or immune-rich versus immune-poor niches.
- Reconstruct or impute measured Xenium-panel signals at region level with imputation flags.
- Predict or score spatial niches and structure labels better than simple image-only, gene-only, and spatial-only baselines.
- Explain a contour or region using traceable morphology, cell membership, marker summaries, and neighboring-region evidence.
- Preserve useful behavior across held-out slides, held-out cases, batches, stains, and registration conditions.
- Ship with a checkpoint card documenting training data, panel, modalities, split fingerprint, known failure modes, intended use, and prohibited claims.

Non-goals for the first checkpoint:

- Do not claim clinical diagnosis or treatment recommendation.
- Do not claim whole-transcriptome prediction from a targeted Xenium panel.
- Do not treat imputed or reconstructed expression as measured expression.
- Do not publish random cell-level splits as evidence of region-level generalization.
- Do not emit biological conclusions without QC and provenance.

## 4. Core Data Contracts

The contract layer should make contour/region modeling explicit. The preferred module is `stgpt.contracts`.

Required contract objects:

- `XeniumSlide`: one validated spatial transcriptomics case with cell expression, coordinates, metadata, and panel information.
- `GenePanel`: measured genes, panel version, compatible checkpoints, missing-gene behavior, and panel fingerprints.
- `SpatialTransform`: transforms between Xenium physical coordinates, registered analysis coordinates, and H&E pixel space.
- `HEPatch`: image patch reference, source image, crop coordinates, magnification or MPP, mask, and registration provenance.
- `ContourRegion`: contour geometry, mask, bounding box, source segmentation, region label, and parent slide.
- `CellEvidenceSet`: member cells, boundary cells, nearby cells, cell weights, and cell-level QC flags for one contour or region.
- `NicheContext`: neighboring contours, cell-cell or region-region graph, boundary rings, and local tissue context.
- `RegionMolecularSummary`: measured-panel aggregate expression, marker summaries, missingness, imputation flags, and aggregation policy.
- `RegionEmbedding`: embedding vector plus checkpoint, region, cell evidence, data, transform, and QC provenance.
- `EvidenceId`: stable identifiers for every derived artifact used by spatho reports.

Higher layers should import these contracts instead of creating ad hoc dictionaries for contour rows, patch rows, region summaries, or embedding tables.

## 5. Region Construction

Each contour/region should be built through a reproducible assignment pipeline.

Required inputs:

- Contour geometry from spatho, HistoSeg, or another segmentation source.
- Cell centroids and, when available, cell boundaries.
- H&E image reference and registration transform.
- Xenium expression matrix and panel metadata.
- Optional structure labels, pathology labels, cluster labels, and manual review labels.

Required construction steps:

1. Validate contour geometry and coordinate system.
2. Assign cells inside each contour.
3. Assign boundary-ring cells within configurable micrometer distances.
4. Assign neighboring contours or regions.
5. Extract contour image crop and optional binary mask.
6. Aggregate measured expression for inner, boundary, and context cells.
7. Record assignment policy, thresholds, transform, source files, and fingerprints.

Recommended aggregation channels:

- `inner_cells`: cells inside the contour.
- `boundary_cells`: cells near the contour boundary.
- `context_cells`: cells in a surrounding spatial ring.
- `neighbor_regions`: adjacent or nearby contours.
- `structure_prior`: optional spatho/HistoSeg/manual structure label.

The model may use all channels internally, but exported artifacts must preserve which evidence came from measured expression and which came from model-derived outputs.

## 6. Model Architecture Direction

The preferred architecture is a region-centered multimodal Transformer.

A region example should contain:

- Region image token from H&E crop or mask-aware patch encoder.
- Shape and geometry tokens: area, perimeter, compactness, eccentricity, boundary features, and coordinates.
- Cell tokens for sampled member cells.
- Molecular summary tokens from measured Xenium expression.
- Boundary/context tokens from nearby cells or neighboring contours.
- Optional structure or pathology-context tokens.
- Optional slide, batch, stain, scanner, organ, and panel tokens for domain tracking.

The model should produce:

- `region_emb`: the primary reusable embedding.
- `cell_emb`: optional internal or exported supporting embeddings.
- `image_emb`: region morphology embedding for image-region retrieval.
- `molecular_emb`: measured-panel molecular embedding for gene-region retrieval.
- `region_pred`: panel reconstruction or imputation outputs.
- `structure_logits`: optional weak-supervision output for region or structure labels.

Cell-level information should be used as evidence and attention context, not discarded by simple averaging unless a baseline explicitly requires that simplification.

## 7. Pretraining Objectives

Required objectives for contour/region foundation modeling:

- Region masked gene reconstruction: recover masked measured-panel expression summaries for a region.
- Cell-to-region reconstruction: recover region molecular summaries from sampled member cells and context cells.
- Neighborhood reconstruction: predict nearby region or boundary molecular summaries from spatial context.
- Image-region contrastive alignment: align H&E contour crops with region molecular/context embeddings.
- Gene-to-region retrieval: retrieve regions from marker or panel-expression queries.
- Region-to-region contrastive learning: align related regions across augmentations, scales, or neighboring context.
- Structure prediction: use weak spatho, HistoSeg, or manual labels when available.
- Panel-aware imputation: infer missing or masked measured-panel genes with explicit imputation flags.

Required ablations:

- Image only.
- Molecular summary only.
- Cell tokens only.
- Spatial/geometry only.
- Image + molecular summary.
- Image + cell tokens.
- Image + molecular summary + spatial geometry.
- Full region model with cell, image, molecular, spatial, boundary, and structure context.

## 8. Runtime API Target

The runtime API should expose contour/region workflows first.

Required Python tools:

- `embed_regions`: embed contours or regions and write region-level artifacts.
- `retrieve_regions`: find similar regions from an example region, image crop, marker query, or embedding.
- `compare_regions`: compare two or more regions with molecular, morphology, and context summaries.
- `impute_region_panel`: produce panel-aware region-level imputation with flags.
- `score_niche`: score a contour or neighborhood for a configured niche signature.
- `explain_region`: produce a traceable evidence explanation for one region.
- `embed_cells`: keep available as a supporting and compatibility tool.
- `evaluate_checkpoint`: evaluate region and cell tasks with fixed split files.
- `export_spatho_artifacts`: export spatho-compatible region and cell evidence artifacts.

The first-class CLI should mirror this runtime shape:

```bash
stgpt embed-regions --config ... --checkpoint ... --output ...
stgpt retrieve-regions --model ... --query-region ... --output ...
stgpt compare-regions --model ... --regions ... --output ...
stgpt impute-region-panel --model ... --regions ... --output ...
stgpt explain-region --model ... --region-id ... --output ...
```

Implementation priority:

1. Stabilize `embed_regions` and `export_spatho_artifacts` first, because they form the initial handshake with `spatho`.
2. Add `retrieve_regions` and `compare_regions` once region embeddings and QC artifacts are reliable.
3. Add `explain_region` only after evidence IDs, guardrail verdicts, and measured-vs-model-derived fields are present in exported artifacts.

The spatho consumption path should remain:

```text
stGPT export artifacts -> spatho evidence graph -> report and manifest -> human review
```

## 9. Exported Artifacts

The contour/region export should become the main downstream artifact set.

Required files:

- `region_embeddings.parquet`: one row per contour/region with `region_id`, geometry fields, labels, QC flags, and `emb_*`.
- `region_molecular_summary.parquet`: measured-panel aggregate expression, marker summaries, aggregation policy, missingness, and imputation flags.
- `region_cell_membership.parquet`: mapping from `region_id` to member cells, boundary cells, context cells, and weights.
- `region_neighbors.parquet`: region-region adjacency, distances, boundary relationships, and graph provenance.
- `region_image_manifest.json`: H&E crop, mask, transform, patch extraction, and source-image provenance.
- `region_qc_report.json`: region coverage, image coverage, registration quality, molecular coverage, cell assignment quality, and warnings.
- `evidence_manifest.json`: fingerprints and evidence IDs for every exported object.

Compatibility files may still include:

- `cell_embeddings.parquet`
- `structure_summary.parquet`
- `structure_embedding_summary.csv`

The region artifacts should be sufficient for spatho to create a reproducible evidence graph without re-running training.

## 10. Evaluation Requirements

Evaluation must prove that region embeddings are reusable on unseen tissue.

Required split strategies:

- Held-out contour within slide for smoke testing only.
- Held-out spatial block for early debugging.
- Held-out slide for paper-facing single-study results.
- Held-out case or patient for foundation-model claims.
- Held-out batch, stain, scanner, or organ when data permits.

Required metrics:

- Region masked-gene reconstruction MSE and correlation.
- Region image-to-molecular and molecular-to-image retrieval top-k.
- Region-to-region same-label retrieval.
- Structure/niche annotation performance.
- Marker-signature ranking and enrichment quality.
- Batch, stain, scanner, and case mixing metrics.
- Domain-shift metrics for held-out slide/case/stain.
- Cell membership and region coverage QC.
- Registration and image coverage failure analysis.
- Panel compatibility and missing-gene failure analysis.

Every result table should include ablation baselines from the same split file.

## 11. Pathology Vocabulary and RAG Layer

Pathology textbook or reference material should not be treated as the primary training data for the stGPT foundation embedding. The core representation should be learned from registered H&E morphology, measured Xenium molecular evidence, cell-to-contour membership, and spatial context.

Pathology RAG tooling, such as [`hutaobo/pathology-rag-workbench`](https://github.com/hutaobo/pathology-rag-workbench), should instead be used as a vocabulary, ontology, explanation, and citation layer around the model.

Recommended division of responsibility:

```text
stGPT
  learns contour/region morpho-molecular embeddings from tissue data

pathology-rag-workbench
  provides pathology vocabulary, definitions, synonyms, and citation-backed context

spatho
  combines model outputs and pathology language into evidence graphs and reports
```

Appropriate uses:

- Build a controlled pathology vocabulary for region labels such as tumor nest, stroma, necrosis, gland, duct, lymphoid aggregate, invasion front, boundary, and vessel.
- Normalize synonyms, descriptive phrases, and hierarchy relationships into a stable ontology used by stGPT exports and spatho reports.
- Support text-to-region retrieval, for example querying for regions resembling a named histologic pattern or structure.
- Support `explain_region` by grounding generated explanations in retrieved definitions and citations.
- Help design weak-label rules or candidate structure labels that are later calibrated against measured data and expert review.

Inappropriate uses:

- Do not embed copyrighted pathology textbook text directly into released stGPT model weights.
- Do not treat retrieved textbook passages as measured evidence from the slide.
- Do not let RAG-generated language override QC, registration, panel compatibility, or human-review guardrails.
- Do not use textbook-derived terms to make clinical diagnostic claims without expert validation.

The stable contract should keep model evidence and textual reference evidence separate. A region explanation may cite both, but it must distinguish:

- measured slide evidence from Xenium and H&E;
- model-derived reconstruction, imputation, embedding, retrieval, or scores;
- reference-language evidence from pathology books, notes, or curated ontology entries.

## 12. Guardrails

The region model must preserve evidence boundaries.

Initial guardrails:

- `QC_BEFORE_REGION_CLAIMS`: block or escalate region claims when data QC fails.
- `REGISTRATION_REQUIRED_FOR_IMAGE_CONTEXT`: block image-conditioned region explanations when registration is missing or low confidence.
- `IMPUTATION_NOT_MEASURED`: mark every imputed or reconstructed value as model-derived.
- `CELL_COVERAGE_MINIMUM`: warn or block region embeddings when too few cells support the contour.
- `BOUNDARY_AMBIGUITY`: flag contours whose cell assignment changes under small boundary perturbations.
- `PANEL_COMPATIBILITY`: block or warn when checkpoint and case panel are incompatible.
- `HUMAN_ESCALATION`: require expert review for novel structures, low-confidence explanations, or clinical-facing summaries.

Guardrails should be versioned, tested, and included in exported evidence manifests.

## 13. Development Order

Recommended implementation order:

1. Add `stgpt.contracts` with `ContourRegion`, `CellEvidenceSet`, `RegionMolecularSummary`, and `RegionEmbedding`.
2. Add deterministic cell-to-contour assignment from spatho/HistoSeg contours.
3. Export `region_cell_membership.parquet` and `region_molecular_summary.parquet`.
4. Add `RegionDataset` with image crop, region geometry, molecular summary, cell tokens, and context rings.
5. Add `embed_regions` runtime and CLI.
6. Add `region_embeddings.parquet`, `region_qc_report.json`, and `evidence_manifest.json`.
7. Add region reconstruction and image-region retrieval metrics.
8. Add `retrieve_regions` and `compare_regions`.
9. Add a controlled pathology vocabulary and ontology export, optionally backed by pathology-rag-workbench.
10. Add panel-aware `impute_region_panel` with imputation flags.
11. Add `explain_region` with evidence IDs, reference citations, and guardrail verdicts.
12. Run region-level ablations from the same fixed split.
13. Package checkpoints with region-focused checkpoint cards.

## 14. v0 Acceptance Criteria

The first contour/region foundation prototype is acceptable when:

- A real Xenium case can be converted into validated contour regions.
- Every region has deterministic cell membership and recorded assignment policy.
- Every region embedding links to its H&E crop, transform, member cells, molecular summary, checkpoint, and QC status.
- `stgpt embed-regions` writes region artifacts usable by spatho.
- `stgpt evaluate` reports region reconstruction, retrieval, embedding QC, batch/domain checks, and failure analysis.
- Full model beats image-only, molecular-only, spatial-only, and simple pooled baselines on at least two region-level tasks.
- Region labels and explanations can use a controlled pathology vocabulary without mixing reference text with measured slide evidence.
- A packaged checkpoint includes a region-focused model card and known failure modes.
- spatho can cite a region embedding in an evidence graph without losing provenance.

## 15. One-Line North Star

stGPT should make every tissue contour a reusable, auditable morpho-molecular object: morphology on the surface, measured cells underneath, spatial context around it, and provenance attached to every claim.

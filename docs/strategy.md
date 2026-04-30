# Development Strategy: Xenium-First Image-Gene GPT for Spatial Transcriptomics

This document captures the development direction for `stGPT` after reviewing closely related spatial transcriptomics, histopathology, and foundation-model methods. It is a working engineering guide, not a full literature review. The landscape snapshot is current as of 2026-04-30.

## Project Positioning

`stGPT` should be developed as a Xenium-first image-gene GPT for spatial transcriptomics. The model should treat each cell or local spatial unit as a multimodal sequence built from:

- gene identity tokens and expression-value/bin embeddings
- trainable H&E patch embeddings
- spatial coordinate embeddings
- optional structure or pathology-context tokens from spatho-style evidence

The core pretraining and fine-tuning objectives should remain aligned with this positioning:

- masked gene reconstruction to learn within-cell molecular structure
- neighborhood reconstruction to capture spatial co-localization and tissue context
- image-gene contrastive learning to align morphology with expression
- compact cell or region embeddings for downstream pathology and spatial biology workflows

This makes `stGPT` different from a pure H&E-to-expression regressor. The goal is to build a reusable representation model for Xenium-centered spatial pathology, with expression prediction as one important evaluation task rather than the whole product.

## Current Method Landscape

### Closest strategic neighbors

- [`scGPT-spatial`](https://github.com/bowang-lab/scGPT-spatial) extends scGPT through continual pretraining for spatial transcriptomics, with spatially aware sampling and neighborhood-oriented objectives. The relevant preprint is [scGPT-spatial: Continual Pretraining of Single-Cell Foundation Model for Spatial Transcriptomics](https://doi.org/10.1101/2025.02.05.636714). This is the closest gene-token foundation-model reference, but it does not make trainable H&E patch context the central input.
- [`STPath`](https://www.nature.com/articles/s41746-025-02020-3) is a generative foundation model for integrating spatial transcriptomics and whole-slide images. It uses a geometry-aware Transformer and masked gene expression prediction over large-scale WSI-ST data. It validates that masked generative objectives are now a strong direction for ST-pathology models.
- [`STORM`](https://arxiv.org/abs/2604.03630) is a multimodal foundation model of spatial transcriptomics and histology for biological discovery and clinical prediction. Its platform-agnostic framing across Visium, Xenium, Visium HD, and CosMx is an important signal that cross-platform evaluation will matter.
- [`ST-Align`](https://arxiv.org/abs/2411.16793) is an image-gene alignment foundation model for spatial transcriptomics. It emphasizes spatial context, spot-niche alignment, multi-scale alignment, and few-shot/zero-shot transfer. This supports the need for image-gene alignment in `stGPT`, while `stGPT` should keep a tighter Xenium-native reconstruction objective.
- [`OmiCLIP/Loki`](https://www.nature.com/articles/s41592-025-02707-1) builds a visual-omics foundation model that bridges H&E histology and spatial transcriptomics, then uses the aligned space for tissue alignment, annotation, retrieval, cell-type decomposition, and ST expression prediction. This is the strongest CLIP-style reference for cross-modal image-expression retrieval.
- [`SEAL`](https://arxiv.org/abs/2602.14177) performs Spatial Expression-Aligned Learning as parameter-efficient ST-guided fine-tuning of pathology vision encoders. The gated model card is available at [`MahmoodLab/SEAL`](https://huggingface.co/MahmoodLab/SEAL). SEAL supports the idea that localized molecular supervision improves pathology encoders, but its primary product is a better vision model rather than a gene-token GPT.

### Xenium-specific and task-specific neighbors

- [`H&Enium`](https://openreview.net/forum?id=W64NsKUpMy) aligns H&E image embeddings and transcriptomic foundation embeddings at single-cell resolution with contrastive learning. It is an important single-cell Xenium-adjacent reference for alignment, but it is closer to an embedding-alignment framework than a unified generative sequence model.
- [`xMINT`](https://openreview.net/forum?id=hnYLq2lwOv) is a Multimodal Integration Transformer for Xenium gene imputation. It is directly relevant to Xenium panel expansion and imputation, but its task scope is narrower than the desired `stGPT` representation-learning agenda.
- [`DiffBulk`](https://pubmed.ncbi.nlm.nih.gov/42048193/) uses diffusion-based training to improve spatial transcriptomic prediction. It is useful as a generative baseline and a reminder that expression-space generation may compete with Transformer reconstruction objectives.
- [`PAST`](https://arxiv.org/abs/2507.06418) is a multimodal single-cell foundation model for histopathology and spatial transcriptomics in cancer. It overlaps with `stGPT` on high-resolution image-to-expression prediction and virtual molecular staining, but is broader in pan-cancer scope.

### Related spatial foundation models

- [`SToFM`](https://arxiv.org/abs/2507.11588) is a multi-scale foundation model for spatial transcriptomics that highlights macro tissue morphology, microenvironment, and gene-scale modeling.
- [`Nicheformer`](https://www.nature.com/articles/s41592-025-02814-z) is a foundation model for single-cell and spatial omics that transfers spatial context into cell representations.
- [`Novae`](https://www.nature.com/articles/s41592-025-02899-6) is a graph-based foundation model for spatial transcriptomics, trained across large multi-tissue cell collections.
- [`CellNiche`](https://www.nature.com/articles/s41467-026-71759-4) represents cellular microenvironments in atlas-scale spatial omics data with contrastive learning.

These methods may not all use H&E as a core modality, but they define the baseline expectations for spatial context, niche representation, graph structure, and cross-tissue generalization.

### Benchmarks and datasets to track

- [`HESCAPE`](https://arxiv.org/abs/2508.01490) is a large-scale benchmark for cross-modal learning between histology and gene expression in spatial transcriptomics. The Hugging Face dataset is [`Peng-AI/hescape-pyarrow`](https://huggingface.co/datasets/Peng-AI/hescape-pyarrow). Its key warning for `stGPT` is that gene encoders and batch effects can dominate cross-modal learning quality.
- [`HEST-1k`](https://huggingface.co/datasets/MahmoodLab/hest) provides a large histology-ST dataset with aligned whole-slide images and spatial transcriptomics profiles.
- [`STimage-1K4M`](https://huggingface.co/datasets/jiawennnn/STimage-1K4M) provides histopathology image-gene expression pairs for spatial transcriptomics research.

## Strategic Judgments

1. `stGPT` should not compete as another generic H&E-to-expression predictor. That space already includes strong large-scale models and specialized prediction methods.
2. The project should differentiate on Xenium-native modeling: cell-level or subcellular-resolution assumptions, panel-aware gene vocabularies, imaging-based ST quirks, and practical adapter quality.
3. The model should fuse H&E, spatial, structure/context, and gene tokens inside a unified Transformer rather than relying only on late feature concatenation.
4. The training recipe should preserve both reconstruction and alignment objectives: masked gene reconstruction, neighborhood reconstruction, and image-gene contrastive loss should remain first-class.
5. spatho-derived H&E patch manifests and structure assignments should become a strategic advantage, because they provide explicit pathology context that many image-gene models leave implicit.
6. Benchmarking should separate three claims: expression prediction, representation quality, and pathology/spatial biology utility. A single aggregate score will hide important failures.

## Development Priorities

- Build robust Xenium ingestion and validation first: coordinates, gene names, panel metadata, cell IDs, optional morphology assets, and reproducible AnnData export.
- Treat `stgpt validate-data` as the first real-data gate: it should write a case manifest, QC reports, and deterministic splits before any paper-facing training run.
- Treat `stgpt evaluate` as the second gate: it should consume the QC split file and write reconstruction, retrieval, and embedding-quality artifacts for every paper-facing checkpoint.
- Make patch and structure manifests reproducible: every embedding should be traceable to image coordinates, patch extraction parameters, registration metadata, and any spatho-derived structure labels.
- Implement baseline comparisons against the closest method families: scGPT-spatial-style gene/spatial objectives, STPath/STORM-style masked expression prediction, ST-Align/OmiCLIP-style contrastive alignment, and xMINT-style Xenium imputation.
- Treat objective ablations as required evidence: `stgpt train --ablation gene_only`, `image_only`, `spatial_only`, `image_gene`, `image_gene_spatial`, and `full` should be run from the same data split before making claims.
- Add explicit handling for batch effects and domain shift: case-level splits, slide-level splits, organ/tissue holdouts, platform holdouts where possible, and staining variation checks.
- Define a panel and vocabulary strategy: fixed panel vocabularies for Xenium smoke tests, configurable gene vocabularies for real studies, and clear behavior for missing or out-of-panel genes.
- Keep failure analysis next to metrics: every evaluation should report patch coverage, missing images, registration traceability, panel mismatch, and available batch/slide/domain keys.
- Keep the public package practical: CPU smoke tests, small synthetic fixtures, documented real-data adapters, and compact exported embeddings for downstream pathology workflows.

## Risks and Evaluation

- Batch effects may dominate image-gene alignment. HESCAPE-style evaluation should be used to detect whether the model learns biology or site/platform artifacts.
- Platform heterogeneity matters. Visium spots, Visium HD bins, Xenium cells, CosMx cells, and MERFISH-style assays differ in resolution, panel design, sparsity, segmentation, and image registration assumptions.
- Xenium is not whole-transcriptome by default. Gene reconstruction and imputation claims must distinguish panel reconstruction from whole-transcriptome prediction.
- H&E registration quality is a major failure mode. The development workflow should record image alignment assumptions and expose quality-control hooks rather than treating image patches as automatically correct.
- Ablation comparisons are only valid when they reuse the same QC-generated split file, seed, panel policy, and patch provenance contract.
- Gated and non-commercial datasets/models may limit reproducibility. Public smoke tests and open synthetic fixtures should remain part of the core repo even when larger benchmarks use restricted assets.
- Large foundation models may outperform `stGPT` on generic expression prediction. The project should win by being transparent, Xenium-aware, easy to run, and useful for downstream spatial pathology evidence generation.

## Near-Term Development Definition of Done

The next development phase should be considered successful when `stGPT` can:

- load a real Xenium case through the optional adapter
- validate the case with `stgpt validate-data` and inspect the QC report before training
- attach reproducible H&E patch and structure/context metadata
- train the image-gene Transformer with reconstruction and contrastive objectives
- evaluate the checkpoint with the QC split file instead of ad hoc random splits
- write a failure-analysis artifact covering patch, registration, panel, and split/domain risks
- export cell or region embeddings with enough metadata for downstream analysis
- run smoke tests without private data
- report baseline and ablation results that make the strategic claims above testable

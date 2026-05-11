# L3-43 v0.1 Paper-Facing Result Summary

Milestone: `L3-43 v0.1 evidence milestone`.
Data version: `l3_20260507_43case` with 43 cases and 293,678 exported contour/region records.
Code provenance: stGPT `df9cad2`, pyXenium `c039a91`, pyXenium `0.4.5`.
Evidence provenance: private Hugging Face dataset `hutaobo/stgpt-l3-evidence-20260504` at `ba960dae59852b9e94dbe2c699ebaa4d61bb0396`.
Run status summary: `pass`. Pointer errors across summarized runs: `0`.

## Main Result

Full M6 contour-store training produced a strong image-gene aligned region space while preserving gene reconstruction. The Full M6 image-to-gene top-1 retrieval is 0.9417, compared with 0.000003 for the gene+spatial baseline. Gene correlation remains high for both runs (0.9988 for Full M6 and 0.9989 for baseline).

The current result should not be framed as a complete foundation model. Label/structure retrieval remains weaker in Full M6 than in the baseline (Label@1 0.0630 vs 0.1283), and the L3-43 training configs did not optimize a structure objective.

## Metrics Table

| run_id                                | steps | checkpoint_role | gene_mse | gene_correlation | image_to_gene_top1 | gene_to_image_top1 | label_top1 | label_top5 | silhouette | prototype_usage_global | pointer_errors |
| ------------------------------------- | ----- | --------------- | -------- | ---------------- | ------------------ | ------------------ | ---------- | ---------- | ---------- | ---------------------- | -------------- |
| smoke_5case_full_m6_lambda_0_01_500   | 500   | best_alignment  | 0.0506   | 0.8545           | 0.0027             | 0.0031             | 0.3892     | 0.7658     | -0.2610    | 43/64                  | 0              |
| full_m6_contour_store_lambda_0_01_20k | 20000 | best_alignment  | 0.0005   | 0.9988           | 0.9417             | 0.9563             | 0.0630     | 0.2577     | -0.2280    | 127/128                | 0              |
| gene_spatial_contour_unit_20k         | 20000 | best_loss       | 0.0005   | 0.9989           | 0.0000             | 0.0000             | 0.1283     | 0.3905     | -0.4666    | N/A                    | 0              |

## Interpretation

- Evidence for the core contour-level pipeline is positive: smoke, Full M6, and baseline all pass, use `contour_store` image evidence, and have zero pointer errors.
- The main measurable gain of Full M6 is cross-modal alignment, not structure classification.
- Gene reconstruction is not materially degraded by the Full M6 multimodal objective, but the gene+spatial baseline remains a slightly stronger pure reconstruction control.
- Prototype coverage is broad globally, but low mean prototype confidence indicates diffuse assignments that need interpretation-focused tuning before biological claims.

## Prohibited Claims

- Do not claim clinical diagnosis, treatment prediction, or pathology-grade structure annotation.
- Do not present reconstructed or imputed expression as measured Xenium expression.
- Do not claim a finished foundation model from this milestone alone.
- Do not infer that weak label retrieval is a data failure without separating objective design, weak labels, and registration/image-gene conflicts.

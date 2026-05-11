# L3-43 v0.1 Failure Modes and Next Experiment Decision

This note separates observed failures into objective-design gaps, weak-supervision limitations, and possible true data or registration conflicts. It uses only lightweight evaluation and evidence artifacts.

## Observed Failure Pattern

- Full M6 label retrieval is weaker than the gene+spatial baseline: Label@1 0.0630 vs 0.1283, Label@5 0.2577 vs 0.3905.
- Full M6 image-gene retrieval is strong, so the failure is not a global image-gene alignment collapse.
- Pointer audit reports `0` pointer errors, so the summarized evidence chain itself is not the leading failure explanation.
- Full M6 prototype global usage is `127/128`, but mean prototype confidence is 0.0237, which is consistent with diffuse prototype assignment.
- Full M6 failure-analysis rows: `18`. Baseline failure-analysis rows: `18`.
- Full M6 region image coverage is 0.9830; missing image count is 0.0; low-cell region count is 0.0.
- Registration metadata checks are present: patch coordinates 1.0, registration metadata 1.0.

## Detailed Retrieval Diagnostics

| diagnostic                 | full_m6 | gene_spatial_baseline |
| -------------------------- | ------- | --------------------- |
| structure_label@1          | 0.0630  | 0.1283                |
| structure_label@5          | 0.2577  | 0.3905                |
| structure_id@1             | 0.0655  | 0.1312                |
| structure_id@5             | 0.2689  | 0.4003                |
| cluster@1                  | 0.0863  | 0.1377                |
| cluster@5                  | 0.3299  | 0.4155                |
| structure_label silhouette | -0.2526 | -0.5153               |
| structure_id silhouette    | -0.2057 | -0.4012               |

## Failure Classes

### 1. Objective did not optimize structure

The L3-43 Full M6 run was optimized for molecular reconstruction, spatial neighborhood reconstruction, image-gene alignment, and prototype organization. Structure context was not enabled in the published L3-43 configs, so weak label retrieval should be treated as an expected objective gap rather than a model defect by itself.

### 2. Structure labels are weak supervision, not gold pathology labels

The current structure labels are useful for retrieval diagnostics, but they are not pathologist gold labels. A label retrieval miss can mean the embedding ignores the weak label, the label is too coarse, or the region has mixed molecular/morphology evidence.

### 3. True image/gene/registration conflicts remain possible

Because image-gene retrieval is strong overall and pointer errors are zero, broad registration failure is unlikely. The correct next step is targeted inspection of high gene-MSE, low prototype-confidence, and label-mismatch regions rather than a data-wide rebuild.

## Recommended Next Experiments

1. Run `structure_context_m6` on the frozen 43-case data first. Success criterion: improve Label@1 or Label@5 over Full M6 while keeping gene correlation at or above 0.995.
2. Run a small Virchow/UNI smoke only after failure review confirms that image encoder capacity is a plausible bottleneck.
3. Defer data expansion and contour repacking until structure-context and image-encoder smoke results are interpreted.

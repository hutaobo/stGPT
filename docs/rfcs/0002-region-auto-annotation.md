# RFC 0002: Region Auto-Annotation from Sparse Labels

**Status:** Draft
**Author:** stGPT Team
**Created:** 2026-05-20

## 1. Motivation

Pathologists annotate a small number of structurally meaningful regions on
one slide. Marking every region by hand on the same slide, and on every
future slide, is not feasible — especially for structures that are hard to
recognize visually without molecular context.

stGPT already trains a multimodal region encoder over H&E, gene
expression, spatial coordinates, and structure-context tokens, and emits
per-region embeddings plus an auxiliary structure-classification head.
This RFC turns that latent capability into a stable user-facing workflow:

> Given an annotated subset of regions on a slide (or set of slides),
> produce confidence-scored structure labels for the remaining regions
> on the same slide, and later for regions on a held-out slide.

The product claim is intentionally narrow:

```text
stGPT propagates sparse expert structure labels to unannotated regions
via measured morpho-molecular embeddings, with per-region confidence,
abstain semantics, and full provenance.
```

This RFC is annotation propagation, not online learning. The trained
checkpoint is frozen; only a lightweight classifier on top of region
embeddings is fit per case.

## 2. Non-goals

- Not an unsupervised structure-discovery method. The 5 classes (or
  whatever the user provides) come from a labeled `structure_assignments`
  contract.
- Not a clinical classifier. Output is auditable evidence with confidence
  and abstain flags, never a diagnosis.
- Not a replacement for `structure_head` retraining. This is a
  per-case calibration layer; full retraining still happens through
  `stgpt train`.
- Not a cross-slide H&E-only path in v0. Phase 2 covers that, and
  needs a separate image-only inference contract (see §7).
- Not a stain-normalization or contour-segmentation tool. Both are
  upstream of stGPT and must be settled in pyXenium / spatho first.

## 3. Feasibility

stGPT today already carries the building blocks:

- `ImageGeneSTGPT.structure_head` (`src/stgpt/models.py:564`) is a
  linear head over the region embedding, trained jointly with masked
  gene reconstruction, neighborhood reconstruction, and the image-gene
  contrastive loss when `structure_loss_weight > 0` and
  `n_structures > 1` (`src/stgpt/training.py:445`,
  `src/stgpt/config.py:165`).
- `ImageGeneSTGPT.forward` emits `structure_logits` per region
  (`src/stgpt/models.py:716`), so both the embedding and a 5-way logit
  vector are available at inference with no architectural change.
- `embed-regions` already writes `region_embeddings.parquet` with
  `structure_label`, `qc_flag`, and `emb_*` columns
  (`README.md:75-82`), giving us a stable seed for the unannotated
  pool.
- `evaluate` already computes `label_retrieval_metrics.csv` and
  `image_gene_retrieval_metrics.csv`, which is the same machinery a
  nearest-neighbor propagator needs.

What is missing is a runtime entry point that consumes the seed labels
and writes per-region predictions for the unlabeled regions. This RFC
adds exactly that.

## 4. Scope, in order

**Phase 1 — same-slide propagation.**
One Xenium case, partial expert labels, predict the rest. Reuses the
existing `xenium_slide.zarr` pipeline end-to-end. This is the v0
deliverable.

**Phase 2 — cross-slide propagation (Xenium + H&E + Xenium).**
A second labeled or unlabeled Xenium case in the same panel. Same
artifacts, but with a domain-shift QC gate and explicit warnings.

**Phase 3 — cross-slide propagation (H&E only).**
A target slide with no Xenium molecules. Requires an `image_only`
inference path that builds region tokens from H&E contour segmentation
alone and masks gene tokens. Separately RFC'd; flagged here so the v0
schema reserves room for `expression_present: bool`.

This RFC scopes implementation to Phase 1.

## 5. Data Contract

### 5.1 Seed labels (input)

The annotator supplies a CSV with at minimum:

```
region_id,structure_label,confidence
R000123,gland,1.0
R000456,stroma,1.0
R000789,necrosis,0.8
...
```

- `region_id` must resolve against the case's `region_embeddings.parquet`
  (built by `stgpt embed-regions`). Unknown ids are rejected before
  fitting.
- `structure_label` must be one of the strings declared in the
  case's `structure_assignments.csv`, or a superset declared in the
  config. Unknown labels are rejected.
- `confidence` is optional (defaults to 1.0). Values below a
  config threshold become sample weights when fitting, never become
  pseudo-labels for unlabeled regions.

A typo or unknown label is a fatal error, not a warning. We do not
silently drop expert annotations.

### 5.2 Region pool (input)

The unlabeled pool defaults to "all regions in this case where the seed
file does not provide a label and `qc_flag == 'ok'`". Two opt-ins:

- `--include-no-image`: also score regions where the H&E patch was
  missing (`qc_flag == 'no_image'`). Off by default; predictions on
  these regions are marked `qc_flag: no_image` in the output and the
  classifier never sees them at fit time.
- `--region-ids <file>`: restrict the pool to a caller-provided list,
  useful for "score these specific regions only" workflows.

### 5.3 Predictions (output)

`outputs/<case>/auto_annotation/region_predictions.parquet` with one
row per region in the pool:

| Column | Type | Description |
|---|---|---|
| `region_id` | str | Joins back to `region_embeddings.parquet` |
| `predicted_label` | str | Top-1 label, or `__abstain__` if confidence below threshold |
| `predicted_prob` | float | Top-1 calibrated probability |
| `entropy` | float | Predictive entropy across labels |
| `nearest_seed_region_id` | str | Closest annotated region in embedding space |
| `nearest_seed_distance` | float | Cosine distance to that seed |
| `qc_flag` | str | Forwarded from `region_embeddings.parquet`, plus `seed` for seed regions |
| `classifier` | str | `structure_head` or `prototype_knn` |
| `evidence_id` | str | Deterministic hash of inputs |

Alongside it, two summary files:

- `region_predictions_per_class.parquet` — one row per (`region_id`,
  `class`) with the full probability vector. Optional, gated by
  `--write-probabilities`.
- `auto_annotation_report.json` — fit/inference fingerprints, seed
  counts per class, abstain rate, confusion matrix on held-out seed
  folds, QC gates, warnings.

Provenance fields on every row (model checkpoint fingerprint, config
fingerprint, seed file hash, classifier kind, threshold) live in
`auto_annotation_report.json`, not duplicated per row.

## 6. Algorithm

Two paths, both emitted by default for v0 so they can be compared on
the same case.

### 6.1 Path A — `structure_head` direct (recommended primary)

1. Load checkpoint via `ImageGeneSTGPT.from_pretrained`.
2. Run `embed-regions` if `region_embeddings.parquet` is not already
   present; otherwise reuse it.
3. Re-run a forward pass over the pool to collect `structure_logits`
   (already produced by the head, see `src/stgpt/models.py:716`).
4. **Temperature-scale the logits on the seed set.** Single scalar
   temperature, fit by minimizing NLL on held-out seed folds. This is
   the standard calibration step and avoids over-confidence on classes
   the head was already biased toward at training time.
5. Probabilities → top-1 + entropy → abstain rule.

Pros: zero new training, exactly the supervision signal we already
trained on, fast.

Cons: only works if the checkpoint was trained with the same label
vocabulary. Adding a new class needs Path B.

### 6.2 Path B — prototype/k-NN on the embedding (sanity check & cold start)

1. Pull `emb_*` columns for seed and pool regions.
2. L2-normalize embeddings.
3. Compute per-class prototypes (mean of seed embeddings, weighted by
   seed `confidence`). For very low seed counts (< 5 per class), use
   k-NN with k=min(5, seed_count) instead.
4. Score pool regions by cosine similarity → softmax over similarities
   with a fitted temperature → top-1 + entropy.

Pros: lets the user redefine label vocabulary without retraining,
robust to small seed sets via k-NN.

Cons: ignores `structure_head` weights; quality depends purely on how
discriminative the embedding is for the label set.

### 6.3 Abstain rule

A prediction becomes `__abstain__` if any of the following are true:

- `predicted_prob < tau_p` (default 0.5, configurable).
- `entropy > tau_h` (default `log(n_classes) - 0.5`).
- `nearest_seed_distance > tau_d` measured against seed-distance
  quantiles fit on the seed set itself (default 99th percentile of
  seed-to-nearest-other-seed distances).

Abstain is not a failure; it is the model saying "ask a human." The
fraction of abstained regions is logged in the report.

### 6.4 Seed validation

Before fitting, run a `k`-fold cross-validation on seeds alone
(default k=5, capped at seed_count_per_class - 1). Write:

- per-class precision/recall/F1
- macro F1
- a small confusion matrix

If any class has fewer than 5 seeds, the report flags `low_seed_count`
and recommends the user add more before trusting Phase 2 cross-slide
runs.

## 7. Reservations for Phase 2 / 3

The v0 schema reserves these fields so Phase 2 and 3 do not break the
contract:

- `expression_present: bool` — false for Phase 3 image-only regions.
- `source_case_id: str` — the case that produced the region, for
  cross-slide propagation reports.
- `propagation_kind: enum { same_slide, cross_slide_full, cross_slide_image_only }`.

In Phase 1 these are filled with `True`, the current case id, and
`same_slide`.

## 8. Runtime and CLI Surface

New runtime entry point in `stgpt.runtime`:

```python
def annotate_regions(
    *,
    config: StGPTConfig | str | Path,
    checkpoint: str | Path,
    seed_labels: str | Path,
    output_dir: str | Path,
    region_ids: str | Path | None = None,
    include_no_image: bool = False,
    classifier: Literal["structure_head", "prototype_knn", "both"] = "both",
    abstain_prob: float = 0.5,
    write_probabilities: bool = False,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    """Propagate sparse expert structure labels to unannotated regions."""
```

New CLI command (in `src/stgpt/cli.py`):

```
stgpt annotate-regions \
  --config configs/atera_wta_breast_slide.yaml \
  --checkpoint outputs/.../checkpoints/last.pt \
  --seed-labels seed_labels.csv \
  --classifier both \
  --output outputs/.../auto_annotation
```

`stgpt embed-regions` is invoked transparently if its outputs are not
present, so the user can run `annotate-regions` cold.

## 9. QC Gates

Fatal (run aborts):

- Any seed `region_id` not present in the case.
- Any seed `structure_label` not in the declared vocabulary.
- Fewer than 2 distinct seed classes (nothing to propagate).
- Checkpoint label vocabulary mismatches the seed vocabulary under
  `--classifier structure_head` (Path A is unavailable; the run
  proceeds with Path B only and the report records the downgrade).

Warning (run continues, report flags it):

- Any class with fewer than 5 seeds.
- Abstain rate over 50% of the pool.
- The held-out seed macro-F1 below 0.6.
- Embedding QC: any `emb_*` column with > 5% NaN or > 1% zero-vectors.

## 10. Reproducibility and Provenance

Every artifact carries:

- `model_fingerprint`: sha256 of the checkpoint payload bytes.
- `config_fingerprint`: sha256 of the resolved StGPTConfig as JSON.
- `seed_fingerprint`: sha256 of the seed CSV.
- `evidence_id`: sha256 of the above three concatenated, used as the
  per-run identifier in spatho.

Re-running with identical inputs must produce identical outputs;
random seed for k-fold splits is fixed (default 42, configurable).

## 11. Testing Plan

- A unit test that builds a tiny synthetic case (extending the smoke
  fixtures), assigns labels to 30% of regions, and asserts that:
  - `region_predictions.parquet` has the right columns and dtypes;
  - all seed regions appear with `qc_flag == 'seed'`;
  - the report's seed-fold macro F1 is above 0.5 on the synthetic
    data;
  - re-running with the same inputs yields a byte-identical output
    file.
- A smoke CLI test: `stgpt annotate-regions --config configs/smoke.yaml
  --seed-labels <generated>` runs to completion in under 30 seconds
  on CPU.

No tests against real Atera data live in the public repo, matching
the existing convention.

## 12. Out-of-Scope Decisions Logged Here

- **Active learning hooks** (next batch of regions to label) are
  intentionally deferred. Once the abstain set exists, picking
  high-uncertainty + high-coverage seeds is straightforward, but it
  is its own RFC because it affects the spatho UI.
- **Stain normalization for Phase 2/3** stays in pyXenium/spatho. The
  stGPT side records which normalization was applied as a
  fingerprint, but does not pick one.
- **A "merge labels" or "rename label" step** is out of scope. If the
  user wants `tumor + necrosis -> tumor_or_necrosis`, they do it
  upstream in the seed CSV.

## 13. Open Questions

1. Should `nearest_seed_distance` use cosine on the trained embedding
   or on the post-`structure_head` logits? Current proposal: cosine
   on embedding, because it remains meaningful when Path A is
   downgraded.
2. How do we surface "two paths disagree" in `auto_annotation_report.json`?
   Proposal: a `path_agreement.csv` side file with one row per
   region and `agree: bool` plus both top-1s, gated by `--classifier both`.
3. For Phase 2, do we require the second slide to share the same
   contour segmentation parameters with the seed slide, or do we
   accept any pyXenium contour set and rely on QC?  Proposal:
   require, until cross-segmentation calibration is studied.

## 14. Status

This RFC is design-only. No code lands until the contract above is
signed off. Implementation order, once approved:

1. `annotate_regions` Python entry point with Path A and Path B.
2. CLI command and smoke test.
3. Seed-fold report and abstain rule.
4. Real-data dry run on one Atera breast case, paired with a short
   evidence note in `docs/results/`.

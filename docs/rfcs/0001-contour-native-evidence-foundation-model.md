# RFC 0001: Contour-Native Evidence Foundation Model

**Status:** Draft  
**Author:** stGPT Team  
**Created:** 2026-05-01

## 1. Motivation

stGPT should become a contour-native, morphology-grounded spatial foundation model for Xenium. The project is not only learning tissue representations; it is building a reproducible loop from contour-level evidence tokens to human-reviewed pathology knowledge.

Existing graph-based spatial transcriptomics models, such as Novae, show that spatial domain foundation models can learn reusable tissue organization from molecular neighborhoods. stGPT should use that lesson without becoming a graph model clone. Its differentiator is the native fusion of measured Xenium expression, contour-grounded H&E morphology, physical spatial context, provenance, and agentic evidence reporting.

The target platform narrative is:

```text
stGPT-core learns contour-level morpho-molecular evidence tokens.
stGPT-evidence validates, benchmarks, audits, and versions those tokens.
spatho-agent turns evidence tokens into queryable, human-reviewed pathology evidence chains.
```

The short research claim is:

```text
stGPT is a morphology-grounded spatial foundation model for Xenium.
```

The product claim is:

```text
stGPT turns contour-level morpho-molecular representations into auditable pathology evidence.
```

## 2. Non-goals

This RFC intentionally excludes several tempting directions:

- stGPT is not a GAT/GCN clone of graph-based spatial foundation models.
- stGPT is not a simple H&E-to-expression regressor.
- stGPT is not an online continuously learning medical system.
- stGPT does not let human feedback mutate a deployed model in place.
- stGPT does not mix physical spatial adjacency and semantic molecular similarity inside the data loader.
- stGPT does not use millions of tiny PNG or JSON files as the training data path.
- stGPT does not treat model-derived imputation, embedding, retrieval, or prototype labels as measured biological evidence.

Online learning is especially out of scope. Human review creates a versioned teaching set; model and prototype-bank updates happen offline, under explicit version control.

## 3. Data Contract

The first engineering law is:

```text
row_index is the global join key; everything else is metadata.
```

Every training and evidence artifact must be joinable through `row_index`, `contour_id`, `slide_id`, and an artifact fingerprint. The physical data contract is:

```text
Xenium AnnData/Zarr      -> measured molecular evidence
Contour image store      -> object/context/shape evidence
Contour manifest Parquet -> provenance, graph indices, QC, and row indexing
stGPT data loader        -> late materialization into bottleneck tokens
```

The second engineering law is:

```text
Physical adjacency is fixed in the manifest; semantic similarity is learned by the model.
```

The neighbor graph written by preprocessing is slide-local and based only on physical coordinates. Cross-slide similarity, biological domain assignment, and retrieval belong to the embedding space and prototype bank, not to the physical neighbor table.

The training path must use packed contour image stores. Debug and export paths may write PNG or JPEG previews, but training must not depend on small image files.

## 4. Contour Image Store

H&E input is contour-native. The model should not use naive square crops around cell centroids as the primary morphology evidence. A training unit is a contour or region:

```text
contour / region
  -> member cells
  -> aggregated measured expression
  -> contour-grounded H&E object evidence
  -> context H&E evidence
  -> local neighbor contours
  -> structure and provenance metadata
```

The logical image evidence has three views:

```text
object_rgb   = mask-focused H&E evidence for the contour itself
context_rgb  = surrounding tissue context with margin
soft_mask    = contour mask or soft segmentation confidence
geometry     = explicit shape measurements
```

The physical storage should be a packed store, preferably Zarr for cloud-native and distributed training, or LMDB for local high-throughput random access:

```text
contour_image_store.zarr/
  attrs:
    slide_id
    source_wsi
    stain_normalization
    coordinate_transform
    patch_size
    context_margin
    fill_policy
    store_fingerprint

  arrays:
    object_rgb    [N, H, W, 3] uint8
    context_rgb   [N, H, W, 3] uint8
    soft_mask     [N, H, W, 1] uint8 or bool
    geometry      [N, G] float32
    contour_ids   [N]
```

The `row_index` order must be spatially sorted within each slide before arrays are written. Morton Z-order is the default because it is simple and fast. Hilbert order is optional when higher locality preservation is worth the extra dependency or implementation complexity.

The store chunking policy must align with the sampler:

```text
row_index = spatially sorted index
chunk_n   = sampler locality window or an integer multiple of it
```

This minimizes read amplification when the data loader fetches an anchor contour and its local neighbors.

## 5. Neighbor Graph Manifest

The manifest is the single source of truth for contour metadata, spatial graph indices, and provenance. It should be written as Parquet with an explicit PyArrow schema rather than relying on Pandas type inference.

Recommended columns:

```text
contour_id
row_index
slide_id
spatial_sort_key
chunk_id
centroid_x
centroid_y
bbox_level0_xy
neighbor_row_indices
neighbor_distances
neighbor_offsets_xy
neighbor_valid_mask
area
perimeter
eccentricity
qc_flag
transform_fingerprint
```

The neighbor table must have fixed shape:

```text
neighbor_row_indices : int32[max_neighbors]
neighbor_distances   : float32[max_neighbors]
neighbor_offsets_xy  : float32[max_neighbors * 2]
neighbor_valid_mask  : bool[max_neighbors]
```

Padding uses `-1` for invalid neighbor indices and `False` in `neighbor_valid_mask`. The data loader must never perform KNN or radius search during training.

The preprocessing order is:

```text
1. collect contours per slide
2. compute spatial_sort_key
3. sort rows and assign row_index
4. write contour image arrays in row_index order
5. build cKDTree on sorted slide-local coordinates
6. write fixed-shape neighbor arrays into Parquet
7. train only from row_index plus fixed arrays
```

The default graph backend is `scipy.spatial.cKDTree`. It is the best default for two-dimensional physical coordinates because it is deterministic, fast, CPU-friendly, CI-friendly, and has native radius-query support. Optional large-scale backends may include FAISS or cuML, but they should not be required for the default real-data path.

Neighbor candidate selection:

```text
radius candidates
-> distance sort
-> optional angular coverage when candidates exceed max_neighbors
-> fixed max_neighbors
-> padding with -1 and valid_mask=False
```

The config should expose:

```text
neighbor_mode: knn | radius
max_neighbors: int
neighbor_radius: float | null
neighbor_sampling: nearest | angular
graph_backend: auto | scipy | faiss_cpu | faiss_gpu | cuml
```

Neighbors must be slide-local. A corpus-global `row_index` may point across slides for storage identity, but physical neighbor indices may not cross slide boundaries.

## 6. Model Architecture

The model should use bottlenecked mid-fusion with gated cross-attention.

The guiding slogan is:

```text
Gene is the subject, image is the evidence, spatial neighborhood is the grammar.
```

The high-level flow is:

```text
Gene sparse tokens -> Gene stem --------\
Object image view  -> Image adapter -----\
Context image view -> Image adapter ------> Gated fusion blocks -> fused evidence token
Shape features     -> Shape MLP --------/
Neighbor summaries -> Local bottleneck -/
Spatial bias/masks --------------------^
```

The gene stream remains the central sequence. Image and neighbor information should be compressed into a small number of bottleneck tokens before entering the main Transformer. This prevents sequence length explosion and keeps attention cost bounded.

Recommended token groups:

```text
gene tokens                 = measured expression and gene identity
object image tokens          = contour-intrinsic morphology
context image tokens         = surrounding tissue context
shape tokens                 = area, perimeter, eccentricity, bbox, mask statistics
neighbor bottleneck tokens   = pooled local physical neighborhood
spatial bias                 = relative position, distance, and valid-neighbor masks
```

The default fusion mechanism should use gated residual cross-attention:

```text
fused = gene_state + sigmoid(gate) * CrossAttention(gene_state, image_or_context_tokens)
```

The gate should be initialized close to zero. Early training behaves like a stable gene/spatial model; image evidence enters gradually as alignment improves.

The architecture should avoid a heavy GNN tower. Graph information enters through fixed neighbor tables, bottleneck pooling, relative spatial bias, and attention masks.

## 7. Prototype Bank

The prototype layer provides global domain consistency without requiring full-slide graph training.

The principle is:

```text
Locality belongs to the input; globality belongs to the assignment.
```

The data loader can sample local anchor neighborhoods, but prototype assignment must be stabilized with cross-slide memory and reference sets.

The prototype head should use:

- SwAV-style prototype assignment.
- Optimal transport through Sinkhorn balancing.
- A detached memory queue to reduce mode collapse.
- A fixed reference set for periodic calibration.
- Hierarchical domain levels.
- Versioned prototype banks.

The Sinkhorn assignment pool should include:

```text
current batch embeddings
+ detached memory queue embeddings
+ optional fixed reference-set embeddings
```

Only the current batch receives gradients. The queue and reference set provide global distribution pressure.

Queue staleness is a risk during warmup. A future implementation should consider a momentum encoder or short queue during early training. Periodic full-corpus refresh is too expensive for large training sets, so prototype refresh should run on a fixed, stratified reference set.

Prototype-bank updates are versioned:

```text
prototype_bank_v4 -> prototype_bank_v5
```

Supported update operations:

```text
relabel prototype
split prototype
merge prototypes
create prototype
```

Historical reports must stay bound to the prototype-bank version that produced them.

## 8. Spatho-agent Evidence Workflow

spatho-agent is an evidence orchestrator, not an authority that emits unsupported biological conclusions.

The agent cannot emit a pathology or biology claim unless it can attach:

```text
row_index
contour_id
slide_id
checkpoint_id
tool_call_id
QC verdict
supporting evidence
artifact fingerprints
```

The stable pointer type is:

```text
EvidencePointer(row_index, contour_id, slide_id, checkpoint_id, artifact_fingerprint)
```

The inference workflow is:

```text
1. validate case
   -> check AnnData, manifest, contour store, registration, checkpoint card

2. embed contours
   -> read gene/object/context/shape/neighbor by row_index
   -> emit fused, gene, object, context, and shape tokens

3. assign domains and score niches
   -> assign prototype domain, confidence, hierarchy level
   -> compare gene-only, image-only, and fused outputs

4. retrieve evidence
   -> retrieve similar contours or regions from fused tokens
   -> load H&E object crop, context crop, mask, geometry, expression, neighbors

5. judge claim
   -> QC judge
   -> modality-conflict judge
   -> uncertainty judge
   -> provenance judge

6. generate report
   -> every conclusion links to evidence_id and tool-call trace
```

Doctor-facing evidence should be shown in four panes:

```text
Contour H&E evidence:
  object crop, context crop, mask overlay

Molecular evidence:
  measured genes, marker signatures, panel coverage

Spatial evidence:
  local neighbors, domain boundary, angular neighborhood

Model evidence:
  fused retrieval, prototype assignment, uncertainty, checkpoint card
```

The evidence model must distinguish:

```text
measured_evidence      = measured genes, raw H&E crops, contour geometry, physical neighbors
model_derived_evidence = embeddings, imputation, prototype domains, retrieval scores
```

Model-derived evidence must never be reported as measured expression or measured pathology.

## 9. Human Feedback Loop

Human feedback is event-sourced. A doctor correction amends an evidence record; it does not mutate a deployed model online.

The rule is:

```text
Human feedback corrects evidence records first.
Prototype and model updates happen only after curation and versioning.
```

A correction event should be append-only:

```json
{
  "event_type": "doctor_correction",
  "claim_id": "claim_001",
  "row_index": 10231,
  "contour_id": "contour_A",
  "slide_id": "slide_07",
  "checkpoint_id": "stgpt-xenium-v0.3",
  "prototype_bank_id": "pb-v4",
  "ai_domain": "tumor_stroma_interface",
  "human_label": "inflammatory_infiltrate",
  "correction_type": "domain_label_error",
  "rationale": "Morphology and marker pattern support inflammation, not tumor edge.",
  "confidence": "high",
  "reviewer_id_hash": "...",
  "timestamp": "...",
  "previous_event_hash": "...",
  "event_hash": "..."
}
```

Correction types:

```text
domain_label_error
contour_boundary_error
registration_error
modality_conflict
novel_domain
report_text_error
```

Only `domain_label_error` and `novel_domain` are candidates for prototype-bank updates. `registration_error` and `contour_boundary_error` belong to QC and failure registries.

The medium-term annotation bank is:

```text
human_annotation_bank.parquet
  row_index
  contour_id
  fused_embedding_ref
  human_label
  reviewer_confidence
  qc_status
  checkpoint_id
  prototype_bank_id
```

Future inference can retrieve against this bank before the model is retrained:

```text
This region resembles prior human-corrected inflammatory infiltrate examples.
```

Long-term training can add supervised auxiliary losses without replacing self-supervised learning:

```text
loss =
  self_supervised_loss
  + supervised_domain_ce
  + supervised_contrastive_loss
  + prototype_alignment_loss
  + disagreement_calibration_loss
```

## 10. Failure Modes and Guardrails

The system must treat failure modes as first-class evidence, not as hidden caveats.

### Sequence length explosion

Xenium neighborhoods can be dense. Neighbor cells or contours must not be flattened into unbounded Transformer tokens. Use local pooling or 1 to 4 neighborhood bottleneck tokens.

### H&E and Xenium registration failures

Registration quality gates must run before image-gene claims. Low registration confidence should block or downgrade morphology-grounded claims.

### Modality conflict

The model and agent should detect when image, gene, spatial, and prototype evidence disagree. Conflicts should trigger uncertainty or human review rather than being hidden by fused embeddings.

### Prototype mode collapse

Prototype learning must use Sinkhorn balancing, memory/reference context, and utilization metrics. Evaluation should report prototype entropy and empty-prototype counts.

### Queue staleness

Memory queues can become stale during fast early training. Warmup should use smaller queues, shorter retention, or a momentum encoder.

### Read amplification

Zarr chunk size and row ordering must match the sampler locality window. Random row order is not acceptable for real training stores.

### Evidence mislabeling

Every output must label measured evidence and model-derived evidence separately. Imputed or reconstructed values must never be represented as measured expression.

## 11. Implementation Milestones

- [ ] **M1:** Manifest and Zarr contour store contract.
- [ ] **M2:** `cKDTree` neighbor graph builder.
- [ ] **M3:** `RegionDataset` reads packed contour stores.
- [ ] **M4:** Object, context, and shape bottleneck tokens.
- [ ] **M5:** Gated mid-fusion implementation.
- [ ] **M6:** Prototype head and Sinkhorn loss.
- [ ] **M7:** Evidence chain export for spatho-agent.
- [ ] **M8:** Human feedback event log.

## 12. Acceptance Criteria

The RFC is considered implemented when:

- A real Xenium case can be validated through a packed contour image store and Parquet manifest.
- Training does not read per-contour PNG or JSON files except in smoke or debug mode.
- The data loader retrieves contour image, gene, geometry, and neighbor evidence through `row_index`.
- Neighbor graph retrieval during training is O(1) lookup from fixed-shape arrays.
- Model outputs include fused token, modality-specific tokens, checkpoint identity, and evidence pointers.
- Prototype-bank version, assignment confidence, and utilization metrics are exported.
- spatho-agent evidence chains separate measured evidence from model-derived evidence.
- Human corrections are append-only events and can be replayed into a curated annotation bank.


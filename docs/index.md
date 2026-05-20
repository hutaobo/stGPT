# stGPT

::::{div} stgpt-hero
Platform-aware morpho-molecular foundation-model backend in development for spatial transcriptomics.
::::

`stGPT` learns reusable tissue representations from spatial transcriptomics expression, H&E morphology, spatial context, and structure evidence across Xenium and Atera data. The project is designed to stay independent as a model package while exporting auditable evidence artifacts for downstream workbenches such as `spatho`.

The platform narrative is: stGPT learns reusable contour/region morpho-molecular representations; spatho plans, validates, and turns them into auditable spatial pathology evidence.

::::{grid} 1 1 2 3
:gutter: 2
:class-container: stgpt-card-grid

:::{grid-item-card} Atera XeniumSlide Data Foundation
:link: tutorials/atera_xeniumslide_data_foundation
:link-type: doc

Reproduce the completed Breast and Cervical Atera XeniumSlide build, QC, contour assignment, and stGPT validation record.
:::

:::{grid-item-card} Development Strategy
:link: strategy
:link-type: doc

Read the platform-aware development guide, agentic runtime contract, and method-landscape judgments for stGPT.
:::

:::{grid-item-card} Foundation Requirements
:link: foundation_model_requirements
:link-type: doc

Track the foundation-model, schema-first runtime, evidence graph, and reproducibility requirements.
:::

:::{grid-item-card} Contour/Region Vision
:link: contour_region_foundation_model
:link-type: doc

Plan the contour- and region-level foundation-model target and spatho downstream artifact contract.
:::

:::{grid-item-card} RFC 0001
:link: rfcs/0001-contour-native-evidence-foundation-model
:link-type: doc

Freeze the contour-native evidence foundation-model design, including packed contour stores, fixed neighbor manifests, mid-fusion, prototype banks, and human feedback.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: Tutorials

tutorials/atera_xeniumslide_data_foundation
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Project Notes

strategy
foundation_model_requirements
contour_region_foundation_model
rfcs/0001-contour-native-evidence-foundation-model
```

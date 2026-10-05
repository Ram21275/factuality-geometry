# A runnable sequence and interpretation rules

This study compares explanations; it does not assume a factuality manifold exists. All geometry is a property of the selected encoder and sampled dataset, not automatically a property of truth itself.

## Study A: establish text-only structure

1. Inspect the author archive. Identify claims, original verdicts, evidence links, source/date fields and duplicate-event groups. Provide an explicit `columns` mapping; keep explanations/rulings out of `text` and `evidence_texts`.
2. Run `audit,embed` with the pilot. Check `audit/result.json`, original split overlap, missing modalities, encoder revision and ID alignment. Complete the near-duplicate/image reuse audit before confirmatory results.
3. Run `experiments,evaluate`. Compare raw K-means/GMM/PC1 against residual GMM/PC1 and the shared axis. A `no_shared_axis_found` result is valid, not an implementation failure.
4. Change to the full preset for prespecified seeds/ranks and repeated group holdouts. Keep encoder and embedding shard size unchanged to reuse embedding shards.
5. In separately named configs set `split_mode=topic` and `split_mode=source`. Repeat with the second encoder. Do not search for the best test-scoring config.

Proposed main gate: shared-axis AUC* improves by at least .03 over a prespecified strong baseline, paired group interval excludes zero, conditional association survives the null, and the effect transfers across topics/sources and encoders. These are proposed practical criteria; uncertainty and sample sizes matter more than crossing one arbitrary threshold.

## Study B: class-associated regions or multiple surfaces?

Run `experiments,geometry,evaluate` with the graph grid enabled. Inspect:

- Surface count/rank versus held-out reconstruction error and post-hoc component association. More surfaces give greater capacity, so describe the capacity tradeoff; do not interpret raw reconstruction gains as proof of separate class manifolds.
- Diffusion versus ordinary PCA/GMM predictions on the same test IDs. Confirm across k values and report connected-component coverage.
- `geometry_report.json`: same-topic, distance/length/source-stratified geodesic-stretch and local-normal effects. Sparse strata may produce no estimate; do not relax matching after seeing a desired label result.
- `paired_comparisons.json`: incremental association and intervals rather than 2D plot appearance.

Graph distance diagnostics are exploratory training-graph measurements; they are not claimed as held-out geodesic verification. Local normal diagnostics use training neighborhoods and held-out pair endpoints. Their confidence intervals use event-group pair weighting. Spectral coordinates are projected to held-out examples through fixed training-neighbor interpolation, not through a full-test refit.

Interpretation candidates:

| Observation | Supported scope of interpretation |
|---|---|
| Stable class-associated local-surface components after controls | Representation contains geometric heterogeneity related to labels |
| Diffusion improves but shared axis fails | A nonlinear regional interpretation is worth developing |
| Shared axis transfers but graph methods do not add value | A simple cross-topic coordinate is sufficient in this representation |
| Raw associations disappear after topic/source controls | Apparent geometry was largely confounded |
| No reliable text-only association | Test evidence/multimodal relationships; do not declare factuality impossible |

Finite noisy samples generally cannot prove “different manifolds” versus “different regions” uniquely. These experiments compare operational models, not establish topology.

## Study C: evidence and multimodal relations

Normalize original evidence paragraphs into explicit `evidence_texts` lists. Run `evidence,evaluate`. The three conditions are supplied evidence (oracle linkage), dense retrieval from training-linked evidence, and within-topic shuffled evidence. Compare full displacement/reconstruction features with nearest-distance only. Inspect retrieval corpus limitations and evidence leakage manually. No entailment/contradiction reasoning is claimed from cosine distance.

Normalize image links into `image_paths` lists. Run `multimodal,evaluate --image-root ...`. All unimodal/joint views use the same paired claim IDs. Compare interaction features against image-only, CLIP text-only, MPNet text-only, concatenation and similarity. Check the shuffled-image condition, provenance overlays, OCR verdicts, reused images and missing-modality selection before attributing improvement to relational reasoning.

Do not subtract MPNet vectors from CLIP image vectors; interactions are constructed in a common CLIP space. Image-only evidence experiments are not image-forgery detection. An independent VERITE task requires a dataset-specific adapter and a separately locked transfer protocol.

## Study D: independently reviewed edits

Create a panel CSV with `family_id,text,edit_type,verified_level`, exactly one `original` per family and reviewed paraphrase/style/factual changes. Keep it disjoint from discovery families. Run `interventions --intervention-panel ...` after text experiments. Inspect frozen-axis score changes, factual-versus-style responses and ordinal correlation only for genuinely graded levels. The code does not infer factuality labels from an edit operation.

## Continuation and preregistration

Before the confirmatory run, freeze the config, data/split hashes, model revisions and Git tag. Preserve checkpoints between sessions. Changes create a different run directory; old outputs remain available. Group/source/date/encoder replications are separate tests, not independent repetitions of the same statistical sample. If a method is selected using labeled development results, identify it as label-assisted selection and evaluate it on a fresh untouched set.

Paper claims should describe what survived controls: a label-associated coordinate, regional geometry, evidence-dependent structure, or a negative result. Do not claim a universal truth direction or proven fiber bundle.

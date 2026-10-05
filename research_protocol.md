# Testing topic-invariant factuality structure: a Kaggle research protocol

Prepared 5 October 2026. This is an experimental design, not a report of results. The dataset has not yet been supplied or run. Numerical gates below are proposed practical criteria, not established performance guarantees.

## 1. Verdict and defensible research question

The proposal is credible as an empirical representation-learning hypothesis. It is not yet a validated method, and its novelty is not established. The most defensible starting question is:

> Do frozen text embeddings contain a reproducible, low-dimensional factor associated with claim truthfulness that generalizes across topics and sources after removing topic variation?

The stronger follow-up is:

> Does evidence-conditioned geometry or image–text interaction improve that factor beyond ordinary similarity and clustering baselines?

Fake news is a reasonable application, but use **claim verification** as the primary task definition. Whether a news story is intentionally deceptive, whether a claim is false, whether an image is manipulated, and whether a caption misrepresents an authentic image are different targets.

### What the literature does and does not establish

- [DEAR, TACL 2025](https://aclanthology.org/2025.tacl-1.16/) separates event-related and authenticity-related representations with a learned architecture. It motivates testing event invariance; it does not establish an unsupervised veracity manifold.
- [Unsupervised Domain-agnostic Fake News Detection using Multi-modal Weak Signals, 2023](https://arxiv.org/abs/2305.11349) already studies unsupervised multimodal detection and unseen domains. Consequently, unsupervised detection, multimodal fusion, and domain invariance alone are not sufficient novelty claims.
- [Locatello et al., ICML 2019](https://proceedings.mlr.press/v97/locatello19a.html) shows limitations of identifying disentangled factors without inductive assumptions. Here, topic removal and recurrent mixture directions are assumptions to test, not guarantees that a discovered factor means truthfulness.

This targeted review does not establish that the exact algorithm below is novel. Before making a publication claim, compare its objective, assumptions, training signals, and evaluation with prior disentanglement, deep clustering, unsupervised fact-checking, and multimodal weak-signal methods. The unnamed 2026 benchmark in the pasted proposal is insufficiently specified to verify and should not be cited as supplied.

### Corrections to the pasted mathematical interpretation

1. **Topic centering does not identify veracity.** If a topic contains almost only one class, centering can remove the signal being sought. Report topic–label composition after fitting; never use it to select the fitting data in the strict label-free track.
2. **A repeated direction can be a nuisance.** Source, writing style, chronology, quoted speech, claim length, or embedding anisotropy can produce the same result. Alignment must be followed by truthfulness association and held-out transfer.
3. **A fiber is not necessarily a normal direction.** In a product model, variation within the veracity factor can be tangent to the overall manifold. Local PCA on mixed observations estimates high-variance directions, not a known topic tangent. Call Phase 4 a test of transverse variation relative to an estimated local subspace.
4. **Graph separation is not proof of distinct manifolds.** Sparse sampling, k, hubness, duplicates, and graph construction can all create disconnected components. Finite embeddings cannot by themselves establish a product topology or fiber bundle.
5. **Semantic closeness is not factual support.** “The trial reduced mortality” and “The trial did not reduce mortality” can be close in embedding space. A convex-hull reconstruction error or nearest-evidence distance is a baseline, not a truth criterion.
6. **Distortion is not meaning-preserving.** A paraphrase preserves a proposition; changing a number, date, entity, or causal claim changes it. Counterfactual editing supplies an intervention prior and must be reported separately from purely observational label-free fitting.
7. **Truthfulness need not be bimodal or ordinal.** Not-enough-information is an epistemic state, not the midpoint between true and false. Binary labels cannot validate a continuous graded factuality scale.
8. **Frozen does not mean free of supervision or memorization.** Pretrained encoders have training objectives and may have seen public claims. The claim is “no task labels used to fit our discovery method,” not learning truth from zero prior knowledge.

## 2. Dataset recommendation and upload instructions

### Primary choice: MOCHEG

Use the original [MOCHEG release and documentation](https://github.com/PLUM-Lab/Mocheg). The [dataset paper](https://arxiv.org/abs/2205.12487) describes 15,601 claims with truthfulness labels, 33,880 evidence paragraphs, and 12,112 evidence images. It supports a text-first study and a later claim-plus-multimodal-evidence study in one resource.

**Important task distinction:** these are claims with evidence. The image counts refer to evidence images; do not assume every claim has its own attached image or that every record supports all modalities. This is not automatically an image-only fake-news or image-forgery benchmark.

**Access:** the official repository's version-1 link currently points to a [dataset request form](https://docs.google.com/forms/d/e/1FAIpQLScAGehM6X9ARZWW3Fgt7fWMhc_Cec6iiAAN4Rn1BHAk6KOfbw/viewform). Obtain the authors' release, then upload the archive to Kaggle. I verified the repository and access link, not the archive contents or a successful download. Exact split sizes, raw field names, missing-image rates, and domain coverage must be audited after access.

Upload the claims, original splits, text evidence, images, claim–evidence mapping, dataset statement, and provenance information. Preserve the unmodified release alongside processed data. Prefer a private Kaggle dataset initially. The repository declares CC BY 4.0 for the dataset; preserve attribution and review the released assets' notices before redistribution.

### Alternatives and later benchmarks

| Dataset | Role | Why / limitation |
|---|---|---|
| [AVeriTeC](https://github.com/MichSchli/AVeriTeC) | Best text-and-evidence fallback; independent replication | [Paper](https://arxiv.org/abs/2305.13117): 4,568 real-world claims from 50 fact-checking organizations, with web evidence and question–answer annotations. Stronger evidence provenance, but smaller and not a paired image–caption benchmark. Use accessible labeled splits; do not assume hidden test labels are available. |
| [Fakeddit](https://github.com/entitize/Fakeddit) | Optional large engineering pilot | Over one million Reddit samples; labels use distant supervision. Good for pipeline scaling, weaker for proving factuality. Source/subreddit shortcuts can dominate. Use the official multimodal subset for multimodal comparisons and identify the label task explicitly. |
| [VERITE](https://github.com/stevejpapad/image-text-verification) | Later independent image–caption robustness test | 1,000 pairs with truthful, miscaptioned, and out-of-context labels; modality balancing addresses unimodal shortcuts. It is small and suited to external evaluation. The release provides image URLs and CLIP features, not a complete packaged image collection. |
| [NewsCLIPpings](https://aclanthology.org/2021.emnlp-main.545/) | Optional image–caption mismatch training/discovery benchmark | Automatically creates out-of-context pairings. Measures contextual compatibility, not general claim truthfulness. An authentic image can accompany a false caption. |

**Recommendation:** obtain MOCHEG first. If access delays the project, start Phases 0–3 on AVeriTeC. Use Fakeddit only as a pilot or additional stress test. A convenient Fake.csv/True.csv collection should not be the main evidence for a topic-invariant factuality factor.

## 3. Formal hypotheses and scientific claims

Let z=f(x) denote an embedding from a frozen encoder, t an inferred topic, s a source group, and y an evaluation-only truthfulness label.

| Hypothesis | Operational test | Finding that weakens it |
|---|---|---|
| H1: raw embeddings contain factuality association | Cluster–label association and within-topic neighbor purity beat conditional permutation nulls | Association vanishes when controlling topic/source |
| H2: a shared factor remains after topic removal | An axis fitted without y transfers to withheld topics/sources and exceeds raw/PCA baselines | Only one topic/source carries the effect; bootstrap axis is unstable |
| H3: graph geometry adds useful information | Diffusion/graph methods exceed matched Euclidean/PCA baselines across neighborhood sizes | Improvements depend on a single k or graph fragmentation |
| H4: factuality variation is more transverse locally | Distance-matched same-topic opposite-label pairs have larger normal-energy ratios | Effect disappears after matching or depends on tangent rank |
| H5: controlled factual distortions have recurrent trajectories | Held-out edit families/topics show coherent movement relative to paraphrase/style controls | Trajectory tracks edit syntax, magnitude, or template alone |
| H6: evidence-conditioned features improve discrimination | Improvement over semantic-distance and randomized-evidence controls | Shuffled evidence works equally well; verdict text explains success |
| H7: multimodal interaction adds factuality information | On the same records, joint features beat text-only/image-only/concatenation controls | Improvement comes from missing-image pattern, OCR verdicts, or source artifacts |

H1–H7 are related but distinct. Failure of text-only H2 does not imply evidence or multimodal verification is impossible. Conversely, a successful evidence model does not validate a universal text-only veracity axis.

## 4. Evaluation firewall and split design

### Strict discovery track

- Fit the encoder outputs, topic model, PCA, graph, axis, and clustering without truthfulness labels, verdicts, ruling statements, justifications, fact-check headlines, or metadata encoding verdicts.
- Infer topics from claim text using TF–IDF plus K-means, separate from the sentence-embedding axis being tested. Metadata topics, if present, can be a sensitivity analysis.
- Keep truthfulness labels in a separate evaluation table. Write scores, clusters, configuration, row IDs, and checksums before joining labels.
- Fit all learned transforms on discovery/training records only. Use fixed transforms for new records. Building a graph on the full dataset is permissible only as a separately named **transductive** experiment.
- Specify cluster counts in advance: K=2 as a binary-task prior, with K=1,2,3,4 and BIC as a sensitivity analysis. A binary prior is not evidence that two natural components exist.
- Topic counts and all parameter grids are fixed before inspecting truthfulness results. Unlabeled validation likelihood/stability may select parameters. If labeled development performance selects anything, report that as a separate label-assisted model-selection track.

### Partitions

1. Keep official train/development/test membership for a comparable reference experiment, after auditing cross-split leakage.
2. Group normalized duplicates, near-duplicates about the same claim/event, and reused images before creating new robustness partitions. A group must never cross train/test.
3. Fit the primary model on all truthfulness categories, without labels. After scoring, evaluate supported versus refuted as the primary binary contrast. Report excluded-category coverage. Do not treat NEI as false or force it onto an ordinal continuum.
4. Add five group-based folds with seed 13; create them without truthfulness stratification in the strict track. Repeat method randomness using seeds 13, 29, 47, 71, 101. These seeds are not five independent datasets.
5. For unseen-topic testing, infer topic groups on development/discovery data without y and reserve whole groups. Using all unlabeled claim texts to define topic partitions is a split-design/transductive use that must be disclosed; refit the actual representation pipeline excluding each withheld group.
6. Add leave-one-source-group-out and, if dates are trustworthy, chronological evaluation. Fact-checking publisher and original claimant/source are different metadata; report which you use. Skip unsupported source/time analyses instead of inventing metadata.

The primary topic-residual projection is a fixed global transform. Do not estimate a held-out topic's centroid from its test samples in an inductive run. A separately reported batch adaptation run may do that without labels.

### Metrics and uncertainty

- Primary clustering association: adjusted Rand index (ARI). Secondary: adjusted mutual information and confusion tables. Ordinary NMI may describe nuisance associations but compare against permutation nulls because category counts differ.
- Continuous-axis association: AUC* = max(AUC, 1−AUC), which evaluates association without naming the axis direction. Apply this same orientation maximization inside every permutation; its null is not exactly 0.5 in small samples.
- A semantic “fake/true” orientation requires an anchor. If needed, use a reserved calibration partition's labels solely to orient the already frozen score; disclose this and use standard AUROC on untouched test data. Never flip the score using test labels for a deployment metric.
- Cluster-to-class matching computed on the test set is an oracle descriptive metric, not deployable classification accuracy. Prefer ARI; if reporting matched accuracy/F1, label it explicitly.
- Use 1,000 group/event bootstrap replicates for 95% confidence intervals, pairing resamples when comparing methods. Bootstrap groups rather than individual correlated claims. Report the number of groups and per-topic results.
- Use 1,000 label permutations within topic and sufficiently populated source strata for conditional association tests. When strata cannot exchange labels, report the limitation and use a coarser predeclared test; do not interpret an uninformative null as evidence.
- Apply Holm correction to the small prespecified family of primary hypothesis tests. For parameter sweeps, publish all results rather than picking the best labeled run.
- Report class proportions, supported/refuted coverage, missing-image coverage, and per-domain/source sample counts. AUROC complements rather than replaces precision–recall metrics in an imbalanced binary task.

## 5. Phase-by-phase experiments

### Phase 0 — data audit, leakage controls, and feasibility

**Question:** can the dataset support a cross-topic factuality test rather than a source-style classification task?

**Procedure:** inspect the release schema; create a clean claim table and separate evidence/label tables. Normalize whitespace for duplicate hashes; group near duplicates using character n-grams or embeddings plus manual review. Inspect a small blinded sample for fact-check boilerplate. Preserve negation, numbers, dates, named entities, and punctuation: they can encode factual distinctions. Group repeated images using file hashes and perceptual hashes when images are available. Check split overlap, image decode failures, language, dates, and metadata completeness.

**Evaluation-only audit:** after fitting topic partitions, tabulate topic×truthfulness and source×truthfulness. Estimate how many topics have at least 50 supported and 50 refuted evaluation examples. This is a diagnostic, not a label-based rule for filtering the primary fitted corpus.

**Gate:** aim for at least four reasonably populated mixed-label topics for a shared-factor claim. If the release cannot supply them, reduce the claim or add a replication corpus; do not invent broad topic coverage. Complete the audit regardless of whether this target is met.

**Artifacts:** dataset manifest/checksums, mapping of raw to normalized fields, group assignments, split manifest, leakage report, label audit held outside training.

### Phase 1 — frozen text baselines

**Question:** what is already present before any new geometry?

**Inputs:** claim text only. Primary encoder: [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), 768-dimensional embeddings. Replicate with [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), 384 dimensions. Pin the resolved model revision in the manifest. Normalize embeddings for cosine comparisons. Log token truncation; if long documents are introduced later, use a prespecified chunk-and-mean baseline rather than silently truncating them.

**Baselines:** TF–IDF→TruncatedSVD→K-means; sentence embedding→K-means; training PCA32→diagonal GMM; raw PC1→one-dimensional GMM; shuffled labels and random axes. K=2 is the primary binary contrast. Fit GMM with regularized covariance and 10 restarts. A supervised logistic regression on training labels may be included as a separately labeled reference, not as an unsupervised method or information-theoretic upper bound.

**Measurements:** ARI, AUC* for scalar scores, nuisance associations, within-topic neighbor purity at k=10,20,40. Compare purity with topic/source-conditional label permutations; raw purity alone is inflated by class imbalance. Two-dimensional PCA/UMAP plots are illustration only, never primary evidence.

**Artifact:** complete baseline table with seeds, intervals, runtime, row coverage, and fitted configurations. A negative result is useful and does not block testing topic removal.

### Phase 2 — topic removal and a concrete shared-axis estimator

**Question:** does a stable cross-topic factor emerge after removing between-topic variation?

**Topic model:** TF–IDF on discovery text→K-means, T=8 primary; T=4,12 sensitivity. Audit discovered top terms without reading truthfulness labels. Topic assignments are approximate content partitions, not ground-truth domains.

**Two ablations:**

1. Centroid residual: r_i=z_i−μ_(t_i), using training centroids and fixed topic assignment. Primarily useful for in-domain comparison.
2. Global topic-subspace residual: compute the centered, weighted topic-centroid matrix from training embeddings, take its leading q_T singular vectors U_T, and set r_i=(I−U_T U_Tᵀ)(z_i−μ). Use q_T=2 primary, q_T=0,1,4 sensitivity. q_T=0 is the no-removal control. This fixed transform also applies to unseen topics.

Do not claim these operations exactly compute a manifold quotient. Fit a common PCA32 on training residuals to obtain h_i. Avoid automatic whitening in the primary run; whitening can amplify weak/noisy directions.

**Label-free axis estimator:**

1. For each training topic with at least 200 records, fit diagonal GMMs with one and two components to h.
2. Compute ΔBIC_t = BIC_1−BIC_2. Retain a topic's candidate direction only if ΔBIC_t>0 and both estimated component masses are at least 0.10. Require at least three valid topic directions. Fit failure or insufficient topics yields “no shared axis found.”
3. Define u_t=(m_t,1−m_t,2)/||m_t,1−m_t,2||. These are unlabeled component means, not fake/true centroids.
4. With equal topic weights, form M=(1/T_valid) Σ_t u_t u_tᵀ. Its leading eigenvector u is the proposed common direction. Outer products avoid arbitrary component-sign differences.
5. Score new records a_i=uᵀh_i; optionally fit a global one-dimensional GMM to training scores for cluster assignments.

This is a proposed estimator, not an established optimal objective. BIC favoring two components does not necessarily imply two visible modes or factuality.

**Controls:** raw PC1; residual PC1; raw and residual PCA→GMM; per-topic mixture clustering without axis aggregation; random axis in the identical residual space; random topic partitions with the same sizes, refitting the full estimator. Compare centroid removal with subspace removal. For multi-axis sensitivity, compare leading subspaces using principal angles rather than individually matched eigenvectors.

**Measurements:** pairwise |u_tᵀu_s|, spectrum of M, bootstrap axis/subspace stability, AUC*, ARI, nuisance association, and held-out topic/source transfer. Alignment must beat the random-partition control, which preserves much of the encoder's global anisotropy. After the estimator is frozen, label-conditioned difference vectors can be computed as a diagnostic; do not feed them back into training.

**Proposed gate:** a paired improvement of at least 0.03 in held-out AUC* over the strongest prespecified simple baseline, with a group-bootstrap interval excluding zero, conditional permutation support after correction, positive transfer on most evaluable topics, and replication with the second encoder. AUC*≥0.65 is a useful planning target, not a universal publication threshold. If the study is underpowered, report uncertainty rather than declaring no geometry exists. Do not retune until the gate passes.

**Artifact:** frozen axis/projection, per-topic alignment matrix, full ablation table, and go/no-go assessment for the shared-factor extension.

### Phase 3 — graph and intrinsic-distance analysis

**Question:** does a graph representation add anything beyond ordinary embedding similarity?

Build symmetric union k-NN graphs on training raw embeddings and residual PCA coordinates. Use k=10,20,40. For normalized raw embeddings use Euclidean chord distances as edge lengths; use PCA-coordinate Euclidean distances for residuals. For diffusion/spectral analysis convert distances to a prespecified Gaussian affinity using a training-data bandwidth. Distance weights and similarity affinities have opposite meanings; do not interchange them.

Compute connected-component coverage, diffusion/spectral coordinates, and graph clustering. Estimate shortest-path distances from 128 training landmarks, rather than constructing every pairwise shortest path. For out-of-sample inference use a fixed nearest-training-neighbor interpolation of training diffusion coordinates; keep full-test graph experiments separate as transductive.

For explanatory geodesic analysis, sample same-topic same-label and opposite-label pairs matched on Euclidean distance and claim length. Compare graph stretch d_G/(d_E+ε) and connectivity rates. Disconnected distances remain undefined/infinite; report them separately and do not replace them with an arbitrary large value or add bridges to obtain a desired result. Pair sampling uses labels only for this evaluation analysis.

**Controls:** raw/residual PCA with the same output dimensions, graph variants, duplicate-free records, conditional permutations, and a fixed set of distance-matched pairs. Show whether increased class separation is simply fragmentation.

**Gate:** retain graph features only if their incremental benefit survives graph choices and held-out evaluation. Otherwise prefer the simpler axis method. Class-related connectivity is evidence of graph association, not proof of two manifolds.

**Artifacts:** graph diagnostics, component coverage, distance-matched comparison, and spectral-versus-PCA ablation.

### Phase 4 — local tangent/transverse analysis

**Question:** is truthfulness-associated variation unusually transverse to a local subspace after matching semantic proximity?

Use training PCA coordinates with dimension 32 or 64. Estimate each local subspace from same-topic training neighbors, k=20,40,80. Use thin SVD of centered neighbor matrices, with r=2,4,8 and r<k−1. For evaluated pair displacement Δ, compute R_perp=||(I−UUᵀ)Δ||²/(||Δ||²+ε). Avoid zero-distance duplicates. Exclude the compared endpoint from the neighborhood used to estimate U; otherwise the displacement partly defines its own tangent.

Compare same-topic same-label and opposite-label pairs matched on Euclidean distance, length, source, and date where feasible. Cross-topic comparisons are secondary. Use group-bootstrap effect differences and conditional permutation tests. Analyze independent query points without fitting a tangent on test labels.

**Controls:** random subspaces with the same rank, global PCA subspace, shuffled topic partitions, and k/r sensitivity. Higher normal energy can arise from a larger displacement or a poor tangent estimate, so matching is essential.

**Gate:** proceed with a transverse interpretation only if the matched effect repeats across topics and rank choices. Otherwise drop that interpretation, even if a useful shared axis remains.

**Artifacts:** matched-pair manifests, R_perp distributions, effect intervals, and sensitivity table.

### Phase 5 — controlled interventions and trajectory coherence

**Question:** does the discovered factor respond to factual changes beyond style or edit syntax?

Prepare a separate panel of approximately 200 supported, evidence-backed claims across at least four available topics. This is an evaluation/intervention panel with explicitly disclosed label/evidence use, not part of strict unlabeled discovery. Remove duplicate claim families from discovery data.

For each original create (a) a meaning-preserving paraphrase, (b) a style/sensationalism edit preserving the proposition, and (c) minimally edited entity, date, quantity, negation, or attribution variants. Verify the changed propositions against evidence; a changed number is not automatically false. Where possible construct a graded distortion series with at least three reviewed levels. Use two reviewers blinded to geometric scores and record disagreements; if that is unavailable, call it a synthetic stress test rather than validated factuality ground truth.

Do not pool unrelated edits into a presumed severity order. “Refuted,” “unsupported,” and “misleading” need not form one scale. Split by original claim, edit template, topic, and editor so repeated wording cannot drive transfer.

Evaluate the frozen Phase-2 axis first: paired score changes, paraphrase stability, alignment of verified factual-change displacements, and ordinal correlation only for independently graded sequences. Compare cosine coherence and path length with matched style and randomly ordered controls. A path through edited samples is not necessarily a shortest geodesic; show graph coverage and endpoints before making geodesic claims.

If intervention vectors are subsequently used to learn an attribute subspace, name it a separate intervention-informed method and evaluate on untouched claim families/edit types. It is no longer the same strictly observational estimator.

**Gate:** require factual-change effects to exceed paraphrase/style effects and transfer beyond the editing template. Synthetic success alone cannot establish natural-news factuality geometry.

**Artifact:** intervention panel, evidence checks, review agreement, paired effect table, and frozen-axis trajectory plots.

### Phase 6 — evidence-conditioned geometry

**Question:** does factuality become more identifiable when the representation includes evidence?

Start with two explicitly separate conditions: (1) supplied/gold-linked evidence, an oracle retrieval setting, and (2) a fixed offline text-evidence corpus queried using BM25 or frozen dense retrieval without verdict fields. In retrieval evaluation measure recall@5 and @10 against supplied links. Freeze corpus version, record evidence dates when known, and exclude ruling/justification text and leaked fact-check verdicts. Prefer source material predating the ruling; report unverifiable timestamps. Shared fact-check pages can leak across splits even when claim strings differ.

Use top m=5 evidence passages per claim as primary. Compare:

- nearest evidence cosine distance;
- nonnegative simplex-constrained convex-hull reconstruction error, as a baseline only;
- claim-to-evidence displacement features, evidence embedding dispersion, numeric/date/entity match flags, and missing-evidence indicators;
- the same features after topic removal and/or graph transformation.

Scale features using training statistics, fit a diagonal GMM without task labels, and compare with the strict text-only baseline. For counts/matches, specify extraction rules and publish them. These are consistency proxies, not logically sufficient support/contradiction detectors.

An optional pretrained NLI model supplies entailment/contradiction probabilities. It brings supervised transfer and must be a separate track with a model card and training-data disclosure; improvements cannot be attributed solely to unsupervised geometry.

**Controls:** evidence shuffled within topic and source when possible; same-topic irrelevant evidence; text-only; distance-only; raw residual features without graph; evidence-count-only and availability-only models. Audit a blinded sample for verdict leakage. Human-selected gold evidence is informative annotation even if class labels are hidden.

**Gate:** improvement must exceed distance-only and randomized-evidence controls, remain in retrieved-evidence conditions, and survive leakage auditing. If only oracle evidence works, report that limitation. Do not estimate a claim/evidence tangent angle unless an actual sequence of related observations defines each tangent; one claim and a bag of passages do not provide trajectories.

**Artifacts:** evidence retrieval manifest, retrieval recall/coverage, oracle-versus-retrieved results, feature ablations, and leakage audit.

### Phase 7 — image and multimodal extension

**Question:** does visual evidence or image–text interaction add information beyond text alone?

For MOCHEG, formulate the experiment as **claim verification using image evidence**, not image-forgery detection. Fix an image-evidence subset and compare every modality on those same claim IDs. Report coverage and label/source composition separately from whole-corpus text results. Use one-to-many claim–image relationships as released; do not manufacture a single “claim image” field.

Freeze a CLIP encoder such as [ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32). Record model revision and preprocessing. Embed text and images in the same CLIP space for cross-modal differences. MPNet and CLIP vectors cannot be subtracted directly. Treat OCR as its own text-bearing condition; screenshots containing verdicts/rating badges must be identified and excluded or masked under a documented rule.

Compare on identical examples:

1. Claim-text embedding only (MPNet and CLIP-text controls).
2. Image-evidence embedding only, with permutation-invariant mean pooling for multiple images.
3. Concatenated separately normalized text/image features with equal total block scaling.
4. CLIP text–image cosine similarity only.
5. Interaction features: signed differences, elementwise products, similarity, and dispersion across evidence images; use fixed pooling and fit a training-only PCA→GMM or shared-axis estimator.
6. Text-evidence features plus image-evidence interactions, with nuisance-removal and no-removal ablations.

Do not assume a text-space axis equals an image-space axis. Refit the analogous unsupervised estimator in a defined common feature space, then test its association. Weight selection using truthfulness labels belongs to a separate label-assisted track.

**Controls:** within-topic image shuffling, same-event mismatches, image-only/text-only scores, missing-modality-only features, OCR on/off, source overlays, duplicate-image groups, and equal paired-subset coverage. A model that works on image-only evidence may encode source selection rather than relational verification.

**Independent paired-image benchmark:** evaluate the method family on VERITE with its original-label interpretation and grouped examples. It tests image–caption contextual misinformation rather than the same target as MOCHEG, so report it as a separate transfer task. For a locked external test, do not fit PCA, GMM, thresholds, or graph bandwidth on VERITE. If unlabeled target adaptation is needed, disclose it separately. An image-only classifier should be challenged by modality-balanced pairs; weak image-only performance is not evidence that a relational method failed.

**Gate:** retain multimodal geometry only if it exceeds the strongest paired-subset baseline by a prespecified practical margin (e.g., 0.03 AUC*), with a paired group interval excluding zero, and loses useful relational information under image shuffling. Replicate or clearly limit the claim when the independent dataset does not support it.

**Artifacts:** modality comparison, image availability and corruption report, shuffled-image results, OCR/overlay audit, and external-task results.

### Phase 8 — locked replication and paper claim

Freeze the protocol after exploration, reserve new claims/domains or an independent dataset, and rerun the selected primary estimator without searching labeled scores. Replicate text findings on accessible AVeriTeC partitions. Report all failed hypotheses and baseline comparisons.

Possible justified outcomes:

- “Topic removal exposes a reproducible label-associated coordinate in two frozen encoders.”
- “A common axis transfers across tested topics but remains source-sensitive.”
- “Text-only geometry is weak; evidence-conditioned relational features improve verification.”
- “Geodesic/transverse assumptions do not survive controls.”

Avoid claiming discovery of a universal veracity fiber, a proven factuality manifold, or truth verification from semantic distance. A careful negative result is more informative than an attractive UMAP plot.

## 6. Kaggle execution layout

The following normalized layout is a proposed adapter interface, not a claim about MOCHEG's original schema:

```text
/kaggle/input/factuality-study/
    raw_release/                 # original archive contents and documentation
    claims.parquet               # id, claim_text, official_split, group_id
    evaluation_labels.parquet    # id, original_label; never used in fit functions
    audit_metadata.parquet       # id, source/date if available; no model features by default
    evidence_text.parquet        # evidence_id, evidence_text, source/date if available
    claim_evidence.parquet       # claim_id, evidence_id, modality, link_provenance
    images/                      # original files, if supplied
    image_manifest.parquet       # image_id, relative_path, hash, decode_status
    dataset_manifest.json
/kaggle/working/factuality-study/
    embeddings/ models/ predictions/ figures/ reports/
```

Suggested notebooks, to implement after inspecting the release:

| Notebook | Scope | Saved outputs |
|---|---|---|
| 00_data_audit | Normalize and group records, freeze splits | manifests and leakage report |
| 01_text_discovery | Phases 1–2, with evaluation in separate final cells | embeddings, transforms, axis, scores |
| 02_graph_local_geometry | Phases 3–4 | graph coverage, landmark distances, pair metrics |
| 03_interventions_evidence | Phases 5–6 | intervention checks, retrieved evidence, scores |
| 04_multimodal_replication | Phases 7–8 | paired modality results and replication report |

Use NumPy, pandas, SciPy, scikit-learn, sentence-transformers, PyTorch, Pillow, and matplotlib. Optional UMAP is for visualization. Log installed package versions rather than assuming a particular Kaggle environment. Use Kaggle Internet access for the initial package/model fetch or attach model files and compatible wheels as offline inputs; do not assume Internet or GPU is enabled.

A single available GPU is sufficient for frozen embedding extraction in principle; geometry runs mainly on CPU. Start with a label-blind, group-sampled 2,000-claim smoke run, then run the full released claims. The pilot checks implementation; it is not the primary confirmatory result. Batch size 32 is a starting value; reduce on memory failure. Cache embeddings and row IDs so geometry ablations do not re-encode text.

For scale: 15,601×768 float32 embeddings are about 48 MB. A full 15,601² float64 distance matrix is about 1.95 GB before algorithmic overhead. Avoid all-pairs shortest paths, store sparse neighbor graphs, use 128 landmarks, and compute local thin SVD in reduced space rather than thousands of 768×768 covariance eigendecompositions. Runtime estimates depend on hardware and must be measured, not promised.

Every run should save model revisions, package versions, seeds, input hashes, split IDs, fit-record IDs, configuration, scores before label joining, evaluation coverage, and wall time. Keep GPU embedding extraction separate from CPU parameter sweeps.

## 7. Minimal experiment matrix and priorities

| Priority | Experiment | Comparators | Decision |
|---|---|---|---|
| Required first | Data/leakage audit | topic/source label distributions | Is cross-topic evaluation feasible? |
| Required first | Raw text clustering | TF–IDF, PCA, random controls | How much signal is already present? |
| Main contribution candidate | Topic projection + shared axis | raw/residual PC1, GMM, random topics | Does a stable factor transfer? |
| Conditional | Diffusion/geodesic features | matched PCA and graph variants | Does graph geometry add value? |
| Conditional | Local transverse analysis | matched pairs and random subspaces | Is the normal-direction claim supported? |
| Conditional | Reviewed intervention trajectories | paraphrase/style/edit controls | Is the factor sensitive to factual change? |
| Strong fake-news follow-up | Evidence-conditioned geometry | distance-only and shuffled evidence | Does relational evidence improve identification? |
| After text findings | Image evidence and joint interactions | same-subset unimodal/concat/shuffle | Is there incremental multimodal information? |
| Before publication | Independent replication | frozen protocol and baselines | What scope of claim survives? |

Begin with Phases 0–2, not a learned Riemannian model. If Phase 2 survives its controls, it supplies the most focused method-development direction. If only evidence helps, reframe the project around evidence-conditioned factuality rather than claiming that text alone has a universal truth direction.

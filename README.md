# Factuality geometry: resumable Kaggle experiments

Investigate whether supported/refuted claims occupy different regions, different local surfaces, or a shared cross-topic coordinate in frozen representations. Fit discovery methods **without truthfulness labels**, then evaluate saved predictions. This repository is an experiment suite, not a validated fake-news detector.

Repository: https://github.com/Ram21275/factuality-geometry

Start with [`notebooks/kaggle_start.ipynb`](notebooks/kaggle_start.ipynb). Full scientific rationale and decision gates are in [`research_protocol.md`](research_protocol.md). Runtime configuration is in `configs/pilot.json` and `configs/full.json`; the older `experiment_config.json` describes the research design and is **not** a CLI config.

## Quick start on Kaggle

Enable Internet for installation/model download and select an available GPU for sentence/CLIP extraction. CPU-only extraction also works, more slowly. Upload the original MOCHEG author release as a Kaggle input; dataset files and checkpoints do not belong in this public repository.

```python
import subprocess, sys
subprocess.run(["git", "clone", "https://github.com/Ram21275/factuality-geometry.git",
                "/kaggle/working/factuality-geometry"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e",
                "/kaggle/working/factuality-geometry[encoders]"], check=True)
```

Then select your claim table and adapt the config after inspecting its columns:

```python
import json
from pathlib import Path
repo = Path("/kaggle/working/factuality-geometry")
config = json.loads((repo / "configs/pilot.json").read_text())
# Example only: replace values with actual column names in the author release.
config["columns"] = {"id": "claim_id", "text": "claim_text", "label": "truthfulness"}
# Numeric ratings require an explicit mapping. NEI is not false.
config["label_map"] = {"supported": 1, "refuted": 0, "nei": None}
Path("/kaggle/working/study-config.json").write_text(json.dumps(config, indent=2))
```

Run a bounded chunk and export a snapshot on completion/pause:

```python
subprocess.run([
    sys.executable, "-m", "fgeometry.cli", "run",
    "--data", "/kaggle/input/YOUR_DATASET/YOUR_CLAIMS.csv",
    "--config", "/kaggle/working/study-config.json",
    "--output", "/kaggle/working/study-runs",
    "--stages", "audit,embed,experiments,geometry,evaluate",
    "--max-seconds", "3300",
    "--snapshot-to", "/kaggle/working/study-checkpoint.tar.gz"
], check=True)
```

Repeat exactly the same command to continue. A soft time budget is checked **between completed units**, not during a single model fit or encoding shard. Begin with the pilot before launching the full multi-seed suite. The full suite can require multiple sessions; no hardware-independent runtime is promised.

## Preparing the dataset

The actual author archive has not yet been supplied for this implementation. The loader supports CSV, TSV, Parquet, JSON record arrays, and JSONL with explicit field mappings. It does not pretend an uninspected raw archive matches a particular schema. Normalize separate claims/evidence/image link tables before running optional branches.

Canonical columns:

| Column | Required? | Meaning |
|---|---|---|
| `id` | Recommended | Unique claim ID; generated row IDs otherwise |
| `claim_text` | Yes | Original claim only; never a ruling or explanation |
| `label` | Evaluation only | Original verdict; ignored by all fit functions |
| `group_id` | Strongly recommended | Near-duplicate claim/event family |
| `source` | Optional | Audited source group for robustness tests |
| `split` | Optional | `train`, `dev`, `test` for official evaluation |
| `evidence_texts` | Optional | JSON list of original evidence passages |
| `image_paths` | Optional | JSON list of local image-evidence paths relative to `--image-root` |

Automatic aliases include `claim`, `Claim`, `claim_id`, `truthfulness`, `Truthfulness`, and `official_split`. Explicit mappings win. The adapter preserves numbers, negation and entities. It unions exact normalized duplicates with supplied event groups. **Near duplicates, reused images, fact-check boilerplate, and verdict badges still require a dataset-specific audit.** Provide `group_id` from that audit before confirmatory evaluation. Exact duplicate grouping alone does not eliminate all leakage.

Do not use `ruling`, `justification`, fact-check headlines, publisher rating fields, or verdict-containing screenshots as evidence. MOCHEG images are evidence images, not necessarily original attached claim images. Some claims may have no images.

For already computed embeddings, supply a `.npy` matrix and a JSON array of matching unique IDs:

```bash
fgeometry run --data claims.csv --embedding-file embeddings.npy \
  --embedding-ids ids.json --config configs/pilot.json --output runs
```

The matrix is normalized and reordered by IDs; anonymous row-order matching is rejected. Record the original encoder/checkpoint provenance yourself. Unverified external embeddings are not accepted for the evidence or intervention projection branches, where a common encoder is necessary.

## Implemented experiments

| Experiment | What it tests | Controls / outputs |
|---|---|---|
| Raw TF–IDF, K-means, GMM, PC1 | Ordinary text structure | Random direction, held-out predictions |
| Topic-subspace projection | Removing between-topic variation | Rank 0/1/2/4, residual PC1 and GMM |
| Shared mixture axis | Recurrent unlabeled mixture directions | BIC/component-mass filter, random topics, explicit no-axis outcome |
| Mixtures of local PCA surfaces | Multiple local geometric surfaces | 1/2/4 surfaces, several tangent ranks, held-out reconstruction |
| Diffusion coordinates | Class-associated graph regions | Several k values; fixed test interpolation and PCA baselines |
| Landmark graph distances | Geodesic stretch and connectivity | Sharded shortest paths; disconnected pairs remain infinite |
| Local transverse displacement | Variation outside estimated local subspaces | Held-out endpoints, distance/length/source-stratified post-hoc comparisons |
| Evidence conditioning | Claim–evidence geometry | Provided vs training-corpus retrieved vs within-topic shuffled evidence; distance-only baseline |
| Multimodal interactions | Incremental image-evidence information | Same paired IDs; text/image/concatenation/similarity/interaction/shuffled-image views |
| Intervention panel | Response to reviewed factual edits | Distance-from-original and frozen shared-axis projections, grouped by claim family |

More local surfaces have more capacity. A lower reconstruction error is **not** evidence of separate true/false manifolds. Graph components are not manifold proofs. Shared axes may encode style/source. Supplied gold evidence is an oracle-retrieval condition, not end-to-end unsupervised fact-checking. Frozen encoders bring pretraining knowledge.

The compact implementation does not yet include automatic web retrieval, NLI support/contradiction reasoning, source timestamp validation, human evidence review, density-matched topology inference, or an external VERITE adapter. These remain explicit protocol tasks, not silently claimed results.

## Stages and robustness runs

```bash
# Longest GPU task; shard results can be reused across pilot/full configs.
fgeometry run --data claims.csv --config configs/pilot.json --output runs --stages audit,embed

# CPU discovery and graph diagnostics.
fgeometry run --data claims.csv --config configs/pilot.json --output runs --stages experiments,geometry

# Evaluation only; fits nothing and uses already saved predictions.
fgeometry run --data claims.csv --config configs/pilot.json --output runs --stages evaluate

# Optional branches, then evaluate saved predictions.
fgeometry run --data claims.csv --config configs/pilot.json --output runs --stages evidence,evaluate
fgeometry run --data claims.csv --config configs/pilot.json --output runs \
  --stages multimodal,evaluate --image-root /kaggle/input/YOUR_DATASET/images
fgeometry run --data claims.csv --config configs/pilot.json --output runs \
  --stages interventions --intervention-panel reviewed_edits.csv
```

Set `split_mode` to `group`, `official`, `topic`, or `source`. Group mode uses fixed label-blind group holdouts across seeds. Topic mode infers topic groups on the full unlabeled corpus **for split design only**, then refits the actual model excluding each held-out topic. Source mode requires trustworthy metadata. Official mode excludes dev records from fitting and requires both train/test without duplicate/event overlap.

Group folds are repeated holdouts rather than disjoint cross-validation folds. Seeds and folds are correlated. Topic counts, ranks and methods are prespecified sweeps; do not select them based on test truthfulness results. To test topic-count/encoder sensitivity, create named config copies before evaluation.

Intervention CSV: `family_id,text,edit_type,verified_level`. Every family needs one `original`; other edit types may include `paraphrase`, `style`, and independently verified factual changes. Levels require evidence review and are only meaningful where severity is actually ordinal. Keep intervention families disjoint from discovery data. No synthetic edits are generated and mislabeled automatically.

## Resuming and keeping results safe

Every completed unit is written to a temporary file, flushed, atomically renamed and recorded with SHA-256 checksums. Partial files are never treated as completed. Corrupted artifacts are recomputed. Run IDs include input hashes, code, config and scientific package versions. Encoder checkpoints resolve to recorded model revisions, which are pinned for partial continuation. A shared embedding cache ignores experimental settings and labels, permitting reuse when only the experiment config changes (keep shard size and encoder fixed).

Resume units:

- Text/image/evidence embedding shards.
- Each split × seed × method × parameter combination.
- Each landmark-distance batch and local-pair batch.
- Each evidence-feature shard.
- Each bootstrap/permutation batch, using deterministic replicate-index seeds.

A single run owner is enforced with OS locks. SIGKILL releases the lock automatically. The interrupted unit is recomputed; completed units survive as long as the output filesystem survives. Iterative model fitting itself is not checkpointed mid-iteration.

**Kaggle working storage is not durable cloud backup.** An unsaved/discarded session can lose the entire working directory. Snapshots are also just local files until you save/download them. Persist notebook outputs or download `study-checkpoint.tar.gz` before ending the session; reattach it as an input next time. This suite cannot guarantee recovery from the destruction of an unsaved disk.

```bash
# Explicit export (destination must be outside the run directory).
fgeometry snapshot --output /kaggle/working/study-runs \
  --destination /kaggle/working/study-checkpoint.tar.gz

# Restore your saved archive into an EMPTY output directory in a new session.
fgeometry restore --archive /kaggle/input/YOUR_SAVED_OUTPUT/study-checkpoint.tar.gz \
  --output /kaggle/working/study-runs

fgeometry status --output /kaggle/working/study-runs
```

Keep the original config and Git commit, plus the versions in `manifest.json`. Changed code/package versions create a new run rather than silently interpreting old checkpoints. Archives exclude datasets, but may contain record IDs, labels in the supplied input identity only indirectly through hashes, and evidence-derived features; treat your saved study outputs according to dataset terms. Restore extracts regular files only, rejects traversal/symlinks, and never overwrites a live directory.

Use `--max-units 1` to exercise interruption recovery. Use `--max-seconds 3300 --snapshot-to ...` for bounded work. Ctrl-C preserves committed units and attempts the requested snapshot. A hard kernel termination cannot run a cleanup handler.

## Evaluation and result files

`report.json` contains ARI, AMI, orientation-free AUC*, per-topic results, coverage, conditional permutation results, group-bootstrap intervals, nuisance association, and explicit failed-axis runs. `paired_comparisons.json` contains paired bootstrap AUC* differences between shared-axis candidates and prespecified baselines. `geometry_report.json` contains matched local/geodesic effects and connectivity coverage. Predictions are saved before any class-label join.

AUC* means `max(AUC, 1-AUC)`: association with labels under unknown axis orientation, **not** a deployable fake/true score. Every null permutation applies the same maximization. Conditional permutations use topic×source strata; if strata have no label variation, their null is uninformative. Holm correction is reported across all scored experiment runs, not advertised as a separately preregistered hypothesis family. Geometry intervals use event-group pair weighting and are exploratory.

Binary evaluation excludes unmapped verdicts, including NEI, while all categories remain in label-free fitting. Numeric labels are never guessed. No classifier is semantically oriented using test labels. Report class imbalance, excluded coverage and dataset-selection limits alongside metrics. Do not rank a parameter sweep and present its best test result as confirmation.

## Local engineering test

No dataset/model downloads are required for this CPU smoke test:

```bash
pip install -e '.[test]'
pytest -q
fgeometry demo --output data/demo
fgeometry run --data data/demo/claims.csv --embedding-file data/demo/embeddings.npy \
  --embedding-ids data/demo/ids.json --config configs/pilot.json --output runs \
  --stages audit,embed,experiments,geometry,evaluate
```

The demo is a synthetic engineering fixture with an intentionally planted coordinate. Its results do not validate the factuality hypothesis. Tests cover end-to-end CPU experiments, label-isolated fitting, exact resume, corruption detection, duplicate-group leakage, deterministic resampling and safe snapshot restore. GPU encoders and the real author archive require integration testing on Kaggle.

Code is MIT licensed. Dataset licenses and third-party model terms remain separate. Cite the original MOCHEG authors and the methods/datasets in the scientific protocol.

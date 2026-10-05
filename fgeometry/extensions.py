"""Evidence/image branches: explicit optional inputs and cached frozen encoders."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import models
from .storage import atomic_json, digest, file_hash


def frozen_encoder(runner, modality="text"):
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    cfg = runner.config
    model = cfg.get("clip_encoder", "sentence-transformers/clip-ViT-B-32") if modality == "clip" else cfg["encoder"]
    pinfile = runner.root / ("clip_revision.json" if modality == "clip" else "encoder_revision.json")
    pin = json.loads(pinfile.read_text()) if pinfile.exists() else None
    revision = pin["resolved_revision"] if pin else cfg.get("clip_revision" if modality == "clip" else "encoder_revision")
    path = snapshot_download(model, revision=revision, local_files_only=cfg.get("offline", False))
    atomic_json(pinfile, {"model": model, "resolved_revision": Path(path).name})
    return SentenceTransformer(path, device=cfg.get("device"))


def encode_cached(runner, items, namespace, modality="text", image_root=None):
    """One shard at a time. Image content hashes enter shard identity."""
    batch = runner.config.get("embedding_shard_size", 256)
    result, encoder = [], None
    for start in range(0, len(items), batch):
        part = items[start:start + batch]
        images = None
        if image_root is not None:
            paths = [Path(image_root) / str(v) for v in part]
            hashes = [file_hash(p) for p in paths]
        else:
            hashes = part
        content = digest(hashes)[:16]
        key = f"{namespace}/{start:08d}-{content}"
        if not runner.has(key):
            if encoder is None:
                encoder = frozen_encoder(runner, modality)
            if image_root is not None:
                from PIL import Image
                images = []
                try:
                    for path in paths:
                        with Image.open(path) as im:
                            images.append(im.convert("RGB"))
                    matrix = encoder.encode(images, batch_size=runner.config.get("batch_size", 32),
                                            normalize_embeddings=True, convert_to_numpy=True)
                finally:
                    for im in images:
                        im.close()
            else:
                matrix = encoder.encode(part, batch_size=runner.config.get("batch_size", 32),
                                        normalize_embeddings=True, convert_to_numpy=True)
            runner.save(key, arrays={"x": np.asarray(matrix, np.float32)}, metadata={"content_hash": content})
        result.append(runner.store.arrays(key)["x"])
    if not result:
        raise ValueError(f"No items for {namespace}")
    return np.concatenate(result)


def simplex_error(claim, evidence):
    # Semantic reconstruction baseline; it cannot establish entailment.
    n = len(evidence)
    solution = minimize(lambda a: np.sum((claim - a @ evidence) ** 2), np.ones(n) / n,
                        method="SLSQP", bounds=[(0., 1.)] * n,
                        constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1}],
                        options={"maxiter": 100, "ftol": 1e-7})
    return float(solution.fun), bool(solution.success)


def evidence(runner):
    if runner.external:
        raise ValueError("Evidence branch requires encoder-generated claim embeddings, not unverified external vectors")
    texts = list(dict.fromkeys(str(t) for r in runner.records for t in r["evidence"] if str(t).strip()))
    if not texts:
        raise ValueError("Supply explicit evidence_texts lists; explanations/rulings are not evidence inputs")
    e = encode_cached(runner, texts, "evidence_embeddings")
    if e.shape[1] != runner.x.shape[1]:
        raise ValueError("Claim/evidence embedding spaces differ")
    index = {t: i for i, t in enumerate(texts)}
    for seed in runner.config.get("seeds", [13]):
        for split, train, test in runner.partitions(seed):
            train_evidence = sorted({index[str(t)] for i in train for t in runner.records[i]["evidence"] if str(t) in index})
            if not train_evidence:
                raise ValueError("No training evidence for a fixed retrieval corpus")
            topics = models.topic_fit(runner.texts, train, runner.config.get("topics", 8), seed)
            rng = np.random.default_rng(seed)
            # Shuffle claim evidence within inferred topics (without labels).
            shuffled = np.arange(len(runner.records))
            for topic in np.unique(topics):
                at = np.flatnonzero(topics == topic)
                shuffled[at] = rng.permutation(at)
            for condition in ("provided", "retrieved", "shuffled"):
                pieces = []
                shard = runner.config.get("feature_shard", 128)
                for start in range(0, len(runner.records), shard):
                    key = f"evidence_features/{split}/seed-{seed}/{condition}/{start:08d}"
                    if not runner.has(key):
                        feature_rows, statuses = [], []
                        for i in range(start, min(start + shard, len(runner.records))):
                            if condition == "retrieved":
                                similarity = e[train_evidence] @ runner.x[i]
                                chosen = np.asarray(train_evidence)[np.argsort(similarity)[-runner.config.get("evidence_top_k", 5):]]
                            else:
                                origin = shuffled[i] if condition == "shuffled" else i
                                chosen = [index[str(t)] for t in runner.records[origin]["evidence"] if str(t) in index][:runner.config.get("evidence_top_k", 5)]
                            if not len(chosen):
                                feature_rows.append(np.zeros(runner.x.shape[1] + 4))
                                statuses.append(False)
                                continue
                            selected = e[chosen]
                            error, solved = simplex_error(runner.x[i], selected)
                            displacement = runner.x[i] - selected.mean(0)
                            feature_rows.append(np.r_[displacement, 1 - np.max(selected @ runner.x[i]),
                                                       error, np.var(selected, axis=0).mean(), len(selected)])
                            statuses.append(solved)
                        runner.save(key, arrays={"features": np.asarray(feature_rows), "solver_ok": statuses},
                                    metadata={"condition": condition, "corpus": "training-linked evidence only for retrieval"})
                    pieces.append(runner.store.arrays(key)["features"])
                features = np.concatenate(pieces)
                mean, scale = features[train].mean(0), features[train].std(0)
                scaled = (features - mean) / np.maximum(scale, 1e-6)
                prefix = f"experiments/{split}/seed-{seed}"
                runner.experiment(f"{prefix}/evidence-{condition}", scaled, train, test, seed, "gmm", topics)
                runner.experiment(f"{prefix}/evidence-{condition}-distance-only", features[:, -4:-3],
                                  train, test, seed, "gmm", topics)


def multimodal(runner, image_root):
    if not image_root:
        raise ValueError("multimodal requires --image-root and explicit image_paths lists")
    paths = list(dict.fromkeys(str(p) for r in runner.records for p in r["images"]))
    if not paths:
        raise ValueError("No image_paths supplied")
    # Fail clearly on broken files rather than silently selecting a label-biased subset.
    images = encode_cached(runner, paths, "image_embeddings", modality="clip", image_root=image_root)
    text = encode_cached(runner, runner.texts, "clip_text_embeddings", modality="clip")
    lookup = {p: i for i, p in enumerate(paths)}
    mask = np.asarray([bool(r["images"]) for r in runner.records])
    pooled = np.zeros_like(text)
    for i in np.flatnonzero(mask):
        pooled[i] = images[[lookup[str(p)] for p in runner.records[i]["images"]]].mean(0)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    image_identity = digest({p: file_hash(Path(image_root) / p) for p in paths})[:12]
    for seed in runner.config.get("seeds", [13]):
        for split, train, test in runner.partitions(seed):
            train, test = train[mask[train]], test[mask[test]]
            if len(train) < 10 or len(test) < 4:
                raise ValueError("Too few paired examples for modality comparison")
            topics = models.topic_fit(runner.texts, train, runner.config.get("topics", 8), seed)
            shuffled = pooled.copy()
            rng = np.random.default_rng(seed)
            for t in np.unique(topics):
                at = np.flatnonzero((topics == t) & mask)
                shuffled[at] = pooled[rng.permutation(at)]
            views = {"mpnet-text": runner.x, "clip-text": text, "image": pooled,
                     "concat": np.c_[text, pooled] / np.sqrt(2),
                     "similarity": np.sum(text * pooled, axis=1)[:, None],
                     "interaction": np.c_[text - pooled, text * pooled],
                     "shuffled-interaction": np.c_[text - shuffled, text * shuffled]}
            for name, x in views.items():
                key = f"experiments/{split}/seed-{seed}/multimodal-{image_identity}-{name}"
                runner.experiment(key, x, train, test, seed, "gmm", topics)


def interventions(runner, panel):
    import pandas as pd
    frame = pd.read_csv(panel)
    required = {"family_id", "text", "edit_type", "verified_level"}
    if not required <= set(frame.columns):
        raise ValueError(f"Panel CSV needs {sorted(required)}; levels require independent evidence review")
    if runner.external:
        raise ValueError("Intervention projection requires encoder-generated embeddings in a verified common space")
    from scipy.stats import spearmanr
    # Content enters the namespace; edits cannot silently reuse old vectors.
    panel_id = file_hash(panel)[:12]
    x = encode_cached(runner, frame["text"].astype(str).tolist(), f"interventions/{panel_id}/embeddings")
    if x.shape[1] != runner.x.shape[1]:
        raise ValueError("Panel and claim embedding dimensions differ")
    axes = {}
    for path in sorted((runner.root / "experiments").rglob("arrays.npz")):
        key = str(path.parent.relative_to(runner.root))
        if not runner.store.valid(key, runner.key_signature(key)):
            continue
        arr = runner.store.arrays(key)
        if "axis" not in arr:
            continue
        rank = runner.store.metadata(key)["parameters"]["rank"]
        combined = np.r_[runner.x, x]
        topics = np.r_[arr["topics"], np.zeros(len(x), dtype=int)]
        residual = models.remove_topic(combined, arr["train"], topics, rank)[len(runner.x):]
        mapped = (residual - arr["map_mean"]) @ arr["map_components"].T
        axes[key] = mapped @ arr["axis"]
    rows = []
    for family, group in frame.groupby("family_id"):
        indices = group.index.to_numpy()
        original = group.index[group["edit_type"] == "original"].tolist()
        if len(original) != 1:
            raise ValueError("Each intervention family needs exactly one original")
        distance = np.linalg.norm(x[indices] - x[original[0]], axis=1)
        levels = pd.to_numeric(group["verified_level"], errors="coerce").to_numpy()
        valid = np.isfinite(levels)
        rho = spearmanr(levels[valid], distance[valid]).statistic if valid.sum() >= 3 else np.nan
        axis_results = {}
        for key, score in axes.items():
            changed = score[indices] - score[original[0]]
            correlation = spearmanr(levels[valid], score[indices][valid]).statistic if valid.sum() >= 3 else np.nan
            axis_results[key] = {"score_change_from_original": changed.tolist(),
                                 "signed_level_correlation": float(correlation) if np.isfinite(correlation) else None}
        rows.append({"family": str(family), "distance_from_original": distance.tolist(),
                     "edit_types": group["edit_type"].tolist(),
                     "graded_distance_spearman": float(rho) if np.isfinite(rho) else None,
                     "frozen_axes": axis_results})
    atomic_json(runner.root / f"intervention_report_{panel_id}.json", {
        "results": rows, "interpretation": "Synthetic/reviewed edit diagnostics, not natural factuality proof",
        "limitation": "Panel must be independently reviewed and disjoint from discovery claim families; axis sign has no semantic orientation"})

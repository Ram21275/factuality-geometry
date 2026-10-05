"""Small, content-addressed units for CPU experiments and GPU extraction."""
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors

from . import models
from .data import normalize, binary_labels
from .evaluation import association, resample_metrics, uncertainty_summary, holm
from .storage import Store, atomic_json, digest, file_hash


def code_fingerprint():
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def environment():
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers", "sentence-transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "packages": packages}


class Runner:
    def __init__(self, data, config, output, budget, embedding_file=None, embedding_ids=None):
        self.config, self.budget = config, budget
        self.output = Path(output)
        self.data = Path(data)
        self.external = Path(embedding_file) if embedding_file else None
        self.external_ids = Path(embedding_ids) if embedding_ids else None
        if self.external and not self.external_ids:
            raise ValueError("Precomputed embeddings require --embedding-ids JSON array for alignment")
        identity = {"data_sha256": file_hash(data), "config": config,
                    "code": code_fingerprint(), "environment": environment(),
                    "external_embeddings": file_hash(self.external) if self.external else None,
                    "external_ids": file_hash(self.external_ids) if self.external_ids else None}
        self.signature = digest(identity)
        self.root = Path(output) / self.signature[:16]
        self.store = Store(self.root)
        atomic_json(self.root / "manifest.json", identity)
        self.records, self.labels, columns = normalize(data, config.get("columns"))
        self.ids = np.asarray([r["id"] for r in self.records])
        self.texts = [r["text"] for r in self.records]
        self.groups = np.asarray([r["group"] for r in self.records])
        self.sources = np.asarray([r["source"] for r in self.records])
        self.columns = columns

    def key_signature(self, key):
        return digest([self.signature, key])

    def has(self, key):
        valid = self.store.valid(key, self.key_signature(key))
        if valid:
            print("reuse", key, flush=True)
        return valid

    def save(self, key, arrays=None, metadata=None):
        self.store.commit(key, self.key_signature(key), arrays, metadata)
        atomic_json(self.root / "progress.json", {"last_completed": key, "time": time.time()})
        print("saved", key, flush=True)
        self.budget.tick()

    def audit(self):
        if self.has("audit"):
            return
        splits = {}
        for r in self.records:
            splits.setdefault(r["group"], set()).add(r["split"])
        leaks = [g for g, s in splits.items() if len(s - {""}) > 1]
        report = {"records": len(self.records), "groups": len(set(self.groups)), "columns": self.columns,
                  "exact_duplicate_count": len(self.records) - len({r["duplicate"] for r in self.records}),
                  "cross_official_split_groups": leaks,
                  "records_with_images": sum(bool(r["images"]) for r in self.records),
                  "records_with_evidence": sum(bool(r["evidence"]) for r in self.records),
                  "near_duplicate_and_image_reuse_audit": "Manual/semantic audit still required before confirmatory claims",
                  "fitting_fields": ["text"], "labels_used_for_fit": False}
        self.save("audit", metadata=report)

    def embed(self):
        cfg = self.config
        shard = cfg.get("embedding_shard_size", 256)
        pieces = []
        # Experiment settings/labels are intentionally absent from this cache identity.
        cache_sig = digest({"ids": self.ids.tolist(), "texts": self.texts,
                            "encoder": cfg["encoder"], "requested_revision": cfg.get("encoder_revision"),
                            "external": file_hash(self.external) if self.external else None,
                            "external_ids": file_hash(self.external_ids) if self.external_ids else None,
                            "code": code_fingerprint(), "environment": environment()})
        shared = Store(self.output / "shared-cache" / cache_sig[:20])
        pinfile = shared.root / "encoder_revision.json"
        if self.external:
            ids = json.loads(self.external_ids.read_text())
            matrix = np.load(self.external, mmap_mode="r", allow_pickle=False)
            if len(set(ids)) != len(ids) or len(matrix) != len(ids):
                raise ValueError("Precomputed IDs must be unique and match matrix rows")
            lookup = {str(v): i for i, v in enumerate(ids)}
            if any(v not in lookup for v in self.ids):
                raise ValueError("Precomputed embeddings missing claim IDs")
            encoder = None
        else:
            matrix = None
            encoder = None
        for start in range(0, len(self.records), shard):
            end = min(start + shard, len(self.records))
            key = f"embeddings/{start:08d}-{end:08d}"
            if not self.has(key):
                if shared.valid(key, cache_sig):
                    arrays = shared.arrays(key)
                    if pinfile.exists():
                        atomic_json(self.root / "encoder_revision.json", json.loads(pinfile.read_text()))
                    self.save(key, arrays=arrays)
                    pieces.append(arrays["x"])
                    continue
                elif matrix is not None:
                    x = np.asarray(matrix[[lookup[v] for v in self.ids[start:end]]], dtype=np.float32)
                else:
                    if encoder is None:
                        from huggingface_hub import snapshot_download
                        from sentence_transformers import SentenceTransformer
                        pin = json.loads(pinfile.read_text()) if pinfile.exists() else None
                        revision = pin["resolved_revision"] if pin else cfg.get("encoder_revision")
                        path = snapshot_download(cfg["encoder"], revision=revision,
                                                 local_files_only=cfg.get("offline", False))
                        resolved = Path(path).name
                        atomic_json(pinfile, {"model": cfg["encoder"], "resolved_revision": resolved})
                        atomic_json(self.root / "encoder_revision.json", {"model": cfg["encoder"], "resolved_revision": resolved})
                        encoder = SentenceTransformer(path, device=cfg.get("device"))
                    x = encoder.encode(self.texts[start:end], batch_size=cfg.get("batch_size", 32),
                                       normalize_embeddings=True, convert_to_numpy=True,
                                       show_progress_bar=True).astype(np.float32)
                if x.ndim != 2 or not np.isfinite(x).all():
                    raise ValueError("Invalid embeddings")
                x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
                arrays = {"x": x, "ids": self.ids[start:end]}
                shared.commit(key, cache_sig, arrays=arrays)
                self.save(key, arrays=arrays)
            piece = self.store.arrays(key)
            if not np.array_equal(piece["ids"], self.ids[start:end]):
                raise ValueError("Cached embedding ID alignment failed")
            pieces.append(piece["x"])
        self.x = np.concatenate(pieces)
        return self.x

    def partitions(self, seed):
        mode = self.config.get("split_mode", "group")
        if mode == "official":
            split = np.asarray([r["split"] for r in self.records])
            train = np.flatnonzero(np.isin(split, ["train", "training"]))
            test = np.flatnonzero(np.isin(split, ["test", "testing"]))
            if not len(train) or not len(test):
                raise ValueError("Official split requires train and test records; dev is not used for fitting")
            if set(self.groups[train]) & set(self.groups[test]):
                raise ValueError("Official splits contain duplicate/event leakage; repair or use group split")
            yield "official", train, test
        elif mode in ("topic", "source"):
            # Topic assignment is a label-free split-design use of the full corpus, disclosed in README.
            group = models.topic_fit(self.texts, np.arange(len(self.records)),
                                     self.config["topics"], 13) if mode == "topic" else self.sources
            if len(np.unique(group)) < 2:
                raise ValueError("Need at least two source/topic groups for holdout")
            for ordinal, g in enumerate(np.unique(group)):
                train, test = np.flatnonzero(group != g), np.flatnonzero(group == g)
                if set(self.groups[train]) & set(self.groups[test]):
                    # Remove all training members related to held-out events.
                    train = train[~np.isin(self.groups[train], self.groups[test])]
                if len(train) >= 10 and len(test) >= 4:
                    yield f"{mode}-{ordinal:03d}", train, test
        elif mode == "group":
            splitter = GroupShuffleSplit(n_splits=self.config.get("folds", 1),
                                         test_size=self.config.get("test_fraction", .25), random_state=13)
            for i, (train, test) in enumerate(splitter.split(self.x, groups=self.groups)):
                yield f"group-{i:02d}", train, test
        else:
            raise ValueError("split_mode must be group, official, topic, or source")

    def experiment(self, key, x, train, test, seed, kind, topics, **parameters):
        if self.has(key):
            return
        dimension = self.config.get("pca_dimension", 32)
        metadata = {"kind": kind, "seed": seed, "parameters": parameters, "labels_used_for_fit": False}
        h, pca = models.pca_coordinates(x, train, dimension, seed)
        extra = {}
        if kind == "tfidf":
            vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
            tr = vectorizer.fit_transform([self.texts[i] for i in train])
            svd = TruncatedSVD(n_components=min(dimension, tr.shape[1] - 1, len(train) - 1), random_state=seed)
            svd.fit(tr)
            h = svd.transform(vectorizer.transform(self.texts))
        if kind == "raw_kmeans" or kind == "tfidf":
            h = x if kind == "raw_kmeans" else h
            model = KMeans(2, n_init=10, random_state=seed).fit(h[train])
            clusters = model.predict(h)
            distance = model.transform(h)
            scores = distance[:, 0] - distance[:, 1]
        elif kind in ("pc1", "random_axis"):
            u = np.eye(h.shape[1])[0] if kind == "pc1" else np.random.default_rng(seed).normal(size=h.shape[1])
            scores = h @ (u / np.linalg.norm(u))
            clusters, _, _ = models.mixture_prediction(scores[:, None], train, seed)
        elif kind == "shared_axis":
            u, details = models.shared_axis(h, train, topics, seed,
                                           self.config.get("minimum_topic_records", 200),
                                           self.config.get("minimum_valid_topics", 3),
                                           self.config.get("gmm_restarts", 5))
            metadata.update(details)
            if u is None:
                self.save(key, metadata=metadata)
                return
            scores = h @ u
            clusters, _, _ = models.mixture_prediction(scores[:, None], train, seed)
            metadata["axis_in_residual_pca_space"] = u.tolist()
            extra = {"map_components": pca.components_, "map_mean": pca.mean_, "axis": u}
        elif kind == "diffusion":
            h, graph, details = models.diffusion_fit(h, train, parameters["k"],
                                                   self.config.get("diffusion_dimensions", 8), seed)
            components, membership = connected_components(graph)
            metadata.update(details)
            metadata.update({"connected_components": int(components),
                             "largest_component_fraction": float(np.bincount(membership).max() / len(train))})
            clusters, scores, _ = models.mixture_prediction(h, train, seed)
        else:
            clusters, scores, _ = models.mixture_prediction(h, train, seed, n_init=self.config.get("gmm_restarts", 5))
        self.save(key, arrays={"ids": self.ids, "scores": scores, "clusters": clusters,
                              "test": test, "train": train, "topics": topics, **extra}, metadata=metadata)

    def experiments(self):
        for seed in self.config.get("seeds", [13]):
            for split, train, test in self.partitions(seed):
                prefix = f"experiments/{split}/seed-{seed}"
                if len(train) < 10:
                    continue
                topics = models.topic_fit(self.texts, train, self.config.get("topics", 8), seed)
                # Same underlying folds across methods and seeds for paired comparisons.
                for kind in self.config.get("baselines", ["tfidf", "raw_kmeans", "gmm", "pc1", "random_axis"]):
                    self.experiment(f"{prefix}/raw-{kind}", self.x, train, test, seed, kind, topics)
                for rank in self.config.get("topic_ranks", [0, 2]):
                    residual = models.remove_topic(self.x, train, topics, rank)
                    for kind in ("gmm", "pc1", "shared_axis"):
                        self.experiment(f"{prefix}/residual-r{rank}-{kind}", residual, train, test,
                                        seed, kind, topics, rank=rank)
                if self.config.get("random_topic_control", True):
                    random_topics = np.random.default_rng(seed).permutation(topics)
                    residual = models.remove_topic(self.x, train, random_topics, 2)
                    self.experiment(f"{prefix}/random-topics-shared_axis", residual, train, test,
                                    seed, "shared_axis", random_topics, rank=2)
                if self.config.get("surfaces", True):
                    self.surfaces(prefix, train, test, seed, topics)
                for k in self.config.get("graph_k", []):
                    self.experiment(f"{prefix}/diffusion-k{k}", self.x, train, test, seed,
                                    "diffusion", topics, k=k)

    def surfaces(self, prefix, train, test, seed, topics):
        h, _ = models.pca_coordinates(self.x, train, self.config.get("pca_dimension", 32), seed)
        for count in self.config.get("surface_counts", [1, 2, 4]):
            for rank in self.config.get("surface_ranks", [2, 4]):
                key = f"{prefix}/surfaces-c{count}-r{rank}"
                if self.has(key):
                    continue
                error, component, fitted = models.local_surface_error(h[train], h, count, rank, seed)
                self.save(key, arrays={"ids": self.ids, "scores": error, "clusters": component,
                                      "test": test, "train": train, "topics": topics},
                          metadata={"kind": "local_surfaces", "surface_count": fitted, "rank": rank,
                                    "test_reconstruction_mse": float(error[test].mean()),
                                    "note": "More surfaces have greater capacity; error alone cannot identify class manifolds"})

    def geometry(self):
        seed = self.config.get("seeds", [13])[0]
        # Geometry diagnostics use the first partition and are exploratory, not all-fold confirmation.
        split, train, test = next(self.partitions(seed))
        topics = models.topic_fit(self.texts, train, self.config.get("topics", 8), seed)
        h, _ = models.pca_coordinates(self.x, train, self.config.get("pca_dimension", 32), seed)
        for k in self.config.get("graph_k", [10, 20, 40]):
            graph = models.neighbor_graph(h[train], k)
            landmarks = np.random.default_rng(seed).choice(len(train), min(self.config.get("landmarks", 128), len(train)), replace=False)
            block = self.config.get("landmark_batch", 8)
            for start in range(0, len(landmarks), block):
                key = f"geometry/{split}/geodesic-k{k}/{start:04d}"
                if self.has(key):
                    continue
                at = landmarks[start:start + block]
                distance = dijkstra(graph, directed=False, indices=at)
                euclidean = np.linalg.norm(h[train[at]][:, None, :] - h[train][None, :, :], axis=2)
                self.save(key, arrays={"graph_distance": distance.astype(np.float32),
                                      "euclidean_distance": euclidean.astype(np.float32),
                                      "landmark_ids": train[at], "target_ids": train,
                                      "landmark_topics": topics[train[at]], "target_topics": topics[train]},
                          metadata={"interpretation": "Training-graph exploratory diagnostic; infinities retained"})
        anchors = np.random.default_rng(seed).choice(test, min(len(test), self.config.get("tangent_anchors", 500)), replace=False)
        for k in self.config.get("tangent_k", [20, 40]):
            for rank in self.config.get("tangent_ranks", [2, 4]):
                for start in range(0, len(anchors), self.config.get("pair_shard", 50)):
                    key = f"geometry/{split}/tangent-k{k}-r{rank}/{start:04d}"
                    if self.has(key):
                        continue
                    result = []
                    for i in anchors[start:start + self.config.get("pair_shard", 50)]:
                        candidates = train[topics[train] == topics[i]]
                        peers = test[(topics[test] == topics[i]) & (test != i)]
                        if len(candidates) <= rank + 1 or not len(peers):
                            continue
                        local_idx = candidates[np.argsort(np.linalg.norm(h[candidates] - h[i], axis=1))[:k]]
                        if rank >= len(local_idx) - 1:
                            continue
                        local = h[local_idx] - h[local_idx].mean(0)
                        _, _, vt = np.linalg.svd(local, full_matrices=False)
                        u = vt[:rank].T
                        # Endpoints are held-out, never used to estimate the tangent.
                        peers = peers[np.argsort(np.linalg.norm(h[peers] - h[i], axis=1))[:10]]
                        for j in peers:
                            delta = h[j] - h[i]
                            total = np.dot(delta, delta)
                            if total <= 1e-12:
                                continue
                            normal = delta - u @ (u.T @ delta)
                            result.append([i, j, total ** .5, np.dot(normal, normal) / total, topics[i]])
                    self.save(key, arrays={"pairs": np.asarray(result, float).reshape(-1, 5)},
                              metadata={"columns": ["anchor_index", "peer_index", "distance", "normal_ratio", "topic"],
                                        "matching_note": "Distance/source/length matching is performed post-hoc; raw ratios are not causal evidence"})

    def evaluate(self):
        y = binary_labels(self.labels, self.config.get("label_map", {}))
        summaries = []
        predictions = sorted((self.root / "experiments").rglob("arrays.npz"))
        for path in predictions:
            prediction_key = str(path.parent.relative_to(self.root))
            if not self.store.valid(prediction_key, self.key_signature(prediction_key)):
                continue
            arr = self.store.arrays(prediction_key)
            test = arr["test"]
            selected = test[y[test] >= 0]
            observed = association(y[selected], arr["scores"][selected], arr["clusters"][selected])
            if observed is None:
                summaries.append({"experiment": prediction_key, "status": "insufficient_binary_labels"})
                continue
            topics = arr["topics"][selected]
            groups = self.groups[selected]
            sources = self.sources[selected]
            strata = np.asarray([f"{t}|{s}" for t, s in zip(topics, sources)])
            blocks = []
            count = self.config.get("uncertainty_replicates", 1000)
            batch = self.config.get("uncertainty_batch", 50)
            for start in range(0, count, batch):
                key = f"evaluation/{prediction_key}/{start:06d}"
                if not self.has(key):
                    values = [resample_metrics(y[selected], arr["scores"][selected], arr["clusters"][selected],
                                              groups, strata, 13, i) for i in range(start, min(start + batch, count))]
                    self.save(key, arrays={"samples": np.asarray(values)})
                blocks.append(self.store.arrays(key)["samples"])
            uncertainty = uncertainty_summary(np.concatenate(blocks), observed) if blocks else {}
            nuisance = {"topic_ami": float(__import__("sklearn.metrics", fromlist=["adjusted_mutual_info_score"]).adjusted_mutual_info_score(topics, arr["clusters"][selected]))}
            per_topic = {str(t): association(y[selected[topics == t]], arr["scores"][selected[topics == t]],
                                             arr["clusters"][selected[topics == t]]) for t in np.unique(topics)}
            summaries.append({"experiment": prediction_key, "status": "ok", **observed, **uncertainty,
                              **nuisance, "binary_coverage": len(selected) / len(test), "per_topic": per_topic,
                              "evaluation_population_hash": digest(self.ids[selected].tolist())})
        valid = [s for s in summaries if "auc_star_permutation_p" in s]
        adjusted = holm([s["auc_star_permutation_p"] for s in valid])
        for s, p in zip(valid, adjusted):
            s["auc_star_p_holm_all_reported_runs"] = p
        atomic_json(self.root / "report.json", {"results": summaries,
                    "semantic_orientation": "unknown; AUC* is label association, not deployable fake/true AUROC",
                    "replicate_note": "Seeds/folds are correlated; do not treat them as independent samples",
                    "no_shared_axis_runs": [str(p.parent.relative_to(self.root)) for p in
                      (self.root / "experiments").rglob("result.json") if json.loads(p.read_text()).get("status") == "no_shared_axis_found"]})
        # Paired method comparisons use the same groups and replicate indices.
        comparisons = []
        by_split = {}
        for s in valid:
            parent = str(Path(s["experiment"]).parent)
            by_split.setdefault(parent, []).append(s)
        for split, results in by_split.items():
            for a in results:
                if "shared_axis" not in a["experiment"]:
                    continue
                for b in results:
                    if b is a or "shared_axis" in b["experiment"]:
                        continue
                    if a["evaluation_population_hash"] != b["evaluation_population_hash"]:
                        # Different modality coverage cannot yield a paired method comparison.
                        continue
                    def samples(result):
                        base = self.root / "evaluation" / result["experiment"]
                        return np.concatenate([self.store.arrays(str(p.parent.relative_to(self.root)))["samples"]
                                               for p in sorted(base.rglob("arrays.npz"))])[:, 0]
                    da, db = samples(a), samples(b)
                    delta = da - db
                    delta = delta[np.isfinite(delta)]
                    comparisons.append({"split": split, "candidate": a["experiment"], "baseline": b["experiment"],
                                        "auc_star_delta": a["auc_star"] - b["auc_star"],
                                        "paired_ci95": np.quantile(delta, [.025, .975]).tolist() if len(delta) else None})
        atomic_json(self.root / "paired_comparisons.json", comparisons)
        if (self.root / "geometry").exists():
            from .diagnostics import summarize
            summarize(self, y)
        print("Report:", self.root / "report.json", flush=True)

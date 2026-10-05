"""Evaluation-only geometric diagnostics, conditional on semantic distance."""
import numpy as np

from .storage import atomic_json


def matched_difference(metric, different, strata, weights=None):
    weights = np.ones(len(metric)) if weights is None else weights
    total, mass, used = 0., 0., 0
    for s in np.unique(strata):
        a = (strata == s) & different
        b = (strata == s) & ~different
        na, nb = weights[a].sum(), weights[b].sum()
        if na <= 0 or nb <= 0:
            continue
        weight = min(na, nb)
        difference = np.average(metric[a], weights=weights[a]) - np.average(metric[b], weights=weights[b])
        total += weight * difference
        mass += weight
        used += int(np.sum(a | b))
    return (float(total / mass) if mass else None), used


def summarize(runner, y):
    base = runner.root / "geometry"
    reports = []
    for folder in sorted(p for p in base.glob("*/*") if p.is_dir()):
        pairs, cover = [], []
        for path in sorted(folder.rglob("arrays.npz")):
            key = str(path.parent.relative_to(runner.root))
            if not runner.store.valid(key, runner.key_signature(key)):
                continue
            arr = runner.store.arrays(key)
            if "pairs" in arr:
                pairs.extend(arr["pairs"].tolist())
            else:
                gd, ed = arr["graph_distance"], arr["euclidean_distance"]
                a, b = np.broadcast_arrays(arr["landmark_ids"][:, None], arr["target_ids"][None, :])
                same_topic = arr["landmark_topics"][:, None] == arr["target_topics"][None, :]
                known = (y[a] >= 0) & (y[b] >= 0) & (ed > 1e-8) & same_topic
                cover.append({"eligible_pairs": int(known.sum()),
                              "finite_pairs": int((known & np.isfinite(gd)).sum())})
                # Deterministic subsample independent of labels; report coverage separately.
                flat = np.flatnonzero(known & np.isfinite(gd))
                if len(flat) > 2000:
                    flat = np.random.default_rng(13).choice(flat, 2000, replace=False)
                for at in flat:
                    i, j = a.ravel()[at], b.ravel()[at]
                    topic = np.broadcast_to(arr["target_topics"][None, :], ed.shape).ravel()[at]
                    pairs.append([i, j, ed.ravel()[at], gd.ravel()[at] / ed.ravel()[at], topic])
        if not pairs:
            continue
        pairs = np.asarray(pairs)
        i, j = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
        mask = (y[i] >= 0) & (y[j] >= 0)
        pairs, i, j = pairs[mask], i[mask], j[mask]
        if not len(pairs):
            continue
        # Pair strata: topic (where available), distance, length discrepancy and source equality.
        bins = np.unique(np.quantile(pairs[:, 2], np.linspace(0, 1, 6)))
        distance_bin = np.digitize(pairs[:, 2], bins[1:-1])
        length_delta = abs(np.asarray([len(runner.texts[a]) - len(runner.texts[b]) for a, b in zip(i, j)]))
        length_bin = np.digitize(length_delta, np.unique(np.quantile(length_delta, [1/3, 2/3])))
        strata = np.asarray([f"{int(t)}:{d}:{l}:{'|'.join(sorted([runner.sources[a], runner.sources[b]]))}"
                             for t, d, l, a, b in zip(pairs[:, 4], distance_bin, length_bin, i, j)])
        different = y[i] != y[j]
        effect, used = matched_difference(pairs[:, 3], different, strata)
        group_names, inverse = np.unique(runner.groups, return_inverse=True)
        g1, g2 = inverse[i], inverse[j]
        blocks = []
        replicates = runner.config.get("uncertainty_replicates", 1000)
        batch = runner.config.get("uncertainty_batch", 50)
        condition = str(folder.relative_to(runner.root))
        for start in range(0, replicates, batch):
            key = f"geometry_evaluation/{condition}/{start:06d}"
            if not runner.has(key):
                effects = []
                for replicate in range(start, min(start + batch, replicates)):
                    rng = np.random.default_rng(np.random.SeedSequence([13, replicate]))
                    counts = np.bincount(rng.integers(0, len(group_names), len(group_names)), minlength=len(group_names))
                    weights = np.where(g1 == g2, counts[g1], counts[g1] * counts[g2])
                    value, _ = matched_difference(pairs[:, 3], different, strata, weights)
                    effects.append(np.nan if value is None else value)
                runner.save(key, arrays={"effects": np.asarray(effects)})
            blocks.append(runner.store.arrays(key)["effects"])
        samples = np.concatenate(blocks) if blocks else np.array([])
        samples = samples[np.isfinite(samples)]
        reports.append({"condition": condition, "pairs": len(pairs), "matched_pairs_used": used,
                        "opposite_minus_same_label_matched_difference": effect,
                        "group_pair_bootstrap_ci95": np.quantile(samples, [.025, .975]).tolist() if len(samples) else None,
                        "connectivity_coverage": cover,
                        "interpretation": "Exploratory distance/length/source-matched association; finite graph pairs only, not manifold proof"})
    atomic_json(runner.root / "geometry_report.json", reports)

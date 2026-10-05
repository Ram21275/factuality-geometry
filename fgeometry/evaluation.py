"""Post-hoc evaluation. Predictions must already exist before this module is called."""
import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, roc_auc_score


def association(y, scores, clusters, include_ami=True):
    if len(y) < 4 or len(np.unique(y)) < 2:
        return None
    auc = roc_auc_score(y, scores)
    result = {"n": len(y), "positive_fraction": float(y.mean()),
            "auc_star": float(max(auc, 1 - auc)),
            "ari": float(adjusted_rand_score(y, clusters))}
    if include_ami:
        result["ami"] = float(adjusted_mutual_info_score(y, clusters))
    return result


def resample_metrics(y, scores, clusters, groups, strata, seed, replicate):
    # Replicate-index seeding permits exact interrupted/resumed continuation.
    rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
    unique = np.unique(groups)
    draw = rng.choice(unique, size=len(unique), replace=True)
    idx = np.concatenate([np.flatnonzero(groups == g) for g in draw])
    metrics = association(y[idx], scores[idx], clusters[idx], include_ami=False)
    shuffled = y.copy()
    exchangeable = 0
    for s in np.unique(strata):
        at = np.flatnonzero(strata == s)
        if len(np.unique(y[at])) > 1:
            exchangeable += len(at)
        shuffled[at] = rng.permutation(y[at])
    null = association(shuffled, scores, clusters, include_ami=False)
    return [np.nan if metrics is None else metrics["auc_star"],
            np.nan if metrics is None else metrics["ari"],
            np.nan if null is None else null["auc_star"],
            np.nan if null is None else null["ari"], exchangeable]


def uncertainty_summary(samples, observed):
    summary = {}
    for j, name in enumerate(("auc_star", "ari")):
        valid = samples[:, j][np.isfinite(samples[:, j])]
        summary[name + "_ci95"] = np.quantile(valid, [.025, .975]).tolist() if len(valid) else None
        null = samples[:, j + 2]
        null = null[np.isfinite(null)]
        summary[name + "_permutation_p"] = float((1 + np.sum(null >= observed[name])) / (1 + len(null)))
    summary["exchangeable_records"] = int(samples[0, 4]) if len(samples) else 0
    summary["null_warning"] = "Conditional null is uninformative if no strata exchange labels"
    return summary


def holm(pvalues):
    values = np.asarray(pvalues, float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    current = 0.
    for i, idx in enumerate(order):
        current = max(current, (len(values) - i) * values[idx])
        adjusted[idx] = min(1., current)
    return adjusted.tolist()

"""Unsupervised estimators; signatures deliberately do not accept class labels."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def dimensions(n, d, requested):
    return max(1, min(requested, n - 1, d))


def topic_fit(texts, train, count, seed):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)
    xt = vec.fit_transform([texts[i] for i in train])
    topic = KMeans(n_clusters=min(count, len(train)), random_state=seed, n_init=10).fit(xt)
    return topic.predict(vec.transform(texts))


def pca_coordinates(x, train, dimension, seed):
    pca = PCA(n_components=dimensions(len(train), x.shape[1], dimension), random_state=seed)
    pca.fit(x[train])
    return pca.transform(x), pca


def remove_topic(x, train, topics, rank, mode="subspace"):
    mean = x[train].mean(0)
    unique = np.unique(topics[train])
    centers = {int(t): x[train[topics[train] == t]].mean(0) for t in unique}
    if mode == "centroid":
        return np.asarray([v - centers.get(int(t), mean) for v, t in zip(x, topics)])
    weighted = np.asarray([
        (centers[int(t)] - mean) * np.sqrt(np.sum(topics[train] == t)) for t in unique
    ])
    _, _, vt = np.linalg.svd(weighted, full_matrices=False)
    u = vt[:min(rank, max(0, len(unique) - 1))].T
    centered = x - mean
    return centered - (centered @ u) @ u.T


def mixture_prediction(h, train, seed, components=2, n_init=5):
    model = GaussianMixture(components, covariance_type="diag", reg_covar=1e-5,
                            n_init=n_init, random_state=seed).fit(h[train])
    return model.predict(h), model.predict_proba(h)[:, -1], model


def shared_axis(h, train, topics, seed, minimum=200, min_valid=3, n_init=5):
    directions, details = [], []
    for t in np.unique(topics[train]):
        ht = h[train[topics[train] == t]]
        if len(ht) < minimum:
            continue
        models = [GaussianMixture(k, covariance_type="diag", reg_covar=1e-5,
                  n_init=n_init, random_state=seed).fit(ht) for k in (1, 2)]
        delta = float(models[0].bic(ht) - models[1].bic(ht))
        difference = models[1].means_[0] - models[1].means_[1]
        valid = delta > 0 and models[1].weights_.min() >= .1 and np.linalg.norm(difference) > 1e-10
        details.append({"topic": int(t), "n": len(ht), "delta_bic": delta, "retained": bool(valid)})
        if valid:
            directions.append(difference / np.linalg.norm(difference))
    if len(directions) < min_valid:
        return None, {"status": "no_shared_axis_found", "topics": details}
    directions = np.asarray(directions)
    vals, vectors = np.linalg.eigh(directions.T @ directions / len(directions))
    u = vectors[:, -1]
    # Deterministic sign without truthfulness labels.
    if u[np.argmax(abs(u))] < 0:
        u = -u
    return u, {"status": "ok", "topics": details, "spectrum": vals.tolist(),
               "alignment": abs(directions @ directions.T).tolist()}


def neighbor_graph(x, k):
    k = min(k, len(x) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    dist, idx = nn.kneighbors(x)
    # Equal-distance duplicates can place self anywhere, not necessarily first.
    chosen = [np.flatnonzero(idx[i] != i)[:k] for i in range(len(x))]
    indices = np.asarray([idx[i, at] for i, at in enumerate(chosen)])
    distances = np.asarray([dist[i, at] for i, at in enumerate(chosen)])
    rows = np.repeat(np.arange(len(x)), k)
    # Retain tiny positive duplicate edges; zero sparse entries mean absent edges.
    graph = sparse.csr_matrix((np.maximum(distances.ravel(), 1e-8),
                              (rows, indices.ravel())), shape=(len(x), len(x)))
    return graph.maximum(graph.T)


def diffusion_fit(x, train, k, dimensions_out, seed):
    graph = neighbor_graph(x[train], k)
    sigma = max(float(np.median(graph.data)), 1e-8)
    affinity = graph.copy()
    affinity.data = np.exp(-affinity.data ** 2 / (2 * sigma ** 2))
    degree = np.asarray(affinity.sum(1)).ravel()
    normalizer = sparse.diags(1 / np.sqrt(np.maximum(degree, 1e-12)))
    normalized = normalizer @ affinity @ normalizer
    ncoords = min(dimensions_out, len(train) - 2)
    vals, vecs = eigsh(normalized, k=ncoords + 1, which="LA",
                      v0=np.random.default_rng(seed).normal(size=len(train)))
    order = np.argsort(vals)[::-1]
    coords = (vecs[:, order[1:]] / np.sqrt(np.maximum(degree[:, None], 1e-12))) * vals[order[1:]]
    # Fixed nearest-training interpolation, never fitting on test observations.
    nn = NearestNeighbors(n_neighbors=min(5, len(train))).fit(x[train])
    distances, indices = nn.kneighbors(x)
    weights = 1 / np.maximum(distances, 1e-8)
    weights /= weights.sum(1, keepdims=True)
    full = (coords[indices] * weights[:, :, None]).sum(1)
    full[train] = coords
    return full, graph, {"bandwidth": sigma, "eigenvalues": vals[order].tolist()}


def local_surface_error(train_x, query_x, count, rank, seed):
    # Models different unlabeled local surfaces, not known true/false surfaces.
    clustering = KMeans(count, n_init=10, random_state=seed).fit(train_x)
    models = []
    for c in range(count):
        part = train_x[clustering.labels_ == c]
        if len(part) < 3:
            continue
        mean = part.mean(0)
        _, _, vt = np.linalg.svd(part - mean, full_matrices=False)
        basis = vt[:min(rank, len(part) - 1)].T
        models.append((mean, basis))
    error = np.column_stack([
        np.sum((query_x - mean - ((query_x - mean) @ basis) @ basis.T) ** 2, axis=1)
        for mean, basis in models
    ])
    return error.min(1), error.argmin(1), len(models)

import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from fgeometry.cli import make_demo
from fgeometry.data import normalize, binary_labels
from fgeometry.evaluation import resample_metrics
from fgeometry.models import shared_axis, neighbor_graph
from fgeometry.pipeline import Runner
from fgeometry.storage import Store, Budget, PauseRun, restore, snapshot


def configuration():
    cfg = json.loads(Path("configs/pilot.json").read_text())
    cfg.update(topics=4, pca_dimension=8, minimum_topic_records=10,
               minimum_valid_topics=2, gmm_restarts=2, surface_counts=[1, 2],
               surface_ranks=[2], graph_k=[8], diffusion_dimensions=3,
               uncertainty_replicates=8, uncertainty_batch=4, embedding_shard_size=40,
               landmarks=8, tangent_anchors=8, tangent_k=[8], tangent_ranks=[2])
    return cfg


def runner(tmp, output="runs", budget=None, config=None):
    data = tmp / "demo"
    if not data.exists():
        make_demo(data, 120)
    return Runner(data / "claims.csv", config or configuration(), tmp / output,
                  budget or Budget(), data / "embeddings.npy", data / "ids.json")


def test_checkpoint_corruption_is_rejected(tmp_path):
    store = Store(tmp_path)
    store.commit("unit", "sig", arrays={"x": np.arange(4)})
    assert store.valid("unit", "sig")
    assert not store.valid("unit", "other")
    (tmp_path / "unit/arrays.npz").write_bytes(b"partial")
    assert not store.valid("unit", "sig")


def test_interrupted_embedding_and_exact_resume(tmp_path):
    interrupted = runner(tmp_path, budget=Budget(max_units=1))
    with pytest.raises(PauseRun):
        interrupted.embed()
    first = interrupted.root / "embeddings/00000000-00000040/arrays.npz"
    before = first.stat().st_mtime_ns
    resumed = runner(tmp_path)
    actual = resumed.embed()
    assert first.stat().st_mtime_ns == before
    uninterrupted = runner(tmp_path, output="clean")
    np.testing.assert_array_equal(actual, uninterrupted.embed())


def test_full_cpu_pipeline_and_resume(tmp_path):
    with threadpool_limits(limits=1):
        run = runner(tmp_path)
        run.audit()
        run.embed()
        run.experiments()
        run.geometry()
        run.evaluate()
        report = (run.root / "report.json").read_text()
        assert json.loads(report)["results"]
        markers = {p: p.stat().st_mtime_ns for p in run.root.rglob("done.json")}
        again = runner(tmp_path)
        again.embed()
        again.experiments()
        again.geometry()
        again.evaluate()
        assert (again.root / "report.json").read_text() == report
        assert all(p.stat().st_mtime_ns == time for p, time in markers.items())


def test_fitting_unchanged_when_labels_flip(tmp_path):
    with threadpool_limits(limits=1):
        first = runner(tmp_path)
        first.embed()
        first.experiments()
        frame = pd.read_csv(tmp_path / "demo/claims.csv")
        frame["label"] = frame["label"].map({"supported": "refuted", "refuted": "supported"})
        frame.to_csv(tmp_path / "demo/claims.csv", index=False)
        second = runner(tmp_path)
        second.embed()
        second.experiments()
        for path in (first.root / "experiments").rglob("arrays.npz"):
            a, b = np.load(path), np.load(second.root / path.relative_to(first.root))
            np.testing.assert_allclose(a["scores"], b["scores"], atol=1e-12)
            np.testing.assert_array_equal(a["train"], b["train"])


def test_group_leakage_and_numeric_label_map(tmp_path):
    frame = pd.DataFrame({"id": list("abcdefghij"), "claim": ["same", "same"] + [str(i) for i in range(8)],
                          "group_id": ["a", "b"] + [str(i) for i in range(8)], "label": [1] * 10})
    path = tmp_path / "claims.csv"
    frame.to_csv(path, index=False)
    records, labels, _ = normalize(path)
    assert records[0]["group"] == records[1]["group"]
    assert np.all(binary_labels(labels, {}) == -1)
    assert np.all(binary_labels(labels, {"1": 1}) == 1)


def test_snapshot_roundtrip_and_unsafe_archive(tmp_path):
    root = tmp_path / "run"
    store = Store(root)
    store.commit("job", "a", arrays={"x": np.arange(4)})
    archive = snapshot(root, tmp_path / "checkpoint.tar.gz")
    restored = tmp_path / "restored"
    restore(archive, restored)
    assert Store(restored).valid("job", "a")
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as f:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        f.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        restore(bad, tmp_path / "unsafe")


def test_resampling_index_is_restart_deterministic():
    y = np.array([0, 1] * 10)
    scores = np.arange(20)
    groups = np.repeat(np.arange(10), 2)
    strata = np.zeros(20, dtype=int)
    a = resample_metrics(y, scores, y, groups, strata, 13, 17)
    b = resample_metrics(y, scores, y, groups, strata, 13, 17)
    np.testing.assert_array_equal(a, b)


def test_no_axis_is_valid_outcome():
    h = np.random.default_rng(0).normal(size=(12, 4))
    u, info = shared_axis(h, np.arange(12), np.zeros(12), 13, minimum=200)
    assert u is None and info["status"] == "no_shared_axis_found"


def test_neighbor_graph_excludes_self_even_with_duplicates():
    graph = neighbor_graph(np.zeros((8, 3)), 3)
    assert np.all(graph.diagonal() == 0)
    assert (graph > 0).sum() >= 8 * 3


def test_embedding_reuse_across_experiment_configs(tmp_path):
    a = runner(tmp_path)
    a.embed()
    config = configuration()
    config["topic_ranks"] = [0, 1, 2]
    b = runner(tmp_path, config=config)
    assert a.root != b.root
    b.embed()
    np.testing.assert_array_equal(a.x, b.x)
    assert len(list((tmp_path / "runs/shared-cache").iterdir())) == 1


def test_optional_evidence_images_and_interventions(tmp_path, monkeypatch):
    from fgeometry import extensions
    class FakeEncoder:
        def encode(self, values, **kwargs):
            import hashlib
            rows = []
            for value in values:
                content = value.tobytes() if hasattr(value, "tobytes") else str(value).encode()
                seed = int.from_bytes(hashlib.sha256(content).digest()[:4], "big")
                x = np.random.default_rng(seed).normal(size=16)
                rows.append(x / np.linalg.norm(x))
            return np.asarray(rows)
    monkeypatch.setattr(extensions, "frozen_encoder", lambda *args, **kwargs: FakeEncoder())
    with threadpool_limits(limits=1):
        run = runner(tmp_path)
        run.embed()
        run.external = None  # Fixture vectors simulate a verified encoder space for this test only.
        for record in run.records:
            record["evidence"] = ["original source paragraph " + record["id"]]
        extensions.evidence(run)
        # Pillow is optional; absence only skips the image part of this integration test.
        pil = pytest.importorskip("PIL.Image")
        image_root = tmp_path / "images"
        image_root.mkdir()
        for i in range(3):
            pil.new("RGB", (8, 8), (i * 50, 100, 20)).save(image_root / f"{i}.png")
        for i, record in enumerate(run.records):
            record["images"] = [f"{i % 3}.png"]
        extensions.multimodal(run, image_root)
        panel = tmp_path / "edits.csv"
        pd.DataFrame({"family_id": ["family"] * 3,
                      "text": ["original claim", "style claim", "changed claim"],
                      "edit_type": ["original", "style", "factual"],
                      "verified_level": [0, 0, 1]}).to_csv(panel, index=False)
        extensions.interventions(run, panel)
        run.evaluate()
        names = [row["experiment"] for row in json.loads((run.root / "report.json").read_text())["results"]]
        assert any("evidence-retrieved" in name for name in names)
        assert any("shuffled-interaction" in name for name in names)


def test_uncertainty_batches_resume_exactly(tmp_path):
    with threadpool_limits(limits=1):
        first = runner(tmp_path)
        first.embed()
        first.experiments()
        first.budget = Budget(max_units=1)
        with pytest.raises(PauseRun):
            first.evaluate()
        shards = list((first.root / "evaluation").rglob("arrays.npz"))
        assert len(shards) == 1
        saved = shards[0].read_bytes()
        resumed = runner(tmp_path)
        resumed.evaluate()
        assert shards[0].read_bytes() == saved
        clean = runner(tmp_path, output="uninterrupted")
        clean.embed()
        clean.experiments()
        clean.evaluate()
        assert (resumed.root / "report.json").read_text() == (clean.root / "report.json").read_text()


def test_topic_and_source_holdouts_do_not_mix_events(tmp_path):
    with threadpool_limits(limits=1):
        for mode in ("topic", "source"):
            cfg = configuration()
            cfg["split_mode"] = mode
            r = runner(tmp_path, config=cfg)
            r.embed()
            partitions = list(r.partitions(13))
            assert partitions
            for _, train, test in partitions:
                assert not set(r.groups[train]) & set(r.groups[test])

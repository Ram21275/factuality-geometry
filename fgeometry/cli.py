"""CLI entry point. Repeating an identical command resumes completed units."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .storage import Budget, PauseRun, atomic_json, run_lock, snapshot, restore


def make_demo(destination, n=240):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(13)
    topics = np.arange(n) % 4
    classes = rng.integers(0, 2, n)
    centers = rng.normal(size=(4, 16)) * 2
    centers[:, -1] = 0
    x = centers[topics] + rng.normal(scale=.2, size=(n, 16))
    x[:, -1] += (classes * 2 - 1) * 1.5
    # Repeated news-like terms allow a realistic TF-IDF topic partition, no label token in text.
    names = ["health medical vaccine hospital", "finance bank stocks trading",
             "politics election minister parliament", "science research telescope space"]
    records = [{"id": f"demo-{i}", "claim_text": f"{names[topics[i]]} reported event number {i}",
                "label": "supported" if classes[i] else "refuted", "group_id": f"event-{i // 2}",
                "source": f"publisher-{i % 3}"} for i in range(n)]
    pd.DataFrame(records).to_csv(destination / "claims.csv", index=False)
    np.save(destination / "embeddings.npy", x.astype(np.float32))
    atomic_json(destination / "ids.json", [r["id"] for r in records])
    print("Synthetic engineering fixture only:", destination)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fgeometry")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run/resume stages with label-isolated fitting")
    run.add_argument("--data", required=True)
    run.add_argument("--config", default="configs/pilot.json")
    run.add_argument("--output", default="runs")
    run.add_argument("--stages", default="audit,embed,experiments,evaluate")
    run.add_argument("--embedding-file")
    run.add_argument("--embedding-ids")
    run.add_argument("--image-root")
    run.add_argument("--intervention-panel")
    run.add_argument("--max-seconds", type=float, default=0, help="soft budget checked after each checkpoint")
    run.add_argument("--max-units", type=int, default=0)
    run.add_argument("--snapshot-to", help="export output tree on graceful stop/completion; outside --output")
    run.add_argument("--threads", type=int, default=2)
    status = sub.add_parser("status")
    status.add_argument("--output", default="runs")
    snap = sub.add_parser("snapshot")
    snap.add_argument("--output", required=True)
    snap.add_argument("--destination", required=True)
    res = sub.add_parser("restore")
    res.add_argument("--archive", required=True)
    res.add_argument("--output", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--output", default="data/demo")
    args = parser.parse_args(argv)
    if args.command == "demo":
        make_demo(args.output)
        return 0
    if args.command == "restore":
        restore(args.archive, args.output)
        print("Restored; repeat the original run command with this output directory")
        return 0
    if args.command == "snapshot":
        with run_lock(args.output):
            print(snapshot(args.output, args.destination))
        return 0
    if args.command == "status":
        for marker in sorted(Path(args.output).rglob("progress.json")):
            print(marker.parent.name, marker.read_text())
        return 0
    from threadpoolctl import threadpool_limits
    from .pipeline import Runner
    from . import extensions
    config = json.loads(Path(args.config).read_text())
    stages = args.stages.split(",")
    allowed = {"audit", "embed", "experiments", "geometry", "evidence", "multimodal", "interventions", "evaluate"}
    if not set(stages) <= allowed:
        parser.error("Unknown stage: " + str(set(stages) - allowed))
    exit_status = 0
    with run_lock(args.output), threadpool_limits(limits=args.threads):
        runner = Runner(args.data, config, args.output, Budget(args.max_seconds, args.max_units),
                        args.embedding_file, args.embedding_ids)
        print("Run:", runner.root, flush=True)
        try:
            for stage in stages:
                print("Stage:", stage, flush=True)
                if stage == "audit":
                    runner.audit()
                elif stage == "embed":
                    runner.embed()
                elif stage == "evaluate":
                    runner.evaluate()
                else:
                    if not hasattr(runner, "x"):
                        runner.embed()
                    if stage == "evidence":
                        extensions.evidence(runner)
                    elif stage == "multimodal":
                        extensions.multimodal(runner, args.image_root)
                    elif stage == "interventions":
                        if not args.intervention_panel:
                            raise ValueError("interventions requires --intervention-panel")
                        extensions.interventions(runner, args.intervention_panel)
                    else:
                        getattr(runner, stage)()
        except (PauseRun, KeyboardInterrupt) as e:
            print("Paused safely; committed units will be reused.", str(e), flush=True)
            exit_status = 0 if isinstance(e, PauseRun) else 130
        finally:
            if args.snapshot_to:
                print("Snapshot:", snapshot(args.output, args.snapshot_to), flush=True)
    return exit_status


if __name__ == "__main__":
    sys.exit(main())

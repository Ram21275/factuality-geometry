"""Content-verified atomic checkpoints. One process owns a run at a time."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, allow_nan=False, default=str).encode())


def atomic_npz(path, **arrays):
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    atomic_bytes(path, buf.getvalue())


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def valid(self, key, signature):
        folder = self.root / key
        marker = folder / "done.json"
        try:
            m = json.loads(marker.read_text())
            return m["signature"] == signature and all(
                (folder / name).is_file() and file_hash(folder / name) == h
                for name, h in m["files"].items()
            )
        except (OSError, ValueError, KeyError):
            return False

    def commit(self, key, signature, arrays=None, metadata=None):
        folder = self.root / key
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        if arrays is not None:
            atomic_npz(folder / "arrays.npz", **arrays)
            files["arrays.npz"] = file_hash(folder / "arrays.npz")
        if metadata is not None:
            atomic_json(folder / "result.json", metadata)
            files["result.json"] = file_hash(folder / "result.json")
        # Completion is published only after all data are durable and hashed.
        atomic_json(folder / "done.json", {"signature": signature, "files": files})

    def arrays(self, key):
        with np.load(self.root / key / "arrays.npz", allow_pickle=False) as f:
            return dict(f)

    def metadata(self, key):
        return json.loads((self.root / key / "result.json").read_text())


@contextlib.contextmanager
def run_lock(root):
    # OS releases advisory locks even after SIGKILL; no stale-lock deletion needed.
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".run.lock", "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("Another process owns this output directory") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def snapshot(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    # Never archive an output inside itself or an earlier snapshot archive.
    if destination == root or root in destination.parents:
        raise ValueError("Snapshot destination must be outside the run directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".pending")
    try:
        with tarfile.open(tmp, "w:gz") as tar:
            for path in sorted(root.rglob("*")):
                if path.is_file() and not path.is_symlink() and not path.name.startswith((".pending-", ".run.lock")):
                    tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()
    return destination


def restore(archive, destination):
    destination = Path(destination).resolve()
    # Restore into an empty directory; never overwrite live work.
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Restore destination must be empty")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not member.isfile() or destination not in target.parents:
                raise ValueError("Unsafe archive member: " + member.name)
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            with tar.extractfile(member) as f:
                atomic_bytes(destination / member.name, f.read())


class Budget:
    def __init__(self, seconds=0, max_units=0):
        self.start = time.monotonic()
        self.seconds, self.max_units, self.units = seconds, max_units, 0

    def tick(self):
        self.units += 1
        if (self.seconds and time.monotonic() - self.start >= self.seconds) or (
            self.max_units and self.units >= self.max_units
        ):
            raise PauseRun("Requested budget reached after a committed unit")


class PauseRun(Exception):
    pass

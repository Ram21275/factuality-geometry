"""Explicit dataset adapter; rating explanations are never model inputs."""
import ast
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .storage import digest


ALIASES = {
    "id": ["id", "claim_id", "Claim ID", "claimId"],
    "text": ["claim_text", "claim", "Claim", "text", "clean_title"],
    "label": ["label", "truthfulness", "Truthfulness", "verdict"],
    "group": ["group_id", "event_id"],
    "source": ["source", "reporting_source"],
    "split": ["split", "official_split"],
    "images": ["image_paths"],
    "evidence": ["evidence_texts"],
}


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if not isinstance(value, list):
            raise ValueError("JSON must be an array of claim records; supply a column mapping")
        return pd.DataFrame(value)
    return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    value = str(value).strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = ast.literal_eval(value)
        if not isinstance(parsed, list):
            raise ValueError("Expected a list")
        return parsed
    return [value]


def normalize(path, mapping=None):
    frame = read_table(path)
    mapping = mapping or {}
    cols = {}
    for field, aliases in ALIASES.items():
        chosen = mapping.get(field)
        if chosen is not None and chosen not in frame:
            raise ValueError(f"Mapped {field} column {chosen!r} not found")
        cols[field] = chosen or next((c for c in aliases if c in frame), None)
    if cols["text"] is None:
        raise ValueError(f"No claim-text column. Available columns: {list(frame.columns)}")
    records, labels = [], []
    seen = set()
    for position, row in frame.iterrows():
        raw = row[cols["text"]]
        if pd.isna(raw) or not str(raw).strip():
            continue
        text = re.sub(r"\s+", " ", str(raw)).strip()
        record_id = str(row[cols["id"]]) if cols["id"] else f"row-{position}"
        if record_id in seen:
            raise ValueError(f"Duplicate record ID {record_id!r}; merge split files with unique IDs")
        seen.add(record_id)
        def scalar(field, default=""):
            value = row[cols[field]] if cols[field] else default
            return default if pd.isna(value) else str(value)
        duplicate = digest(text.casefold())[:20]
        records.append({
            "id": record_id, "text": text, "duplicate": duplicate,
            "group": scalar("group", duplicate), "source": scalar("source", "unknown"),
            "split": scalar("split").lower(),
            "images": as_list(row[cols["images"]]) if cols["images"] else [],
            "evidence": as_list(row[cols["evidence"]]) if cols["evidence"] else [],
        })
        # Labels reside in a different object and are never accepted by fit functions.
        labels.append({"id": record_id, "label": scalar("label")})
    if len(records) < 10:
        raise ValueError("Need at least 10 nonempty claim records")
    # Union explicit event groups and exact normalized duplicates, even when IDs differ.
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for field in ("group", "duplicate"):
        first = {}
        for i, record in enumerate(records):
            value = record[field]
            if value in first:
                parent[find(i)] = find(first[value])
            else:
                first[value] = i
    for i, record in enumerate(records):
        record["group"] = "group-" + records[find(i)]["id"]
    return records, labels, cols


def binary_labels(labels, label_map):
    # Numeric labels require an explicit mapping; NEI remains excluded from binary evaluation.
    values = []
    for row in labels:
        key = str(row["label"]).strip().lower()
        value = label_map.get(key)
        values.append(-1 if value is None else int(value))
    if any(v not in (-1, 0, 1) for v in values):
        raise ValueError("label_map values must be 0, 1, or null")
    return np.asarray(values)

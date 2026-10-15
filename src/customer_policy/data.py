"""Download, validate and split the original Hillstrom customer experiment."""
from __future__ import annotations
import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

ORIGINAL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
MIRROR = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"
MIRROR_MD5 = "a68a81291f53a14f4e29002629803ba3"
ORIGINAL_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
FEATURES = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL = ["zip_code", "channel"]
OUTCOMES = ["visit", "conversion", "spend"]
ARM_NAMES = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
EXPECTED_COLUMNS = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel", "segment", "visit", "conversion", "spend"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire(root: Path) -> Path:
    """Prefer the checksum-verified HTTPS mirror; never accept unpinned bytes."""
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    original = raw / "hillstrom.csv"
    if original.exists():
        if sha256(original) != ORIGINAL_SHA256:
            raise ValueError("Existing original data checksum mismatch; refusing to overwrite it.")
        return original
    cached = raw / "hillstrom_verified.csv.gz"
    if not cached.exists():
        with urllib.request.urlopen(MIRROR, timeout=60) as response:
            content = response.read()
        if hashlib.md5(content).hexdigest() != MIRROR_MD5:
            raise ValueError("HTTPS mirror does not match the maintainer's published checksum.")
        cached.write_bytes(content)
    if hashlib.md5(cached.read_bytes()).hexdigest() != MIRROR_MD5:
        raise ValueError("Cached mirror checksum mismatch.")
    return cached


def load_checked(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        if hashlib.md5(path.read_bytes()).hexdigest() != MIRROR_MD5:
            raise ValueError("Mirror checksum mismatch.")
        frame = pd.read_csv(io.BytesIO(gzip.decompress(path.read_bytes())))
    else:
        if sha256(path) != ORIGINAL_SHA256:
            raise ValueError("Original checksum mismatch.")
        frame = pd.read_csv(path)
    validate(frame)
    return frame


def validate(frame: pd.DataFrame) -> None:
    if list(frame.columns) != EXPECTED_COLUMNS or len(frame) != 64000:
        raise ValueError("Unexpected columns or row count.")
    if frame.isna().any().any():
        raise ValueError("Missing values require an explicit protocol amendment.")
    if set(frame.segment) != set(ARM_NAMES):
        raise ValueError("Unknown treatment arm.")
    for name in ["mens", "womens", "newbie", "visit", "conversion"]:
        if not frame[name].isin([0, 1]).all():
            raise ValueError(f"Nonbinary {name}.")
    if not np.isfinite(frame[NUMERIC + OUTCOMES].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite numeric data.")
    if (frame[["history", "spend"]] < 0).any().any():
        raise ValueError("Unexpected negative spending.")
    if not frame.recency.between(1, 12).all():
        raise ValueError("Unexpected recency range.")
    if (frame.conversion > frame.visit).any() or ((frame.spend > 0) != (frame.conversion == 1)).any():
        raise ValueError("Outcome consistency check failed.")


def split_indices(n: int, seed: int) -> dict[str, np.ndarray]:
    shuffled = np.random.default_rng(seed).permutation(n)
    n_train, n_dev = int(n * 0.6), int(n * 0.2)
    splits = {"train": shuffled[:n_train], "development": shuffled[n_train:n_train + n_dev], "test": shuffled[n_train + n_dev:]}
    joined = np.concatenate(list(splits.values()))
    if len(np.unique(joined)) != n or set(joined) != set(range(n)):
        raise AssertionError("Split coverage/disjointness failure.")
    return splits


def treatment(frame: pd.DataFrame) -> np.ndarray:
    return frame.segment.map(dict(zip(ARM_NAMES, range(3)))).to_numpy(dtype=int)


def audit(frame: pd.DataFrame, source: Path, splits: dict[str, np.ndarray]) -> dict:
    return {
        "provider": "Kevin Hillstrom / MineThatData", "source_file": source.name,
        "sha256": sha256(source), "rows": len(frame), "columns": list(frame),
        "missing_cells": int(frame.isna().sum().sum()),
        "arm_counts": {k: int(v) for k, v in frame.segment.value_counts().items()},
        "features": FEATURES, "post_treatment_excluded": ["segment"] + OUTCOMES,
        "split": {k: {"rows": len(v), "sha256_row_positions": hashlib.sha256(v.astype("<i8").tobytes()).hexdigest()} for k, v in splits.items()},
        "raw_redistribution": "Excluded: no express redistribution license established",
        "population": "Historical randomized experiment; no present-day transport claim",
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    path = acquire(root)
    frame = load_checked(path)
    splits = split_indices(len(frame), 20261015)
    result = audit(frame, path, splits)
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports" / "data_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": len(frame), "missing_cells": result["missing_cells"], "split_sizes": {k: len(v) for k, v in splits.items()}}, indent=2))

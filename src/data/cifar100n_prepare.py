"""Pinned official CIFAR-100N preparation; no JAX, Torch or TF runtime required."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import ssl
import tarfile
import urllib.request

import numpy as np

# Filled from the official release, not a mirror or a generated noise file.
REVISION = "49df7d8a69e355470c77c1c2f2424916325a394b"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
ASSETS = {
    "cifar-100-python.tar.gz": (CIFAR_URL, "md5", "eb9058c3a382ffc7106e4002c42a8d85"),
    "CIFAR-100_human_ordered.npy": ("https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/49df7d8a69e355470c77c1c2f2424916325a394b/data/CIFAR-100_human_ordered.npy", "sha256", "603b16e5f88f77a0db27ffe41b62cf59ba438da2b8bd4a085d502ad72e2b3dbf"),
    "image_order_c100.npy": ("https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/49df7d8a69e355470c77c1c2f2424916325a394b/image_order_c100.npy", "sha256", "423c3118865b4d912ff9082423b992e43fa94ef0f503c4180bd415f915eec018"),
}
FORMAT = "dpco-cifar100n-v1"
TRAIN_COUNT, TEST_COUNT = 50000, 10000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_digest(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def preparation_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def download_asset(url, destination, algorithm, expected, *, progress=None):
    """A bad existing source fails closed; publish only a verified complete file."""
    destination = Path(destination)
    if destination.exists():
        if file_digest(destination, algorithm) != expected:
            raise ValueError(f"Source checksum mismatch: {destination}; preserve/replace the bad source explicitly")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    received = 0
    with urllib.request.urlopen(url, context=context, timeout=60) as response, partial.open("wb") as target:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            target.write(chunk)
            received += len(chunk)
            if progress:
                progress(event="download_progress", file=destination.name, bytes=received)
        target.flush()
        os.fsync(target.fileno())
    if file_digest(partial, algorithm) != expected:
        raise ValueError(f"Downloaded checksum mismatch: {destination.name}; incomplete file retained at {partial}")
    os.replace(partial, destination)


def labels_vector(value, count, *, name):
    value = np.asarray(value)
    if value.shape != (count,) or value.dtype.kind not in "iu" or np.any((value < 0) | (value >= 100)):
        raise ValueError(f"{name} must contain {count} integer labels in [0, 99]")
    return value.astype(np.int32)


def align_official_labels(clean_python, ordered, order):
    """Official mapping: TFDS index i corresponds to Python index order[i]."""
    count = len(clean_python)
    order = np.asarray(order)
    if order.shape != (count,) or order.dtype.kind not in "iu" or not np.array_equal(np.sort(order), np.arange(count)):
        raise ValueError("image_order_c100 must be a complete permutation")
    clean_tf = labels_vector(ordered["clean_label"], count, name="ordered clean_label")
    # The official TFDS npy key is noise_label, unlike noisy_label in the PT file.
    human_tf = labels_vector(ordered["noise_label"], count, name="ordered noise_label")
    if not np.array_equal(clean_tf, np.asarray(clean_python)[order]):
        raise ValueError("Official clean labels do not match CIFAR images in the declared order")
    human = np.empty(count, np.int32)
    human[order] = human_tf
    return human


def publish_arrays(root, train_images, clean, human, test_images, test_labels, provenance):
    """Publish an immutable non-pickled prepared cache, also usable by test fixtures."""
    root = Path(root)
    destination = root / "prepared"
    if destination.exists():
        raise FileExistsError(f"Prepared CIFAR cache already exists: {destination}")
    staging = root / ".building"
    staging.mkdir(parents=True, exist_ok=True)
    clean = labels_vector(clean, len(train_images), name="clean labels")
    human = labels_vector(human, len(train_images), name="human labels")
    test_labels = labels_vector(test_labels, len(test_images), name="test labels")
    for name, images in (("train_images", train_images), ("test_images", test_images)):
        images = np.asarray(images)
        if images.dtype != np.uint8 or images.shape != (len(images), 32, 32, 3) or len(images) == 0:
            raise ValueError("CIFAR images must be nonempty uint8 [N,32,32,3]")
        np.save(staging / (name + ".npy"), images, allow_pickle=False)
    np.savez(staging / "labels.npz", clean_label=clean, noise_label=human, test_label=test_labels)
    manifest = {"format": FORMAT, "provenance": provenance,
        "counts": {"train": len(clean), "validation": len(test_labels)},
        "split_semantics": {"train": "official_training", "validation": "official_test_not_for_model_selection"},
        "human_mismatch_count": int(np.count_nonzero(clean != human)),
        "human_noise_percent": 100 * float(np.mean(clean != human)),
        "files": {name: file_digest(staging / name) for name in ("train_images.npy", "test_images.npy", "labels.npz")}}
    digest = hashlib.sha256(canonical(manifest)).hexdigest()
    atomic_json(staging / "manifest.json", manifest)
    (staging / "manifest.sha256").write_text(digest + "\n")
    atomic_json(staging / "complete.json", {"manifest_sha256": digest})
    os.replace(staging, destination)
    return manifest, digest


def verify_prepared(root):
    directory = Path(root) / "prepared"
    manifest = json.loads((directory / "manifest.json").read_bytes())
    digest = hashlib.sha256(canonical(manifest)).hexdigest()
    if (manifest.get("format") != FORMAT or (directory / "manifest.sha256").read_text().strip() != digest
            or json.loads((directory / "complete.json").read_bytes()) != {"manifest_sha256": digest}):
        raise ValueError("CIFAR completion/manifest identity mismatch")
    if set(manifest["files"]) != {"train_images.npy", "test_images.npy", "labels.npz"}:
        raise ValueError("Unexpected CIFAR cache members")
    for name, expected in manifest["files"].items():
        if file_digest(directory / name) != expected:
            raise ValueError(f"Prepared CIFAR checksum mismatch: {name}")
    return manifest, digest


def prepare(root, *, download=False, progress=None):
    root = Path(root).expanduser().resolve()
    with preparation_lock(root):
        if (root / "prepared").exists():
            return verify_prepared(root)
        raw = root / "raw"
        for name, (url, algorithm, expected) in ASSETS.items():
            path = raw / name
            if not path.exists() and not download:
                raise FileNotFoundError(f"Missing {path}; run scripts/prepare_cifar100n.py --stage download")
            if download:
                download_asset(url, path, algorithm, expected, progress=progress)
            elif file_digest(path, algorithm) != expected:
                raise ValueError(f"Source checksum mismatch: {path}")
        # Exact named members only: do not extract an archive into the filesystem.
        with tarfile.open(raw / "cifar-100-python.tar.gz", "r:gz") as archive:
            records = []
            for split, expected_count in (("train", TRAIN_COUNT), ("test", TEST_COUNT)):
                with archive.extractfile("cifar-100-python/" + split) as handle:
                    record = pickle.load(handle, encoding="latin1")
                pixels = np.asarray(record["data"])
                labels = labels_vector(record["fine_labels"], expected_count, name=split)
                if pixels.shape != (expected_count, 3072) or pixels.dtype != np.uint8:
                    raise ValueError(f"Invalid original CIFAR-100 {split} images")
                if not np.array_equal(np.bincount(labels, minlength=100), np.full(100, expected_count // 100)):
                    raise ValueError("Original CIFAR class counts are not balanced")
                records.append((pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), labels))
        # Object-array input is loaded only after its pinned official SHA-256 check.
        ordered = np.load(raw / "CIFAR-100_human_ordered.npy", allow_pickle=True).item()
        order = np.load(raw / "image_order_c100.npy", allow_pickle=False)
        human = align_official_labels(records[0][1], ordered, order)
        provenance = {"dataset": "cifar100n", "official_revision": REVISION,
            "canonical_order": "cifar100_python", "input_label_order": "official_tfds_ordered_npy",
            "sources": {name: {"url": asset[0], "sha256": file_digest(raw / name)} for name, asset in ASSETS.items()}}
        manifest, digest = publish_arrays(root, records[0][0], records[0][1], human,
                                          records[1][0], records[1][1], provenance)
        if progress:
            progress(event="prepared", manifest_sha256=digest, **manifest["counts"],
                     human_noise_percent=manifest["human_noise_percent"])
        return manifest, digest

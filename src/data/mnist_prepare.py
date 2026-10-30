"""MNIST IDX sources, integrity checks and immutable NumPy cache; no ML runtime."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from src.data.cifar100n_prepare import (atomic_json, canonical, download_asset,
                                      file_digest, preparation_lock)

# Source digests agree with torchvision.datasets.MNIST.resources.
SOURCE = "https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/mnist.py"
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
ASSETS = {name: (BASE_URL + name, "md5", digest) for name, digest in (
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"))}
FORMAT = "dpco-mnist-v1"
FILES = {"train_images.npy", "test_images.npy", "labels.npz"}


def read_idx(path, count, *, images):
    header_size = 16 if images else 8
    size = count * (28 * 28 if images else 1)
    with gzip.open(path, "rb") as handle:
        header = handle.read(header_size)
        expected = (2051, count, 28, 28) if images else (2049, count)
        if len(header) != header_size or struct.unpack(">" + "I" * len(expected), header) != expected:
            raise ValueError(f"Invalid MNIST IDX header: {path}")
        data = handle.read(size + 1)
    if len(data) != size:
        raise ValueError(f"Invalid MNIST IDX payload length: {path}")
    values = np.frombuffer(data, np.uint8)
    if not images and np.any(values > 9):
        raise ValueError("MNIST labels must be in [0,9]")
    return values.reshape(count, 28, 28) if images else values.astype(np.int32)


def publish_arrays(root, train_images, train_labels, test_images, test_labels, provenance):
    """Publish clean source arrays only; corruption is a fingerprinted reader setting."""
    root = Path(root)
    destination = root / "prepared"
    if destination.exists():
        raise FileExistsError(f"Prepared MNIST cache already exists: {destination}")
    arrays = {}
    for name, pixels, labels in (("train", train_images, train_labels), ("test", test_images, test_labels)):
        pixels, labels = np.asarray(pixels), np.asarray(labels)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[1:] != (28, 28) or len(pixels) == 0:
            raise ValueError("MNIST images must be nonempty uint8 [N,28,28]")
        if labels.shape != (len(pixels),) or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels > 9)):
            raise ValueError("MNIST label shape/range is invalid")
        arrays[name] = (pixels, labels.astype(np.int32))
    staging = root / ".building"
    staging.mkdir(parents=True, exist_ok=True)
    for name, (pixels, _) in arrays.items():
        np.save(staging / (name + "_images.npy"), pixels, allow_pickle=False)
    np.savez(staging / "labels.npz", train_label=arrays["train"][1], test_label=arrays["test"][1])
    manifest = {"format": FORMAT, "provenance": provenance,
        "counts": {"train": len(train_labels), "validation": len(test_labels)},
        "files": {name: file_digest(staging / name) for name in sorted(FILES)}}
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
        raise ValueError("MNIST completion/manifest identity mismatch")
    if set(manifest["files"]) != FILES:
        raise ValueError("Unexpected MNIST cache members")
    for name, expected in manifest["files"].items():
        if file_digest(directory / name) != expected:
            raise ValueError(f"Prepared MNIST checksum mismatch: {name}")
    return manifest, digest


def prepare(root, *, download=False, progress=None):
    root = Path(root).expanduser().resolve()
    with preparation_lock(root):
        if (root / "prepared").exists():
            return verify_prepared(root)
        for name, (url, algorithm, expected) in ASSETS.items():
            path = root / "raw" / name
            if download:
                download_asset(url, path, algorithm, expected, progress=progress)
            elif not path.exists():
                raise FileNotFoundError(f"Missing {path}; run scripts/prepare_mnist.py --stage download")
            elif file_digest(path, algorithm) != expected:
                raise ValueError(f"MNIST source checksum mismatch: {name}")
        raw = root / "raw"
        train_images = read_idx(raw / "train-images-idx3-ubyte.gz", 60000, images=True)
        train_labels = read_idx(raw / "train-labels-idx1-ubyte.gz", 60000, images=False)
        test_images = read_idx(raw / "t10k-images-idx3-ubyte.gz", 10000, images=True)
        test_labels = read_idx(raw / "t10k-labels-idx1-ubyte.gz", 10000, images=False)
        provenance = {"dataset": "mnist", "canonical_order": "original_idx", "digest_source": SOURCE,
            "sources": {name: {"url": item[0], "sha256": file_digest(raw / name)} for name, item in ASSETS.items()}}
        result = publish_arrays(root, train_images, train_labels, test_images, test_labels, provenance)
        if progress:
            progress(event="prepared", dataset="mnist", manifest_sha256=result[1], train=60000, test=10000)
        return result

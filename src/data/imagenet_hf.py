"""Pinned HF ImageNet streaming -> local image-byte SQLite -> exact epoch shuffles.

Only selected original image bytes are retained; no Arrow copy or decoded image
dump is made. Full ImageNet-100 preparation can still scan the much larger 1K
source over the network. Tokens are reused by HF libraries, never serialized.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

# Re-export the established interface for existing callers.
from src.data.images import PreparedImages, ImageBatchSource, evaluation_batches, transform


def label_mapping(config, base_dir):
    if config.name == "imagenet1k":
        return {i: i for i in range(1000)}, hashlib.sha256(b"identity_ilsvrc2012_1000").hexdigest()
    path = Path(config.class_manifest)
    if not path.is_absolute():
        path = Path(base_dir) / path
    content = path.read_bytes()
    rows = [line.split() for line in content.decode().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) != 100 or any(len(row) != 2 for row in rows):
        raise ValueError("ImageNet-100 manifest must have 100 synset/index rows")
    indices = [int(row[1]) for row in rows]
    if len(set(indices)) != 100 or any(not 0 <= i < 1000 for i in indices):
        raise ValueError("Invalid or duplicate original ImageNet label indices")
    return {source: label for label, source in enumerate(indices)}, hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class DataPlan:
    repo_id: str
    revision: str
    mapping: dict
    manifest_sha256: str
    train_limit: int | None = None
    validation_limit: int | None = None

    def __post_init__(self):
        import re
        if not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("Pin the HF dataset to a full commit SHA")
        if set(self.mapping.values()) != set(range(len(self.mapping))):
            raise ValueError("Selected labels must map to consecutive classifier indices")
        for limit in (self.train_limit, self.validation_limit):
            if limit is not None and limit <= 0:
                raise ValueError("Smoke sample limits must be positive")

    @property
    def identity(self):
        return {"repo_id": self.repo_id, "revision": self.revision,
            "mapping": {str(k): v for k, v in self.mapping.items()},
            "manifest_sha256": self.manifest_sha256,
            "train_limit": self.train_limit, "validation_limit": self.validation_limit,
            "format_version": 1, "source_resolution": "original_source_bytes"}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()


def hf_stream(plan, split):
    from datasets import Image as HFImage, load_dataset
    from huggingface_hub import get_token
    kwargs = {}
    if len(plan.mapping) < 1000:
        kwargs["filters"] = [("label", "in", sorted(plan.mapping))]
    # Explicitly use the saved account token or HF_TOKEN. No login/mutation is
    # attempted; accepting gated-dataset terms remains a browser action.
    token = get_token()
    stream = load_dataset(plan.repo_id, revision=plan.revision, split=split,
        streaming=True, token=token, **kwargs)
    names = getattr(stream.features["label"], "names", None)
    if names is None or len(names) != 1000 or not names[0].startswith("tench"):
        raise ValueError("Expected original 1000-class ILSVRC label order")
    return stream.cast_column("image", HFImage(decode=False)), list(names)


def prepare(plan, cache_root, *, stream_factory=hf_stream, progress=None):
    """Resume partial preparation; completed caches require no HF login/network.

    Replaying an interrupted preparation may re-read its source prefix. Training
    resumes from the completed local cache without network iterator replay.
    """
    from filelock import FileLock
    root = Path(cache_root) / plan.fingerprint[:24]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "images.sqlite"
    with FileLock(str(root / ".prepare.lock")):
        connection = sqlite3.connect(path)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS samples (split TEXT, id INTEGER, label INTEGER, image BLOB, source_index INTEGER, PRIMARY KEY(split,id))")
            def get(key, default=None):
                value = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
                return default if value is None else json.loads(value[0])
            def put(key, value):
                connection.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))
            identity = get("identity")
            if identity is not None and identity != plan.identity:
                raise ValueError("Prepared data identity mismatch")
            put("identity", plan.identity)
            connection.commit()
            if get("complete", False):
                return PreparedImages(path, plan.fingerprint)
            limited = plan.train_limit is not None and plan.validation_limit is not None
            reserve = 3_000_000_000
            expected = ((plan.train_limit + plan.validation_limit) * 600_000 if limited
                        else (19_000_000_000 if len(plan.mapping) == 100 else 170_000_000_000))
            if shutil.disk_usage(root).free < reserve + max(0, expected - path.stat().st_size):
                raise OSError("Insufficient free space for selected ImageNet bytes plus a 3 GB checkpoint reserve; change data.prepared_dir")
            for split, limit in (("train", plan.train_limit), ("validation", plan.validation_limit)):
                count = connection.execute("SELECT COUNT(*) FROM samples WHERE split=?", (split,)).fetchone()[0]
                if get(split + "_complete", False):
                    continue
                if limit is not None and count == limit:
                    put(split + "_complete", True)
                    connection.commit()
                    continue
                try:
                    stream, names = stream_factory(plan, split)
                    put("original_label_names", names)
                    cursor = get(split + "_cursor", 0)
                    for index, record in enumerate(stream):
                        if index < cursor:
                            continue
                        source_label = int(record["label"])
                        if source_label not in plan.mapping:
                            continue
                        image = record["image"]
                        blob = image.get("bytes") if isinstance(image, dict) else None
                        if blob is None:
                            raise ValueError("Expected embedded original image bytes, not external paths")
                        connection.execute("INSERT INTO samples VALUES (?,?,?,?,?)",
                            (split, count, plan.mapping[source_label], blob, index))
                        count += 1
                        put(split + "_cursor", index + 1)
                        if count % 128 == 0 or (limit is not None and count >= limit):
                            connection.commit()
                            if progress:
                                progress(split=split, selected_samples=count)
                        if limit is not None and count >= limit:
                            break
                    if limit is not None and count != limit:
                        raise ValueError(f"Expected {limit} selected {split} samples, received {count}")
                    if limit is None:
                        expected_count = ({"train": 126689, "validation": 5000} if len(plan.mapping) == 100
                                          else {"train": 1281167, "validation": 50000})[split]
                        if count != expected_count:
                            raise ValueError(f"Unexpected full {split} count {count}; expected {expected_count}")
                        classes = connection.execute("SELECT COUNT(DISTINCT label) FROM samples WHERE split=?", (split,)).fetchone()[0]
                        if classes != len(plan.mapping):
                            raise ValueError("Full preparation is missing selected classes")
                    put(split + "_complete", True)
                    connection.commit()
                except Exception as error:
                    connection.rollback()
                    raise RuntimeError(f"ImageNet preparation failed ({type(error).__name__}). Check disk/network and `hf auth whoami`; approve access at https://huggingface.co/datasets/{plan.repo_id}. No synthetic fallback is used.") from error
            put("complete", True)
            connection.commit()
        finally:
            connection.close()
    return PreparedImages(path, plan.fingerprint)

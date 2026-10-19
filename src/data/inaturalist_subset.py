"""Reproducible official iNat2021 FULL subsets; no training or JAX imports."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import heapq
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tarfile
import urllib.request


ALGORITHM = "inat2021-proportional-sha256-v1"
BASE = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021"


@dataclass(frozen=True)
class Source:
    annotation_url: str
    annotation_md5: str
    image_url: str
    image_md5: str
    count: int


SOURCES = {
    "train": Source(f"{BASE}/train.json.tar.gz", "38a7bb733f7a09214d44293460ec0021",
                    f"{BASE}/train.tar.gz", "e0526d53c7f7b2e3167b2b43bb2690ed", 2686843),
    "validation": Source(f"{BASE}/val.json.tar.gz", "4d761e0f6a86cc63e8f7afc91f6a8f0b",
                         f"{BASE}/val.tar.gz", "f6f6e0e242e3d4c9569ba56400938afc", 100000),
}


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode("utf-8")


def atomic_bytes(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StorageBudget:
    def __init__(self, root, max_bytes=15_000_000_000, reserve_bytes=10_000_000_000):
        self.root = Path(root)
        self.max_bytes, self.reserve_bytes = max_bytes, reserve_bytes
        if max_bytes <= 0 or reserve_bytes < 0:
            raise ValueError("Storage ceiling must be positive and reserve nonnegative")

    def check(self, extra=0):
        used = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())
        if used + extra > self.max_bytes:
            raise OSError("Preparation exceeds its storage ceiling; partial data retained, no resampling")
        if shutil.disk_usage(self.root).free < self.reserve_bytes + extra:
            raise OSError("Insufficient free space above the configured reserve; partial data retained")


@contextmanager
def preparation_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another preparation process owns this output directory") from error
        yield


def safe_archive_path(name):
    # Normalize only the harmless leading ./ prefix, never parent traversal.
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    if str(path) != name or not name.endswith(".jpg"):
        raise ValueError(f"Unexpected image path: {name!r}")
    return name


def proportional_quotas(counts, total):
    """Hamilton quotas using integers, category ID breaks exact remainder ties."""
    population = sum(counts.values())
    if not counts or any(n <= 0 for n in counts.values()) or not 0 < total <= population:
        raise ValueError("Invalid source counts or subset size")
    quotas = {category: total * n // population for category, n in counts.items()}
    remaining = total - sum(quotas.values())
    order = sorted(counts, key=lambda c: (-(total * counts[c] % population), c))
    for category in order[:remaining]:
        quotas[category] += 1
    if min(quotas.values()) == 0:
        raise ValueError("Requested subset would omit a category; increase the sample count")
    return quotas


def image_rank(seed, split, category, image_id, path):
    key = [ALGORITHM, seed, split, category, image_id, path]
    return int.from_bytes(hashlib.sha256(canonical(key)).digest(), "big")


def select_rows(rows, quotas, seed, split):
    """Bounded heaps retain smallest hash ranks; row order cannot change selection."""
    heaps = {c: [] for c in quotas}
    for image_id, category, path in rows:
        path = safe_archive_path(path)
        rank = image_rank(seed, split, category, image_id, path)
        # IDs are unique in the validated annotation table. Negated integer keys
        # make heap[0] the largest retained (rank, ID); paths enter the hash.
        entry = (-rank, -image_id, image_id, path)
        heap = heaps[category]
        if len(heap) < quotas[category]:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    chosen = []
    for category in sorted(heaps):
        if len(heaps[category]) != quotas[category]:
            raise ValueError("Annotation rows do not satisfy the class quota")
        chosen.extend((image_id, category, path) for _, _, image_id, path in heaps[category])
    return sorted(chosen, key=lambda row: (row[1], row[0], row[2]))


def open_url(url):
    return urllib.request.urlopen(url, timeout=120)


def download_annotations(source, destination, budget, opener=open_url):
    """Only the small annotation archive is retained; never an image archive."""
    if destination.exists():
        if file_hash(destination, "md5") != source.annotation_md5:
            raise ValueError(f"Annotation checksum mismatch: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    # A failed annotation transfer restarts; image preparation has its own resume.
    with opener(source.annotation_url) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            budget.check(len(chunk))
            output.write(chunk)
    if file_hash(temporary, "md5") != source.annotation_md5:
        raise ValueError("Downloaded annotations failed their official MD5 check")
    temporary.replace(destination)


def annotation_objects(stream):
    """Stream individual top-level array records, never materialize the full JSON."""
    try:
        import ijson
    except ImportError as error:
        raise RuntimeError("Install the data extra: python -m pip install -e '.[data]'") from error
    builder, active = None, None
    for prefix, event, value in ijson.parse(stream):
        if builder is None and event == "start_map" and prefix in (
                "images.item", "annotations.item", "categories.item"):
            builder, active = ijson.ObjectBuilder(), prefix
        if builder is not None:
            builder.event(event, value)
            if prefix == active and event == "end_map":
                yield active.split(".")[0], builder.value
                builder, active = None, None


def read_annotation_index(archive_path, connection, source, split, category_count, budget):
    connection.executescript("DROP TABLE IF EXISTS source_images; DROP TABLE IF EXISTS source_labels;"
        "CREATE TABLE source_images(id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL);"
        "CREATE TABLE source_labels(id INTEGER PRIMARY KEY, category INTEGER NOT NULL);")
    categories, record_count, json_count = {}, 0, 0
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            json_count += 1
            if json_count != 1:
                raise ValueError("Expected exactly one JSON file in annotation archive")
            with archive.extractfile(member) as stream:
                for kind, item in annotation_objects(stream):
                    if kind == "images":
                        path = safe_archive_path(item["file_name"])
                        prefix = "train/" if split == "train" else "val/"
                        if not path.startswith(prefix):
                            raise ValueError("Annotation path is not in the expected official split")
                        connection.execute("INSERT INTO source_images VALUES (?,?)", (int(item["id"]), path))
                    elif kind == "annotations":
                        connection.execute("INSERT INTO source_labels VALUES (?,?)",
                                           (int(item["image_id"]), int(item["category_id"])))
                    else:
                        category = int(item["id"])
                        if category in categories:
                            raise ValueError("Duplicate source category")
                        categories[category] = str(item["name"])
                    record_count += 1
                    if record_count % 4096 == 0:
                        budget.check(8 * 1024 * 1024)
                        connection.commit()
    connection.commit()
    if json_count != 1 or len(categories) != category_count:
        raise ValueError("Unexpected category count or missing annotation JSON")
    images = connection.execute("SELECT COUNT(*) FROM source_images").fetchone()[0]
    labels = connection.execute("SELECT COUNT(*) FROM source_labels").fetchone()[0]
    joined = connection.execute("SELECT COUNT(*) FROM source_images JOIN source_labels USING(id)").fetchone()[0]
    counts = dict(connection.execute("SELECT category,COUNT(*) FROM source_labels GROUP BY category"))
    if (images, labels, joined) != (source.count,) * 3 or set(counts) != set(categories):
        raise ValueError("Official source counts, labels or categories do not match")
    return categories, counts


def build_manifest(root, *, seed=42, train_count=140000, validation_count=20000,
                   budget=None, sources=SOURCES, category_count=10000, opener=open_url, progress=None):
    root = Path(root)
    if seed < 0:
        raise ValueError("Selection seed must be nonnegative")
    budget = budget or StorageBudget(root)
    identity = {"algorithm": ALGORITHM, "selection_seed": seed,
        "subset_counts": {"train": train_count, "validation": validation_count},
        "sources": {split: asdict(source) for split, source in sources.items()},
        "num_classes": category_count, "image_encoding": "official_original_bytes"}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest, digest = load_manifest(root, repair=True)
        if manifest["identity"] != identity:
            raise ValueError("Output already belongs to a different selection; use another directory")
        return manifest, digest
    metadata = root / "metadata"
    metadata.mkdir(exist_ok=True)
    marker = root / "selection.json"
    if marker.exists() and json.loads(marker.read_bytes()) != identity:
        raise ValueError("Partial output belongs to a different selection")
    atomic_bytes(marker, canonical(identity))
    rows, distributions, common_categories = [], {}, None
    index_path = metadata / "annotation_index.sqlite"
    connection = sqlite3.connect(index_path)
    try:
        # Keep metadata transactions bounded and prevent memory-backed journals.
        connection.execute("PRAGMA journal_mode=DELETE")
        for split in ("train", "validation"):
            source = sources[split]
            annotation_path = metadata / (split + ".json.tar.gz")
            if progress:
                progress(event="annotation_start", split=split)
            download_annotations(source, annotation_path, budget, opener)
            categories, counts = read_annotation_index(annotation_path, connection, source, split,
                                                       category_count, budget)
            if common_categories is None:
                common_categories = categories
            elif common_categories != categories:
                raise ValueError("Train/validation category IDs or names differ")
            mapping = {c: label for label, c in enumerate(sorted(categories))}
            quotas = proportional_quotas(counts, identity["subset_counts"][split])
            cursor = connection.execute("SELECT source_images.id,category,path FROM source_images "
                                        "JOIN source_labels USING(id)")
            selected = select_rows(cursor, quotas, seed, split)
            for index, (image_id, category, path) in enumerate(selected):
                rows.append({"split": split, "id": index, "image_id": image_id,
                    "category_id": category, "label": mapping[category], "path": path})
            distributions[split] = [{"category_id": c, "source_count": counts[c], "selected_count": quotas[c]}
                                    for c in sorted(categories)]
            if progress:
                progress(event="selection_complete", split=split, selected_samples=len(selected))
    finally:
        connection.close()
    manifest = {"identity": identity, "categories": [{"category_id": c, "label": i,
        "name": common_categories[c]} for i, c in enumerate(sorted(common_categories))],
        "distributions": distributions, "samples": rows}
    content = canonical(manifest)
    digest = hashlib.sha256(content).hexdigest()
    budget.check(len(content) + 4096)
    atomic_bytes(manifest_path, content)
    atomic_bytes(root / "manifest.sha256", (digest + "\n").encode("ascii"))
    return manifest, digest


def load_manifest(root, *, repair=False):
    root = Path(root)
    content = (root / "manifest.json").read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if (root / "manifest.sha256").exists():
        if (root / "manifest.sha256").read_text().strip() != digest:
            raise ValueError("Manifest SHA-256 mismatch")
    elif repair:
        # Recover publication interrupted after the complete manifest was renamed.
        atomic_bytes(root / "manifest.sha256", (digest + "\n").encode("ascii"))
    else:
        raise ValueError("Manifest checksum file is missing; rerun the plan stage to recover publication")
    manifest = json.loads(content)
    if canonical(manifest) != content or manifest["identity"]["algorithm"] != ALGORITHM:
        raise ValueError("Noncanonical or unsupported manifest")
    return manifest, digest


class HashingReader:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.md5()

    def read(self, size=-1):
        chunk = self.stream.read(size)
        self.digest.update(chunk)
        return chunk


def stored_rows(connection, manifest, *, validate_bytes):
    expected = {(row["split"], row["id"]): row for row in manifest["samples"]}
    present, payload = set(), 0
    for split, index, image_id, label, path, size, digest, blob in connection.execute(
            "SELECT split,id,image_id,label,path,length(image),sha256,image FROM samples"):
        row = expected.get((split, index))
        if row is None or (image_id, label, path) != (row["image_id"], row["label"], row["path"]):
            raise ValueError("Stored image identity differs from the manifest")
        if size <= 0 or (validate_bytes and hashlib.sha256(blob).hexdigest() != digest):
            raise ValueError("Stored image content checksum mismatch")
        present.add((split, index))
        payload += size
    return present, payload


def prepare_images(root, *, budget=None, opener=open_url, progress=None):
    root = Path(root)
    manifest, fingerprint = load_manifest(root)
    budget = budget or StorageBudget(root)
    connection = sqlite3.connect(root / "images.sqlite")
    try:
        connection.executescript("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS samples(split TEXT,id INTEGER,image_id INTEGER,label INTEGER,"
            "path TEXT,image BLOB,sha256 TEXT,PRIMARY KEY(split,id),UNIQUE(path));")
        def get(key):
            row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return None if row is None else json.loads(row[0])

        def put(key, value):
            connection.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))

        if get("manifest_sha256") not in (None, fingerprint):
            raise ValueError("Image database belongs to another manifest")
        put("manifest_sha256", fingerprint)
        connection.commit()
        present, payload = stored_rows(connection, manifest, validate_bytes=True)
        selected = {row["path"]: row for row in manifest["samples"]}
        for split in ("train", "validation"):
            expected = sum(row["split"] == split for row in manifest["samples"])
            if get(split + "_verified"):
                if sum(s == split for s, _ in present) != expected:
                    raise ValueError("Verified split is missing samples")
                continue
            source = manifest["identity"]["sources"][split]
            budget.check(8 * 1024 * 1024)
            count, scanned = sum(s == split for s, _ in present), 0
            if progress:
                progress(event="image_scan_start", split=split, selected_samples=count)
            with opener(source["image_url"]) as response:
                reader = HashingReader(response)
                with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                    for member in archive:
                        scanned += 1
                        name = member.name
                        while name.startswith("./"):
                            name = name[2:]
                        row = selected.get(name)
                        if row is None:
                            if progress and scanned % 10000 == 0:
                                progress(event="image_scan", split=split, scanned_entries=scanned,
                                         selected_samples=count)
                            continue
                        if row["split"] != split or not member.isfile() or member.size <= 0:
                            raise ValueError("Selected image has an invalid archive entry")
                        if member.size > 64 * 1024 * 1024:
                            raise ValueError("Unexpectedly large selected image")
                        with archive.extractfile(member) as stream:
                            blob = stream.read()
                        digest = hashlib.sha256(blob).hexdigest()
                        key = (split, row["id"])
                        if key in present:
                            saved = connection.execute("SELECT sha256 FROM samples WHERE split=? AND id=?", key).fetchone()[0]
                            if saved != digest:
                                raise ValueError("Source image changed during preparation replay")
                        else:
                            budget.check(2 * len(blob) + 1024 * 1024)
                            connection.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?)", (split,
                                row["id"], row["image_id"], row["label"], name, blob, digest))
                            present.add(key)
                            payload += len(blob)
                            count += 1
                            if count % 64 == 0:
                                connection.commit()
                                if progress:
                                    progress(event="image_prepare", split=split,
                                             selected_samples=count, payload_bytes=payload)
                # Consume the response tail so checksum includes the entire gz file.
                while reader.read(1024 * 1024):
                    pass
                if reader.digest.hexdigest() != source["image_md5"]:
                    raise ValueError("Official image archive MD5 mismatch; split remains unverified")
            if count != expected:
                raise ValueError(f"Missing selected {split} images: received {count}, expected {expected}")
            put(split + "_verified", source["image_md5"])
            connection.commit()
        present, payload = stored_rows(connection, manifest, validate_bytes=True)
        if len(present) != len(manifest["samples"]):
            raise ValueError("Selected sample count mismatch")
        budget.check()
        put("complete", True)
        connection.commit()
        summary = {"manifest_sha256": fingerprint, "sample_count": len(present), "payload_bytes": payload}
        atomic_bytes(root / "complete.json", canonical(summary))
        if progress:
            progress(event="completed", **summary)
        return summary
    finally:
        # A failed transfer preserves verified inserts, but never a completion flag.
        connection.commit()
        connection.close()


def verify_images(root):
    """Offline verification of selected bytes; never contact a source or use JAX."""
    root = Path(root)
    manifest, fingerprint = load_manifest(root)
    connection = sqlite3.connect(f"file:{(root / 'images.sqlite').resolve()}?mode=ro", uri=True)
    try:
        meta = {key: json.loads(value) for key, value in connection.execute("SELECT key,value FROM meta")}
        if meta.get("manifest_sha256") != fingerprint or not meta.get("complete"):
            raise ValueError("Image database is incomplete or does not match its manifest")
        for split in ("train", "validation"):
            if meta.get(split + "_verified") != manifest["identity"]["sources"][split]["image_md5"]:
                raise ValueError("Source archive verification is missing")
        present, payload = stored_rows(connection, manifest, validate_bytes=True)
        if len(present) != len(manifest["samples"]):
            raise ValueError("Missing selected images")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        return {"manifest_sha256": fingerprint, "sample_count": len(present), "payload_bytes": payload}
    finally:
        connection.close()

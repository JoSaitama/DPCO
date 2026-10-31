"""CNWL Red Mini-ImageNet release splits; never generate or repair labels.

ImageNet ordinal names and 84px/JPEG preprocessing follow the official TFDS
builder. Source images must be supplied under the ImageNet access agreement.
"""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import ssl
import tarfile
import urllib.request
import zipfile

import numpy as np

from src.data.cifar100n_prepare import atomic_json, canonical, file_digest, preparation_lock

HOMEPAGE = "https://google.github.io/controlled-noisy-web-labels/index.html"
METADATA_URL = "https://storage.googleapis.com/cnlw/dataset_no_images.zip"
FORMAT = "dpco-cnwl-red-v1"
LEVELS = (0, 5, 10, 15, 20, 30, 40, 50, 60, 80)
# Published TFDS split counts, which differ slightly from the paper's rounded 50K.
RED_COUNTS = {0: 50000, 5: 50000, 10: 50000, 15: 50000, 20: 50000,
              30: 49985, 40: 50010, 50: 49962, 60: 50000, 80: 50008}
BLUE_COUNT, VALIDATION_COUNT = 60000, 5000
TFDS_BUILDER_BLOB = "bbbc776dca8a8e704a8651831d0e1fd700665419"


def level(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value not in LEVELS:
        raise ValueError(f"cnwl_noise_percent selects an official Red split: {LEVELS}")
    return int(value)


def cache_dir(root, percent):
    return Path(root) / "prepared" / f"red_{level(percent):02d}"


def download_file(url, path, *, max_bytes=64 * 1024 * 1024):
    """Atomic complete-file downloads. Existing verified images are reused later.

    Interrupted individual files restart; this is not HTTP byte-range resume.
    """
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise ValueError("CNWL image URI must be an HTTP(S) URL")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=30, context=context) as response, partial.open("wb") as out:
        received = 0
        for block in iter(lambda: response.read(1024 * 1024), b""):
            received += len(block)
            if received > max_bytes:
                raise ValueError("CNWL source exceeds the bounded download size")
            out.write(block)
        if not received:
            raise ValueError("Empty CNWL source download")
        out.flush()
        os.fsync(out.fileno())
    os.replace(partial, path)


def metadata_directory(manual, *, download=False):
    manual = Path(manual)
    destination = manual / "dataset_no_images"
    required = ["mini-imagenet-annotations.json", "mini-imagenet/split/blue_noise_nl_0.0",
                "mini-imagenet/split/clean_validation"]
    required += [f"mini-imagenet/split/red_noise_nl_{p / 100}" for p in LEVELS]
    if all((destination / name).is_file() for name in required) and any(
            (destination / "mini-imagenet" / name).is_file() for name in ("class_names.txt", "class_name.txt")):
        return destination
    archive = manual / "dataset_no_images.zip"
    if not archive.exists():
        if not download:
            raise FileNotFoundError(f"Place official dataset_no_images.zip in {manual}, or use --download")
        download_file(METADATA_URL, archive)
    # Read only named release metadata; never extract arbitrary archive paths.
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        class_names = [n for n in names if n.endswith(("mini-imagenet/class_names.txt", "mini-imagenet/class_name.txt"))]
        if len(class_names) != 1:
            raise ValueError("Official CNWL class-name file is missing or ambiguous")
        required.append("mini-imagenet/" + Path(class_names[0]).name)
        for relative in required:
            matches = [n for n in names if n == relative or n.endswith("/" + relative)]
            if len(matches) != 1 or source.getinfo(matches[0]).file_size > 64 * 1024 * 1024:
                raise ValueError(f"Missing/ambiguous/oversized CNWL release member: {relative}")
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(source.read(matches[0]))
    return destination


def read_split(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 2 or not re.fullmatch(r"[A-Za-z0-9_-]+\.jpg", parts[0]):
            raise ValueError(f"Invalid CNWL split row in {path}")
        label = int(parts[1])
        if not 0 <= label < 100:
            raise ValueError("CNWL class IDs must be in [0,99]")
        rows.append((parts[0], label))
    if not rows or len({name for name, _ in rows}) != len(rows):
        raise ValueError("Empty CNWL split or duplicate image IDs; do not silently deduplicate")
    return rows


def read_release(directory, percent):
    """Keep exact split order, labels and IDs; validate real release cardinality."""
    percent = level(percent)
    directory = Path(directory)
    splits = directory / "mini-imagenet/split"
    train_file, val_file = splits / f"red_noise_nl_{percent / 100}", splits / "clean_validation"
    blue_file = splits / "blue_noise_nl_0.0"
    train, val, blue = read_split(train_file), read_split(val_file), read_split(blue_file)
    if (len(train), len(val), len(blue)) != (RED_COUNTS[percent], VALIDATION_COUNT, BLUE_COUNT):
        raise ValueError("CNWL split size differs from the published release; incomplete/custom data is not an official split")
    names_file = directory / "mini-imagenet/class_names.txt"
    if not names_file.exists():
        names_file = directory / "mini-imagenet/class_name.txt"
    names = names_file.read_text().splitlines()
    if len(names) != 100 or any(not name.strip() for name in names):
        raise ValueError("CNWL needs the author's 100 class names in original order")
    original = dict(blue)
    if {label for _, label in blue} != set(range(100)) or any(
            not re.fullmatch(r"n\d{16}\.jpg", name) for name in original):
        raise ValueError("Invalid clean mini-ImageNet canonical IDs/classes")
    annotation_file = directory / "mini-imagenet-annotations.json"
    data = json.loads(annotation_file.read_bytes())
    web = {}
    for group in data["data"]:
        item = group[0]  # Author/TFDS annotation schema: each image is a list of records.
        name = item["image/id"] + ".jpg"
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.jpg", name) or name in web:
            raise ValueError("Unsafe or duplicate CNWL web image ID")
        web[name] = item
    if set(original) & set(web) or set(dict(train)) & set(dict(val)):
        raise ValueError("CNWL clean/web IDs or train/validation IDs overlap")
    records = []
    for name, label in train:
        if name in original and original[name] == label:
            records.append((name, label, True))
        elif name in web:
            item = web[name]
            if item["image/class/label/is_clean"] != 0 or int(item["image/class/label"]) != label:
                raise ValueError("Red split does not match its released incorrect web-label annotation")
            records.append((name, label, False))
        else:
            raise ValueError("Red split image is neither an unchanged clean image nor released web noise")
    if percent == 0 and any(not clean for _, _, clean in records):
        raise ValueError("The official zero-noise split contains web noise")
    files = [train_file, val_file, blue_file, names_file, annotation_file]
    provenance = {"homepage": HOMEPAGE, "metadata_url": METADATA_URL,
        "files_sha256": {str(p.relative_to(directory)): file_digest(p) for p in files},
        "noise_type": "red", "cnwl_noise_percent": percent,
        "train_split": f"red_noise_nl_{percent / 100}", "validation_split": "clean_validation",
        "class_names": names, "tfds_builder_blob": TFDS_BUILDER_BLOB}
    return {"train": records, "validation": [(n, y, True) for n, y in val]}, web, provenance


def resize_image(data):
    # Same conversion as the official TFDS builder (including JPEG quality).
    try:
        import cv2
    except ImportError as error:
        raise ImportError("CNWL preparation requires pip install -e '.[data,cnwl]'; training reads the prepared cache without OpenCV") from error
    pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags=3)
    if pixels is None:
        raise ValueError("CNWL source is not a decodable color image")
    pixels = cv2.resize(pixels, (84, 84), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", pixels, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        raise ValueError("CNWL JPEG conversion failed")
    return encoded.tobytes()


def train_archive_images(path, wanted):
    """Resolve mini IDs by numeric rank within each original ImageNet synset.

    nXXXXXXXX00000009.jpg denotes the NINTH numeric-sorted image, not *_9.JPEG.
    """
    by_class = {}
    for name in wanted:
        by_class.setdefault(name[:9], {})[int(name[9:-4])] = name
    with tarfile.open(path, "r:") as outer:
        for member in outer:
            synset = Path(member.name).stem
            if synset not in by_class or not member.isfile():
                continue
            with outer.extractfile(member) as handle:
                body = BytesIO(handle.read())
            with tarfile.open(fileobj=body, mode="r:") as inner:
                entries = [m for m in inner.getmembers() if m.isfile() and re.fullmatch(
                    synset + r"_\d+\.JPEG", Path(m.name).name)]
                entries.sort(key=lambda m: int(Path(m.name).stem.split("_")[-1]))
                for rank, name in sorted(by_class[synset].items()):
                    if not 1 <= rank <= len(entries):
                        raise ValueError(f"CNWL mini-ImageNet ordinal outside original class archive: {name}")
                    with inner.extractfile(entries[rank - 1]) as handle:
                        yield name, handle.read()


def validation_archive_images(path, wanted):
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            name = Path(member.name).stem + ".jpg"
            if member.isfile() and name in wanted:
                with archive.extractfile(member) as handle:
                    yield name, handle.read()


def verify_prepared(root, percent):
    directory = cache_dir(root, percent)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    digest = hashlib.sha256(canonical(manifest)).hexdigest()
    if (manifest.get("format") != FORMAT or manifest.get("provenance", {}).get("cnwl_noise_percent") != level(percent)
            or json.loads((directory / "complete.json").read_bytes()) != {"manifest_sha256": digest}
            or file_digest(directory / "images.sqlite") != manifest["database_sha256"]):
        raise ValueError("CNWL cache completion/identity/checksum mismatch")
    return manifest, digest


def prepare(root, *, percent=20, manual_dir=None, download=False, workers=8, progress=None, report_path=None):
    percent = level(percent)
    root = Path(root).expanduser().resolve()
    manual = Path(manual_dir).expanduser().resolve() if manual_dir else root / "raw"
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise ValueError("workers must be in [1,32]")
    with preparation_lock(root):
        destination = cache_dir(root, percent)
        if destination.exists():
            return verify_prepared(root, percent)
        release = metadata_directory(manual, download=download)
        records, web, provenance = read_release(release, percent)
        staging = destination.parent / ("." + destination.name + ".building")
        staging.mkdir(parents=True, exist_ok=True)
        # Resume only against exactly the same split metadata and local source layout.
        source_paths = [manual / n for n in ("ILSVRC2012_img_train.tar", "ILSVRC2012_img_val.tar")]
        provenance["source_archives"] = {str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                                         for p in source_paths if p.exists()}
        identity = {"provenance": provenance, "manual_directory": str(manual),
                    "preprocessing": "opencv_imdecode_color_INTER_AREA84_JPEG72"}
        connection = sqlite3.connect(staging / "images.sqlite")
        failures = []
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS samples (split TEXT, id INTEGER, source_id TEXT, label INTEGER, is_clean INTEGER, image BLOB, PRIMARY KEY(split,id), UNIQUE(split,source_id))")
            connection.execute("CREATE TABLE IF NOT EXISTS meta (identity TEXT)")
            previous = connection.execute("SELECT identity FROM meta").fetchone()
            if previous is not None and previous[0] != canonical(identity).decode():
                raise ValueError("Partial CNWL cache belongs to different sources; preserve it and choose another data directory")
            if previous is None:
                connection.execute("INSERT INTO meta VALUES (?)", (canonical(identity).decode(),))
                connection.commit()

            def insert(split, index, row, data):
                name, label, clean = row
                connection.execute("INSERT INTO samples VALUES (?,?,?,?,?,?)", (split, index, name, label, int(clean), resize_image(data)))

            def emit(**fields):
                if progress:
                    progress(**fields)

            missing = {split: {row[0]: (i, row) for i, row in enumerate(rows)
                       if connection.execute("SELECT 1 FROM samples WHERE split=? AND id=?", (split, i)).fetchone() is None}
                       for split, rows in records.items()}
            for split, needed in missing.items():
                # Optional already-obtained images must retain the official image IDs.
                folders = ([manual / "noisy_images", release / "noisy_images", manual / "all_images"] if split == "train"
                           else [manual / "validation_all"])
                for name, (index, row) in list(needed.items()):
                    is_web = name in web
                    candidates = folders if is_web or split == "validation" else [manual / "clean_images", manual / "all_images"]
                    local = next((folder / name for folder in candidates if (folder / name).is_file()), None)
                    if local is not None:
                        insert(split, index, row, local.read_bytes())
                        del needed[name]
                archive_needed = {name for name in needed if name not in web}
                archive = source_paths[0 if split == "train" else 1]
                if archive_needed:
                    if not archive.exists():
                        raise FileNotFoundError(f"Missing original ImageNet source {archive}; obtain it under ImageNet terms, or supply canonical clean_images/validation_all")
                    iterator = train_archive_images if split == "train" else validation_archive_images
                    for name, data in iterator(archive, archive_needed):
                        index, row = needed.pop(name)
                        insert(split, index, row, data)
                        if index % 100 == 0:
                            connection.commit()
                            emit(event="cnwl_prepare_images", split=split, cached=len(records[split]) - len(needed))
                    unresolved = archive_needed & set(needed)
                    if unresolved:
                        raise ValueError(f"Original ImageNet archive is missing {len(unresolved)} selected images, e.g. {min(unresolved)}")
                connection.commit()
            selected_web = list(missing["train"].items())

            def fetch(item):
                name, _ = item
                path = manual / "noisy_images" / name
                try:
                    if not download:
                        raise FileNotFoundError(f"Missing CNWL web image {name}; use --download to fetch its released URI")
                    download_file(web[name]["image/uri"], path)
                    data = path.read_bytes()
                    resize_image(data)  # Validate before accepting this completed file.
                    return name, data, None
                except Exception as error:
                    return name, None, f"{type(error).__name__}: {error}"

            with ThreadPoolExecutor(max_workers=workers) as pool:
                done = 0
                # Bound in-flight image bytes even if an early URI responds slowly.
                for offset in range(0, len(selected_web), workers * 4):
                    for name, data, error in pool.map(fetch, selected_web[offset:offset + workers * 4]):
                        done += 1
                        if error:
                            failures.append({"image_id": name, "error": error})
                        else:
                            index, row = missing["train"].pop(name)
                            insert("train", index, row, data)
                        if done % 100 == 0:
                            connection.commit()
                            emit(event="cnwl_web_progress", completed=done, requested=len(selected_web), failures=len(failures))
            connection.commit()
            if failures:
                raise RuntimeError(f"{len(failures)} CNWL images unavailable; no samples dropped. Retry after restoring sources; see preparation report")
            counts = dict(connection.execute("SELECT split,COUNT(*) FROM samples GROUP BY split"))
            if counts != {split: len(rows) for split, rows in records.items()}:
                raise ValueError("Incomplete CNWL prepared cache")
        except BaseException as error:
            connection.commit()  # Preserve completed rows for a later retry.
            if report_path:
                atomic_json(Path(report_path), {"status": "failed", "error": str(error), "failures": failures,
                    "requested_noise_percent": percent, "no_samples_dropped": True})
            raise
        finally:
            connection.close()
        corrupted = sum(not clean for _, _, clean in records["train"])
        manifest = {"format": FORMAT, "provenance": provenance, "counts": counts,
            "corrupted_count": corrupted, "actual_noise_percent": 100 * corrupted / counts["train"],
            "preprocessing": identity["preprocessing"], "database_sha256": file_digest(staging / "images.sqlite"),
            "original_labels": "known_for_clean_only; no_closed_set_truth_for_web_images"}
        digest = hashlib.sha256(canonical(manifest)).hexdigest()
        atomic_json(staging / "manifest.json", manifest)
        atomic_json(staging / "complete.json", {"manifest_sha256": digest})
        os.replace(staging, destination)
        if report_path:
            atomic_json(Path(report_path), {"status": "completed", "manifest_sha256": digest, **manifest})
        if progress:
            progress(event="cnwl_prepared", manifest_sha256=digest, **counts, actual_noise_percent=manifest["actual_noise_percent"])
        return manifest, digest

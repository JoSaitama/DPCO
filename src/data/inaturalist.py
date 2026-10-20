"""Read the completed proportional iNaturalist subset without network or copies."""

import hashlib
import json
from pathlib import Path

from src.data.images import PreparedImages
from src.data.inaturalist_subset import canonical, load_manifest


class INaturalistImages(PreparedImages):
    """Check selection identities on open; full byte verification is a separate CLI."""

    def __init__(self, root, *, num_classes=10000, limit=None):
        root = Path(root).expanduser().resolve()
        required = ("manifest.json", "manifest.sha256", "images.sqlite", "complete.json")
        for name in required:
            if not (root / name).is_file():
                raise FileNotFoundError(f"Incomplete local iNaturalist subset: missing {root / name}. "
                    "Finish scripts/prepare_inaturalist.py before training; no automatic download is performed.")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError("Smoke sample limit must be a positive integer")
        manifest, digest = load_manifest(root)
        identity = manifest["identity"]
        categories = manifest["categories"]
        if (identity["num_classes"] != num_classes or len(categories) != num_classes
                or [row["label"] for row in categories] != list(range(num_classes))
                or len({row["category_id"] for row in categories}) != num_classes):
            raise ValueError("iNaturalist class mapping does not match the classifier head")
        counts = identity["subset_counts"]
        if set(counts) != {"train", "validation"} or any(
                isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in counts.values()):
            raise ValueError("Invalid iNaturalist split counts")
        expected = {}
        mapping = {row["category_id"]: row["label"] for row in categories}
        split_labels = {split: set() for split in counts}
        for row in manifest["samples"]:
            key = (row["split"], row["id"])
            if (key in expected or row["split"] not in counts
                    or mapping.get(row["category_id"]) != row["label"]):
                raise ValueError("Invalid or duplicate iNaturalist manifest sample identity")
            expected[key] = (row["image_id"], row["label"], row["path"])
            split_labels[row["split"]].add(row["label"])
        if (set(expected) != {(split, i) for split, n in counts.items() for i in range(n)}
                or any(len(labels) != num_classes for labels in split_labels.values())):
            raise ValueError("iNaturalist manifest has missing split members or classes")
        summary = json.loads((root / "complete.json").read_bytes())
        if summary.get("manifest_sha256") != digest or summary.get("sample_count") != len(expected):
            raise ValueError("iNaturalist completion record does not match the manifest")
        super().__init__(root / "images.sqlite", digest)
        try:
            meta = {key: json.loads(value) for key, value in self.connection.execute("SELECT key,value FROM meta")}
            if meta.get("manifest_sha256") != digest or meta.get("complete") is not True:
                raise ValueError("iNaturalist database is incomplete or belongs to another selection")
            for split in counts:
                if meta.get(split + "_verified") != identity["sources"][split]["image_md5"]:
                    raise ValueError("iNaturalist source archive verification is missing or mismatched")
            if self.counts != counts:
                raise ValueError("iNaturalist database split counts differ from the manifest")
            # Read identities and byte lengths only, not the 14+ GB image payload.
            payload, seen = 0, set()
            for split, index, image_id, label, path, size in self.connection.execute(
                    "SELECT split,id,image_id,label,path,length(image) FROM samples"):
                key = (split, index)
                if key in seen or expected.get(key) != (image_id, label, path) or not size or size < 0:
                    raise ValueError("Stored iNaturalist sample differs from the manifest or has empty bytes")
                seen.add(key)
                payload += size
            if seen != set(expected) or payload != summary.get("payload_bytes"):
                raise ValueError("iNaturalist database members or payload size differ from completion record")
            self.full_counts = dict(self.counts)
            self.identity = {"dataset": "inaturalist2021", "manifest_sha256": digest,
                "selection": identity, "sample_limit_per_split": limit}
            if limit is not None:
                self.counts = {split: min(n, limit) for split, n in counts.items()}
                self.fingerprint = hashlib.sha256(canonical(self.identity)).hexdigest()
        except BaseException:
            self.close()
            raise

    def get(self, split, index):
        if split not in self.counts or not 0 <= int(index) < self.counts[split]:
            raise IndexError(f"Image outside the configured iNaturalist view: {split}/{index}")
        return super().get(split, index)

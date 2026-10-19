"""Offline source fixtures verify exact, host-independent subset preparation."""

from collections import Counter
from contextlib import closing
from dataclasses import replace
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import unittest

from src.data.inaturalist_subset import (Source, StorageBudget, build_manifest, canonical,
    load_manifest, preparation_lock, prepare_images, proportional_quotas, select_rows, verify_images)


def tar_bytes(files):
    raw = BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, content in files:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, BytesIO(content))
    return gzip.compress(raw.getvalue(), mtime=0)


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.contents, self.sources = {}, {}
        self.images = {}
        for split, counts in (("train", [10, 20, 30, 40]), ("validation", [10] * 4)):
            images, annotations, files = [], [], []
            for category, count in enumerate(counts):
                for offset in range(count):
                    image_id = category * 1000 + offset
                    path = f"{'train' if split == 'train' else 'val'}/category_{category}/{image_id}.jpg"
                    blob = f"original-encoded-image-{split}-{image_id}".encode()
                    images.append({"id": image_id, "file_name": path})
                    annotations.append({"image_id": image_id, "category_id": category})
                    files.append((path, blob))
                    self.images[path] = blob
            # Labels deliberately precede images, and source rows are reversed.
            document = {"annotations": list(reversed(annotations)), "images": list(reversed(images)),
                "categories": [{"id": c, "name": f"species_{c}"} for c in reversed(range(4))]}
            annotation = tar_bytes([(split + ".json", json.dumps(document).encode())])
            archive = tar_bytes(files)
            a_url, i_url = f"fixture://{split}/metadata", f"fixture://{split}/images"
            self.contents[a_url], self.contents[i_url] = annotation, archive
            self.sources[split] = Source(a_url, hashlib.md5(annotation).hexdigest(), i_url,
                                         hashlib.md5(archive).hexdigest(), sum(counts))

    def opener(self, url):
        return BytesIO(self.contents[url])

    def budget(self, root):
        return StorageBudget(root, max_bytes=100_000_000, reserve_bytes=0)

    def plan(self, root, seed=42, **kwargs):
        root.mkdir(exist_ok=True)
        return build_manifest(root, seed=seed, train_count=40, validation_count=8,
            budget=self.budget(root), sources=self.sources, category_count=4,
            opener=self.opener, **kwargs)

    def test_proportional_quota_not_equal_per_class_and_integer_ties(self):
        self.assertEqual(proportional_quotas({0: 10, 1: 20, 2: 30, 3: 40}, 40),
                         {0: 4, 1: 8, 2: 12, 3: 16})
        self.assertEqual(proportional_quotas({3: 10, 2: 10, 1: 10, 0: 10}, 6),
                         {3: 1, 2: 1, 1: 2, 0: 2})
        for total in (0, -1, 101, 1):
            with self.assertRaises(ValueError):
                proportional_quotas({0: 10, 1: 90}, total)

    def test_selection_is_independent_of_order_and_python_hash_seed(self):
        rows = [(i, i % 4, f"train/c{i % 4}/{i}.jpg") for i in range(100)]
        quotas = {i: 4 for i in range(4)}
        chosen = select_rows(rows, quotas, 42, "train")
        random.Random(321).shuffle(rows)
        self.assertEqual(chosen, select_rows(rows, quotas, 42, "train"))
        self.assertNotEqual(chosen, select_rows(rows, quotas, 5, "train"))
        script = ("from src.data.inaturalist_subset import select_rows,canonical; "
                  f"print(canonical(select_rows({rows!r},{quotas!r},42,'train')).hex())")
        import os
        outputs = [subprocess.check_output([sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": value}) for value in ("1", "99")]
        self.assertEqual(*outputs)

    def test_two_hosts_produce_byte_identical_manifests(self):
        first, second = self.root / "node_ziao2", self.root / "node_ziao1"
        manifest, fingerprint = self.plan(first)
        other, other_fingerprint = self.plan(second)
        self.assertEqual(manifest, other)
        self.assertEqual(fingerprint, other_fingerprint)
        self.assertEqual((first / "manifest.json").read_bytes(), (second / "manifest.json").read_bytes())
        for split, counts in (("train", {0: 4, 1: 8, 2: 12, 3: 16}),
                              ("validation", {i: 2 for i in range(4)})):
            self.assertEqual(Counter(r["category_id"] for r in manifest["samples"] if r["split"] == split), counts)
        third, _ = self.plan(self.root / "different_seed", seed=5)
        self.assertNotEqual(manifest["samples"], third["samples"])
        self.assertEqual(manifest["distributions"], third["distributions"])

    def test_plan_reuse_and_identity_guard_and_checksum_recovery(self):
        root = self.root / "reuse"
        self.plan(root)
        with self.assertRaisesRegex(ValueError, "different selection"):
            self.plan(root, seed=5)
        (root / "manifest.sha256").unlink()
        self.plan(root)
        self.assertTrue((root / "manifest.sha256").is_file())
        (root / "manifest.json").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_manifest(root)

    def test_stream_selected_original_bytes_and_reuse_offline(self):
        root = self.root / "prepare"
        manifest, fingerprint = self.plan(root)
        summary = prepare_images(root, budget=self.budget(root), opener=self.opener)
        self.assertEqual(summary["sample_count"], 48)
        self.assertEqual(summary["manifest_sha256"], fingerprint)
        with closing(sqlite3.connect(root / "images.sqlite")) as connection:
            rows = list(connection.execute("SELECT path,image FROM samples"))
        self.assertEqual(set(path for path, _ in rows), {row["path"] for row in manifest["samples"]})
        self.assertTrue(all(self.images[path] == blob for path, blob in rows))
        def offline(url):
            raise AssertionError("Completed cache must not use the network")
        self.assertEqual(summary, prepare_images(root, budget=self.budget(root), opener=offline))
        self.assertEqual(summary, verify_images(root))
        self.assertFalse(list(root.glob("*.tar.gz")))
        self.assertEqual(len(list((root / "metadata").glob("*.tar.gz"))), 2)

    def test_interrupted_transfer_replays_and_preserves_selection(self):
        root = self.root / "interrupted"
        self.plan(root)
        events = []
        def interrupted(**event):
            events.append(event)
            if event["event"] == "image_scan_start" and event["split"] == "validation":
                raise OSError("simulated network outage")
        with self.assertRaises(OSError):
            prepare_images(root, budget=self.budget(root), opener=self.opener, progress=interrupted)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify_images(root)
        self.assertEqual(prepare_images(root, budget=self.budget(root), opener=self.opener)["sample_count"], 48)

    def test_archive_checksum_failure_does_not_publish_completion(self):
        root = self.root / "bad_archive"
        self.plan(root)
        url = self.sources["train"].image_url
        original = self.contents[url]
        self.contents[url] = original + b"unexpected-trailing-content"
        with self.assertRaisesRegex(ValueError, "MD5"):
            prepare_images(root, budget=self.budget(root), opener=self.opener)
        self.assertFalse((root / "complete.json").exists())
        self.contents[url] = original
        self.assertEqual(prepare_images(root, budget=self.budget(root), opener=self.opener)["sample_count"], 48)

    def test_resume_after_mid_split_committed_batch_and_archive_order_independence(self):
        roots = [self.root / "resume", self.root / "other_host"]
        for root in roots:
            root.mkdir()
            build_manifest(root, seed=42, train_count=80, validation_count=8,
                budget=self.budget(root), sources=self.sources, category_count=4, opener=self.opener)
        def interrupt(**event):
            if event["event"] == "image_prepare" and event["split"] == "train":
                raise OSError("simulated interrupted preparation after 64 durable images")
        with self.assertRaises(OSError):
            prepare_images(roots[0], budget=self.budget(roots[0]), opener=self.opener, progress=interrupt)
        with closing(sqlite3.connect(roots[0] / "images.sqlite")) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 64)
        self.assertEqual(prepare_images(roots[0], budget=self.budget(roots[0]), opener=self.opener)["sample_count"], 88)
        prepare_images(roots[1], budget=self.budget(roots[1]), opener=self.opener)
        self.assertEqual(verify_images(roots[0]), verify_images(roots[1]))
        def logical_rows(root):
            with closing(sqlite3.connect(root / "images.sqlite")) as connection:
                return list(connection.execute("SELECT * FROM samples ORDER BY split,id"))
        self.assertEqual(logical_rows(roots[0]), logical_rows(roots[1]))

    def test_unsafe_paths_and_missing_members_fail_without_completion(self):
        from src.data.inaturalist_subset import safe_archive_path
        for name in ("../x.jpg", "/tmp/x.jpg", "train/../x.jpg", "train\\x.jpg"):
            with self.assertRaises(ValueError):
                safe_archive_path(name)
        root = self.root / "missing"
        manifest, _ = self.plan(root)
        omitted = next(row["path"] for row in manifest["samples"] if row["split"] == "train")
        archive = tar_bytes([(path, blob) for path, blob in self.images.items()
                             if path.startswith("train/") and path != omitted])
        # Keep a matching source checksum to exercise the independent count guard.
        manifest["identity"]["sources"]["train"]["image_md5"] = hashlib.md5(archive).hexdigest()
        content = canonical(manifest)
        (root / "manifest.json").write_bytes(content)
        (root / "manifest.sha256").write_text(hashlib.sha256(content).hexdigest() + "\n")
        self.contents[self.sources["train"].image_url] = archive
        with self.assertRaisesRegex(ValueError, "Missing selected"):
            prepare_images(root, budget=self.budget(root), opener=self.opener)
        self.assertFalse((root / "complete.json").exists())

    def test_annotation_checksum_or_count_mismatch_rejected(self):
        root = self.root / "bad_annotations"
        root.mkdir()
        sources = dict(self.sources)
        sources["train"] = replace(sources["train"], annotation_md5="0" * 32)
        with self.assertRaisesRegex(ValueError, "MD5"):
            build_manifest(root, train_count=40, validation_count=8, budget=self.budget(root),
                sources=sources, category_count=4, opener=self.opener)
        root2 = self.root / "bad_counts"
        root2.mkdir()
        sources["train"] = replace(self.sources["train"], count=101)
        with self.assertRaisesRegex(ValueError, "counts"):
            build_manifest(root2, train_count=40, validation_count=8, budget=self.budget(root2),
                sources=sources, category_count=4, opener=self.opener)

    def test_corrupt_local_image_fails_offline_verification(self):
        root = self.root / "corrupt"
        self.plan(root)
        prepare_images(root, budget=self.budget(root), opener=self.opener)
        with closing(sqlite3.connect(root / "images.sqlite")) as connection:
            connection.execute("UPDATE samples SET image=? WHERE split='train' AND id=0", (b"corrupt",))
            connection.commit()
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_images(root)

    def test_capacity_failure_preserves_manifest_and_lock_excludes_writers(self):
        root = self.root / "capacity"
        self.plan(root)
        original = (root / "manifest.json").read_bytes()
        with self.assertRaises(OSError):
            prepare_images(root, budget=StorageBudget(root, max_bytes=1, reserve_bytes=0), opener=self.opener)
        self.assertEqual((root / "manifest.json").read_bytes(), original)
        self.assertFalse((root / "complete.json").exists())
        with preparation_lock(root):
            with self.assertRaisesRegex(RuntimeError, "Another preparation"):
                with preparation_lock(root):
                    pass

    def test_standalone_help_and_module_do_not_import_jax(self):
        subprocess.check_call([sys.executable, "-c", "import sys; import src.data.inaturalist_subset; "
            "assert 'jax' not in sys.modules; assert 'numpy' not in sys.modules"])
        result = subprocess.run([sys.executable, "scripts/prepare_inaturalist.py", "--help"],
                                capture_output=True, text=True, check=True)
        self.assertIn("--selection-seed", result.stdout)
        result = subprocess.run([sys.executable, "scripts/prepare_inaturalist.py", "--selection-seed", "-1"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()

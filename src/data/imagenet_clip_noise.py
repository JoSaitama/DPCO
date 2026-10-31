"""Offline, immutable CLIP-induced noise; no torch/JAX import for artifact tooling.

This is a controlled synthetic protocol, not a release of human noisy labels.
Prediction caches resume by committed batches. Published labels are authoritative:
floating-point predictions need not be bitwise equal on different hardware.
"""

import csv
from contextlib import contextmanager, closing
import hashlib
import html
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
from PIL import Image


FORMAT = "dpco_imagenet_clip_idn_v1"
ALGORITHM = "margin_gumbel_weighted_without_replacement_v1"
CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
WEIGHT_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
WEIGHT_URL = f"https://openaipublic.azureedge.net/clip/models/{WEIGHT_SHA256}/ViT-L-14.pt"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def readonly(path):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        yield connection


def source_description(path):
    with readonly(path) as connection:
        meta = {key: json.loads(value) for key, value in connection.execute("SELECT key,value FROM meta")}
        if not meta.get("complete") or "identity" not in meta:
            raise ValueError("ImageNet cache must be completely prepared before CLIP scoring")
        mapping = {int(k): v for k, v in meta["identity"]["mapping"].items()}
        names = meta.get("original_label_names", [])
        if set(mapping.values()) != set(range(len(mapping))) or len(mapping) < 2:
            raise ValueError("Invalid source class mapping")
        if len(names) <= max(mapping):
            raise ValueError("Prepared ImageNet cache is missing original_label_names")
        classes = [None] * len(mapping)
        for original, local in mapping.items():
            classes[local] = {"label": local, "imagenet_label": original, "name": names[original]}
        rows = connection.execute("SELECT id,label,source_index FROM samples WHERE split='train' ORDER BY id").fetchall()
        if not rows or [r[0] for r in rows] != list(range(len(rows))):
            raise ValueError("Training sample IDs must be contiguous and nonempty")
        labels = np.array([r[1] for r in rows], dtype=np.int32)
        if np.any((labels < 0) | (labels >= len(classes))):
            raise ValueError("Source labels outside selected class space")
        counts = dict(connection.execute("SELECT split,COUNT(*) FROM samples GROUP BY split"))
    return {"identity": meta["identity"], "fingerprint": hashlib.sha256(canonical(meta["identity"])).hexdigest(),
        "classes": classes, "counts": counts,
        "sample_index_sha256": hashlib.sha256(canonical(rows)).hexdigest()}


def prediction_summary(logits, original):
    logits = np.asarray(logits, dtype=np.float64)
    original = np.asarray(original, dtype=np.int64)
    if (logits.ndim != 2 or logits.shape[0] != len(original) or logits.shape[1] < 2
            or not np.all(np.isfinite(logits)) or np.any((original < 0) | (original >= logits.shape[1]))):
        raise ValueError("Invalid CLIP logits/original labels")
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    top1 = probabilities.argmax(axis=1)
    true = probabilities[np.arange(len(original)), original].copy()
    probabilities[np.arange(len(original)), original] = -1
    wrong = probabilities.argmax(axis=1)
    wrong_probability = probabilities[np.arange(len(original)), wrong]
    return wrong, true, wrong_probability, top1


def selection_order(difficulty, seed, temperature):
    difficulty = np.asarray(difficulty, dtype=np.float64)
    if (difficulty.ndim != 1 or not len(difficulty) or not np.all(np.isfinite(difficulty))
            or np.any(np.abs(difficulty) > 1)):
        raise ValueError("Difficulty must be a nonempty finite probability margin in [-1,1]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("noise_seed must be a uint32 integer")
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature < 1e-6:
        raise ValueError("sampling_temperature must be finite and >= 1e-6")
    # Gumbel top-k implements weighted sampling without replacement, w=exp(d/tau).
    # One full ordering also gives nested 10%/20% masks for the same seed/tau.
    keys = difficulty / temperature + np.random.Generator(np.random.PCG64(seed)).gumbel(size=len(difficulty))
    return np.lexsort((np.arange(len(difficulty)), -keys))


def exact_count(count, percent):
    if isinstance(percent, bool) or not math.isfinite(percent) or not 0 <= percent <= 100:
        raise ValueError("label_noise_percent must be finite and in [0,100]")
    return int(math.floor(count * percent / 100 + .5))


class OpenAIClipPredictor:
    """Original OpenAI L/14, FP32 deterministic eval, CPU/CUDA only; no training."""

    def __init__(self, classes, *, weights_dir, checkpoint=None, download=False,
                 device="cpu", prompt="a photo of a {class_name}."):
        # Must be set before CUDA context creation for deterministic cuBLAS GEMMs.
        if device.startswith("cuda"):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import importlib.metadata
        import clip
        import torch
        import torchvision
        import PIL
        if not hasattr(clip, "load") or not hasattr(clip, "tokenize"):
            raise RuntimeError("Install the pinned OpenAI CLIP extra, not the unrelated PyPI clip package")
        if device != "cpu" and device != "cuda" and not device.startswith("cuda:"):
            raise ValueError("CLIP preparation supports cpu or cuda[:index]; it does not claim TPU support")
        if prompt.count("{class_name}") != 1:
            raise ValueError("prompt must contain exactly one {class_name}")
        texts = [prompt.format(class_name=row["name"]) for row in classes]
        weights_dir = Path(weights_dir).expanduser().resolve()
        path = Path(checkpoint).expanduser().resolve() if checkpoint else weights_dir / "ViT-L-14.pt"
        if not path.exists():
            if not download or checkpoint:
                raise FileNotFoundError(f"Missing official checkpoint {path}; use --download-weights to obtain it")
            weights_dir.mkdir(parents=True, exist_ok=True)
            # The pinned official downloader verifies the URL's SHA256 too.
            from clip.clip import _download
            path = Path(_download(WEIGHT_URL, str(weights_dir)))
        if sha256(path) != WEIGHT_SHA256:
            raise ValueError("Checkpoint is not the official OpenAI ViT-L/14 weight (SHA256 mismatch)")
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        model, self.preprocess = clip.load(str(path), device=device, jit=False)
        self.model, self.torch, self.device = model.eval().float(), torch, device
        self.model.requires_grad_(False)
        if int(model.visual.input_resolution) != 224:
            raise ValueError("Expected ViT-L/14 at 224 pixels")
        with torch.inference_mode():
            # Bound text-encoder memory when the candidate space grows to 1K.
            features = torch.cat([model.encode_text(clip.tokenize(texts[start:start + 64]).to(device)).float()
                for start in range(0, len(texts), 64)])
            self.text_features = features / features.norm(dim=-1, keepdim=True)
        direct_url = importlib.metadata.distribution("clip").read_text("direct_url.json")
        self.identity = {"model": "OpenAI CLIP ViT-L/14", "checkpoint_sha256": WEIGHT_SHA256,
            "checkpoint_url": WEIGHT_URL, "input_resolution": 224, "dtype": "float32",
            "prompt": prompt, "texts": texts, "logit_scale": float(model.logit_scale.exp().item()),
            "text_batch_size": 64,
            "implementation_files_sha256": {name: sha256(Path(clip.__file__).parent / name)
                for name in ("clip.py", "model.py", "simple_tokenizer.py")},
            "clip_distribution_source": json.loads(direct_url) if direct_url else None,
            "torch_version": torch.__version__, "torchvision_version": torchvision.__version__, "pillow_version": PIL.__version__,
            "numpy_version": np.__version__, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else "cpu",
            "preprocess": repr(self.preprocess), "deterministic_algorithms": True}

    def __call__(self, images):
        torch = self.torch
        with torch.inference_mode():
            features = self.model.encode_image(torch.stack([self.preprocess(image) for image in images]).to(self.device)).float()
            features /= features.norm(dim=-1, keepdim=True)
            return (self.model.logit_scale.exp() * features @ self.text_features.T).cpu().numpy()


def score_images(image_db, scores_dir, predictor, *, batch_size=32, progress=None):
    """Transactional restart: complete committed batches are reused, never shuffled."""
    from filelock import FileLock
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = source_description(image_db)
    identity = {"format": FORMAT, "source": source, "predictor": predictor.identity,
        "inference_batch_size": batch_size}
    directory = Path(scores_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".scores.lock")), readonly(image_db) as images, closing(sqlite3.connect(directory / "scores.sqlite")) as scores:
        scores.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        scores.execute("CREATE TABLE IF NOT EXISTS scores(id INTEGER PRIMARY KEY,original INTEGER,source_index INTEGER,image_sha256 TEXT,wrong INTEGER,p_true REAL,p_wrong REAL,top1 INTEGER)")
        stored = scores.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if stored and json.loads(stored[0]) != identity:
            raise ValueError("Score cache identity differs (source/model/prompt/device/batch size); use a new scores-dir")
        scores.execute("INSERT OR IGNORE INTO meta VALUES('identity',?)", (canonical(identity).decode(),))
        done = scores.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        if done and scores.execute("SELECT MIN(id),MAX(id) FROM scores").fetchone() != (0, done - 1):
            raise ValueError("Partial score cache has noncontiguous IDs")
        total = source["counts"]["train"]
        if done > total:
            raise ValueError("Score cache has too many samples")
        scores.commit()
        for start in range(done, total, batch_size):
            rows = images.execute("SELECT id,label,source_index,image FROM samples WHERE split='train' AND id>=? ORDER BY id LIMIT ?", (start, batch_size)).fetchall()
            pixels = []
            for row in rows:
                with Image.open(BytesIO(row[3])) as image:
                    pixels.append(image.convert("RGB"))
            wrong, true, false, top1 = prediction_summary(predictor(pixels), [r[1] for r in rows])
            scores.executemany("INSERT INTO scores VALUES(?,?,?,?,?,?,?,?)", [
                (r[0], r[1], r[2], hashlib.sha256(r[3]).hexdigest(), int(w), float(p), float(q), int(t))
                for r, w, p, q, t in zip(rows, wrong, true, false, top1)])
            scores.commit()
            if progress:
                progress(event="clip_score_progress", completed=start + len(rows), total=total)
        scores.execute("INSERT OR REPLACE INTO meta VALUES('complete','true')")
        scores.commit()
    return directory / "scores.sqlite"


def read_scores(scores_dir):
    with readonly(Path(scores_dir) / "scores.sqlite") as connection:
        meta = {key: json.loads(value) for key, value in connection.execute("SELECT key,value FROM meta")}
        if not meta.get("complete"):
            raise ValueError("Score cache is incomplete; resume the score stage first")
        rows = connection.execute("SELECT * FROM scores ORDER BY id").fetchall()
    if len(rows) != meta["identity"]["source"]["counts"]["train"] or [r[0] for r in rows] != list(range(len(rows))):
        raise ValueError("Score cache sample count/IDs differ from source")
    original = np.array([r[1] for r in rows], np.int32)
    wrong = np.array([r[4] for r in rows], np.int32)
    true = np.array([r[5] for r in rows], np.float64)
    false = np.array([r[6] for r in rows], np.float64)
    classes = len(meta["identity"]["source"]["classes"])
    if (np.any(original == wrong) or np.any((wrong < 0) | (wrong >= classes))
            or not np.all(np.isfinite(true)) or not np.all(np.isfinite(false))
            or np.any((true < 0) | (false < 0) | (true > 1) | (false > 1) | (true + false > 1 + 1e-12))):
        raise ValueError("Invalid score cache probabilities/candidates")
    return meta["identity"], rows, original, wrong, true, false


def publish_noise(scores_dir, output_dir, *, percent=20., seed=0, temperature=.2):
    from filelock import FileLock
    identity, rows, original, wrong, true, false = read_scores(scores_dir)
    order = selection_order(false - true, seed, temperature)
    count = exact_count(len(rows), percent)
    noisy = original.copy()
    noisy[order[:count]] = wrong[order[:count]]
    corrupted = noisy != original
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".publish.lock")):
        # Existing artifacts are immutable; even equal requests must use verify.
        names = ("labels.npz", "labels.csv", "summary.json", "transition.csv", "noise_manifest.json")
        if any((directory / name).exists() for name in names):
            raise FileExistsError("Noise output exists; verify/reuse it or choose a new output-dir")
        with tempfile.TemporaryDirectory(prefix=".publish-", dir=directory) as tmp:
            staging = Path(tmp)
            np.savez_compressed(staging / "labels.npz", sample_id=np.arange(len(rows), dtype=np.int64),
                source_index=np.array([r[2] for r in rows], np.int64), original_label=original,
                noisy_label=noisy, is_corrupted=corrupted, candidate_label=wrong,
                p_original=true, p_wrong=false, difficulty=false - true,
                image_sha256=np.array([r[3] for r in rows], dtype="U64"))
            classes = identity["source"]["classes"]
            with (staging / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("sample_id", "source_index", "original_label", "original_name", "noisy_label", "noisy_name", "is_corrupted", "candidate_label", "p_original", "p_wrong", "difficulty", "image_sha256"))
                for i, row in enumerate(rows):
                    writer.writerow((i, row[2], int(original[i]), classes[original[i]]["name"], int(noisy[i]),
                        classes[noisy[i]]["name"], int(corrupted[i]), int(wrong[i]), true[i], false[i], false[i] - true[i], row[3]))
            transition = np.zeros((len(classes), len(classes)), np.int64)
            np.add.at(transition, (original[corrupted], noisy[corrupted]), 1)
            with (staging / "transition.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("original_label", "original_name", "noisy_label", "noisy_name", "count"))
                for a, b in zip(*np.nonzero(transition)):
                    writer.writerow((a, classes[a]["name"], b, classes[b]["name"], transition[a, b]))
            summary = {"sample_count": len(rows), "corrupted_count": count,
                "target_noise_percent": percent, "actual_noise_percent": 100 * count / len(rows),
                "clip_natural_top1_error_percent": 100 * float(np.mean(np.array([r[7] for r in rows]) != original)),
                "per_class": [{**c, "train_count": int(np.count_nonzero(original == c["label"])),
                    "corrupted_count": int(np.count_nonzero(corrupted & (original == c["label"]))) } for c in classes],
                "noise_is": "controlled_model_induced_synthetic; not human annotations",
                "warning": "Candidates are forced non-original classes; exact rate does not guarantee semantic plausibility."}
            (staging / "summary.json").write_bytes(canonical(summary))
            manifest = {"format": FORMAT, "algorithm": ALGORITHM, "score_identity": identity,
                "selection": {"label_noise_percent": percent, "noise_seed": seed, "sampling_temperature": temperature,
                    "rng": "numpy.Generator(PCG64); gumbel_top_k",
                    "difficulty": "max_non_original_probability - original_probability",
                    "weights": "exp(difficulty / sampling_temperature); Gumbel top-k; sample-id tie break",
                    "count_rounding": "floor(N*percent/100 + 0.5)", "numpy_version": np.__version__},
                "sample_count": len(rows), "corrupted_count": count,
                "actual_noise_percent": summary["actual_noise_percent"],
                "files": {name: sha256(staging / name) for name in names if name != "noise_manifest.json"}}
            (staging / "noise_manifest.json").write_bytes(canonical(manifest))
            # Publish marker last; interrupted partial publication fails closed.
            for name in names:
                (staging / name).replace(directory / name)
    return directory / "noise_manifest.json"


def verify_noise(manifest_path):
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_bytes())
    if manifest.get("format") != FORMAT or manifest.get("algorithm") != ALGORITHM:
        raise ValueError("Unsupported ImageNet noise manifest")
    if set(manifest.get("files", {})) != {"labels.npz", "labels.csv", "summary.json", "transition.csv"}:
        raise ValueError("Noise manifest has an invalid file set")
    for name, digest in manifest["files"].items():
        if sha256(path.parent / name) != digest:
            raise ValueError(f"Noise artifact checksum mismatch: {name}")
    with np.load(path.parent / "labels.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    n = manifest["sample_count"]
    k = len(manifest["score_identity"]["source"]["classes"])
    required = {"sample_id", "source_index", "original_label", "noisy_label", "is_corrupted", "candidate_label", "p_original", "p_wrong", "difficulty", "image_sha256"}
    if set(arrays) != required or any(a.shape != (n,) for a in arrays.values()) or n <= 0:
        raise ValueError("Noise arrays have invalid keys/shapes")
    if not np.array_equal(arrays["sample_id"], np.arange(n)):
        raise ValueError("Noise sample IDs differ from canonical source order")
    for key in ("original_label", "noisy_label", "candidate_label"):
        a = arrays[key]
        if a.dtype.kind not in "iu" or np.any((a < 0) | (a >= k)):
            raise ValueError("Noise labels have invalid dtype/range")
    original, noisy = arrays["original_label"], arrays["noisy_label"]
    selection = manifest["selection"]
    mask = noisy != original
    expected = exact_count(n, selection["label_noise_percent"])
    if (arrays["is_corrupted"].dtype != bool or not np.array_equal(mask, arrays["is_corrupted"])
            or np.count_nonzero(mask) != expected or manifest["corrupted_count"] != expected
            or not math.isclose(manifest["actual_noise_percent"], 100 * expected / n, abs_tol=1e-12)
            or np.any(arrays["candidate_label"] == original)):
        raise ValueError("Noise mask/count differs from generation protocol")
    difficulty = arrays["p_wrong"] - arrays["p_original"]
    p, q = arrays["p_original"], arrays["p_wrong"]
    if (not np.all(np.isfinite(p)) or not np.all(np.isfinite(q))
            or np.any((p < 0) | (q < 0) | (p > 1) | (q > 1) | (p + q > 1 + 1e-12))):
        raise ValueError("Noise artifact contains invalid probability summaries")
    if not np.array_equal(difficulty, arrays["difficulty"]):
        raise ValueError("Noise difficulty differs from stored probabilities")
    order = selection_order(difficulty, selection["noise_seed"], selection["sampling_temperature"])
    rebuilt = original.copy()
    rebuilt[order[:expected]] = arrays["candidate_label"][order[:expected]]
    if not np.array_equal(rebuilt, noisy):
        raise ValueError("Stored labels do not reproduce from seed/scores/protocol")
    for value in arrays.values():
        value.flags.writeable = False
    return manifest, arrays, sha256(path)


def export_audit(image_db, manifest_path, output_dir, *, count=100, seed=42):
    """Uniform independent samples in changed/unchanged groups; no label editing."""
    manifest, arrays, digest = verify_noise(manifest_path)
    if source_description(image_db) != manifest["score_identity"]["source"]:
        raise ValueError("Audit source differs from scored source")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("audit count must be a positive integer per group")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("audit seed must be uint32")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    thumbnails = directory / "images"
    thumbnails.mkdir()
    rng = np.random.default_rng(seed)
    selected = []
    for changed in (True, False):
        ids = np.flatnonzero(arrays["is_corrupted"] == changed)
        selected.extend(int(i) for i in rng.choice(ids, size=min(count, len(ids)), replace=False))
    classes = manifest["score_identity"]["source"]["classes"]
    cards = []
    with readonly(image_db) as images, (directory / "review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_id", "is_corrupted", "original_label", "original_name", "noisy_label", "noisy_name", "candidate_name", "p_original", "p_wrong", "difficulty", "image", "human_plausible", "notes"))
        for i in selected:
            blob = images.execute("SELECT image FROM samples WHERE split='train' AND id=?", (i,)).fetchone()[0]
            if hashlib.sha256(blob).hexdigest() != arrays["image_sha256"][i]:
                raise ValueError(f"Audit image bytes changed: sample {i}")
            name = f"images/{i:08d}.jpg"
            with Image.open(BytesIO(blob)) as image:
                image = image.convert("RGB")
                image.thumbnail((320, 320))
                image.save(directory / name, quality=90)
            a, b, c = (int(arrays[key][i]) for key in ("original_label", "noisy_label", "candidate_label"))
            true, false, difficulty = (float(arrays[key][i]) for key in ("p_original", "p_wrong", "difficulty"))
            writer.writerow((i, int(arrays["is_corrupted"][i]), a, classes[a]["name"], b, classes[b]["name"], classes[c]["name"], true, false, difficulty, name, "", ""))
            cards.append(f'<article><img src="{name}"><p>ID {i}; changed={bool(arrays["is_corrupted"][i])}</p>'
                f'<p>Original: {html.escape(classes[a]["name"])}</p><p>Training: {html.escape(classes[b]["name"])}</p>'
                f'<p>Candidate: {html.escape(classes[c]["name"])}</p><p>p(original)={true:.4f}; p(wrong)={false:.4f}; margin={difficulty:.4f}</p></article>')
    (directory / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>CLIP noise audit</title>'
        '<style>main{display:flex;flex-wrap:wrap;gap:20px}article{width:330px;border:1px solid #bbb;padding:12px}img{max-width:320px;height:240px;object-fit:contain}</style>'
        '<h1>CLIP-induced synthetic label audit</h1><p>Uniform sample per changed/unchanged group. Record judgments in review.csv; labels remain immutable. These probabilities are not calibrated human error rates.</p><main>'
        + "".join(cards) + '</main>', encoding="utf-8")
    (directory / "audit.json").write_bytes(canonical({"noise_manifest_sha256": digest,
        "audit_seed": seed, "requested_per_group": count, "sample_ids": selected,
        "selection": "independent uniform without replacement within changed/unchanged groups",
        "image_view": "full image thumbnail; CLIP scoring uses official center crop"}))
    return directory / "index.html"


class NoisyImageNetImages:
    """Training labels only in batches; original labels/mask exposed for audits."""

    def __init__(self, images, manifest_path, *, noise_percent=None, limit=None):
        self.images, self.path = images, images.path
        try:
            manifest, arrays, digest = verify_noise(manifest_path)
            source = source_description(images.path)
            if source != manifest["score_identity"]["source"] or images.fingerprint != source["fingerprint"]:
                raise ValueError("Noise artifact source identity differs from prepared ImageNet")
            rows = images.connection.execute("SELECT label,source_index FROM samples WHERE split='train' ORDER BY id").fetchall()
            if (not np.array_equal([r[0] for r in rows], arrays["original_label"])
                    or not np.array_equal([r[1] for r in rows], arrays["source_index"])):
                raise ValueError("Noise artifact sample labels/source indices differ from source")
            self.original_labels, self.observed_labels = arrays["original_label"], arrays["noisy_label"]
            self.reference_labels, self.corrupted = self.observed_labels, arrays["is_corrupted"]
            self.full_counts = dict(images.counts)
            if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
                raise ValueError("Smoke limit must be positive")
            self.counts = {split: min(n, limit) if limit is not None else n for split, n in self.full_counts.items()}
            self.actual_noise_percent = manifest["actual_noise_percent"]
            if noise_percent is not None and (isinstance(noise_percent, bool) or not math.isfinite(noise_percent)
                    or not math.isclose(noise_percent, self.actual_noise_percent, abs_tol=1e-8, rel_tol=0)):
                raise ValueError(f"noise_percent checks actual rate; loaded {self.actual_noise_percent}%")
            self.group_indices = {"clean": np.flatnonzero(~self.corrupted[:self.counts["train"]]),
                "corrupted": np.flatnonzero(self.corrupted[:self.counts["train"]])}
            self.identity = {"source": source, "noise_manifest_sha256": digest,
                "noise_type": "clip_idn", "selection": manifest["selection"],
                "generator": manifest["score_identity"]["predictor"],
                "actual_noise_percent": self.actual_noise_percent, "view_counts": self.counts,
                "full_counts": self.full_counts, "group_counts": {k: len(v) for k, v in self.group_indices.items()},
                "truth_usage": "offline_generation_and_evaluation_only; excluded_from_ClassificationBatch"}
            self.fingerprint = hashlib.sha256(canonical(self.identity)).hexdigest()
        except BaseException:
            images.close()
            raise

    def get(self, split, index):
        if split not in self.counts or not 0 <= int(index) < self.counts[split]:
            raise IndexError("Image outside configured view")
        image, label = self.images.get(split, index)
        return image, int(self.observed_labels[index]) if split == "train" else label

    def label_view(self, kind):
        if kind not in ("original", "reference"):
            raise ValueError("Unknown ImageNet label view")
        return _LabelView(self, kind)

    def close(self):
        self.images.close()


class _LabelView:
    def __init__(self, images, kind):
        self.images, self.kind, self.counts = images, kind, images.counts

    def get(self, split, index):
        image, label = self.images.get(split, index)
        if split == "train" and self.kind == "original":
            label = int(self.images.original_labels[index])
        return image, label

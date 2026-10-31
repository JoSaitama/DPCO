"""Score existing ImageNet bytes with frozen OpenAI CLIP and publish fixed labels."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "score", "generate", "verify", "audit"), default="all")
    parser.add_argument("--config", default="configs/tpu/dpco/imagenet100_resnet50.toml")
    parser.add_argument("--data-dir", help="Existing clean ImageNet cache root; never downloads dataset images")
    parser.add_argument("--image-db", help="Explicit completed images.sqlite; otherwise derive from config/data-dir")
    parser.add_argument("--output-dir", required=True, help="Noise artifact directory, normally results/files/<name>")
    parser.add_argument("--scores-dir", help="Reusable/resumable CLIP score cache; default output-dir/clip_scores")
    parser.add_argument("--label-noise-percent", type=float, default=20., help="Target global changed-label percentage, 0..100")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--sampling-temperature", type=float, default=.2, help="Weight exp(margin/tau); smaller favors harder images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]; CPU works on TPU hosts but does not use TPU")
    parser.add_argument("--weights-dir", default="../data/models/openai_clip")
    parser.add_argument("--checkpoint", help="Local official ViT-L-14.pt; exact official SHA256 required")
    parser.add_argument("--download-weights", action="store_true", help="Allow the pinned official weight download")
    parser.add_argument("--prompt", default="a photo of a {class_name}.")
    parser.add_argument("--audit-count", type=int, default=100, help="Uniform samples per changed/unchanged group")
    parser.add_argument("--audit-seed", type=int, default=42)
    parser.add_argument("--audit-dir", help="New review directory; default output-dir/audit")
    args = parser.parse_args()
    from src.data.imagenet_clip_noise import (OpenAIClipPredictor, source_description,
        score_images, publish_noise, verify_noise, export_audit, exact_count, selection_order)
    # Validate inexpensive controls before starting any network/large computation.
    exact_count(1, args.label_noise_percent)
    selection_order([0.], args.noise_seed, args.sampling_temperature)
    if args.batch_size <= 0 or args.audit_count <= 0 or not 0 <= args.audit_seed < 2**32:
        parser.error("batch-size/audit-count must be positive; audit-seed must be uint32")

    def path(value):
        p = Path(value).expanduser()
        return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()

    output = path(args.output_dir)
    scores = path(args.scores_dir) if args.scores_dir else output / "clip_scores"
    manifest = output / "noise_manifest.json"

    def image_database():
        if args.image_db:
            return path(args.image_db)
        from src.config import load_config
        from src.data.imagenet_hf import DataPlan, label_mapping
        config = load_config(path(args.config))
        if config.data.name not in ("imagenet100", "imagenet1k"):
            parser.error("CLIP noise preparation requires ImageNet100/1K config")
        mapping, digest = label_mapping(config.data, ROOT)
        plan = DataPlan(config.data.hf_repo_id, config.data.hf_revision, mapping, digest)
        return path(args.data_dir or config.data.prepared_dir) / plan.fingerprint[:24] / "images.sqlite"

    def progress(**fields):
        print(json.dumps(fields), flush=True)

    if args.stage in ("all", "score"):
        db = image_database()
        source = source_description(db)
        progress(event="clip_load_start", model="OpenAI CLIP ViT-L/14", device=args.device,
            source=str(db), train_count=source["counts"]["train"], download_weights=args.download_weights)
        predictor = OpenAIClipPredictor(source["classes"], weights_dir=path(args.weights_dir),
            checkpoint=path(args.checkpoint) if args.checkpoint else None,
            download=args.download_weights, device=args.device, prompt=args.prompt)
        score_images(db, scores, predictor, batch_size=args.batch_size, progress=progress)
        del predictor
        progress(event="clip_scores_ready", scores_dir=str(scores))
    if args.stage in ("all", "generate"):
        if manifest.exists():
            old, _, _ = verify_noise(manifest)
            requested = (args.label_noise_percent, args.noise_seed, args.sampling_temperature)
            actual = tuple(old["selection"][key] for key in ("label_noise_percent", "noise_seed", "sampling_temperature"))
            from src.data.imagenet_clip_noise import read_scores
            if actual != requested or old["score_identity"] != read_scores(scores)[0]:
                raise FileExistsError("Existing output has a different protocol/source; choose a new output-dir")
            progress(event="noise_artifact_reused", manifest=str(manifest))
        else:
            publish_noise(scores, output, percent=args.label_noise_percent,
                seed=args.noise_seed, temperature=args.sampling_temperature)
    if args.stage in ("all", "generate", "verify", "audit"):
        data, _, digest = verify_noise(manifest)
        progress(event="noise_artifact_verified", manifest=str(manifest), manifest_sha256=digest,
            target_noise_percent=data["selection"]["label_noise_percent"], actual_noise_percent=data["actual_noise_percent"],
            sample_count=data["sample_count"], corrupted_count=data["corrupted_count"])
    if args.stage in ("all", "audit"):
        audit = path(args.audit_dir) if args.audit_dir else output / "audit"
        if args.stage == "all" and audit.exists():
            existing = json.loads((audit / "audit.json").read_bytes())
            if (existing["noise_manifest_sha256"], existing["audit_seed"], existing["requested_per_group"]) != (digest, args.audit_seed, args.audit_count):
                raise FileExistsError("Existing audit has different controls; use a new --audit-dir")
            progress(event="audit_reused", html=str(audit / "index.html"))
        else:
            index = export_audit(image_database(), manifest, audit, count=args.audit_count, seed=args.audit_seed)
            progress(event="audit_ready", html=str(index), review_csv=str(audit / "review.csv"))


if __name__ == "__main__":
    main()

"""Dataset dispatch; share image iteration, training and recording across sources."""

from pathlib import Path


def open_training_images(config, base_dir, *, limit=None, progress=None):
    directory = Path(config.prepared_dir).expanduser()
    if not directory.is_absolute():
        directory = Path(base_dir) / directory
    directory = directory.resolve()
    if config.name == "cnwl":
        from src.data.cnwl import CNWLImages
        images = CNWLImages(directory, cnwl_noise_percent=config.cnwl_noise_percent,
            noise_percent=config.noise_percent, download=config.download, limit=limit, progress=progress)
        if progress:
            progress(event="data_ready", **images.identity)
        return images, images.identity
    if config.name == "mnist":
        from src.data.mnist import MNISTImages
        images = MNISTImages(directory, label_noise_percent=config.label_noise_percent,
            noise_seed=config.noise_seed, noise_percent=config.noise_percent,
            download=config.download, limit=limit, progress=progress)
        if progress:
            progress(event="data_ready", **images.identity)
        return images, images.identity
    if config.name == "cifar100n":
        from src.data.cifar100n import CIFAR100NImages
        images = CIFAR100NImages(directory, noise_type=config.noise_type, is_human=config.is_human,
            noise_seed=config.noise_seed, noise_percent=config.noise_percent,
            download=config.download, limit=limit, progress=progress)
        if progress:
            progress(event="data_ready", **images.identity)
        return images, images.identity
    if config.name == "inaturalist2021":
        from src.data.inaturalist import INaturalistImages
        if progress:
            progress(event="data_prepare_start", dataset=config.name, source="local_prepared_subset",
                     directory=str(directory), limited_smoke_samples=limit)
        images = INaturalistImages(directory, num_classes=config.num_classes, limit=limit)
        if progress:
            progress(event="data_ready", dataset=config.name, manifest_sha256=images.identity["manifest_sha256"],
                     split_counts=images.counts, full_split_counts=images.full_counts)
        return images, images.identity
    if config.name not in ("imagenet100", "imagenet1k"):
        raise ValueError(f"Unsupported classification dataset: {config.name}")
    from src.data.imagenet_hf import DataPlan, label_mapping, prepare
    mapping, manifest_hash = label_mapping(config, base_dir)
    # Noise artifacts always refer to the full prepared source, even for smoke views.
    source_limit = None if config.noise_manifest else limit
    plan = DataPlan(config.hf_repo_id, config.hf_revision, mapping, manifest_hash, source_limit, source_limit)
    if config.noise_manifest:
        from src.data.images import PreparedImages
        from src.data.imagenet_clip_noise import NoisyImageNetImages
        manifest = Path(config.noise_manifest).expanduser()
        if not manifest.is_absolute():
            manifest = Path(base_dir) / manifest
        # No implicit dataset download or CLIP inference inside training.
        images = PreparedImages(directory / plan.fingerprint[:24] / "images.sqlite", plan.fingerprint)
        images = NoisyImageNetImages(images, manifest, noise_percent=config.noise_percent, limit=limit)
        if progress:
            progress(event="data_ready", **images.identity)
        return images, images.identity
    if progress:
        progress(event="data_prepare_start", dataset=config.name, source=plan.repo_id,
                 revision=plan.revision, limited_smoke_samples=limit,
                 note="ImageNet-100 preparation may scan the larger 1K source; only selected original bytes are retained.")
    images = prepare(plan, directory, progress=(
        (lambda **fields: progress(event="data_prepare", **fields)) if progress else None))
    return images, plan.identity

"""Shared suite CLI for ImageNet and the local iNaturalist shell entry points."""

from launch_tpu import main


if __name__ == "__main__":
    main(default_mode="train", default_config="configs/tpu/imagenet100_resnet50_sage_e.toml",
         default_wait=True)

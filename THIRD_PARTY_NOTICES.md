# Third-party sources

Model base:

- Repository: https://github.com/google/flax
- Relevant directory: `examples/imagenet/`
- Source revision: `81b168d62f151feca9e8e7abc591df80168395b4` for
  `examples/imagenet/models.py`.
- Source license: Apache License 2.0, retained in `licenses/FLAX_APACHE_2.0.txt`.

`src/models/resnet.py` adapts the bottleneck ResNet V1.5 architecture. It retains
the copyright/license notice, exposes 100/1000-class heads and train/eval adapters,
and removes the named pmap BatchNorm axis in favor of host-local GSPMD reductions.
No upstream training loop, pretrained weights, or datasets are redistributed.

Class manifest sources:

- CMC ImageNet-100 synset order:
  https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
- Original 1K label index mapping:
  https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
- `configs/manifests/imagenet100_classes.txt` contains the factual synset/index
  mapping; output labels follow its recorded line order, 0 through 99.

Data source:

- https://huggingface.co/datasets/ILSVRC/imagenet-1k
- Default revision: `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`.
- Downloads remain subject to ImageNet access/usage terms. Existing HF credentials
  are reused; approval, tokens, images, and checkpoints are not part of this repository.

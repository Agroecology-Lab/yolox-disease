"""
Patch torch.load to default weights_only=False.

PyTorch >=2.6 flipped the default, which breaks Trainer.resume_train()
in stock YOLOX when loading older checkpoints (numpy scalar isn't in
the safe-globals allowlist). Patching here avoids editing the vendored
YOLOX source, which would just get overwritten on update/reclone.

Usage: import this as the very first import in tools/train.py, e.g.:

    import torch_load_patch  # noqa: F401  (must be first)
    import torch
    ...
"""
import torch

_orig_load = torch.load


def _load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _load_weights_only_false

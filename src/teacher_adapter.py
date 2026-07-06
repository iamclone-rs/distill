"""Residual output adapter shared by DFN5B fine-tuning and distillation."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DFN_MODEL_NAME = "ViT-H-14-quickgelu"
DFN_PRETRAINED = "dfn5b"


class ResidualAdapter(nn.Module):
    def __init__(self, feature_dim: int, bottleneck_dim: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, feature_dim)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        update = self.up(F.gelu(self.down(self.norm(features))))
        return F.normalize(features + update, dim=-1)


class ModalityAdapters(nn.Module):
    def __init__(self, feature_dim: int, bottleneck_dim: int = 32):
        super().__init__()
        self.sketch = ResidualAdapter(feature_dim, bottleneck_dim)
        self.photo = ResidualAdapter(feature_dim, bottleneck_dim)


def load_adapter_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_feature_dim: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[ModalityAdapters, dict]:
    """Load old or new residual-adapter checkpoints with strict validation."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    required = {"adapter_state_dict", "feature_dim", "bottleneck_dim"}
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(
            f"Invalid adapter checkpoint '{checkpoint_path}'; "
            f"missing keys: {sorted(missing)}"
        )

    adapter_mode = checkpoint.get("adapter_mode", "residual")
    if adapter_mode != "residual":
        raise RuntimeError(
            f"Unsupported adapter_mode='{adapter_mode}' in '{checkpoint_path}'. "
            "The cleaned pipeline only supports residual adapters."
        )

    saved_model = checkpoint.get("model")
    saved_pretrained = checkpoint.get("pretrained")
    if saved_model not in (None, DFN_MODEL_NAME):
        raise RuntimeError(
            f"Adapter model mismatch: checkpoint={saved_model}, expected={DFN_MODEL_NAME}"
        )
    if saved_pretrained not in (None, DFN_PRETRAINED):
        raise RuntimeError(
            "Adapter pretrained mismatch: "
            f"checkpoint={saved_pretrained}, expected={DFN_PRETRAINED}"
        )

    feature_dim = int(checkpoint["feature_dim"])
    if expected_feature_dim is not None and feature_dim != expected_feature_dim:
        raise RuntimeError(
            f"Adapter feature_dim={feature_dim}, expected={expected_feature_dim}."
        )

    adapters = ModalityAdapters(
        feature_dim=feature_dim,
        bottleneck_dim=int(checkpoint["bottleneck_dim"]),
    )
    adapters.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    adapters.eval().requires_grad_(False)
    return adapters.to(device=device, dtype=torch.float32), checkpoint

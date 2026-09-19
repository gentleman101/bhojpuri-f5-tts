import math
import re

import torch
from torch import nn


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen and only A, B trained."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        factory = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, **factory))
        # B starts at zero so the adapted model is exactly the base model at step 0.
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if self.merged:
            return out
        return out + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True


def apply_lora(model: nn.Module, target_modules: list[str], rank: int, alpha: float, dropout: float) -> list[str]:
    patterns = [re.compile(p) for p in target_modules]
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(p.fullmatch(name) for p in patterns)
    ]
    if not targets:
        raise ValueError(f"No nn.Linear modules matched {target_modules}")
    for name in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(getattr(parent, child_name), rank, alpha, dropout))
    return targets


def mark_trainable(model: nn.Module, extra_trainable: list[str]) -> list[str]:
    patterns = [re.compile(p) for p in extra_trainable]
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad = name.endswith((".lora_A", ".lora_B")) or any(p.fullmatch(name) for p in patterns)
        if param.requires_grad:
            trainable.append(name)
    return trainable


def merge_lora(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()

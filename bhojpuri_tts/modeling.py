import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer

from bhojpuri_tts.data import MEL_KWARGS
from bhojpuri_tts.lora import apply_lora, mark_trainable, merge_lora

F5_BASE_ARCH = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
TEXT_EMBED_KEY = "transformer.text_embed.text_embed.weight"


def build_cfm(vocab_file: str, arch: dict) -> CFM:
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
    return CFM(
        transformer=DiT(**arch, text_num_embeds=vocab_size, mel_dim=MEL_KWARGS["n_mel_channels"]),
        mel_spec_kwargs=MEL_KWARGS,
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_char_map,
    )


def _read_checkpoint(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        return load_file(path, device="cpu")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt.get("ema_model_state_dict") or ckpt.get("model_state_dict") or ckpt


def _best_prefix(ckpt_keys, model_keys) -> str:
    # Checkpoints wrap the CFM under varying prefixes ("ema_model.", "model.", ...); pick the one matching most keys.
    counts: dict[str, int] = {}
    for key in ckpt_keys:
        for model_key in model_keys:
            if key.endswith(model_key) and (len(key) == len(model_key) or key[-len(model_key) - 1] == "."):
                prefix = key[: len(key) - len(model_key)]
                counts[prefix] = counts.get(prefix, 0) + 1
                break
    if not counts:
        raise ValueError("Checkpoint keys do not match the CFM model at all")
    return max(counts, key=counts.get)


def load_base_weights(model: CFM, ckpt_path: str) -> None:
    raw = _read_checkpoint(ckpt_path)
    model_state = model.state_dict()
    prefix = _best_prefix(raw.keys(), model_state.keys())
    state = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix) and k[len(prefix) :] in model_state}

    old_embed = state.get(TEXT_EMBED_KEY)
    new_embed = model_state[TEXT_EMBED_KEY]
    if old_embed is not None and old_embed.shape != new_embed.shape:
        # Extended vocab: new characters are appended to vocab.txt, so existing rows keep their indices.
        rows = min(old_embed.shape[0], new_embed.shape[0])
        merged = new_embed.clone()
        merged[:rows] = old_embed[:rows]
        state[TEXT_EMBED_KEY] = merged
        print(f"Text embedding resized {tuple(old_embed.shape)} -> {tuple(new_embed.shape)}")

    missing = [k for k in model_state if k not in state and not k.startswith("mel_spec.")]
    if missing:
        raise ValueError(f"{len(missing)} model weights missing from checkpoint, e.g. {missing[:5]}")
    model.load_state_dict(state, strict=False)
    print(f"Loaded {len(state)} tensors from {ckpt_path} (prefix {prefix!r})")


def add_lora(model: CFM, lora_cfg: dict) -> tuple[list[str], list[str]]:
    targets = apply_lora(model, lora_cfg["target_modules"], lora_cfg["rank"], lora_cfg["alpha"], lora_cfg["dropout"])
    trainable = mark_trainable(model, lora_cfg.get("extra_trainable", []))
    return targets, trainable


def load_model_for_inference(
    vocab_file: str, base_checkpoint: str | None, adapter_dir: str | None, use_ema: bool, arch: dict | None = None
) -> CFM:
    adapter_cfg = None
    if adapter_dir:
        adapter_cfg = json.loads((Path(adapter_dir).parent.parent / "adapter_config.json").read_text())
        arch = adapter_cfg["arch"]
    model = build_cfm(vocab_file, arch or F5_BASE_ARCH)
    if base_checkpoint:
        load_base_weights(model, base_checkpoint)
    if adapter_cfg:
        add_lora(model, adapter_cfg["lora"])
        weights = load_file(str(Path(adapter_dir) / ("ema.safetensors" if use_ema else "trainable.safetensors")))
        unexpected = set(weights) - set(model.state_dict())
        if unexpected:
            raise ValueError(f"Adapter has unknown tensors, e.g. {sorted(unexpected)[:5]}")
        model.load_state_dict(weights, strict=False)
        merge_lora(model)
    return model.eval()

"""LoRA fine-tuning of IndicF5 on prepared manifests.

  python scripts/train_lora.py --config configs/lora_slice.yaml
  python scripts/train_lora.py --config configs/lora_slice.yaml --resume latest
  python scripts/train_lora.py --config configs/smoke_cpu.yaml --overfit-one-batch 200
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bhojpuri_tts import REPO_ROOT
from bhojpuri_tts.data import SAMPLE_RATE, FrameBatchSampler, MelDataset, collate, load_wav, read_manifest
from bhojpuri_tts.modeling import add_lora, build_cfm, load_base_weights


def resolve(path: str | None) -> str | None:
    return None if path is None else str(REPO_ROOT / path)


class TrainableEMA:
    """EMA over trainable tensors only; the frozen base weights never change so need no copy."""

    def __init__(self, params: dict[str, torch.nn.Parameter], decay: float):
        self.decay = decay
        self.shadow = {name: p.detach().clone().float() for name, p in params.items()}

    @torch.no_grad()
    def update(self, params: dict[str, torch.nn.Parameter]) -> None:
        for name, p in params.items():
            self.shadow[name].lerp_(p.detach().float(), 1 - self.decay)

    @torch.no_grad()
    def swap(self, params: dict[str, torch.nn.Parameter]) -> None:
        for name, p in params.items():
            tmp = p.detach().clone()
            p.copy_(self.shadow[name])
            self.shadow[name].copy_(tmp)


def lr_lambda(warmup: int, total: int):
    def fn(update: int) -> float:
        if update < warmup:
            return (update + 1) / warmup
        return max(0.0, (total - update) / max(1, total - warmup))

    return fn


def pick_sample_pairs(rows: list[dict], count: int) -> list[tuple[dict, dict]]:
    """Per speaker: a 3–8 s reference clip and a different val sentence to synthesize in that voice."""
    pairs = []
    for speaker in sorted({r["speaker"] for r in rows}):
        clips = sorted((r for r in rows if r["speaker"] == speaker), key=lambda r: r["utt_id"])
        refs = [r for r in clips if 3.0 <= r["duration"] <= 8.0] or clips
        targets = [r for r in clips if r is not refs[0]]
        pairs += [(refs[0], t) for t in targets[:count]]
    return pairs


@torch.no_grad()
def validation_loss(model, loader, device, seed: int) -> float:
    # Fixed RNG so noise, t and masks are identical at every evaluation and losses are comparable.
    py_state = random.getstate()
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        random.seed(seed)
        model.eval()
        losses = []
        for batch in loader:
            loss, _, _ = model(batch["mel"].to(device), text=batch["text"], lens=batch["lens"].to(device))
            losses.append(loss.item())
        model.train()
    random.setstate(py_state)
    return float(np.mean(losses))


@torch.no_grad()
def synthesize(model, vocoder, ref: dict, target: dict, sampling: dict, device) -> np.ndarray:
    from f5_tts.infer.utils_infer import infer_batch_process

    model.eval()
    torch.manual_seed(0)
    wave, _, _ = infer_batch_process(
        (load_wav(ref["audio_path"])[None], SAMPLE_RATE),
        ref["text"] + " ",
        [target["text"]],
        model,
        vocoder,
        nfe_step=sampling["nfe_step"],
        cfg_strength=sampling["cfg_strength"],
        sway_sampling_coef=sampling["sway_sampling_coef"],
        device=device,
    )
    model.train()
    return wave


def save_checkpoint(out_dir: Path, update: int, epoch: int, params, ema, optimizer, scheduler, keep_last: int):
    ckpt_dir = out_dir / "checkpoints" / f"step_{update:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file({n: p.detach().float().cpu().contiguous() for n, p in params.items()}, ckpt_dir / "trainable.safetensors")
    save_file({n: t.cpu().contiguous() for n, t in ema.shadow.items()}, ckpt_dir / "ema.safetensors")
    torch.save(
        dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), update=update, epoch=epoch),
        ckpt_dir / "training_state.pt",
    )
    (out_dir / "checkpoints" / "latest").write_text(ckpt_dir.name)
    old = sorted((out_dir / "checkpoints").glob("step_*"))[:-keep_last]
    for path in old:
        shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="'latest' or a checkpoint directory")
    parser.add_argument("--overfit-one-batch", type=int, metavar="STEPS", help="Sanity check: train on one batch only")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg, model_cfg, lora_cfg, optim_cfg, log_cfg = (cfg[k] for k in ("data", "model", "lora", "optim", "logging"))
    out_dir = REPO_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])

    accelerator = Accelerator(
        mixed_precision=optim_cfg["mixed_precision"], gradient_accumulation_steps=optim_cfg["grad_accumulation_steps"]
    )
    device = accelerator.device

    model = build_cfm(resolve(model_cfg["vocab_file"]), model_cfg["arch"])
    if model_cfg.get("base_checkpoint"):
        load_base_weights(model, resolve(model_cfg["base_checkpoint"]))
    else:
        print("WARNING: no base_checkpoint — training from random init (smoke tests only)")
    targets, trainable_names = add_lora(model, lora_cfg)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    n_trainable = sum(p.numel() for p in params.values())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA on {len(targets)} linear layers; trainable {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    (out_dir / "adapter_config.json").write_text(
        json.dumps(dict(arch=model_cfg["arch"], lora=lora_cfg, vocab_file=model_cfg["vocab_file"],
                        base_checkpoint=model_cfg.get("base_checkpoint")), indent=2)
    )
    shutil.copy(args.config, out_dir / "config.yaml")

    train_rows = read_manifest(REPO_ROOT / data_cfg["train_manifest"])
    val_rows = read_manifest(REPO_ROOT / data_cfg["val_manifest"])
    train_set, val_set = MelDataset(train_rows), MelDataset(val_rows)
    batch_kwargs = dict(max_frames=data_cfg["max_frames_per_batch"], max_samples=data_cfg["max_samples_per_batch"])
    train_sampler = FrameBatchSampler([train_set.frame_len(i) for i in range(len(train_set))], seed=cfg["seed"], **batch_kwargs)
    val_sampler = FrameBatchSampler([val_set.frame_len(i) for i in range(len(val_set))], seed=0, shuffle=False, **batch_kwargs)
    loader_kwargs = dict(collate_fn=collate, num_workers=data_cfg["num_workers"], persistent_workers=data_cfg["num_workers"] > 0)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, **loader_kwargs)
    print(f"train: {len(train_rows)} clips in {len(train_sampler)} batches; val: {len(val_rows)} clips")

    optimizer = torch.optim.AdamW(params.values(), lr=optim_cfg["learning_rate"], weight_decay=optim_cfg["weight_decay"])
    max_updates = args.overfit_one_batch or optim_cfg["max_updates"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(optim_cfg["warmup_updates"], max_updates))
    model, optimizer = accelerator.prepare(model, optimizer)
    ema = TrainableEMA(params, optim_cfg["ema_decay"])

    update, epoch = 0, 0
    if args.resume:
        ckpt_dir = Path(args.resume)
        if args.resume == "latest":
            ckpt_dir = out_dir / "checkpoints" / (out_dir / "checkpoints" / "latest").read_text().strip()
        weights = load_file(ckpt_dir / "trainable.safetensors")
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(weights[name])
        ema.shadow = {n: t.to(device) for n, t in load_file(ckpt_dir / "ema.safetensors").items()}
        state = torch.load(ckpt_dir / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        update, epoch = state["update"], state["epoch"]
        print(f"Resumed from {ckpt_dir} at update {update}, epoch {epoch}")

    writer = SummaryWriter(out_dir / "tensorboard")
    metrics_path = out_dir / "metrics.jsonl"  # read by scripts/dashboard.py

    def emit(**row):
        with metrics_path.open("a") as f:
            f.write(json.dumps(dict(t=time.time(), **row)) + "\n")

    vocoder = None
    sample_pairs = pick_sample_pairs(val_rows, cfg["sampling"]["num_per_speaker"])
    unwrapped = accelerator.unwrap_model(model)
    trainable_params = list(params.values())

    def batches():
        nonlocal epoch
        if args.overfit_one_batch:
            batch = next(iter(train_loader))
            while True:
                yield batch
        while True:
            train_sampler.set_epoch(epoch)
            yield from train_loader
            epoch += 1

    model.train()
    start, frames_seen, loss_sum, loss_count = time.time(), 0, 0.0, 0
    last_t, last_update = start, update
    emit(kind="start", update=update, max_updates=max_updates, config=str(args.config))
    for batch in batches():
        with accelerator.accumulate(model):
            loss, _, _ = model(batch["mel"].to(device), text=batch["text"], lens=batch["lens"].to(device))
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(trainable_params, optim_cfg["max_grad_norm"]) if accelerator.sync_gradients else None
            optimizer.step()
            optimizer.zero_grad()
        frames_seen += int(batch["lens"].sum())
        loss_sum, loss_count = loss_sum + loss.item(), loss_count + 1
        if not accelerator.sync_gradients:
            continue

        scheduler.step()
        ema.update(params)
        update += 1

        if update % log_cfg["log_every"] == 0 or update == 1:
            elapsed = time.time() - start
            audio_hours_per_hour = frames_seen * 256 / SAMPLE_RATE / elapsed
            lr = scheduler.get_last_lr()[0]
            print(f"update {update}/{max_updates} epoch {epoch} loss {loss_sum / loss_count:.4f} "
                  f"grad_norm {float(grad_norm):.3f} lr {lr:.2e} speed {audio_hours_per_hour:.1f}x realtime")
            writer.add_scalar("train/loss", loss_sum / loss_count, update)
            writer.add_scalar("train/grad_norm", float(grad_norm), update)
            writer.add_scalar("train/lr", lr, update)
            now = time.time()
            sec_per_update = (now - last_t) / max(update - last_update, 1)  # includes val/sample/save time
            last_t, last_update = now, update
            vram_gb = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            emit(kind="train", update=update, max_updates=max_updates, epoch=epoch, loss=loss_sum / loss_count,
                 grad_norm=float(grad_norm), lr=lr, sec_per_update=sec_per_update,
                 speed=audio_hours_per_hour, vram_gb=vram_gb)
            loss_sum, loss_count = 0.0, 0

        if args.overfit_one_batch:
            if update >= max_updates:
                break
            continue

        if update % log_cfg["val_every"] == 0 or update == max_updates:
            ema.swap(params)
            val_loss = validation_loss(unwrapped, val_loader, device, cfg["seed"])
            ema.swap(params)
            print(f"update {update}: val_loss (EMA weights) {val_loss:.4f}")
            writer.add_scalar("val/loss_ema", val_loss, update)
            emit(kind="val", update=update, val_loss=val_loss)

        if update % log_cfg["sample_every"] == 0 or update == max_updates:
            if vocoder is None:
                from f5_tts.infer.utils_infer import load_vocoder

                vocoder = load_vocoder("vocos", device=device)
            ema.swap(params)
            sample_dir = out_dir / "samples" / f"step_{update:07d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for i, (ref, target) in enumerate(sample_pairs):
                wave = synthesize(unwrapped, vocoder, ref, target, cfg["sampling"], device)
                name = f"{target['speaker']}_{i}"
                sf.write(sample_dir / f"{name}.wav", wave, SAMPLE_RATE)
                writer.add_audio(f"samples/{name}", torch.from_numpy(wave)[None], update, SAMPLE_RATE)
            ema.swap(params)
            (sample_dir / "texts.txt").write_text(
                "\n".join(f"{t['speaker']}_{i}\tref={r['utt_id']}\t{t['text']}" for i, (r, t) in enumerate(sample_pairs)),
                encoding="utf-8",
            )

        if update % log_cfg["save_every"] == 0 or update == max_updates:
            save_checkpoint(out_dir, update, epoch, params, ema, optimizer, scheduler, log_cfg["keep_last"])
            print(f"update {update}: saved checkpoint")
            emit(kind="checkpoint", update=update)

        if update >= max_updates:
            break

    writer.close()
    emit(kind="done", update=update)
    print(f"Done: {update} updates in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()

"""Pull a training run's newest checkpoint (and its metadata) back from the Hugging Face model repo, ready for --resume latest.

    python scripts/pull_run.py lora_slice_r32 [--repo gentleman101/bhojpuri-f5-tts] [--step step_0002000]

Downloads <run>/checkpoints/<step>/{trainable,ema}.safetensors + training_state.pt, adapter_config.json, config.yaml,
metrics.jsonl and samples into runs/<run>/, then writes runs/<run>/checkpoints/latest. Defaults to the highest step.
Then:  python scripts/train_lora.py --config runs/<run>/config.yaml --resume latest
"""

import argparse
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from bhojpuri_tts import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--repo", default="gentleman101/bhojpuri-f5-tts")
    parser.add_argument("--step", help="e.g. step_0002000 (default: highest)")
    args = parser.parse_args()

    files = HfApi().list_repo_files(args.repo, repo_type="model")
    steps = sorted({m.group(1) for f in files if (m := re.match(rf"{re.escape(args.run)}/checkpoints/(step_\d+)/", f))})
    if not steps:
        raise SystemExit(f"no checkpoints for run '{args.run}' in {args.repo}")
    step = args.step or steps[-1]
    if step not in steps:
        raise SystemExit(f"{step} not found; available: {', '.join(steps)}")
    tmp = Path("/home/assets/pull_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    snapshot_download(args.repo, repo_type="model", local_dir=tmp, max_workers=8, allow_patterns=[
        f"{args.run}/checkpoints/{step}/*", f"{args.run}/adapter_config.json", f"{args.run}/config.yaml",
        f"{args.run}/metrics.jsonl", f"{args.run}/samples/*"])
    dest = REPO_ROOT / "runs" / args.run
    (dest / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copytree(tmp / args.run, dest, dirs_exist_ok=True)
    (dest / "checkpoints" / "latest").write_text(step)
    (dest / ".pushed_steps").write_text("\n".join(steps[: steps.index(step) + 1]))  # so the uploader won't re-push them
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"restored {args.run} at {step} -> {dest}\nresume with: python scripts/train_lora.py --config {dest}/config.yaml --resume latest")


if __name__ == "__main__":
    main()

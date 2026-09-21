"""Upload each new training checkpoint to a private Hugging Face model repo, so progress survives a wiped disk.

    python scripts/push_checkpoints.py --run runs/lora_10h_r32 --repo <user>/<name> [--interval 120] [--dry-run]

Watches <run>/checkpoints/latest. When it changes, uploads that step folder (trainable + EMA + optimizer state,
enough to resume) plus metrics.jsonl, config.yaml, adapter_config.json and samples, under <run-name>/ in the repo.
Exits after the trainer logs its "done" record and the last checkpoint is pushed.

Resume on a fresh machine:
    hf download <repo> --include "<run-name>/*" --local-dir /tmp/pull
    copy /tmp/pull/<run-name>/* into runs/<run-name>/, then train_lora.py --config ... --resume latest
"""

import argparse
import json
import time
from pathlib import Path

REQUIRED = ("trainable.safetensors", "ema.safetensors", "training_state.pt")


def latest_complete_step(run: Path) -> str | None:
    marker = run / "checkpoints" / "latest"
    if not marker.exists():
        return None
    step = marker.read_text().strip()
    step_dir = run / "checkpoints" / step
    return step if all((step_dir / f).exists() for f in REQUIRED) else None


def trainer_done(run: Path) -> bool:
    path = run / "metrics.jsonl"
    if not path.exists():
        return False
    lines = path.read_text().strip().splitlines()
    return bool(lines) and json.loads(lines[-1]).get("kind") == "done"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="run directory, e.g. runs/lora_10h_r32")
    parser.add_argument("--repo", required=True, help="HF repo id, created private if missing")
    parser.add_argument("--interval", type=int, default=60, help="seconds between checks")
    parser.add_argument("--once", action="store_true", help="upload the newest checkpoint if it is missing, then exit")
    parser.add_argument("--dry-run", action="store_true", help="print what would be uploaded; touch nothing on the Hub")
    args = parser.parse_args()

    run, name = args.run.resolve(), args.run.resolve().name
    pushed_file = run / ".pushed_steps"
    pushed = set(pushed_file.read_text().split()) if pushed_file.exists() else set()

    api = None
    if not args.dry_run:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)

    def push(step: str):
        step_dir = run / "checkpoints" / step
        if args.dry_run:
            size = sum(p.stat().st_size for p in step_dir.rglob("*") if p.is_file()) / 2**20
            print(f"[dry-run] would upload {step_dir} ({size:.0f} MB) -> {args.repo}/{name}/checkpoints/{step}")
            return
        api.upload_folder(folder_path=str(step_dir), path_in_repo=f"{name}/checkpoints/{step}", repo_id=args.repo,
                          commit_message=f"{name} {step}")
        # small run-level files: overwritten each time, cheap
        api.upload_folder(folder_path=str(run), path_in_repo=name, repo_id=args.repo,
                          allow_patterns=["metrics.jsonl", "config.yaml", "adapter_config.json", "samples/**"],
                          commit_message=f"{name} metadata at {step}")

    while True:
        step = latest_complete_step(run)
        if step and step not in pushed:
            try:
                push(step)
                if not args.dry_run:  # a dry run must not mark steps as pushed
                    pushed.add(step)
                    pushed_file.write_text("\n".join(sorted(pushed)))
                    print(f"{time.strftime('%H:%M:%S')} pushed {name}/{step} -> {args.repo}", flush=True)
            except Exception as e:  # network blips must not kill the watcher; retry next tick
                print(f"{time.strftime('%H:%M:%S')} push of {step} FAILED: {e!r} — retrying in {args.interval}s", flush=True)
        elif trainer_done(run) and (step is None or step in pushed):
            print("trainer finished and last checkpoint is pushed; exiting" if step else
                  "WARNING: trainer finished but no complete checkpoint was found — nothing pushed; exiting", flush=True)
            return
        if args.dry_run and step:
            return
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

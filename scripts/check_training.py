"""One-shot health check of a training run, meant to be run on a schedule (e.g. a Claude /loop tick).

    python scripts/check_training.py [--run NAME] [--snapshot OUT.html]

Prints a short report and ends with a line `STATUS: OK`, `STATUS: WARN ...`, `STATUS: FINISHED` or `STATUS: NO RUN`.
Exit code is 0 for OK/FINISHED/NO RUN and 1 for WARN. With --snapshot it also rewrites the artifact page (republish it
afterwards). Checks: tmux windows alive, metrics fresh, loss finite, checkpoints uploaded, disk, GPU.
"""

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

from bhojpuri_tts import REPO_ROOT

RUNS = REPO_ROOT / "runs"


def load_dashboard():
    spec = importlib.util.spec_from_file_location("dashboard", REPO_ROOT / "scripts" / "dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tmux_windows(session: str) -> dict[str, str] | None:
    r = subprocess.run(["tmux", "list-panes", "-t", session, "-s", "-F", "#{window_name}=#{pane_current_command}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return dict(line.split("=", 1) for line in r.stdout.splitlines())


def python_running(script: str) -> bool:
    """True if a python interpreter (not the tmux wrapper shell, whose command line also names the script) runs `script`."""
    out = subprocess.run(["ps", "-eo", "comm=,args="], capture_output=True, text=True).stdout
    return any(line.split(None, 1)[0].startswith("python") and script in line for line in out.splitlines())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="run name under runs/ (default: newest with metrics.jsonl)")
    parser.add_argument("--session", default="train")
    parser.add_argument("--snapshot", type=Path, help="also write the artifact snapshot page here")
    args = parser.parse_args()

    runs = sorted((p.parent.name for p in RUNS.glob("*/metrics.jsonl")),
                  key=lambda n: (RUNS / n / "metrics.jsonl").stat().st_mtime, reverse=True)
    name = args.run or (runs[0] if runs else "")
    now = time.time()
    warns: list[str] = []
    lines: list[str] = []

    if not name:
        lines.append("no run with metrics.jsonl yet")
        status = "NO RUN"
    else:
        run = RUNS / name
        recs = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
        train = [r for r in recs if r["kind"] == "train"]
        val = [r for r in recs if r["kind"] == "val"]
        last = recs[-1]
        finished = last["kind"] == "done"
        age = now - last["t"]
        lines.append(f"run {name}: last event '{last['kind']}' {age / 60:.1f} min ago")
        if train:
            t = train[-1]
            recent = sorted(r["sec_per_update"] for r in train[-10:])
            spu = recent[len(recent) // 2]
            remaining = max(t["max_updates"] - t["update"], 0) * spu
            lines.append(f"update {t['update']}/{t['max_updates']} ({100 * t['update'] / t['max_updates']:.1f}%), "
                         f"loss {t['loss']:.4f}, {spu:.2f} s/update, ETA {remaining / 60:.0f} min, peak VRAM {t['vram_gb']:.1f} GB")
            if not all(math.isfinite(r["loss"]) for r in train[-5:]):
                warns.append("loss is NaN/inf")
            if len(val) >= 3 and val[-1]["val_loss"] > min(v["val_loss"] for v in val) * 1.15:
                warns.append(f"val loss {val[-1]['val_loss']:.4f} is >15% above its best — possible overfit")
            stall_after = max(300, 60 * spu * 25)  # ~25 log intervals, at least 5 min
            if not finished and age > stall_after:
                warns.append(f"no metrics for {age / 60:.0f} min — trainer may have stalled or died")
        if val:
            lines.append(f"val loss {val[-1]['val_loss']:.4f} @ {val[-1]['update']} (best {min(v['val_loss'] for v in val):.4f})")

        # checkpoints vs uploads
        ckpts = sorted(p.name for p in (run / "checkpoints").glob("step_*")) if (run / "checkpoints").exists() else []
        pushed_file = run / ".pushed_steps"
        pushed = set(pushed_file.read_text().split()) if pushed_file.exists() else set()
        unpushed = [c for c in ckpts if c not in pushed]
        lines.append(f"checkpoints on disk {len(ckpts)}, uploaded to HF {len(pushed)}, waiting {len(unpushed)}")
        if unpushed:
            newest_age = now - (run / "checkpoints" / unpushed[-1]).stat().st_mtime
            if newest_age > 900:
                warns.append(f"{unpushed[-1]} saved {newest_age / 60:.0f} min ago and still not on HF")
        push_log = run / "push.log"
        tail = push_log.read_text().splitlines()[-1:] if push_log.exists() else []
        if tail and "FAILED" in tail[0]:
            warns.append("last uploader line reports a FAILED push")

        wins = tmux_windows(args.session)
        if wins is None:
            if not finished:
                warns.append(f"tmux session '{args.session}' not found")
        else:
            lines.append("tmux windows: " + ", ".join(wins))
        if not finished:
            if not python_running("scripts/train_lora.py"):
                warns.append("no train_lora.py process is running")
            if not python_running("scripts/push_checkpoints.py"):
                warns.append("uploader process is not running — checkpoints are not being backed up")
        status = "FINISHED" if finished else ("WARN " + "; ".join(warns) if warns else "OK")

    free_gb = shutil.disk_usage(REPO_ROOT).free / 2**30
    lines.append(f"disk free {free_gb:.0f} GB")
    if free_gb < 20:
        warns.append(f"only {free_gb:.0f} GB disk free")
        status = "WARN " + "; ".join(warns)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    if gpu:
        lines.append("gpu util%, mem MiB used/total, temp C: " + gpu)

    print("\n".join(lines))
    print("STATUS:", status)

    if args.snapshot:
        stamp = time.strftime("%H:%M", time.localtime(now))
        note = f"Claude's check at {stamp}: {status}."
        load_dashboard().build_snapshot(name, args.snapshot, note)
        print("snapshot written to", args.snapshot)
    raise SystemExit(1 if status.startswith("WARN") else 0)


if __name__ == "__main__":
    main()

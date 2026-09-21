"""Run this BEFORE pausing or deleting the instance. Verifies against GitHub and Hugging Face themselves, not local markers.

    python scripts/pre_pause_check.py [--fix]

Checks: nothing uncommitted or unpushed in git; every run's newest local checkpoint is on the HF model repo; the processed
dataset is on the HF dataset repo; the repo and its data/weights links live under /home (the only path that survives a
pause on JarvisLabs); no training process is still running. Prints `SAFE TO PAUSE` or `NOT SAFE: ...` (exit 1).
--fix also pushes committed code and uploads any newest checkpoint that is missing from HF.
"""

import argparse
import re
import subprocess
import sys

from huggingface_hub import HfApi

from bhojpuri_tts import REPO_ROOT

MODEL_REPO = "gentleman101/bhojpuri-f5-tts"
DATA_REPO = "gentleman101/bhojpuri-syspin-24k"
EXPECTED_WAVS = 53155


def sh(*cmd) -> str:
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()
    problems = []

    def check(ok: bool, good: str, bad: str):
        print(("PASS  " if ok else "FAIL  ") + (good if ok else bad))
        if not ok:
            problems.append(bad)

    # git
    dirty = sh("git", "status", "--porcelain")
    check(not dirty, "git: working tree clean", f"git: uncommitted changes:\n{dirty}")
    subprocess.run(["git", "fetch", "-q"], cwd=REPO_ROOT)
    if args.fix and sh("git", "rev-list", "--count", "@{u}..HEAD") not in ("", "0"):
        subprocess.run(["git", "push", "-q"], cwd=REPO_ROOT)
    ahead = sh("git", "rev-list", "--count", "@{u}..HEAD")
    check(ahead == "0", "git: everything pushed to GitHub", f"git: {ahead} commit(s) not pushed (run: git push)")

    # checkpoints
    api = HfApi()
    files = set(api.list_repo_files(MODEL_REPO, repo_type="model"))
    for latest in sorted((REPO_ROOT / "runs").glob("*/checkpoints/latest")):
        run, step = latest.parent.parent.name, latest.read_text().strip()
        key = f"{run}/checkpoints/{step}/training_state.pt"
        if key not in files and args.fix:
            subprocess.run([sys.executable, "scripts/push_checkpoints.py", "--run", f"runs/{run}", "--repo", MODEL_REPO, "--once"],
                           cwd=REPO_ROOT)
            files = set(api.list_repo_files(MODEL_REPO, repo_type="model"))
        check(key in files, f"HF: {run} newest checkpoint {step} is on {MODEL_REPO}", f"HF: {run}/{step} is NOT on {MODEL_REPO}")
        metrics = f"{run}/metrics.jsonl"
        check(metrics in files, f"HF: {run} metrics present", f"HF: {metrics} missing")

    # dataset
    dfiles = api.list_repo_files(DATA_REPO, repo_type="dataset")
    shards = [f for f in dfiles if re.fullmatch(r".*\.tar", f)]
    check("index.json" in dfiles and len(shards) >= 30, f"HF: dataset repo has {len(shards)} shards + index.json ({DATA_REPO})",
          f"HF: dataset {DATA_REPO} incomplete ({len(shards)} shards)")

    # location: only /home survives
    for label, path in (("repo", REPO_ROOT), ("weights", REPO_ROOT / "checkpoints"), ("processed clips", REPO_ROOT / "data" / "processed")):
        real = path.resolve()
        check(str(real).startswith("/home/"), f"location: {label} is at {real} (persistent)",
              f"location: {label} is at {real}, which does NOT survive a pause — move it under /home")

    # running work
    running = subprocess.run(["ps", "-eo", "comm=,args="], capture_output=True, text=True).stdout
    busy = [line for line in running.splitlines() if line.split(None, 1)[0].startswith("python") and "scripts/train_lora.py" in line]
    check(not busy, "no training process running", "a training run is still in progress — pausing now would kill it")

    print("\nSAFE TO PAUSE" if not problems else f"\nNOT SAFE: {len(problems)} problem(s) above")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

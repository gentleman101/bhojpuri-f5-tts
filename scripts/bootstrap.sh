#!/usr/bin/env bash
# Rebuild everything on a fresh box, after the repo has been cloned into /home/bhojpuri-f5-tts. Idempotent: every step
# skips what is already in place, so it is also a health check.
#
#   ./scripts/bootstrap.sh
#
# Needs (manual, once): the GitHub deploy key already used for the clone, and `hf auth login` with a WRITE token
# (the weights are gated; the dataset and checkpoint repos are private). Everything else comes back by itself:
#   code -> GitHub, weights -> ai4bharat/IndicF5, clips -> gentleman101/bhojpuri-syspin-24k, runs -> scripts/pull_run.py
set -euo pipefail

export HOME=/home
cd "$(dirname "$0")/.."
ROOT=$PWD
case "$(pwd -P)" in
  /home/*) ;;
  *) echo "The repo must live under /home — only /home survives a JarvisLabs pause (found $(pwd -P))." >&2; exit 1 ;;
esac

step() { echo; echo "==> $*"; }

step "persistent folders and links"
mkdir -p /home/assets/checkpoints /home/assets/data/processed data
[ -e checkpoints ] || ln -s /home/assets/checkpoints checkpoints
[ -e data/processed ] || ln -s /home/assets/data/processed data/processed
grep -qx '/checkpoints' .git/info/exclude 2>/dev/null || echo '/checkpoints' >> .git/info/exclude

step "python environment"
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import torch,f5_tts" 2>/dev/null; then
  UV=$(command -v uv || true)
  [ -n "$UV" ] || { curl -LsSf https://astral.sh/uv/install.sh | sh; UV="$HOME/.local/bin/uv"; }
  "$UV" venv --python 3.10 .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  [ -d third_party/IndicF5 ] || { mkdir -p third_party && git clone -q https://github.com/AI4Bharat/IndicF5.git third_party/IndicF5; }
  # CUDA torch FIRST: requirements-lock.txt lists plain torch==2.5.1, which would pull the CPU wheel
  "$UV" pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  "$UV" pip install -r requirements-lock.txt
  "$UV" pip install -e third_party/IndicF5 --no-deps
  "$UV" pip install -e . --no-deps
else
  source .venv/bin/activate
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

step "IndicF5 weights"
if [ -f checkpoints/IndicF5/model.safetensors ]; then echo "present"; else
  hf download ai4bharat/IndicF5 --local-dir checkpoints/IndicF5
fi

step "processed clips (53,155 expected)"
N=$(find data/processed/syspin_24k -name '*.wav' 2>/dev/null | wc -l)
if [ "$N" = "53155" ]; then echo "present ($N)"; else
  echo "found $N — restoring from the private HF dataset repo"
  python scripts/restore_data.py
fi

step "manifests unchanged vs git"
git status --short manifests | tee /dev/stderr | grep -q . && { echo "manifests differ from git — investigate before training" >&2; exit 1; } || echo "clean"

echo
echo "Bootstrap OK. Resume a run with:  python scripts/pull_run.py <run>   then   ./scripts/run_training.sh runs/<run>/config.yaml --resume latest"
echo "Before any pause:                 python scripts/pre_pause_check.py --fix"

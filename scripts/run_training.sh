#!/usr/bin/env bash
# Start a training run in tmux so it survives SSH disconnects: trainer, dashboard, HF checkpoint uploader.
#
#   ./scripts/run_training.sh configs/lora_10h.yaml [--resume latest]
#
# Attach:  tmux attach -t train      (Ctrl-b then d to detach; Ctrl-b n / p to switch windows)
# Stop:    tmux kill-session -t train
# Env:     HF_REPO (default gentleman101/bhojpuri-f5-tts, private), DASH_PORT (default 8080), SESSION (default train)
set -euo pipefail

if [ $# -lt 1 ]; then sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 1; fi
CONFIG="$1"; shift
EXTRA="$*"

cd "$(dirname "$0")/.."
ROOT="$PWD"
SESSION="${SESSION:-train}"
HF_REPO="${HF_REPO:-gentleman101/bhojpuri-f5-tts}"
DASH_PORT="${DASH_PORT:-8080}"

[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session '$SESSION' already exists — attach with: tmux attach -t $SESSION" >&2; exit 1; }
RUN_DIR=$(.venv/bin/python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CONFIG")
mkdir -p "$RUN_DIR"

ENV="cd $ROOT && source .venv/bin/activate"
tmux new-session -d -s "$SESSION" -n trainer \
  "$ENV && python scripts/train_lora.py --config $CONFIG $EXTRA 2>&1 | tee -a $RUN_DIR/train.log; echo '[trainer exited — press enter]'; read"
tmux new-window -t "$SESSION" -n dashboard "$ENV && python scripts/dashboard.py --port $DASH_PORT"
tmux new-window -t "$SESSION" -n uploader \
  "$ENV && python scripts/push_checkpoints.py --run $RUN_DIR --repo $HF_REPO 2>&1 | tee -a $RUN_DIR/push.log; echo '[uploader exited — press enter]'; read"
tmux new-window -t "$SESSION" -n shell "$ENV"
tmux select-window -t "$SESSION:trainer"

echo "Started tmux session '$SESSION' (trainer, dashboard, uploader, shell) for $CONFIG"
echo "  attach:    tmux attach -t $SESSION"
echo "  dashboard: http://localhost:$DASH_PORT  (ssh -L $DASH_PORT:localhost:$DASH_PORT <box>)"
echo "  uploads:   $HF_REPO (private)"

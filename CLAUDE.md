# Bhojpuri F5-TTS — working notes

LoRA fine-tune of IndicF5 (F5-TTS, 337M) for Bhojpuri, then distill to a smaller student.
Read `PROJECT_PLAN.md` for frozen decisions and the stop/go gate; `docs/LEARNING_GUIDE.md` explains the concepts.

## First thing on a GPU machine

The venv was built with **CPU PyTorch**. Reinstall the CUDA build before training:

```bash
cd ~/bhojpuri-f5-tts
source .venv/bin/activate
uv pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `.venv` is missing entirely (fresh disk), rebuild it. `requirements-lock.txt` pins the exact
versions that were verified working on CPU:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/bhojpuri-f5-tts
uv venv --python 3.10 .venv && source .venv/bin/activate
git clone https://github.com/AI4Bharat/IndicF5.git third_party/IndicF5
uv pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements-lock.txt
uv pip install -e third_party/IndicF5 --no-deps && uv pip install -e . --no-deps
```

Install torch from the CUDA index *first*; the lock file lists plain `torch==2.5.1`, which would
otherwise pull the CPU wheel from PyPI.

Target machine is an A100 40GB with 16 cores, so configs ship with `max_frames_per_batch: 38400` and `num_workers: 8`. Halve the batch on a 24 GB card. `mixed_precision: bf16` needs Ampere or newer (use `fp16` on V100/T4).

## State as of 2026-09-19

Done: data downloaded and prepared, diagnostics built, weights verified, CPU baseline generated, all code tested end to end on CPU and on the A100 (probe, overfit check, uploader, resume-from-HF, monitoring rehearsal). No real training run has started yet.

| Thing | Where | Notes |
|---|---|---|
| Raw SYSPIN corpus | `data/syspin/` | 48 GB, gitignored |
| Processed 24 kHz clips | `data/processed/syspin_24k/` | 15 GB, gitignored |
| IndicF5 weights | `checkpoints/IndicF5/` | 1.4 GB, gitignored, gated download |
| Manifests | `manifests/syspin_{slice,10h,full}/` | in git — 1.9h / 9.4h / 90.7h train |
| Diagnostics | `manifests/diagnostics.json` | 32 held-out sentences, 8 contrasts |
| Stock baseline audio | `runs/eval/baseline_stock/` | 32 diagnostic clips, GPU run 2026-09-19 (2 min), gitignored |

## Fresh-machine bootstrap (empty disk)

Code comes back from git; data, weights and credentials do not. Run in this order — steps 1–3 need the user.
Paths here use `~`; on the GPU box that is `/root`, not `/home/ubuntu`. The code derives its own paths, so
the repo works from any location.

**0. Clear any partial copy first.** An aborted scp leaves truncated files that look valid — a half-written
WAV still opens, and a partly-copied `data/` silently trains on fewer clips. Inspect, then delete:

```bash
du -sh ~/bhojpuri-f5-tts/* 2>/dev/null          # what actually landed
find ~ -maxdepth 3 -name 'bhojpuri*' -o -maxdepth 3 -name 'syspin*' 2>/dev/null
rm -rf ~/bhojpuri-f5-tts                        # only if it holds nothing but a partial copy
```

Never keep a partial `data/processed/`: `find data/processed -name '*.wav' | wc -l` must equal **53155**.
Anything less means clips are missing, and the manifests reference files that are not there.

1. **Repo access.** The old SSH deploy key is gone. Make a new one and add it at GitHub → repo → Settings → Deploy keys (tick *Allow write access*):
   ```bash
   ssh-keygen -t ed25519 -N "" -C "bhojpuri-f5-tts-deploy" -f ~/.ssh/bhojpuri_f5_tts_deploy
   cat ~/.ssh/bhojpuri_f5_tts_deploy.pub
   printf '\nHost github-bhojpuri-f5-tts\n  HostName github.com\n  User git\n  IdentityFile ~/.ssh/bhojpuri_f5_tts_deploy\n  IdentitiesOnly yes\n' >> ~/.ssh/config
   git clone git@github-bhojpuri-f5-tts:gentleman101/bhojpuri-f5-tts.git ~/bhojpuri-f5-tts
   git -C ~/bhojpuri-f5-tts config user.name "Naman"
   git -C ~/bhojpuri-f5-tts config user.email "50843800+gentleman101@users.noreply.github.com"
   ```
2. **Weights** (gated; terms already accepted on the account): `hf auth login` in a real terminal, then
   `hf download ai4bharat/IndicF5 --local-dir checkpoints/IndicF5`.
3. **Corpus** — links are personal, time-limited and not in git. Either reuse the two wget URLs from the
   SYSPIN email, or request fresh ones at https://spiredatasets.ee.iisc.ac.in/syspincorpus (Bhojpuri,
   Female + Male, Human Checked). Then:
   ```bash
   ./scripts/download_syspin.sh "<female_url>" "<male_url>"     # ~10 min at 200 MB/s
   ```
4. **Rebuild the three manifests** (~25 min, mostly resampling 53k clips):
   ```bash
   python scripts/prepare_syspin.py --name syspin_full  --exclude-file manifests/diagnostics_exclude.txt
   python scripts/prepare_syspin.py --name syspin_10h   --exclude-file manifests/diagnostics_exclude.txt --train-hours-per-speaker 5
   python scripts/prepare_syspin.py --name syspin_slice --exclude-file manifests/diagnostics_exclude.txt --train-hours-per-speaker 1
   ```
   Splits are seeded, so these reproduce the manifests already in git. Confirm with `git status` — the
   manifest files should come back unchanged. **`manifests/diagnostics.json` is already in git; do not
   rebuild it**, or the held-out sentences change and the baseline stops being comparable.
5. **Disk**: needs ~70 GB for raw + processed + weights. Check before downloading.

## Next steps (in order)

1. ~~Probe~~ done: 0.44 s/update, 27.9 GB peak at 38,400 frames — keep the batch size.
2. ~~Sanity~~ done: overfit-one-batch 200 falls 0.75 to 0.60.
3. **Slice run** on `configs/lora_slice.yaml` — pipeline check only; 1.9h is below the quality cliff.
4. **10h sweep**: baseline, then `extra_trainable: text_embed`, then rank 64/16, then lr variants.
5. **Stop/go gate**: `scripts/eval_diagnostics.py` vs the stock baseline. Only scale to 90.7h if it improves.

## Commands

```bash
./scripts/run_training.sh configs/lora_10h.yaml [--resume latest]   # tmux: trainer + dashboard + HF uploader (attach: tmux attach -t train)
python scripts/train_lora.py --config configs/lora_slice.yaml [--resume latest]
python scripts/push_checkpoints.py --run runs/<run> --repo gentleman101/bhojpuri-f5-tts   # private HF repo; full resumable checkpoints
python scripts/check_training.py [--snapshot out.html]   # health check: exit 1 on WARN. Run on each monitoring tick, then republish the artifact
python scripts/dashboard.py --port 8080      # live loss/ETA/GPU page; ssh -L 8080:localhost:8080 <box>; reads runs/*/metrics.jsonl
python scripts/eval_diagnostics.py --name baseline_stock                 # stock model
python scripts/eval_diagnostics.py --name lora_10h --adapter runs/<run>/checkpoints/step_0002000
python scripts/infer.py --ref-audio X.wav --ref-text "..." --text "..." --out out.wav [--adapter DIR]
python scripts/check_vocab.py --vocab checkpoints/IndicF5/checkpoints/vocab.txt --manifests manifests/syspin_full/train.csv
```

## Links and facts worth not losing

- Repo: `git@github-bhojpuri-f5-tts:gentleman101/bhojpuri-f5-tts.git` (private).
- Weights: https://huggingface.co/ai4bharat/IndicF5 (gated, terms already accepted on the account).
- Corpus request form: https://spiredatasets.ee.iisc.ac.in/syspincorpus — Bhojpuri, Female + Male, Human Checked. Links emailed, valid 7 days. The batch fetched on 2026-09-17 expires 2026-09-24.
- Listening-test page (private artifact): https://claude.ai/artifact/NWutZVsd8xYoHVqQKZbUoC — real recording vs. stock model, shared with native speakers for feedback. Needs link sharing enabled from its share menu.
- Base-model paper (IN-F5 = IndicF5): https://arxiv.org/abs/2505.20693 — data ladder, Bhojpuri zero-resource result (MUSHRA 82 from 1h synthetic), and their hyperparameters.
- `docs/decisions-memory.md` mirrors the cross-session memory note. On a new machine, copy it to
  `~/.claude/projects/-root/memory/project_bhojpuri_tts.md` and add a one-line pointer in that folder's `MEMORY.md`.

Verified facts (don't re-derive): IndicF5 is 337,096,804 params; its checkpoint uses the prefix
`ema_model._orig_mod.`; its vocab is 2,545 entries and covers **every** Bhojpuri character in SYSPIN;
the vocoder bundled in the checkpoint is bit-identical to stock `charactr/vocos-mel-24khz`.
LoRA r=32 on 132 layers = 10,092,544 trainable params (2.99%). CPU inference ran at RTF ~29.

## Measured on the A100 (2026-09-19)

Slice config, real batches (38,400 frames): **0.44 s/update**, **27.9 GB peak VRAM** (of 40), 155 MB per checkpoint
(trainable 40 + EMA 40 + optimizer 81). 200-step overfit check passes (loss 0.75 to 0.60). Uploader and resume-from-HF both
verified end to end. Monitoring artifact: https://claude.ai/artifact/8fz6hmA6o2bBS2xn3GQwyL — regenerate with
`check_training.py --snapshot F` and republish F to that URL.

## Gotchas

- `check_training.py` detects a dead trainer/uploader from the python processes, not tmux's `pane_current_command` (reports `bash` even while running).
- `push_checkpoints.py` polls every 120 s and uploads only the newest checkpoint, so a very short run can finish before its first upload.

- IndicF5's tokenizer maps unknown characters to index 0, which is **space** — silently. All current Bhojpuri text is covered (checked), but re-run `check_vocab.py` after any text change.
- Mel settings live once in `bhojpuri_tts/data.py` (`MEL_KWARGS`). Training and inference must use identical values or output is garbage.
- The checkpoint bundles a Vocos vocoder identical to stock `charactr/vocos-mel-24khz` (verified, max diff 0.0), so loading Vocos separately is correct.
- Diagnostic clips are pinned out of training via `manifests/diagnostics_exclude.txt`. If you rebuild diagnostics, re-run `prepare_syspin.py` with that file and re-verify no leakage.
- Flow-matching loss is noisy and plateaus early. Judge checkpoints by listening and by the diagnostic set, not by loss.
- Adapter checkpoints are small (tens of MB); only weights/data are large. Push code often — if the JarvisLabs wallet hits $0 the disk is wiped.

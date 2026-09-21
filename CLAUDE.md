# Bhojpuri F5-TTS — working notes

LoRA fine-tune of IndicF5 (F5-TTS, 337M) for Bhojpuri, then distill to a smaller student.
Read `PROJECT_PLAN.md` for frozen decisions and the stop/go gate; `docs/LEARNING_GUIDE.md` explains the concepts.

## First thing on a GPU machine

Run `./scripts/bootstrap.sh` (idempotent; see the recovery runbook below). It builds `.venv` with **CUDA** PyTorch 2.5.1 installed *before*
`requirements-lock.txt` (the lock lists plain `torch==2.5.1`, which would otherwise pull the CPU wheel), fetches the weights and clips,
and prints `torch ... cuda True <GPU name>` when it is ready. Check that line before training.

Target machine is an A100 40GB with 16 cores, so configs ship with `max_frames_per_batch: 38400` and `num_workers: 8`. Halve the batch on a 24 GB card. `mixed_precision: bf16` needs Ampere or newer (use `fp16` on V100/T4).

## Where things live — READ THIS (lost everything once on 2026-09-21)

**On JarvisLabs only `/home` survives a pause/resume.** `/` (including `/root`) is a container layer that is reset to a
fresh image. On 2026-09-19 the repo, venv, corpus, processed clips and weights all sat in `/root/bhojpuri-f5-tts` and were
wiped by a pause/resume; only GitHub and the Hugging Face checkpoints survived. Everything now lives under `/home`:

| Path | What | Persists? |
|---|---|---|
| `/home/bhojpuri-f5-tts/` | repo + `.venv` (5.8 GB) | yes |
| `/home/assets/checkpoints/` | IndicF5 weights (repo `checkpoints` is a symlink to it, hidden via `.git/info/exclude`) | yes |
| `/home/assets/data/processed/` | 15 GB processed clips (repo `data/processed` symlink) | yes |
| `/root/syspin_raw/` | raw corpus, 48 GB (repo `data/syspin` symlink) — only needed to rebuild processed clips | **no, on purpose** |
| `/home/assets/*.sh`, `logs/` | rebuild scripts (`dl_corpus.sh`, `build_env.sh`, `prep_full.sh`) | yes |

Note that `/home` is 100 GB. Git uses the deploy key `/home/.ssh/bhojpuri_f5_tts_deploy` via the repo's own `core.sshCommand`
(remote is plain `git@github.com:`), so there is no dependence on `/root/.ssh/config`. Hugging Face token: `/home/.cache/huggingface`.
Rebuilding from nothing takes ~20 min now: corpus download (~4 min) + extract (~4 min) + weights (seconds) + venv (~3 min) +
`prepare_syspin.py` for all three manifests (~1 min with one torch thread per worker).

**Full history, results, incident report, open decisions and the post-resume checklist: `docs/PROGRESS_LOG.md` (read it first after any pause).**

## State as of 2026-09-21

Done: data rebuilt and verified (53,155 WAVs, manifests byte-identical to git), diagnostics built, weights verified, stock
baseline regenerated, slice run finished (3000 updates, val 0.6194; checkpoints on HF), slice adapter beat stock on pitch/spectrum
(`scripts/compare_eval.py`). Not yet done: native-speaker listening, the 10h run (compile on, ~40 min), the stop/go gate. Instance was paused right after this state; nothing running.

| Thing | Where | Notes |
|---|---|---|
| Raw SYSPIN corpus | `data/syspin/` → `/root/syspin_raw` | 48 GB, gitignored, ephemeral |
| Processed 24 kHz clips | `data/processed/syspin_24k/` → `/home/assets/data/processed` | 15 GB, gitignored |
| IndicF5 weights | `checkpoints/IndicF5/` → `/home/assets/checkpoints` | 1.4 GB, gitignored, gated download |
| Manifests | `manifests/syspin_{slice,10h,full}/` | in git — 1.9h / 9.4h / 90.7h train |
| Diagnostics | `manifests/diagnostics.json` | 32 held-out sentences, 8 contrasts |
| Stock baseline audio | `runs/eval/baseline_stock/` | 32 diagnostic clips, GPU run 2026-09-19 (2 min), gitignored |

## Backup policy — what is pushed where, and how often

Rule: **nothing exists only on this machine.** Anything on `/` (including `/root`) is lost on pause/resume; a wallet at $0 wipes
everything. Four homes, each with a cadence:

| What | Home | Cadence | Automated by |
|---|---|---|---|
| Code, configs, manifests, docs | GitHub `gentleman101/bhojpuri-f5-tts` | **Commit + push after every meaningful change**, always before starting or ending a run, never more than ~30 min of uncommitted work | you/Claude (`git push`); `pre_pause_check.py` fails on unpushed work |
| Training checkpoints (trainable + EMA + optimizer, ~155 MB each) + `metrics.jsonl`, `config.yaml`, `adapter_config.json`, samples | HF **model** repo `gentleman101/bhojpuri-f5-tts` (private), under `<run>/` | Trainer saves every `save_every` updates (1000 = ~7 min on the 10h config, 500 on slice); uploader polls every **60 s**, so at most ~8 min of training is ever unbacked-up. All checkpoints are kept on HF; local disk keeps the last `keep_last` (4) | `scripts/push_checkpoints.py` (started by `run_training.sh`); `check_training.py` warns if a checkpoint is >15 min un-uploaded |
| Processed 24 kHz clips (53,155 WAVs, 14.8 GiB, 37 tar shards + `index.json` with sha256) | HF **dataset** repo `gentleman101/bhojpuri-syspin-24k` (private, CC-BY-4.0 attribution to IISc in its card) | Pushed once (2026-09-21); immutable. Re-pack and re-push (`scripts/pack_data.py`, then upload `/home/assets/shards`) **only if the manifests or the processing change** | manual; restore verified byte-identical on 2026-09-21 |
| Eval outputs (`runs/eval/*`) | nowhere — regenerable in ~2 min each | — | — |

**Before ANY pause, resume, delete or handover: `python scripts/pre_pause_check.py --fix`.** It checks GitHub and Hugging Face
themselves (not local markers), pushes what is missing, and prints `SAFE TO PAUSE` or `NOT SAFE`. Do not pause on `NOT SAFE`.

## Recovery runbook — regain everything on an empty machine

`/home` survives a pause, so first look there (`ls /home/bhojpuri-f5-tts /home/assets`). If it is empty (new instance, or `/home` lost):

1. **GitHub access — the only step that needs the user.** Make a key, add its *public* half at GitHub → repo → Settings → Deploy keys
   (tick *Allow write access*), then clone **into `/home`** and pin the key to the repo (no `~/.ssh/config`, that lives in the wiped `/root`):
   ```bash
   export HOME=/home; mkdir -p /home/.ssh && chmod 700 /home/.ssh
   ssh-keygen -t ed25519 -N "" -C "bhojpuri-f5-tts-deploy" -f /home/.ssh/bhojpuri_f5_tts_deploy
   cat /home/.ssh/bhojpuri_f5_tts_deploy.pub          # add this at GitHub; if copying from a terminal fails, SendUserFile it or paste from an artifact page
   SSHC="ssh -i /home/.ssh/bhojpuri_f5_tts_deploy -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/.ssh/known_hosts -o StrictHostKeyChecking=accept-new"
   GIT_SSH_COMMAND="$SSHC" git clone git@github.com:gentleman101/bhojpuri-f5-tts.git /home/bhojpuri-f5-tts
   cd /home/bhojpuri-f5-tts && git config core.sshCommand "$SSHC" && git config user.name "Naman" && git config user.email "50843800+gentleman101@users.noreply.github.com"
   ```
   Never use `gh auth login` or a broad token for this; the deploy key is the scoped credential.
2. **Hugging Face token (user, once):** `hf auth login` in a real terminal with a **write** token (fine-grained: write on the two repos is enough).
   The token is stored under `/home/.cache/huggingface`, so it normally survives.
3. **`./scripts/bootstrap.sh`** — idempotent. Builds `.venv` (CUDA torch first), downloads the IndicF5 weights, restores the 53,155 clips from the HF dataset
   repo (sha256-verified, ~2 min), and checks the manifests match git. About 10 minutes in total; it needs no SYSPIN download links.
4. **Resume a run:** `python scripts/pull_run.py <run>` (newest checkpoint, ~12 s), then
   `./scripts/run_training.sh runs/<run>/config.yaml --resume latest`.
5. **Monitoring** (optional): re-create the cron/loop that runs `check_training.py --snapshot` and republishes the artifact.

**Last resort — the HF dataset repo is also gone.** Rebuild from the raw corpus (needs fresh links; the 2026-09-17 batch expires 2026-09-24):
raw archives go on `/root` (fast, disposable, only needed to make the clips), the processed clips on `/home`:
```bash
./scripts/download_syspin.sh "<female_url>" "<male_url>"      # ~4 min download + ~4 min extract at ~200 MB/s; run it with data/syspin -> /root/syspin_raw
python scripts/prepare_syspin.py --name syspin_slice --exclude-file manifests/diagnostics_exclude.txt --train-hours-per-speaker 1
python scripts/prepare_syspin.py --name syspin_10h   --exclude-file manifests/diagnostics_exclude.txt --train-hours-per-speaker 5
python scripts/prepare_syspin.py --name syspin_full  --exclude-file manifests/diagnostics_exclude.txt      # ~20 s total; one torch thread per worker
```
Splits are seeded, so `git status manifests` must come back clean. **`manifests/diagnostics.json` is in git; never rebuild it**, or the held-out sentences
change and the baseline stops being comparable. `find data/processed -name '*.wav' | wc -l` must equal **53155** (never keep a partial copy — an aborted scp
leaves truncated WAVs that still open). Then run `scripts/pack_data.py` and re-upload so the dataset backup exists again.

## Next steps (in order)

1. ~~Probe~~ done: 0.44 s/update, 27.9 GB peak at 38,400 frames — keep the batch size.
2. ~~Sanity~~ done: overfit-one-batch 200 falls 0.75 to 0.60.
3. ~~Slice run~~ done (3000 updates, val 0.6194) — pipeline check only; 1.9h is below the quality cliff.
4. **10h sweep**: baseline, then `extra_trainable: text_embed`, then rank 64/16, then lr variants.
5. **Stop/go gate**: `scripts/eval_diagnostics.py` vs the stock baseline. Only scale to 90.7h if it improves.

## Commands

```bash
./scripts/run_training.sh configs/lora_10h.yaml [--resume latest]   # tmux: trainer + dashboard + HF uploader (attach: tmux attach -t train)
python scripts/train_lora.py --config configs/lora_slice.yaml [--resume latest]
python scripts/push_checkpoints.py --run runs/<run> --repo gentleman101/bhojpuri-f5-tts [--once]   # private HF repo; full resumable checkpoints; polls every 60 s
python scripts/pre_pause_check.py [--fix]   # RUN BEFORE ANY PAUSE: verifies GitHub + HF have everything
python scripts/pull_run.py <run>            # newest HF checkpoint back into runs/<run>, ready for --resume latest
./scripts/bootstrap.sh                      # idempotent full rebuild/health check of the environment, weights and data
python scripts/restore_data.py              # 53,155 clips back from the HF dataset repo
python scripts/pack_data.py                 # re-pack clips into HF-ready tar shards
python scripts/compare_eval.py baseline_stock lora_slice   # pitch/spectrum vs real recordings
python scripts/check_training.py [--snapshot out.html]   # health check: exit 1 on WARN. Run on each monitoring tick, then republish the artifact
python scripts/dashboard.py --port 8080      # live loss/ETA/GPU page; ssh -L 8080:localhost:8080 <box>; reads runs/*/metrics.jsonl
python scripts/eval_diagnostics.py --name baseline_stock                 # stock model
python scripts/eval_diagnostics.py --name lora_10h --adapter runs/<run>/checkpoints/step_0002000
python scripts/infer.py --ref-audio X.wav --ref-text "..." --text "..." --out out.wav [--adapter DIR]
python scripts/check_vocab.py --vocab checkpoints/IndicF5/checkpoints/vocab.txt --manifests manifests/syspin_full/train.csv
```

## Links and facts worth not losing

- Repo: `git@github.com:gentleman101/bhojpuri-f5-tts.git` (private; deploy key pinned via the repo's `core.sshCommand`, see the runbook).
- HF model repo (checkpoints): https://huggingface.co/gentleman101/bhojpuri-f5-tts (private). HF dataset repo (clips): https://huggingface.co/datasets/gentleman101/bhojpuri-syspin-24k (private).
- Weights: https://huggingface.co/ai4bharat/IndicF5 (gated, terms already accepted on the account).
- Corpus request form: https://spiredatasets.ee.iisc.ac.in/syspincorpus — Bhojpuri, Female + Male, Human Checked. Links emailed, valid 7 days. The batch fetched on 2026-09-17 expires 2026-09-24.
- Listening-test page (private artifact): https://claude.ai/artifact/NWutZVsd8xYoHVqQKZbUoC — real recording vs. stock model, shared with native speakers for feedback. Needs link sharing enabled from its share menu.
- Base-model paper (IN-F5 = IndicF5): https://arxiv.org/abs/2505.20693 — data ladder, Bhojpuri zero-resource result (MUSHRA 82 from 1h synthetic), and their hyperparameters.
- `docs/decisions-memory.md` mirrors the cross-session memory note. On a new machine, copy it to the project's Claude memory folder
  (`/home/.claude/projects/<project-dir>/memory/project_bhojpuri_tts.md`; the folder name follows the directory Claude is started in) and add a one-line pointer in that folder's `MEMORY.md`.

Verified facts (don't re-derive): IndicF5 is 337,096,804 params; its checkpoint uses the prefix
`ema_model._orig_mod.`; its vocab is 2,545 entries and covers **every** Bhojpuri character in SYSPIN;
the vocoder bundled in the checkpoint is bit-identical to stock `charactr/vocos-mel-24khz`.
LoRA r=32 on 132 layers = 10,092,544 trainable params (2.99%). CPU inference ran at RTF ~29.

## Speed: torch.compile (2026-09-21)

`perf: {compile: true}` in a config compiles only the transformer's forward (parameter names, checkpoints and EMA are unchanged).
Measured on the slice config: **0.244 s/update vs 0.43 eager (1.8x)**; validation loss at update 500 is the same (0.6744 vs 0.6749; 0.6876 vs 0.6894 at 250).
Cost: ~6 min one-off per run (76 s first step + two recompiles around updates ~50 and ~300), and a persistent Inductor cache did **not** remove it, so it
pays off for runs of more than ~2,000 updates. `configs/lora_10h.yaml` has it on. No gain from TF32/cuDNN-benchmark (already bf16). The GPU is compute-bound
(96-99% util), so data loading is not the bottleneck and running two jobs at once would not help. Estimated: 10h run ~40 min instead of ~57.

## Measured on the A100 (2026-09-19)

Slice config, real batches (38,400 frames): **0.44 s/update**, **27.9 GB peak VRAM** (of 40), 155 MB per checkpoint
(trainable 40 + EMA 40 + optimizer 81). 200-step overfit check passes (loss 0.75 to 0.60). Uploader and resume-from-HF both
verified end to end. Monitoring artifact: https://claude.ai/artifact/8fz6hmA6o2bBS2xn3GQwyL — regenerate with
`check_training.py --snapshot F` and republish F to that URL.

## Gotchas

- `check_training.py` detects a dead trainer/uploader from the python processes, not tmux's `pane_current_command` (reports `bash` even while running).
- `push_checkpoints.py` polls every 60 s and uploads only the newest checkpoint, so a very short run can finish before its first upload; `pre_pause_check.py --fix` catches that.
- `prepare_syspin.py` workers must use one torch thread each (now built in). Without it, many workers x 16 threads each thrash the CPU (28 clips/s vs ~2,500).
- `pkill -f <pattern>` from a tool call matches its own command line and kills the shell — kill by PID instead.
- Killing a training run leaves orphaned `pt_data_worker` processes holding GPU memory; kill them too before starting the next run.

- IndicF5's tokenizer maps unknown characters to index 0, which is **space** — silently. All current Bhojpuri text is covered (checked), but re-run `check_vocab.py` after any text change.
- Mel settings live once in `bhojpuri_tts/data.py` (`MEL_KWARGS`). Training and inference must use identical values or output is garbage.
- The checkpoint bundles a Vocos vocoder identical to stock `charactr/vocos-mel-24khz` (verified, max diff 0.0), so loading Vocos separately is correct.
- Diagnostic clips are pinned out of training via `manifests/diagnostics_exclude.txt`. If you rebuild diagnostics, re-run `prepare_syspin.py` with that file and re-verify no leakage.
- Flow-matching loss is noisy and plateaus early. Judge checkpoints by listening and by the diagnostic set, not by loss.
- Adapter checkpoints are ~155 MB with optimizer state. If the JarvisLabs wallet hits $0 the disk is wiped; the backup policy above is what makes that survivable.

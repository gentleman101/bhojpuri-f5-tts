# Progress log — Bhojpuri F5-TTS

Chronological record of what was done, what was measured, what went wrong, and exactly what to do next. Newest state first.
Read this together with `CLAUDE.md` (layout, backup policy, recovery runbook) and `PROJECT_PLAN.md` (frozen decisions, stop/go gate).
Append to it; do not rewrite history.

---

## 1. State at the pause (2026-09-21, ~12:35 UTC)

**Trained so far: one run, on the smallest stage.** `lora_slice_r32`: LoRA r=32, 1.9 h slice (1,085 clips), 3,000 updates (~88 epochs),
~23 min active, val loss 0.6924 -> 0.6194. Nothing has trained on the 10 h or full data. All probe/benchmark runs were throwaway and deleted.

| Where | What | State |
|---|---|---|
| GitHub `gentleman101/bhojpuri-f5-tts` (private) | code, configs, manifests, docs, scripts | up to date at the last commit; `pre_pause_check.py` = SAFE |
| HF model repo `gentleman101/bhojpuri-f5-tts` (private) | `lora_slice_r32/` (6 checkpoints 500..3000 with optimizer state, metrics, config, samples) and `eval/` (stock baseline, slice adapter clips, comparison JSON) | verified |
| HF dataset repo `gentleman101/bhojpuri-syspin-24k` (private) | 53,155 processed clips, 37 tar shards (14.8 GiB) + `index.json` (sha256) | restore verified byte-identical |
| `/home/bhojpuri-f5-tts`, `/home/assets` | repo + `.venv`, weights, processed clips, scripts, logs | persistent volume (see incident below) |
| `/root/syspin_raw` | raw SYSPIN archives extracted (48 GB) | **disposable on purpose**, not backed up |

**Nothing is running.** No training, no dashboard, no uploader, no cron monitor. The GPU is idle.

**Not backed up anywhere (and how to get it back):** the raw corpus (re-request at https://spiredatasets.ee.iisc.ac.in/syspincorpus; the 2026-09-17 links expire
**2026-09-24**; not needed for training because the processed clips are on HF); `/home/assets/logs/*` (low value); the Claude session scratchpad under `/tmp`.
Credentials live only in `/home`: the GitHub deploy key `/home/.ssh/bhojpuri_f5_tts_deploy` (public half is registered on the repo) and the HF fine-grained token
(`/home/.cache/huggingface`). If `/home` is wiped they must be re-created by the user (see the runbook in `CLAUDE.md`).

## 2. First thing to do after resuming the instance

1. **Persistence test (this settles the incident).** Two marker files were written just before the pause:
   `cat /root/PAUSE_MARKER.txt /home/PAUSE_MARKER.txt`  (both say `Mon Sep 21 12:20:00 UTC 2026`).
   Record which one survived by appending a line to section 6 of this file. Expected (the vendor says so): `/home` survives, `/root` does not.
2. `ls /home/bhojpuri-f5-tts /home/assets`. If present: `cd /home/bhojpuri-f5-tts && ./scripts/bootstrap.sh` (idempotent health check, ~5 s when nothing is missing), then
   `python scripts/pre_pause_check.py` (expects SAFE). If `/home` is empty: follow the recovery runbook in `CLAUDE.md` (needs the user for the GitHub deploy key).
3. If `/home` survived but `data/syspin` points at the now-missing `/root/syspin_raw`, that is fine — nothing reads it after the clips exist.
4. Start Claude Code inside tmux (`tmux new -s claude`) before any long run; the training session (`run_training.sh`) is already its own tmux session.

## 3. Results so far

**Slice adapter (step 3,000) vs stock IndicF5**, 32 held-out diagnostic sentences, same reference clips, seed 0, nfe 32, cfg 2.0, sway -1.0.
Aligned to the real recording of the same sentence (DTW), `scripts/compare_eval.py`. Reproduced digit for digit on the rebuilt machine on 2026-09-21.

| Measure | Stock | Slice adapter |
|---|---|---|
| Pitch-contour correlation (higher better) | 0.502 | **0.569** |
| Pitch RMSE, semitones (lower better) | 3.759 | **3.319** |
| Pitch range ratio vs real (1.0 ideal) | 1.457 | **1.194** |
| Spectral (MFCC) distance (lower better) | 49.355 | **47.027** |

Paired bootstrap over the 32 clips: all four improvements have 95% intervals excluding zero (better on 69-84% of clips). Pitch correlation improves in 7 of 8 categories
(`conjunct` 0.471 -> 0.459 is the exception; `v_to_b` barely moves, 0.476 -> 0.507). Stock IndicF5 has ~46% more pitch variation than the real speakers; the adapter cuts that to ~19%.

**Caveats — read before over-trusting:** these are proxies for melody/spectrum, not pronunciation correctness; 32 clips, one seed, one real recording per sentence; the slice (1.9 h)
is below the data-size quality cliff. **The stop/go gate is NOT passed or failed yet** — it needs the 10 h run plus native-speaker listening. `eval_diagnostics.py`'s
"duration drift" is identical for every model (F5 sets output length from the text) and must be ignored.

**Measured on the A100 40GB (slice config, real batches of 38,400 frames):** 0.43-0.44 s/update eager, **0.244 s/update with `torch.compile`** (val loss at 500 updates 0.6744 vs 0.6749),
27.9 GB peak VRAM, GPU 96-99% busy, checkpoint 155 MB (trainable 40 + EMA 40 + optimizer 81). TF32/cuDNN-benchmark: no gain. Compile costs ~6 min one-off per run
(a persistent Inductor cache did not remove it). Preprocessing all 53,155 clips: ~20 s with one torch thread per worker (was ~25 min). HF upload ~25 MB/s (15 GB in ~10 min),
restore ~200 MB/s (15 GB in ~2 min).

## 4. Plan and estimates (not started)

Frozen plan in `PROJECT_PLAN.md`. Estimated with the compiled speed: 10 h baseline run 8,000 updates ~40 min; five more 10 h sweep variants (text_embed, rank 64, rank 16, two lr variants,
counts other than the first are assumptions) ~3 h; compile warm-up ~6 min x 6; diagnostics ~2 min each; **~4 h 10 m to the stop/go gate**, +~2 h for a full 90.7 h run of 30,000 updates
(assumed, no config exists yet) if the gate passes. Live version: the monitoring artifact (section 7).

**Open decisions (need the user):**
1. Start the 10 h run (`./scripts/run_training.sh configs/lora_10h.yaml`; compile is on).
2. Trim the sweep? Comparing variants at 4,000 instead of 8,000 updates would roughly halve it, but changes the protocol (rankings could differ at 8,000).
3. Native-speaker listening on `eval/baseline_stock` vs `eval/lora_slice` (on HF) — especially `v_to_b`, `sibilant`, `conjunct`. What listeners hear as Bhojpuri "tone" is unmeasured; research found no evidence
   of lexical tone (Trammell 1971: 4 pitch levels, 3 terminal contours; stress reported as phonemic, uncited). Ask them what differs (pitch, rhythm, vowel length, stress) before optimising a metric for it.
4. Full-run update budget (30,000 is a placeholder), and whether to keep the models private (the two SYSPIN voices are real people; consent is required for cloning uses).
5. JarvisLabs wallet balance (cannot be checked from the box).

## 5. Timeline

**2026-09-17** SYSPIN download links issued (7-day validity, expire 2026-09-24).

**2026-09-19** Audit of the GPU box: A100 40GB, code in git, weights present, corpus missing (only the female archive). Downloaded the male archive, extracted both, ran `prepare_syspin.py`
(53,155 clips). Regenerated `syspin_full` because the committed copy was inconsistent with the 10 h/slice manifests (different val/test); no diagnostic clip ever leaked into train.
Built: trainer `metrics.jsonl`, `dashboard.py`, `push_checkpoints.py`, `run_training.sh`, `check_training.py`, monitoring artifact. The HF token was read-only; the user replaced it with a fine-grained
write token; private model repo created. Rehearsals found and fixed: health check false alarms (tmux `pane_current_command` shows `bash`), launcher tmux target clash (`-t train` matched the window
`trainer`), shell window closing instantly, samples player loading 30 s late, HF progress bars flooding logs. Verified upload + resume-from-HF end to end. Slice run 13:36 -> ~14:00 (last checkpoint
uploaded 14:01):
3,000 updates, 6 checkpoints pushed, the monitor ticks I read reported OK, final val 0.6194. Diagnostics + pitch/spectrum comparison (section 3).

**2026-09-21 ~10:24 — INCIDENT.** After a pause/resume, `/root/bhojpuri-f5-tts` (repo, `.venv`, 48 GB corpus, 15 GB processed clips, weights, `runs/`) was gone; `/root` had reverted to its fresh
image (files dated May 5, recreated `/root/.ssh`, `.cache`). `/home` (a separate 100 GB volume, `/dev/rbd1`) survived. The other VM (which had held the original copy) had already been deleted.
A full-filesystem search found no corpus, WAV or weight file anywhere. Root cause, **confirmed by the vendor**: the JarvisLabs pause dialog states "Data stored outside the /home directory will be lost when the instance stops" (screenshot 2026-09-21). **Only `/home` persists**, and the project lived in `/root`. Paused storage is billed at $0.00014/GB/hour (100 GB is ~$0.34/day, so a $16.70 wallet covers ~50 days paused; at $0 everything is wiped).
No command in the shell history or of mine deleted it (my `rm -rf` calls were all on named `runs/*` subfolders). tmux cannot move files.

**2026-09-21 recovery (~1 h, mostly waiting on the GitHub key).** The deploy key was no longer accepted by GitHub; getting the public key out of a terminal failed (Mac + Windows keyboard), so it went
via a file card and a copy-box artifact page; the user added it. A `gh auth login` (broad token) was blocked by the harness and not used. Rebuilt under `/home`: clone (key pinned in the repo's `core.sshCommand`),
CUDA venv, weights, corpus re-downloaded (~3 min) and extracted (~4 min) to `/root/syspin_raw`, clips reprocessed. **Verified faithful:** 53,155 WAVs, all manifest rows resolve, the three manifests came out
byte-identical to git, and the slice comparison numbers reproduced exactly. Then: dataset pushed to HF and restore-tested; `pack_data.py`, `restore_data.py`, `pull_run.py`, `bootstrap.sh`, `pre_pause_check.py` written;
CLAUDE.md, PROJECT_PLAN.md and the memory note corrected; `torch.compile` evaluated and enabled for the 10 h config; eval outputs pushed to HF.

## 6. Persistence test result (fill in after resume)

- 2026-09-21 12:20 UTC markers written to `/root/PAUSE_MARKER.txt` and `/home/PAUSE_MARKER.txt`.
- After resume: `/root` marker: ____   `/home` marker: ____   (record here; also note whether `/home/bhojpuri-f5-tts` and `/home/assets` survived)

## 7. Where everything is

- GitHub: `git@github.com:gentleman101/bhojpuri-f5-tts.git` (deploy key `/home/.ssh/bhojpuri_f5_tts_deploy`, pinned per-repo).
- HF: model repo https://huggingface.co/gentleman101/bhojpuri-f5-tts, dataset repo https://huggingface.co/datasets/gentleman101/bhojpuri-syspin-24k (both private).
- Monitoring artifact (snapshot, republished by Claude): https://claude.ai/artifact/8fz6hmA6o2bBS2xn3GQwyL. Native-speaker listening page: https://claude.ai/artifact/NWutZVsd8xYoHVqQKZbUoC.
  Deploy-key copy page: https://claude.ai/artifact/SUGqTZfF3FBYrDD1xTUc4j (public key only).
- Claude memory notes: `/home/.claude/projects/-root-bhojpuri-f5-tts/memory/` (`jarvislabs-only-home-persists`, `user-cannot-copy-from-terminal`). `docs/decisions-memory.md` mirrors the project note.
- Environment: Python 3.10, torch 2.5.1+cu121, A100-PCIE-40GB, 16 cores, 112 GB RAM (the JarvisLabs console; `free` inside the container shows the host's 503 GB). `/home` is 100 GB (24 GB used).
- JarvisLabs instance `bhojpuri-gpu`, Machine ID 511241, region IN2, 1 x A100, type Container. Wallet was $16.70 on 2026-09-21 16:33 (phone clock).
- Scripts: `bootstrap.sh`, `pre_pause_check.py`, `pull_run.py`, `pack_data.py`, `restore_data.py`, `run_training.sh`, `push_checkpoints.py`, `check_training.py`, `dashboard.py`,
  `eval_diagnostics.py`, `compare_eval.py`, `prepare_syspin.py`, `train_lora.py`, `scripts/dev/*` (compile equivalence check, JS runtime test for the dashboard page).

## 8. Lessons (mistakes made — do not repeat)

1. **Keep everything under `/home`; run `pre_pause_check.py --fix` before any pause.** Never advise deleting a VM before its data is confirmed elsewhere (I told the user the old VM could be deleted; that was wrong).
2. `pkill -f <pattern>` inside a tool call matches its own command line and kills the shell; kill by PID. Killing a run leaves orphan `pt_data_worker` processes holding GPU memory.
3. Many process-pool workers x default torch threads thrash the CPU (14 x 16 threads: 28 clips/s vs ~2,500 with one thread each). Fixed in `prepare_syspin.py`.
4. `tmux new-window -t train` matches a window named `trainer`; use `-t train:`. `pane_current_command` reports `bash` for running windows; check the python process instead.
5. F5 draws condition dropout from Python's `random`; seed random + numpy + torch or even eager-vs-eager comparisons disagree.
6. Duration drift in `eval_diagnostics.py` is model-independent; a metric that is identical across models is not measuring the model.
7. Set `HF_HUB_DISABLE_PROGRESS_BARS=1` for one-off uploads/downloads or the output floods.
8. Don't rely on terminal copy for the user; send files or a copy-box page. Don't use broad credentials (`gh auth login`) when a repo deploy key does the job.

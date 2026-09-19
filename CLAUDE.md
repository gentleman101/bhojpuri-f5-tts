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

If `.venv` is missing entirely (fresh disk), rebuild it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 .venv && source .venv/bin/activate
git clone https://github.com/AI4Bharat/IndicF5.git third_party/IndicF5
uv pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -e third_party/IndicF5 "transformers<4.50" && uv pip install -e .
```

Set `max_frames_per_batch` by VRAM: 19200 for 24 GB, 38400 for 40 GB+. `mixed_precision: bf16` needs Ampere or newer (use `fp16` on V100/T4).

## State as of 2026-09-19

Done: data downloaded and prepared, diagnostics built, weights verified, CPU baseline generated, all code tested end to end on CPU. Nothing has trained on a GPU yet.

| Thing | Where | Notes |
|---|---|---|
| Raw SYSPIN corpus | `data/syspin/` | 48 GB, gitignored |
| Processed 24 kHz clips | `data/processed/syspin_24k/` | 15 GB, gitignored |
| IndicF5 weights | `checkpoints/IndicF5/` | 1.4 GB, gitignored, gated download |
| Manifests | `manifests/syspin_{slice,10h,full}/` | in git — 1.9h / 9.4h / 90.8h train |
| Diagnostics | `manifests/diagnostics.json` | 32 held-out sentences, 8 contrasts |
| CPU baseline audio | `runs/baseline/` | stock IndicF5, gitignored |

## Fresh-machine bootstrap (empty disk)

Code comes back from git; data, weights and credentials do not. Run in this order — steps 1–3 need the user.

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

1. **Probe**: 50 updates on the slice, record seconds/update and peak VRAM, then set batch size.
2. **Sanity**: `python scripts/train_lora.py --config configs/lora_slice.yaml --overfit-one-batch 200` — loss must fall.
3. **Slice run** on `configs/lora_slice.yaml` — pipeline check only; 1.9h is below the quality cliff.
4. **10h sweep**: baseline, then `extra_trainable: text_embed`, then rank 64/16, then lr variants.
5. **Stop/go gate**: `scripts/eval_diagnostics.py` vs the stock baseline. Only scale to 90.8h if it improves.

## Commands

```bash
python scripts/train_lora.py --config configs/lora_slice.yaml [--resume latest]
python scripts/eval_diagnostics.py --name baseline_stock                 # stock model
python scripts/eval_diagnostics.py --name lora_10h --adapter runs/<run>/checkpoints/step_0002000
python scripts/infer.py --ref-audio X.wav --ref-text "..." --text "..." --out out.wav [--adapter DIR]
python scripts/check_vocab.py --vocab checkpoints/IndicF5/checkpoints/vocab.txt --manifests manifests/syspin_full/train.csv
```

## Gotchas

- IndicF5's tokenizer maps unknown characters to index 0, which is **space** — silently. All current Bhojpuri text is covered (checked), but re-run `check_vocab.py` after any text change.
- Mel settings live once in `bhojpuri_tts/data.py` (`MEL_KWARGS`). Training and inference must use identical values or output is garbage.
- The checkpoint bundles a Vocos vocoder identical to stock `charactr/vocos-mel-24khz` (verified, max diff 0.0), so loading Vocos separately is correct.
- Diagnostic clips are pinned out of training via `manifests/diagnostics_exclude.txt`. If you rebuild diagnostics, re-run `prepare_syspin.py` with that file and re-verify no leakage.
- Flow-matching loss is noisy and plateaus early. Judge checkpoints by listening and by the diagnostic set, not by loss.
- Adapter checkpoints are small (tens of MB); only weights/data are large. Push code often — if the JarvisLabs wallet hits $0 the disk is wiped.

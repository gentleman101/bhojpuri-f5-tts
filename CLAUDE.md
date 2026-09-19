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

If the disk is fresh, `data/` and `checkpoints/` must be re-fetched:
- Weights: `hf auth login`, then `hf download ai4bharat/IndicF5 --local-dir checkpoints/IndicF5` (accept terms on the model page first).
- Corpus: request download links at https://spiredatasets.ee.iisc.ac.in/syspincorpus (Bhojpuri, both speakers, Human Checked). Links are emailed and expire after 7 days.
- Then: `python scripts/prepare_syspin.py --name syspin_full --exclude-file manifests/diagnostics_exclude.txt` and the same for `syspin_10h` (`--train-hours-per-speaker 5`) and `syspin_slice` (`--train-hours-per-speaker 1`).

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

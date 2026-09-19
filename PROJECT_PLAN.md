# Bhojpuri TTS — Project Plan

**Goal:** A learning-focused project to fine-tune a lightweight, zero-shot voice-cloning TTS model for Bhojpuri, then distill it into a smaller model. Self-funded, personal learning — not a race to be first.

---

## Core decisions (frozen)

| Decision | Choice | Why |
|---|---|---|
| Base model | **IndicF5** (F5-TTS architecture, ~330M params) | Already knows Devanagari + Indic phonetics via 11-language pretraining (1417h); no dedicated Bhojpuri F5 model exists publicly, so this is genuinely novel work |
| Target language (v1) | **Bhojpuri only** | Real data exists (SYSPIN); Awadhi has none — deferred to a later phase |
| Fine-tuning method | **LoRA** | Cheaper, faster iteration, lower catastrophic-forgetting risk than full fine-tune; well-suited to limited compute |
| Streaming | **Dropped from scope** | Would require porting to CosyVoice2 (different architecture entirely — LLM + causal flow matching). Not needed for the actual use case; adds architecture-swap complexity without matching the "lightweight" learning goal |
| Model compression | **Distillation only** (pruning & quantization dropped) | One technique done well > three done shallowly. Also the most relevant technique to the "make it smaller" goal, and mirrors how SILMA TTS achieves ~150M-scale Arabic TTS |
| Training hardware | **Rented NVIDIA GPU (cloud)** for fine-tuning + distillation | F5-TTS training is native PyTorch/CUDA; the MLX ecosystem's F5 implementation (`f5-tts-mlx`) is inference-only — no training/fine-tuning support exists there |
| Local hardware | **Mac (16GB unified memory)** for inference, demos, data pipeline dev | `f5-tts-mlx` runs pretrained/fine-tuned checkpoints well locally once training is done elsewhere |

---

## Data

- **Primary corpus: SYSPIN_S1.0 Bhojpuri** (IISc) — ~47h male + ~47h female, studio quality, CC-BY-4.0. Source: syspin.iisc.ac.in or AIKosh.
- **Optional speaker-diversity augmentation:** noisy multi-speaker Bhojpuri sources (e.g. AI4Bharat rural ASR sets), cleaned via Demucs (vocal separation) + VAD + quality filtering before use. Needed only if zero-shot cloning to unseen speakers underperforms with just the 2 SYSPIN voices.
- **Optional bonus data: podcaster outreach.** Ask independent Bhojpuri podcasters/creators for pointers to their own existing content (not movies/songs — those are label/studio-owned, not the performer's to license). Requires explicit written consent for AI voice-cloning use, even when they own the recording.

---

## Phased plan

1. **Run the pretrained model first** — get IndicF5 working end-to-end on its existing languages; read the F5-TTS codebase (DiT architecture, flow matching, vocoder) before touching Bhojpuri.
2. **Build the data pipeline on a small slice** — 1–2 hours of SYSPIN Bhojpuri; write the preprocessing/manifest script yourself.
3. **First fine-tune: LoRA on the small slice** — validate the training loop and hyperparameters cheaply before scaling up. The 1.9h slice proves the code runs; it is too small to judge quality (see the data-size ladder below).
4. **Tune on 10h, then scale to the full corpus if it earns it** — run the hyperparameter sweep on `syspin_10h` (9.4h), apply the stop/go rule below, and only then commit to the ~91h run. Evaluate with the diagnostic set, held-out listening tests, and native-speaker feedback.
5. **Test zero-shot cloning honestly** — reference clip from a speaker unseen in training. If it fails or defaults to a SYSPIN voice, that's the key lesson about speaker diversity and generalization. Don't proceed until this is reasonably solid.
6. **Distill into a smaller student model** — design a smaller architecture (SILMA's ~150M config as a reference shape), train with a distillation loss against the fine-tuned IndicF5 as teacher. Compare teacher vs. student on quality, size, and latency — this comparison is the deliverable.
7. **Optional: expand data via podcaster outreach** — bonus round once the core pipeline is proven; compare cloning quality before/after added speaker diversity.

---

## Stop/go rule: does fine-tuning actually help?

Stock IndicF5 already speaks intelligible Bhojpuri with correct voice cloning — its authors reached MUSHRA 82.0 on Bhojpuri from ~1h of *synthetic* audio ([IN-F5 paper](https://arxiv.org/abs/2505.20693)), and our own baseline listening confirms only specific characters are mispronounced. So the gain from fine-tuning must be demonstrated, not assumed.

**The gate.** After the 10h run, compare against the stock baseline on the same 32 diagnostic sentences, same references, same seed and sampling settings:

- **Improvement on diagnostics + ASR error rate + native-speaker ear → scale to the full ~91h run.**
- **No clear improvement → do not pay for the full run.** First try the one variant that targets the observed symptom: unfreeze the text embedding (`extra_trainable`), where character-to-sound mapping lives. If that also fails, stop fine-tuning and proceed to distillation **with stock IndicF5 as the teacher** — the deliverable (teacher vs. student on quality/size/latency) does not depend on the fine-tune succeeding.

A negative result is a real finding worth writing up: "91h of real studio data did not beat 1h of validated synthetic data" is genuinely informative about low-resource TTS. What it would *not* prove is that Bhojpuri fine-tuning is hopeless in general — only that LoRA at this scale did not help.

---

## Data-size ladder (from the IN-F5 paper's data-constrained study)

| Their finding | Hours | MUSHRA | WER |
|---|---|---|---|
| Plenty | 100h | 64.3 | 32.6% |
| **Nearly as good** | **10h** | **61.5** | **31.3%** |
| Collapses | 1h | 33.7 | 59.4% |

They also found 10h trained for 150k steps beat 100h trained for 120k steps — **longer training partly substitutes for more data**. Our three stages follow from this:

| Manifest | Train audio | Purpose |
|---|---|---|
| `syspin_slice` | 1.9h | Pipeline correctness only — below the quality cliff |
| `syspin_10h` | 9.4h | Hyperparameter sweep and the stop/go gate |
| `syspin_full` | 90.8h | Final run, with a large update budget |

Their full fine-tune used AdamW, lr 5e-5, 30k frames/GPU across 32 H100s, up to 150k steps, warmup 48k, checkpoints every 2k. Our single-GPU LoRA runs at a much smaller batch, so lr stays at 1e-4 with warmup ~10% of total updates.

---

## Evaluation

- **Diagnostic set** (`manifests/diagnostics.json`) — 32 held-out sentences, 4 per pronunciation contrast where Bhojpuri diverges from Hindi (व→ब, श/ष→स, ण→न, avagraha, verb endings -ला, copula बा/हवे, retained final vowels, consonant clusters). Every clip is pinned out of training in all stages and has a real recording to compare against. Run with `scripts/eval_diagnostics.py`.
- **Automated scoring** — no official Bhojpuri ASR exists (IndicConformer covers the 22 scheduled languages; Bhojpuri is not one). Community Whisper/wav2vec2 Bhojpuri fine-tunes do exist. Calibrate a candidate on *real* held-out recordings first to establish its error floor on human speech, then read our synthetic audio's CER relative to that floor.
- **Listening tests** — MUSHRA-style (the IN-F5 paper's metric): reference plus shuffled systems scored 0–100 on one screen, with a hidden reference and a degraded anchor to catch inattentive listeners. Far more sensitive to small differences than 1–5 MOS, and usable with ~10–20 listeners. A simple two-clip page is enough for first impressions; build the MUSHRA page once there is a fine-tuned model worth comparing.

---

## Deferred / out of scope for v1

- Awadhi (no usable corpus exists yet; would need synthetic bootstrap via cross-lingual transfer once Bhojpuri model works)
- Streaming / CosyVoice2 port
- Structured pruning and quantization (may revisit after distillation if useful)
- Public-figure studio recording asks (superseded by podcaster/own-content approach)

---

## Workflow: single JarvisLabs VM, pause/resume, switch GPU type when training

One instance, one machine ID, for the entire project. No separate filesystem resource, no multi-machine sync — that complexity was priced out (storage billing is identical whether split into a separate filesystem or kept on one paused instance: $0.00014/GB/hour either way, with no cost advantage to separating them for a sequential, non-concurrent workflow like this one).

**How it works:**

1. **Launch one instance as CPU** for all development — writing/testing preprocessing scripts, building the SYSPIN manifest, data cleaning. Cheap, no GPU meter running.
2. **Pause whenever you step away** — `jl instance pause <machine_id>`. Billing drops to storage-only immediately. Resume later and everything under `/home` is exactly as you left it.
3. **When ready to train, resume the same instance as GPU** — `jl instance resume <machine_id> --gpu A100` (or L4). Same machine ID, same data, no transfer step. This is a native, documented feature (confirmed: `instance.resume(gpu_type=..., num_gpus=1, storage=...)` in their SDK).
4. **Pause again immediately after each training run** — switch back to CPU or fully pause once Phase 3, 4, or 6's run finishes.
5. **`git push` code periodically as backup insurance** — protects against the one real failure mode (wallet hits $0 → all data, including storage, is wiped and unrecoverable). Push scripts/configs/manifests only — never audio or checkpoints.
6. **Download the final checkpoint** via `jl instance download` or scp once there's something worth running locally (e.g. for demos via `f5-tts-mlx` on a Mac, if that's ever wanted) — otherwise everything, including inference, can stay on the same instance.

---

## Open risks to keep in mind

- **Evaluation bottleneck:** no official Bhojpuri ASR for automatic WER scoring. Mitigated (not solved) by the diagnostic set, MUSHRA listening tests, and a community Bhojpuri ASR calibrated against real recordings — see Evaluation above.
- **Speaker diversity:** only 2 SYSPIN speakers may not be enough to prove genuine cloning generalization — augmentation data may become necessary, not optional.
- **Consent scope:** any voice used for cloning (even self-owned content) needs explicit, written, scope-limited consent — separate from ordinary copyright/ownership of the recording.
- **Compute budget:** rented GPU costs estimated ~$150–400 total for LoRA fine-tuning + distillation runs (to be refined once rental pricing is checked).

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
3. **First fine-tune: LoRA on the small slice** — validate the training loop and hyperparameters cheaply before scaling up.
4. **Scale up and evaluate properly** — full ~95h SYSPIN fine-tune; evaluate manually (no strong Bhojpuri ASR exists for automated scoring) via held-out listening tests, ideally with native-speaker feedback.
5. **Test zero-shot cloning honestly** — reference clip from a speaker unseen in training. If it fails or defaults to a SYSPIN voice, that's the key lesson about speaker diversity and generalization. Don't proceed until this is reasonably solid.
6. **Distill into a smaller student model** — design a smaller architecture (SILMA's ~150M config as a reference shape), train with a distillation loss against the fine-tuned IndicF5 as teacher. Compare teacher vs. student on quality, size, and latency — this comparison is the deliverable.
7. **Optional: expand data via podcaster outreach** — bonus round once the core pipeline is proven; compare cloning quality before/after added speaker diversity.

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

- **Evaluation bottleneck:** no reliable Bhojpuri ASR for automatic WER scoring — plan for manual/native-speaker evaluation throughout.
- **Speaker diversity:** only 2 SYSPIN speakers may not be enough to prove genuine cloning generalization — augmentation data may become necessary, not optional.
- **Consent scope:** any voice used for cloning (even self-owned content) needs explicit, written, scope-limited consent — separate from ordinary copyright/ownership of the recording.
- **Compute budget:** rented GPU costs estimated ~$150–400 total for LoRA fine-tuning + distillation runs (to be refined once rental pricing is checked).

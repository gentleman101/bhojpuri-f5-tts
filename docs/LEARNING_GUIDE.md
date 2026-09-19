# Bhojpuri F5-TTS — Learning Guide

A study path to read *before and during* development. It assumes you can write Python but have never trained or fine-tuned a model. Every topic says **why it matters for this project**, **what to learn**, **where it shows up in our code**, and **self-check questions** — if you can answer them, move on.

Code references point at `third_party/IndicF5/f5_tts/` (the IndicF5 fork of F5-TTS, cloned locally).

---

## 0. How to use this guide

- Read the tiers **in order**. Tiers 1–4 are needed before Phase 1–3; Tier 5–6 before Phase 3–4; Tier 7 before Phase 6.
- Don't read everything deeply. For papers: abstract → figures → method section. Skip proofs on first pass.
- The fastest learning happens when you read a concept, then find it in `f5_tts/`. Each tier tells you where.
- Rough time budget, part-time: Tiers 1–2 ≈ 1–2 weeks (skip what you know), Tiers 3–4 ≈ 1 week, Tiers 5–7 ≈ 1 week, spread over the phases.

---

## 1. The big picture (read this first)

### What the finished system does

```
 reference clip (someone's voice, 3–10 s) ─┐
 transcript of that clip ──────────────────┤
 new Bhojpuri text to speak ───────────────┘
                     │
                     ▼
     ┌─────────────────────────────────────┐
     │ 1. Text → characters → token IDs    │  (char tokenizer, vocab.txt)
     │ 2. Reference audio → mel spectrogram│  (24 kHz, 100 mel bins)
     │ 3. DiT + flow matching "fills in"   │  (the ~330M-param model we fine-tune)
     │    the mel frames for the new text, │
     │    copying the reference's voice    │
     │ 4. Vocoder: mel → waveform          │  (Vocos, pretrained, NOT trained by us)
     └─────────────────────────────────────┘
                     │
                     ▼
            Bhojpuri speech in that voice
```

### The core idea of F5-TTS in one paragraph

F5-TTS treats speech synthesis as **infilling**. During training it takes a real clip's mel spectrogram, hides a random 70–100% chunk of it, and learns to regenerate the hidden part given the visible part and the *full* transcript. At inference, the "visible part" is your reference clip and the "hidden part" is the new sentence — so the model continues speaking in the reference voice. There is no separate speaker encoder, no phoneme aligner, no duration model: text characters and audio frames are fed together into one transformer, and alignment is learned implicitly.

### What we actually change

| Component | Pretrained by | What we do |
|---|---|---|
| Vocoder (Vocos) | Charactr / F5 authors | Nothing — reuse as-is |
| Tokenizer / vocab | IndicF5 (character-level, includes Devanagari) | Verify every Bhojpuri character exists in the vocab |
| DiT backbone (~330M) | IndicF5 on 1417h of 11 Indic languages | **Fine-tune with LoRA** on SYSPIN Bhojpuri |
| Smaller student DiT (~150M) | — | **Train by distillation** from our fine-tuned model |

---

## 2. The framework stack

| Layer | Tool | Role in this project |
|---|---|---|
| Language | Python 3.10 | Everything |
| Tensor/autograd library | **PyTorch** 2.5 | Model definition, training loop, GPU execution |
| Audio I/O & features | **torchaudio**, soundfile, librosa | Loading WAVs, resampling 48k→24k, mel spectrograms |
| Model code | **IndicF5 / F5-TTS** (`f5_tts`) | DiT backbone, flow-matching (CFM) wrapper, dataset, trainer, inference |
| Transformer building blocks | **x_transformers** | Rotary position embeddings used inside the DiT |
| ODE solver | **torchdiffeq** | Integrates the flow at inference (`odeint`, 32 steps) |
| Multi-GPU / mixed precision | **Hugging Face Accelerate** | Wraps the training loop (`model/trainer.py`) |
| EMA weights | **ema_pytorch** | Keeps a smoothed copy of weights; this copy is what gets used for inference |
| Datasets on disk | **HF datasets** (Arrow format) | The training data format F5's prepare scripts produce |
| Config | **Hydra** (YAML) | Training configs in `f5_tts/configs/` |
| LoRA | **PEFT** (to be added — IndicF5 has *no* LoRA support built in) | Inject low-rank adapters into DiT attention/FFN layers |
| Model hub | **Hugging Face Hub** | Downloading `ai4bharat/IndicF5` weights (gated) |
| Experiment tracking | wandb or TensorBoard | Loss curves, audio samples during training |
| Environment | `uv` + `.venv` | Package management |
| Compute | JarvisLabs VM (CPU for dev, A100/L4 for training) | See `PROJECT_PLAN.md` |

---

## Tier 1 — Machine-learning foundations

**Why it matters:** fine-tuning is just training that starts from good weights. Every training-run problem you'll debug (loss not decreasing, NaNs, overfitting) is a Tier 1 concept.

### Concepts checklist
- [ ] Tensors, shapes, broadcasting; batch dimension conventions `(batch, time, channels)`
- [ ] Forward pass, loss function, backpropagation, gradients
- [ ] Gradient descent → SGD → **Adam / AdamW** (what weight decay does)
- [ ] **Learning rate**, and why it's the #1 hyperparameter
- [ ] **Learning-rate warmup** and decay schedules
- [ ] Epoch vs. step vs. "update"; **batch size** and its interaction with LR
- [ ] **Gradient accumulation** (simulating a big batch on a small GPU)
- [ ] **Gradient clipping** (`max_grad_norm`)
- [ ] Overfitting vs. underfitting; train/validation/test splits
- [ ] **Checkpoints**: what's saved (weights, optimizer state, step)
- [ ] **Mixed precision** (fp16 vs. bf16) and why it halves memory
- [ ] GPU memory budget: weights + gradients + optimizer state + activations

### Reading
- Andrej Karpathy, *Neural Networks: Zero to Hero* (video series) — https://karpathy.ai/zero-to-hero.html. Watch at least "micrograd" and "makemore part 1–3". Best possible intro to backprop and training loops.
- PyTorch official tutorials, "Learn the Basics" — https://pytorch.org/tutorials/
- 3Blue1Brown, *Neural Networks* series (YouTube) — intuition for gradients.

### In our code
- `model/trainer.py` — the whole training loop: `accelerator.backward(loss)`, `clip_grad_norm_`, optimizer step, warmup scheduler, EMA update, checkpoint save.
- `configs/F5TTS_Base_train.yaml` — `learning_rate`, `num_warmup_updates`, `grad_accumulation_steps`, `max_grad_norm`.

### Self-check
1. If loss goes to NaN at step 200, name three likely causes.
2. Why does Adam need ~2× extra memory on top of the weights?
3. `finetune_cli.py` defaults to LR `1e-5`, pretraining used `7.5e-5`. Why is fine-tuning LR lower?

---

## Tier 2 — Transformers

**Why it matters:** the DiT is a transformer. LoRA attaches to its linear layers — you must know which layers exist to choose where adapters go.

### Concepts checklist
- [ ] Embeddings (turning token IDs into vectors)
- [ ] **Self-attention**: Q, K, V projections, attention weights, multi-head attention
- [ ] Feed-forward (MLP) block, residual connections, LayerNorm
- [ ] **Positional information**: absolute vs. **rotary (RoPE)** embeddings
- [ ] Why attention cost grows with sequence length² (matters: speech has many frames)
- [ ] Encoder-only vs. decoder-only vs. non-autoregressive (F5 is non-autoregressive: generates all frames at once)

### Reading
- Jay Alammar, *The Illustrated Transformer* — https://jalammar.github.io/illustrated-transformer/
- Vaswani et al., *Attention Is All You Need* (2017) — https://arxiv.org/abs/1706.03762
- Su et al., *RoFormer: Rotary Position Embedding* — https://arxiv.org/abs/2104.09864 (skim)
- Karpathy, "Let's build GPT" video (part of Zero to Hero)

### In our code
- `model/modules.py` — `Attention`, `FeedForward`, `ConvPositionEmbedding`, `AdaLayerNormZero`
- `model/backbones/dit.py` — `DiT` class; config `dim=1024, depth=22, heads=16, ff_mult=2`

### Self-check
1. With `dim=1024` and `ff_mult=2`, what are the shapes of the Q/K/V and FFN linear layers?
2. 22 blocks × those layers ≈ how many params? Does it roughly match 330M?
3. Why can't a plain transformer tell frame 10 from frame 500 without positional encoding?

---

## Tier 3 — Speech & audio fundamentals

**Why it matters:** Phase 2 is entirely audio preprocessing. Mistakes here (wrong sample rate, clipped audio, silence) silently ruin training.

### Concepts checklist
- [ ] Waveform, **sample rate** (SYSPIN is 48 kHz, model needs **24 kHz**), bit depth (SYSPIN is 24-bit), mono/stereo
- [ ] **Resampling** and aliasing
- [ ] Short-time Fourier transform (STFT): `n_fft`, `win_length`, **`hop_length`** (256 → 24000/256 ≈ **93.75 frames per second**)
- [ ] **Mel spectrogram**: why mel scale, what 100 mel bins means, log-magnitude
- [ ] **Vocoder**: turning mel back to waveform (Griffin-Lim → neural vocoders → **Vocos**)
- [ ] Loudness normalization / RMS (`target_rms = 0.1` in inference)
- [ ] Silence trimming and **voice activity detection (VAD)**
- [ ] TTS text normalization: numbers, abbreviations, punctuation → spoken form
- [ ] Devanagari in Unicode: consonants, matras, **virama/halant**, nukta, **NFC vs NFD normalization** (same-looking text can be different code points — critical for a character tokenizer)

### Reading
- *librosa* documentation, "Tutorial" and `feature.melspectrogram` — https://librosa.org/doc/latest/
- torchaudio tutorials, "Audio Feature Extractions" and "Audio Resampling" — https://pytorch.org/audio/stable/tutorials/
- Siuzdak, *Vocos: Closing the gap between time-domain and Fourier-based neural vocoders* — https://arxiv.org/abs/2306.00814 (just abstract + figure 1)
- Unicode Standard, Devanagari chapter (search "Unicode Devanagari block") — focus on combining marks and normalization.

### In our code
- `model/modules.py` — `MelSpec` (how mels are computed)
- `infer/utils_infer.py` — `target_sample_rate=24000`, `n_mel_channels=100`, `hop_length=256`, `load_vocoder`, `preprocess_ref_audio_text`
- `train/datasets/prepare_csv_wavs.py` — reference for turning `(wav, text)` pairs into a training dataset

### Self-check
1. A 10-second clip at 24 kHz with hop 256 produces how many mel frames?
2. Why must training mels and inference mels use identical settings?
3. Two Devanagari strings look identical but the tokenizer gives different IDs. What happened?

---

## Tier 4 — Generative models: diffusion → flow matching

**Why it matters:** this is the "brain" of F5-TTS and the hardest topic. You don't need the math proofs, but you must understand what the model predicts, what the loss is, and what "steps" mean at inference — distillation (Tier 7) depends on it.

### Concepts checklist
- [ ] What a generative model does: sample from a data distribution
- [ ] **Diffusion** intuition: gradually add noise, learn to reverse it
- [ ] **Flow matching**: learn a *velocity field* that moves noise → data along a path
- [ ] **Conditional flow matching / rectified flow**: straight-line path `x_t = (1−t)·noise + t·data`, target velocity `= data − noise`
- [ ] The training loss is just **MSE between predicted and target velocity** — on the masked frames only
- [ ] Inference = solving an **ODE** from t=0 to t=1 with N function evaluations (**NFE**, default 32)
- [ ] **Classifier-free guidance (CFG)**: train with conditions sometimes dropped; at inference, push away from the unconditional prediction (`cfg_strength=2.0`)
- [ ] **Sway sampling** (F5-specific): spend more ODE steps early (`sway_sampling_coef=-1`)
- [ ] **EMA** of weights for better samples

### Reading (in this order)
1. Lilian Weng, *What are Diffusion Models?* — https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ (intuition only)
2. Lipman et al., *Flow Matching for Generative Modeling* — https://arxiv.org/abs/2210.02747
3. Liu et al., *Flow Straight and Fast: Rectified Flow* — https://arxiv.org/abs/2209.03003
4. Lipman et al. (Meta), *Flow Matching Guide and Code* — https://arxiv.org/abs/2412.06264 — the most practical reference, with code
5. Ho & Salimans, *Classifier-Free Diffusion Guidance* — https://arxiv.org/abs/2207.12598
6. Peebles & Xie, *Scalable Diffusion Models with Transformers (DiT)* — https://arxiv.org/abs/2212.09748 — where "DiT" and adaptive LayerNorm conditioning come from

### Speech-specific lineage (read after the above)
7. Le et al., *Voicebox* — https://arxiv.org/abs/2306.15687 — introduced speech infilling with flow matching
8. Eskimez et al., *E2 TTS* — https://arxiv.org/abs/2406.18009 — "just pad the text with filler tokens"; F5's direct predecessor
9. **Chen et al., *F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching*** — https://arxiv.org/abs/2410.06885 — **the paper for this project. Read fully.**

### In our code
- `model/cfm.py` — **the most important file.**
  - `forward()` = training: random `t`, build `x_t`, random span mask (`frac_lengths_mask=(0.7, 1.0)`), drop audio 30% / drop text 20% for CFG (`audio_drop_prob`, `cond_drop_prob`), MSE on masked region.
  - `sample()` = inference: `odeint` over sway-sampled timesteps with CFG.
- `model/backbones/dit.py` — `TextEmbedding` (ConvNeXt blocks on characters), `InputEmbedding` (concat noisy mel + reference mel + text), `AdaLayerNormZero` (time conditioning).

### Self-check
1. In one sentence: what exactly does the DiT output, and what is it compared to in the loss?
2. Why does the loss only count masked frames?
3. What happens to quality and speed if you use NFE=8 instead of 32?
4. Why must the model sometimes be trained *without* text for CFG to work?
5. How does F5 know how long the output should be? (Hint: look at how `duration` is computed from reference text/audio length in `utils_infer.py`.)

---

## Tier 5 — Transfer learning, fine-tuning & LoRA

**Why it matters:** this is Phase 3–4. IndicF5 doesn't ship LoRA, so we'll add it — understanding it is required, not optional.

### Concepts checklist
- [ ] **Transfer learning**: why a model pretrained on Hindi/Marathi etc. is a great starting point for Bhojpuri (shared Devanagari script, related phonology)
- [ ] **Full fine-tuning** vs. **parameter-efficient fine-tuning (PEFT)**
- [ ] **Catastrophic forgetting** — and why we may not care much (we want Bhojpuri, not all 11 languages)
- [ ] **LoRA**: freeze weight `W`, learn `W + (α/r)·B·A` with `A ∈ R^{r×d}`, `B ∈ R^{d×r}`, `B` initialized to zero
- [ ] LoRA hyperparameters: **rank `r`**, **alpha `α`**, dropout, **target modules** (attention q/k/v/out? FFN? text embedding?)
- [ ] Which parts to fully train alongside LoRA (e.g. text embedding for new characters, norms)
- [ ] **Merging** LoRA back into base weights for inference (zero runtime cost)
- [ ] Capacity limits: when LoRA is too small to learn a new language → raise rank or train more modules
- [ ] Vocabulary extension: what to do if Bhojpuri text contains characters absent from `vocab.txt` (resize embeddings, new rows trained)
- [ ] Learning-rate for LoRA is usually **higher** than for full fine-tuning (e.g. 1e-4 range vs. 1e-5)

### Reading
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* — https://arxiv.org/abs/2106.09685 — sections 1, 4, 7
- Hugging Face **PEFT** docs, "LoRA" conceptual guide + "Custom models" (applying LoRA to non-HF `nn.Module`s — exactly our case) — https://huggingface.co/docs/peft
- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, appendix on LoRA — optional, excellent hand-written LoRA implementation
- **Exercise:** implement a 20-line `LoRALinear(nn.Module)` yourself before using PEFT. You'll understand it permanently.

### In our code (and what we'll add)
- `train/finetune_cli.py` — existing *full* fine-tuning entry point (`--finetune`, `--pretrain`, `--tokenizer custom`). We'll base our LoRA script on it.
- `model/utils.py` — `get_tokenizer()`: `char`/`custom` tokenizers, **index 0 = space = unknown char** (unknown Bhojpuri chars silently become spaces — must check!)
- To add: wrap the DiT with PEFT `LoraConfig(target_modules=[...])`, freeze base, verify trainable-param count.

### Self-check
1. With `r=16` on a `1024×1024` linear layer, how many trainable params vs. frozen?
2. Why is `B` initialized to zero?
3. Your LoRA fine-tune sounds like Hindi with Bhojpuri words. Name two knobs to turn.
4. What happens at inference if a Bhojpuri character isn't in `vocab.txt`?

---

## Tier 6 — Training engineering, data pipeline & evaluation

**Why it matters:** most real fine-tuning time goes here, not into model code. It's also where cloud money is wasted.

### 6a. Data pipeline (Phase 2)
- [ ] Building a **manifest**: `(audio_path, text, duration)` rows
- [ ] Filtering: clip duration limits (F5 dataset drops very short/long clips), empty/garbled text, loudness outliers
- [ ] Train / validation / test split **by sentence**, and keep a held-out set you *never* train on
- [ ] Text normalization consistency between training and inference
- [ ] Converting to F5's **Arrow dataset + `duration.json` + `vocab.txt`** format (`prepare_csv_wavs.py`)
- [ ] **Frame-based batching** (`batch_size_type: frame`): batch size measured in mel frames, not clips. 38 400 frames ≈ 410 s of audio per GPU
- [ ] Throughput: dataloader workers, disk vs. RAM, preprocessing once vs. on the fly

### 6b. Running training on rented GPUs (Phase 3–4, 6)
- [ ] Smoke test on CPU/tiny data before paying for a GPU; **overfit a single batch** first (loss should go near zero — proves the loop works)
- [ ] GPU choices: A100 40/80GB vs. L4 24GB; bf16 support
- [ ] `accelerate config` / `accelerate launch`
- [ ] Reading loss curves: noisy flow-matching loss plateaus early — **listen to samples**, don't trust loss alone
- [ ] Checkpoint cadence (`save_per_updates`, `last_per_steps`) and **resume after interruption**
- [ ] Logging audio samples every N steps (wandb/TensorBoard)
- [ ] Cost math: $/hour × hours; stop runs that aren't improving
- [ ] `nvidia-smi`, GPU utilization (low util = data-loading bottleneck)

### 6c. Evaluation (Phase 4–5)
- [ ] **Subjective**: MOS (mean opinion score), A/B preference tests, native-speaker listening — our primary method
- [ ] **Speaker similarity**: cosine similarity of speaker embeddings (ECAPA-TDNN / WavLM-based); IndicF5 ships `eval/ecapa_tdnn.py`
- [ ] **Intelligibility**: WER/CER via ASR — weak for Bhojpuri; a Hindi ASR can give a rough CER signal
- [ ] **Zero-shot test design**: reference speakers *never seen in training*, fixed test sentences, fixed seeds, same NFE/CFG for every comparison
- [ ] **Real-time factor (RTF)**: generation time ÷ audio duration (needed for the distillation comparison)

### Reading
- Hugging Face **Accelerate** docs, "Quicktour" and "Gradient accumulation" — https://huggingface.co/docs/accelerate
- `f5_tts/train/README.md` and `f5_tts/eval/README.md` in the repo
- Andrej Karpathy, *A Recipe for Training Neural Networks* (blog post, 2019) — https://karpathy.github.io/2019/04/25/recipe/ — **read this before your first GPU run.**

### Self-check
1. Why overfit one batch before a real run?
2. Loss stopped dropping at step 5k but samples keep improving until 30k. Is that surprising for flow matching?
3. Why is a held-out *speaker* (not just held-out sentences) needed to test zero-shot cloning?

---

## Tier 7 — Knowledge distillation (Phase 6)

**Why it matters:** the final deliverable is a teacher-vs-student comparison.

### Concepts checklist
- [ ] **Teacher / student** setup; soft targets
- [ ] Distillation for **generative / regression** models: student matches the teacher's *outputs* (here: predicted velocity fields) rather than class probabilities
- [ ] Two different things called "distillation" for flow/diffusion models — don't confuse them:
  - **Model-size distillation** (our plan): smaller DiT (fewer layers / smaller `dim`) imitates the teacher at the same NFE
  - **Step distillation** (e.g. progressive distillation, consistency models): same size, far fewer ODE steps
- [ ] Loss design: teacher-velocity MSE, optionally mixed with the normal ground-truth flow-matching loss; **feature/hidden-state matching** between layers
- [ ] Student initialization: random vs. copying every other teacher layer (layer dropping)
- [ ] Teacher outputs computed online vs. pre-computed and cached
- [ ] Comparing fairly: parameters, disk size, RTF on the same hardware, MOS, speaker similarity

### Reading
- Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network* — https://arxiv.org/abs/1503.02531
- Sanh et al., *DistilBERT* — https://arxiv.org/abs/1910.01108 — clean example of layer-dropped student init + combined losses
- Salimans & Ho, *Progressive Distillation for Fast Sampling of Diffusion Models* — https://arxiv.org/abs/2202.00512 — step distillation, for contrast
- `f5_tts/configs/F5TTS_Small_train.yaml` — an official smaller F5 config; a natural student-shape reference alongside SILMA's ~150M design

### Self-check
1. Why might a student trained on teacher velocities beat a student trained on ground-truth data alone, given only ~95h?
2. What's the difference between making the model *smaller* and making sampling *faster*? Which does our plan target?

---

## 8. Project phases → what you'll do technically

| Phase | Technical work | Tiers needed |
|---|---|---|
| 1. Run pretrained IndicF5 | Download gated weights, run inference on CPU with a Hindi prompt, try Bhojpuri text zero-shot as a **baseline**, trace one call through `utils_infer.py → cfm.sample → dit.forward → vocoder` | 1–4 |
| 2. Data pipeline (1–2h slice) | Parse SYSPIN JSON, Unicode-normalize text, check chars against IndicF5 `vocab.txt`, resample 48k→24k mono, trim silence, filter by duration, build manifest, convert to F5 Arrow format, split train/val/test | 3, 6a |
| 3. LoRA on slice | Add PEFT to DiT, pick target modules & rank, overfit-one-batch test, short GPU run, listen to checkpoints | 5, 6b |
| 4. Full ~97h fine-tune | Scale data, tune LR/steps, checkpointing & resume, held-out listening tests | 5, 6b, 6c |
| 5. Zero-shot test | Unseen-speaker references, speaker-similarity scores, honest failure analysis | 6c |
| 6. Distillation | Design student config, distillation loss, train, measure size/RTF/quality vs. teacher | 7 |
| 7. (Optional) more speakers | Consent, cleaning noisy data (Demucs, VAD), retrain, before/after comparison | 3, 6a, 6c |

### Baseline to record in Phase 1
Before fine-tuning anything, generate a few Bhojpuri sentences with **stock IndicF5**. It already reads Devanagari, so it will produce *something* — probably Hindi-accented. Every later result is compared against this baseline; without it you can't show the fine-tune helped.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| **Mel spectrogram** | Time × frequency image of audio on a perceptual pitch scale; what the model generates |
| **Vocoder** | Network turning mels into audio waveforms (Vocos here) |
| **Hop length** | Samples between spectrogram frames (256 → ~94 frames/s at 24 kHz) |
| **DiT** | Diffusion Transformer — transformer backbone conditioned on timestep via adaptive LayerNorm |
| **CFM** | Conditional Flow Matching — the training objective and sampler wrapper (`cfm.py`) |
| **NFE** | Number of function evaluations — model calls per generated sample (default 32) |
| **CFG** | Classifier-free guidance — trades diversity for adherence to text/voice |
| **Sway sampling** | F5's non-uniform timestep schedule for the ODE solver |
| **EMA** | Exponential moving average of weights; used for inference |
| **LoRA** | Low-rank adapters added to frozen linear layers |
| **Rank (r) / alpha (α)** | LoRA capacity and scaling |
| **PEFT** | Parameter-efficient fine-tuning (and the HF library of that name) |
| **Frame-based batching** | Batch size counted in mel frames rather than number of clips |
| **Warmup** | Gradually raising LR at training start to avoid instability |
| **Zero-shot cloning** | Mimicking a voice never seen in training from a short reference |
| **MOS** | Mean Opinion Score — human 1–5 quality rating |
| **Speaker similarity (SIM)** | Cosine similarity of speaker embeddings between reference and output |
| **RTF** | Real-time factor — compute time ÷ audio length (<1 is faster than real time) |
| **Teacher / student** | Big model providing targets / small model being distilled |
| **NFC** | Unicode canonical composition — normalize all text to it before tokenizing |

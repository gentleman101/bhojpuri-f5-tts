---
name: project-bhojpuri-tts
description: Bhojpuri TTS learning project — frozen architecture/scope decisions and GPU workflow
metadata: 
  node_type: memory
  type: project
  originSessionId: 438d9374-a5b8-4112-a149-169dc114c78c
  modified: 2026-09-19T09:48:54.711Z
---

Personal, self-funded learning project at `/home/ubuntu/bhojpuri-f5-tts` (full plan in `PROJECT_PLAN.md` there). Goal: fine-tune IndicF5 (F5-TTS architecture, ~330M params) for zero-shot voice cloning in Bhojpuri, then distill to a smaller student model.

Frozen decisions:
- Base model: IndicF5, fine-tuned via LoRA only (no full fine-tune).
- Language: Bhojpuri only for v1 — Awadhi has no usable corpus, deferred.
- Compression: distillation only — pruning/quantization explicitly dropped ("one technique done well > three done shallowly").
- Streaming dropped — would require porting to CosyVoice2, a different architecture; not worth the complexity for this project's goals.
- Data: SYSPIN_S1.0 Bhojpuri corpus (IISc, ~95h, CC-BY-4.0) is primary. Podcaster outreach for extra speaker diversity is optional/bonus and requires explicit written consent even for self-owned content.
- Stop/go gate (added 2026-09-19): fine-tuning must beat the stock IndicF5 baseline on the 32-sentence diagnostic set after the 10h run, or the full ~91h run is not paid for — fall back to distilling from stock IndicF5, which keeps the deliverable intact. Stock IndicF5 already handles Bhojpuri decently (IN-F5 paper, arXiv 2505.20693, MUSHRA 82 from 1h of synthetic Bhojpuri), so gains must be shown, not assumed.
- Data ladder: syspin_slice 1.9h (pipeline check only), syspin_10h 9.4h (hyperparameter sweep + the gate), syspin_full 90.8h (final run). Follows the IN-F5 finding that 10h ≈ 100h when trained longer, while 1h collapses.
- Evaluation: 32-sentence diagnostic set covering Bhojpuri-vs-Hindi pronunciation contrasts (व→ब, श/ष→स, ण→न, avagraha, verb endings, retained final vowels), all pinned out of training; MUSHRA-style listening tests rather than 1-5 MOS because differences are small; community Bhojpuri ASR calibrated on real recordings for relative CER.
- Infra: single JarvisLabs VM, launched as CPU for dev, resumed as GPU (A100/L4) for training runs, paused when idle. No separate filesystem/multi-machine setup — priced out as unnecessary for this sequential workflow. Only code/configs/manifests get git-pushed as backup; audio/checkpoints never leave the instance (wallet-hits-$0 wipes storage, so this is the real risk being hedged).

**Why:** These are the frozen scope boundaries the user already worked through — don't re-litigate them (e.g. don't suggest quantization, streaming, or multi-machine setups) unless the user reopens that decision.
**How to apply:** When helping with this project, assume these choices are settled; focus effort on implementation within this scope. Phased plan (in PROJECT_PLAN.md) starts with running pretrained IndicF5 end-to-end before touching Bhojpuri-specific work.

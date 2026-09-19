"""Generate speech with stock IndicF5 (no --adapter) or with a LoRA checkpoint merged in.

  python scripts/infer.py --ref-audio ref.wav --ref-text "..." --text "..." --out out.wav
  python scripts/infer.py --adapter runs/lora_slice/checkpoints/step_0003000 ... --out out.wav
"""

import argparse
import time

import soundfile as sf
import torch
import torchaudio

from bhojpuri_tts import REPO_ROOT
from bhojpuri_tts.modeling import F5_BASE_ARCH, load_model_for_inference
from bhojpuri_tts.text import normalize_text, split_sentences

DEFAULT_VOCAB = "checkpoints/IndicF5/checkpoints/vocab.txt"
DEFAULT_BASE = "checkpoints/IndicF5/model.safetensors"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True, help="Exact transcript of the reference clip")
    parser.add_argument("--text", help="Text to speak")
    parser.add_argument("--text-file", help="Read text to speak from a file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter", help="Checkpoint directory from train_lora.py")
    parser.add_argument("--raw-weights", action="store_true", help="Use raw adapter weights instead of EMA")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE, help="Pass \"\" to skip (smoke tests only)")
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from f5_tts.infer.utils_infer import infer_batch_process, load_vocoder

    text = args.text if args.text is not None else open(args.text_file, encoding="utf-8").read()
    text, ref_text = normalize_text(text), normalize_text(args.ref_text) + " "

    model = load_model_for_inference(
        str(REPO_ROOT / args.vocab),
        str(REPO_ROOT / args.base_checkpoint) if args.base_checkpoint else None,
        args.adapter,
        not args.raw_weights,
        F5_BASE_ARCH,
    ).to(args.device)
    vocoder = load_vocoder("vocos", device=args.device)

    ref_audio, sr = torchaudio.load(args.ref_audio)
    ref_seconds = ref_audio.shape[-1] / sr
    if ref_seconds > 15:
        print(f"WARNING: reference is {ref_seconds:.1f}s; 3–10s works best (F5 generates ref + text within ~25s)")
    # Same budget rule as F5's infer_process: keep each chunk plus the reference under ~25 s of audio.
    max_chars = max(20, int(len(ref_text) / ref_seconds * (25 - ref_seconds)))
    chunks = split_sentences(text, max_chars)

    torch.manual_seed(args.seed)
    start = time.time()
    wave, out_sr, _ = infer_batch_process(
        (ref_audio, sr), ref_text, chunks, model, vocoder,
        nfe_step=args.nfe, cfg_strength=args.cfg, sway_sampling_coef=args.sway, speed=args.speed, device=args.device,
    )
    elapsed = time.time() - start
    sf.write(args.out, wave, out_sr)
    audio_seconds = len(wave) / out_sr
    print(f"Wrote {args.out}: {audio_seconds:.2f}s audio in {elapsed:.1f}s ({len(chunks)} chunk(s), RTF {elapsed / audio_seconds:.2f})")


if __name__ == "__main__":
    main()

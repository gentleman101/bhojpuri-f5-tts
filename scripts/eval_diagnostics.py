"""Run the diagnostic sentences through a model and save the audio for listening comparison.

  python scripts/eval_diagnostics.py --name baseline_stock
  python scripts/eval_diagnostics.py --name lora_slice_2k --adapter runs/lora_slice_r32/checkpoints/step_0002000

Every run uses the same sentences, the same reference clips and the same seed, so runs differ
only by the model. Results land in runs/eval/<name>/.
"""

import argparse
import json
import time

import soundfile as sf
import torch
import torchaudio

from bhojpuri_tts import REPO_ROOT
from bhojpuri_tts.data import SAMPLE_RATE
from bhojpuri_tts.modeling import F5_BASE_ARCH, load_model_for_inference

DEFAULT_VOCAB = "checkpoints/IndicF5/checkpoints/vocab.txt"
DEFAULT_BASE = "checkpoints/IndicF5/model.safetensors"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Label for this run, e.g. baseline_stock")
    parser.add_argument("--diagnostics", default="manifests/diagnostics.json")
    parser.add_argument("--adapter", help="LoRA checkpoint directory; omit for stock IndicF5")
    parser.add_argument("--raw-weights", action="store_true", help="Use raw adapter weights instead of EMA")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from f5_tts.infer.utils_infer import infer_batch_process, load_vocoder

    spec = json.loads((REPO_ROOT / args.diagnostics).read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / "runs" / "eval" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_for_inference(
        str(REPO_ROOT / args.vocab), str(REPO_ROOT / args.base_checkpoint), args.adapter, not args.raw_weights, F5_BASE_ARCH
    ).to(args.device)
    vocoder = load_vocoder("vocos", device=args.device)
    refs = {s: torchaudio.load(REPO_ROOT / r["audio_path"]) for s, r in spec["refs"].items()}

    results = []
    for i, item in enumerate(spec["items"], 1):
        ref = spec["refs"][item["speaker"]]
        audio, sr = refs[item["speaker"]]
        torch.manual_seed(args.seed)
        start = time.time()
        wave, out_sr, _ = infer_batch_process(
            (audio, sr), ref["text"] + " ", [item["text"]], model, vocoder,
            nfe_step=args.nfe, cfg_strength=args.cfg, sway_sampling_coef=args.sway, device=args.device,
        )
        elapsed = time.time() - start
        filename = f"{item['category']}_{item['speaker']}_{item['utt_id'].split('_')[-1]}.wav"
        sf.write(out_dir / filename, wave, out_sr)
        results.append(dict(**item, generated=filename, gen_duration=round(len(wave) / out_sr, 2), seconds=round(elapsed, 1)))
        print(f"[{i}/{len(spec['items'])}] {item['category']:12} {filename}  "
              f"{results[-1]['gen_duration']:.2f}s vs real {item['duration']:.2f}s  ({elapsed:.0f}s)")

    meta = dict(name=args.name, adapter=args.adapter, nfe=args.nfe, cfg=args.cfg, sway=args.sway,
                seed=args.seed, device=args.device, refs=spec["refs"], results=results)
    (out_dir / "results.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    drift = [abs(r["gen_duration"] - r["duration"]) / r["duration"] for r in results]
    print(f"\n{len(results)} clips -> {out_dir.relative_to(REPO_ROOT)}")
    print(f"mean duration drift vs real recording: {100 * sum(drift) / len(drift):.1f}%")


if __name__ == "__main__":
    main()

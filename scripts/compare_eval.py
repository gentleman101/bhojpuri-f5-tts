"""Compare eval_diagnostics outputs to the real recordings of the same sentences: pitch contour, pitch range, spectrum.

    python scripts/compare_eval.py baseline_stock lora_slice [more names...]

Each name is a folder under runs/eval/. Every generated clip is time-aligned to its real recording (DTW on MFCCs), then:
  f0_corr    Pearson correlation of the two pitch contours over frames voiced in both (1 = same intonation shape)
  f0_rmse_st RMS pitch difference in semitones after removing each clip's own mean pitch (0 = same contour, speaker offset ignored)
  f0_range   the generated clip's pitch spread (std, semitones) divided by the real clip's (1 = same expressiveness; <1 = flatter)
  mfcc_dist  mean distance between aligned MFCC frames c1-c12 (lower = spectrally closer; relative, not a calibrated MCD)
These are proxies. They say nothing about which sounds are right, only how close the melody and spectrum are to a human
saying the same sentence; listening still decides. Duration is not compared: F5 fixes output length from the text, so it
is identical across models.
"""

import argparse
import json
import sys
from collections import defaultdict

import librosa
import numpy as np

from bhojpuri_tts import REPO_ROOT

SR, HOP = 24000, 256


def f0_track(y: np.ndarray) -> np.ndarray:
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=450, sr=SR, frame_length=1024, hop_length=HOP)
    return f0  # NaN where unvoiced


def mfcc(y: np.ndarray) -> np.ndarray:
    return librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, hop_length=HOP)[1:]  # drop c0 (energy)


def compare_clip(gen: np.ndarray, real: np.ndarray) -> dict:
    mg, mr = mfcc(gen), mfcc(real)
    _, path = librosa.sequence.dtw(X=mg, Y=mr, metric="euclidean")
    path = path[::-1]
    dist = float(np.mean([np.linalg.norm(mg[:, i] - mr[:, j]) for i, j in path]))
    fg, fr = f0_track(gen), f0_track(real)
    n = min(len(fg), len(fr), mg.shape[1], mr.shape[1])
    stg, str_ = 12 * np.log2(fg[:n] / 100), 12 * np.log2(fr[:n] / 100)  # semitones re 100 Hz
    stg, str_ = stg - np.nanmean(stg), str_ - np.nanmean(str_)
    pairs = [(stg[i], str_[j]) for i, j in path if i < n and j < n and np.isfinite(stg[i]) and np.isfinite(str_[j])]
    a, b = np.array(pairs).T if pairs else (np.array([]), np.array([]))
    out = dict(mfcc_dist=dist)
    if len(a) > 10:
        out.update(f0_corr=float(np.corrcoef(a, b)[0, 1]), f0_rmse_st=float(np.sqrt(np.mean((a - b) ** 2))),
                   f0_range=float(np.nanstd(stg) / np.nanstd(str_)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", help="folders under runs/eval/, e.g. baseline_stock lora_slice")
    parser.add_argument("--diagnostics", default="manifests/diagnostics.json")
    args = parser.parse_args()

    items = json.loads((REPO_ROOT / args.diagnostics).read_text(encoding="utf-8"))["items"]
    real = {i["utt_id"]: librosa.load(REPO_ROOT / i["audio_path"], sr=SR)[0] for i in items}
    report = {}
    for name in args.names:
        meta = json.loads((REPO_ROOT / "runs/eval" / name / "results.json").read_text(encoding="utf-8"))
        rows = []
        for item, res in zip(items, meta["results"]):
            assert item["utt_id"] == res["utt_id"], "results.json order differs from diagnostics.json"
            gen = librosa.load(REPO_ROOT / "runs/eval" / name / res["generated"], sr=SR)[0]
            rows.append(dict(category=item["category"], speaker=item["speaker"], utt_id=item["utt_id"],
                             **compare_clip(gen, real[item["utt_id"]])))
            print(f"\r{name}: {len(rows)}/{len(items)}", end="", file=sys.stderr)
        print(file=sys.stderr)
        report[name] = rows
    out = REPO_ROOT / "runs/eval" / ("compare_" + "_vs_".join(args.names) + ".json")
    out.write_text(json.dumps(report, indent=1))

    metrics = ["f0_corr", "f0_rmse_st", "f0_range", "mfcc_dist"]

    def mean(rows, m):
        v = [r[m] for r in rows if m in r]
        return f"{np.mean(v):7.3f}" if v else "    n/a"

    print(f"\n{'':14}" + "".join(f"{n:>26}" for n in args.names))
    print(f"{'overall':14}" + "".join(f"{m:>7}" for _ in args.names for m in ["corr", "rmse", "range", "dist"]))
    print(f"{'':14}" + "".join("".join(mean(report[n], m) for m in metrics) + "  " for n in args.names))
    for cat in sorted({r["category"] for r in report[args.names[0]]}):
        print(f"{cat:14}" + "".join("".join(mean([r for r in report[n] if r["category"] == cat], m) for m in metrics) + "  "
                                     for n in args.names))
    print(f"\n(higher is better for corr; lower for rmse and dist; range closer to 1.0 is better)\nsaved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

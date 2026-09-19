"""SYSPIN Bhojpuri -> 24 kHz mono clips + train/val/test manifests.

Test and val clips are chosen before the train slice, so they stay identical whether you
prepare a 1-hour slice or the full corpus. Processed audio is shared across runs and skipped if it exists.
"""

import argparse
import json
import random
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm import tqdm

from bhojpuri_tts import REPO_ROOT
from bhojpuri_tts.data import SAMPLE_RATE, write_manifest
from bhojpuri_tts.text import normalize_text


def trim_silence(wav: np.ndarray, sr: int, top_db: float, margin_s: float) -> np.ndarray:
    frame = int(0.02 * sr)
    n_frames = len(wav) // frame
    if n_frames == 0:
        return wav
    rms = np.sqrt(np.mean(wav[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1) + 1e-10)
    db = 20 * np.log10(rms)
    voiced = np.flatnonzero(db > db.max() - top_db)
    margin = int(margin_s * sr)
    start = max(voiced[0] * frame - margin, 0)
    end = min((voiced[-1] + 1) * frame + margin, len(wav))
    return wav[start:end]


def process_clip(job: tuple[str, str, float, float]) -> float:
    src, dst, top_db, margin_s = job
    dst = Path(dst)
    if not dst.exists():
        wav, sr = sf.read(src, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1))
        wav = AF.resample(wav, sr, SAMPLE_RATE).numpy()
        wav = trim_silence(wav, SAMPLE_RATE, top_db, margin_s)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.wav")
        sf.write(tmp, wav, SAMPLE_RATE, subtype="PCM_16")
        tmp.rename(dst)
    return sf.info(dst).duration


def discover_speakers(syspin_root: Path) -> dict[str, Path]:
    speakers = {}
    for corpus_dir in sorted(syspin_root.glob("*/IISc_SYSPIN_Data/IISc_SYSPINProject_Bhojpuri_*_HC")):
        gender = corpus_dir.name.split("_")[3].lower()
        speakers[f"bho_{gender[0]}"] = corpus_dir
    if not speakers:
        raise SystemExit(f"No SYSPIN Bhojpuri corpora found under {syspin_root}")
    return speakers


def load_items(speaker: str, corpus_dir: Path) -> list[dict]:
    transcripts = json.loads(next(corpus_dir.glob("*_Transcripts.json")).read_text(encoding="utf-8"))["Transcripts"]
    items = []
    for utt_id, entry in transcripts.items():
        text = normalize_text(entry["Transcript"])
        wav = corpus_dir / "wav" / f"{utt_id}.wav"
        if text and wav.exists() and "|" not in text:
            items.append(dict(utt_id=utt_id, speaker=speaker, text=text, domain=entry["Domain"], src=wav))
    return sorted(items, key=lambda x: x["utt_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syspin-root", default="data/syspin")
    parser.add_argument("--processed-dir", default="data/processed/syspin_24k")
    parser.add_argument("--name", required=True, help="Manifest set name, e.g. syspin_slice or syspin_full")
    parser.add_argument("--train-hours-per-speaker", type=float, default=None, help="Omit to use everything")
    parser.add_argument("--exclude-file", help="utt_ids (one per line) to force into test and never train on")
    parser.add_argument("--val-per-speaker", type=int, default=50)
    parser.add_argument("--test-per-speaker", type=int, default=50)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--trim-top-db", type=float, default=40.0)
    parser.add_argument("--trim-margin", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    syspin_root = REPO_ROOT / args.syspin_root
    processed_dir = REPO_ROOT / args.processed_dir
    out_dir = REPO_ROOT / "manifests" / args.name
    rng = random.Random(args.seed)

    excluded = set()
    if args.exclude_file:
        excluded = {line.strip() for line in open(REPO_ROOT / args.exclude_file, encoding="utf-8") if line.strip()}
        print(f"Forcing {len(excluded)} clip(s) into the test split")

    splits = {"train": [], "val": [], "test": []}
    for speaker, corpus_dir in discover_speakers(syspin_root).items():
        items = load_items(speaker, corpus_dir)
        for item in items:
            item["src_duration"] = sf.info(item["src"]).duration
        items = [x for x in items if args.min_duration <= x["src_duration"] <= args.max_duration + 1.0]
        forced = [x for x in items if x["utt_id"] in excluded]
        items = [x for x in items if x["utt_id"] not in excluded]
        rng.shuffle(items)
        n_test, n_val = args.test_per_speaker, args.val_per_speaker
        splits["test"] += forced + items[:n_test]
        splits["val"] += items[n_test : n_test + n_val]
        train, budget = [], (args.train_hours_per_speaker or float("inf")) * 3600
        for item in items[n_test + n_val :]:
            if budget < item["src_duration"]:
                break
            train.append(item)
            budget -= item["src_duration"]
        splits["train"] += train
        print(f"{speaker}: {len(items)} usable clips, train {len(train)}, val {n_val}, test {n_test + len(forced)}")

    all_items = [item for rows in splits.values() for item in rows]
    for item in all_items:
        item["dst"] = processed_dir / item["speaker"] / f"{item['utt_id']}.wav"
    jobs = [(str(x["src"]), str(x["dst"]), args.trim_top_db, args.trim_margin) for x in all_items]
    with ProcessPoolExecutor(args.workers) as pool:
        durations = list(tqdm(pool.map(process_clip, jobs, chunksize=16), total=len(jobs), desc="Resampling"))

    char_counts: Counter = Counter()
    for item, duration in zip(all_items, durations):
        item["duration"] = duration
        item["audio_path"] = item["dst"].relative_to(REPO_ROOT).as_posix()
    for split, rows in splits.items():
        kept = [x for x in rows if args.min_duration <= x["duration"] <= args.max_duration]
        kept.sort(key=lambda x: (x["speaker"], x["utt_id"]))
        write_manifest(out_dir / f"{split}.csv", kept)
        if split == "train":
            for x in kept:
                char_counts.update(x["text"])
        by_speaker = Counter()
        for x in kept:
            by_speaker[x["speaker"]] += x["duration"] / 3600
        summary = ", ".join(f"{s} {h:.2f}h" for s, h in sorted(by_speaker.items()))
        print(f"{split}: {len(kept)} clips ({len(rows) - len(kept)} dropped after trim) — {summary}")

    with open(out_dir / "charset.tsv", "w", encoding="utf-8") as f:
        f.write("char\tcodepoint\tname\tcount\n")
        for char, count in sorted(char_counts.items(), key=lambda kv: -kv[1]):
            f.write(f"{char}\tU+{ord(char):04X}\t{unicodedata.name(char, '?')}\t{count}\n")
    # Space must be index 0: F5's tokenizer maps unknown characters to 0.
    with open(out_dir / "vocab_data.txt", "w", encoding="utf-8") as f:
        f.write(" \n")
        f.writelines(f"{c}\n" for c in sorted(char_counts) if c != " ")
    print(f"Wrote manifests, charset.tsv and vocab_data.txt to {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

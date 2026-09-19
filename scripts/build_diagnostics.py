"""Pick held-out sentences that stress each Bhojpuri-vs-Hindi pronunciation contrast.

Prefers sentences already in the test split. If a contrast is too rare to appear there, it takes
candidates from the wider corpus and writes them to an exclude file; re-run prepare_syspin.py with
--exclude-file so those clips are forced into test and never trained on.
"""

import argparse
import json
import re
from pathlib import Path

from bhojpuri_tts import REPO_ROOT
from bhojpuri_tts.data import read_manifest
from bhojpuri_tts.diagnostics import CATEGORIES
from bhojpuri_tts.text import normalize_text


def corpus_candidates(syspin_root: Path) -> list[dict]:
    items = []
    for path in sorted(syspin_root.glob("*/IISc_SYSPIN_Data/*/*_Transcripts.json")):
        speaker = "bho_f" if "_Female_" in path.name else "bho_m"
        for utt_id, entry in json.loads(path.read_text(encoding="utf-8"))["Transcripts"].items():
            items.append(dict(utt_id=utt_id, speaker=speaker, text=normalize_text(entry["Transcript"])))
    return sorted(items, key=lambda x: x["utt_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-manifest", default="manifests/syspin_full/test.csv")
    parser.add_argument("--syspin-root", default="data/syspin")
    parser.add_argument("--out", default="manifests/diagnostics.json")
    parser.add_argument("--per-category", type=int, default=4, help="Sentences per contrast, split across speakers")
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=12.0)
    args = parser.parse_args()

    test_rows = [r for r in read_manifest(REPO_ROOT / args.test_manifest) if args.min_duration <= r["duration"] <= args.max_duration]
    by_speaker = sorted({r["speaker"] for r in test_rows})
    # One fixed reference clip per speaker, so voice is held constant across every evaluation.
    refs = {s: min((r for r in test_rows if r["speaker"] == s), key=lambda r: abs(r["duration"] - 6.0)) for s in by_speaker}

    items, missing, used = [], [], set(refs[s]["utt_id"] for s in by_speaker)
    for name, (pattern, note) in CATEGORIES.items():
        regex = re.compile(pattern)
        picked = []
        for speaker in by_speaker:
            hits = [r for r in test_rows if r["speaker"] == speaker and r["utt_id"] not in used and regex.search(r["text"])]
            for row in hits[: args.per_category // len(by_speaker)]:
                used.add(row["utt_id"])
                picked.append(dict(category=name, note=note, speaker=speaker, utt_id=row["utt_id"],
                                   text=row["text"], audio_path=row["audio_path"], duration=row["duration"]))
        items += picked
        per_speaker = args.per_category // len(by_speaker)
        for speaker in by_speaker:
            shortfall = per_speaker - sum(1 for p in picked if p["speaker"] == speaker)
            if shortfall > 0:
                missing.append((name, speaker, shortfall))

    extra_excludes = []
    if missing:
        candidates = corpus_candidates(REPO_ROOT / args.syspin_root)
        for name, speaker, shortfall in missing:
            regex = re.compile(CATEGORIES[name][0])
            hits = [c for c in candidates if c["speaker"] == speaker and c["utt_id"] not in used and regex.search(c["text"])]
            for cand in hits[:shortfall]:
                used.add(cand["utt_id"])
                extra_excludes.append(cand["utt_id"])
            print(f"{name}/{speaker}: {len(hits[:shortfall])} of {shortfall} clip(s) found in wider corpus")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(refs=refs, items=items), ensure_ascii=False, indent=1), encoding="utf-8")
    counts = {}
    for item in items:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    print(f"Wrote {len(items)} diagnostic sentences to {args.out}: {counts}")
    print("References:", {s: r["utt_id"] for s, r in refs.items()})

    # Every clip the diagnostics touch must be excluded from training, not only newly found ones.
    exclude_path = REPO_ROOT / "manifests" / "diagnostics_exclude.txt"
    previous = exclude_path.read_text(encoding="utf-8").split() if exclude_path.exists() else []
    all_ids = sorted(set(previous) | set(extra_excludes) | {i["utt_id"] for i in items} | {r["utt_id"] for r in refs.values()})
    exclude_path.write_text("\n".join(all_ids) + "\n", encoding="utf-8")
    print(f"\n{len(all_ids)} clip(s) pinned out of training in {exclude_path.relative_to(REPO_ROOT)}.")
    if extra_excludes:
        print("Re-run: python scripts/prepare_syspin.py --name syspin_full --exclude-file manifests/diagnostics_exclude.txt")
        print("then re-run this script so those sentences are held out of training.")


if __name__ == "__main__":
    main()

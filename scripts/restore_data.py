"""Restore the processed clips from the private Hugging Face dataset repo (written by scripts/pack_data.py + push_data).

    python scripts/restore_data.py [--repo gentleman101/bhojpuri-syspin-24k] [--dest data/processed/syspin_24k]

Downloads the tar shards, checks each sha256 against index.json, extracts them, then checks the total WAV count and that
every audio_path in manifests/*/{train,val,test}.csv exists. About 15 GB; takes minutes instead of the ~20 min rebuild from
the raw corpus, and works after the SYSPIN download links have expired.
"""

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

from bhojpuri_tts import REPO_ROOT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="gentleman101/bhojpuri-syspin-24k")
    parser.add_argument("--dest", default="data/processed/syspin_24k")
    parser.add_argument("--cache", default="/home/assets/restore_cache", help="where shards are downloaded (deleted after)")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--skip-manifest-check", action="store_true")
    args = parser.parse_args()

    dest, cache = REPO_ROOT / args.dest, Path(args.cache)
    snapshot_download(args.repo, repo_type="dataset", local_dir=cache, max_workers=8)
    index = json.loads((cache / "index.json").read_text())
    dest.mkdir(parents=True, exist_ok=True)
    for name, meta in index["shards"].items():
        if sha256(cache / name) != meta["sha256"]:
            raise SystemExit(f"CORRUPT shard {name}: sha256 mismatch — delete {cache} and rerun")
        with tarfile.open(cache / name) as tar:
            tar.extractall(dest, filter="data")
        print(f"extracted {name} ({meta['files']} files)", flush=True)
    n = sum(1 for _ in dest.rglob("*.wav"))
    if n != index["n_files"]:
        raise SystemExit(f"expected {index['n_files']} WAVs, found {n}")
    print(f"OK: {n} WAVs in {dest}")
    if not args.skip_manifest_check:
        missing = 0
        for manifest in sorted(m for n in ("syspin_slice", "syspin_10h", "syspin_full") for m in (REPO_ROOT / "manifests" / n).glob("*.csv")):
            for row in csv.DictReader(open(manifest, encoding="utf-8"), delimiter="|"):
                if "audio_path" in row and not (REPO_ROOT / row["audio_path"]).exists():
                    missing += 1
        if missing:
            raise SystemExit(f"{missing} manifest rows point to missing files")
        print("OK: every manifest row points to an existing clip")
    if not args.keep_cache:
        import shutil

        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()

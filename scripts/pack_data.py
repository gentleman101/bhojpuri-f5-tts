"""Pack the processed 24 kHz clips into tar shards (one flat file per ~450 MB) for backup on the Hugging Face Hub.

    python scripts/pack_data.py [--src data/processed/syspin_24k] [--out /home/assets/shards]

53k tiny WAVs upload and restore far slower than a few dozen large files, so this writes uncompressed tar shards plus
index.json (file counts, sizes, sha256 per shard). Restore with scripts/restore_data.py, which verifies the hashes.
Shards keep paths relative to the syspin_24k folder, e.g. bho_f/IISc_SYSPINProject_bho_f_AGRI_00001.wav.
"""

import argparse
import hashlib
import json
import tarfile
from multiprocessing import Pool
from pathlib import Path

from bhojpuri_tts import REPO_ROOT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_shard(job):
    src, out, name, files = job
    path = out / name
    with tarfile.open(path, "w") as tar:
        for f in files:
            tar.add(src / f, arcname=f)
    return name, dict(files=len(files), bytes=path.stat().st_size, sha256=sha256(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="data/processed/syspin_24k")
    parser.add_argument("--out", default="/home/assets/shards")
    parser.add_argument("--files-per-shard", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    src, out = REPO_ROOT / args.src, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for speaker_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        files = sorted(f"{speaker_dir.name}/{p.name}" for p in speaker_dir.glob("*.wav"))
        for i in range(0, len(files), args.files_per_shard):
            jobs.append((src, out, f"{speaker_dir.name}_{i // args.files_per_shard:03d}.tar", files[i:i + args.files_per_shard]))
    with Pool(args.workers) as pool:
        shards = dict(pool.map(write_shard, jobs))
    index = dict(n_files=sum(s["files"] for s in shards.values()), total_bytes=sum(s["bytes"] for s in shards.values()),
                 shards=dict(sorted(shards.items())))
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(shards)} shards, {index['n_files']} files, {index['total_bytes'] / 2**30:.1f} GiB -> {out}")


if __name__ == "__main__":
    main()

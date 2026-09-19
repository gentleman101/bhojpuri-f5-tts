#!/usr/bin/env bash
# Download + extract the SYSPIN Bhojpuri human-checked corpus (~41 GB compressed, ~48 GB extracted).
#
# Links are personal and time-limited (7 days), so they are NOT stored in git. Request them at
# https://spiredatasets.ee.iisc.ac.in/syspincorpus  (Bhojpuri, Female + Male, Human Checked);
# they arrive by email as two wget commands.
#
#   ./scripts/download_syspin.sh "<female_url>" "<male_url>"
#
# Archives are downloaded resumably, verified by size, extracted, then deleted, so peak disk is
# ~24 GB above the extracted size rather than double it. Re-running skips speakers already done.
set -euo pipefail

if [ $# -ne 2 ]; then
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

cd "$(dirname "$0")/.."
DEST=data/syspin
mkdir -p "$DEST"
cd "$DEST"

declare -A URL=([Female]="$1" [Male]="$2")

for g in Female Male; do
  f="IISc_SYSPINProject_Bhojpuri_${g}_Spk001_HC.tar.gz"
  if [ -f ".done_$g" ]; then echo "[$g] already extracted, skipping"; continue; fi
  echo "[$g] downloading $(date)"
  wget -c --tries=20 --retry-connrefused --waitretry=30 --progress=dot:giga -O "$f" "${URL[$g]}"
  expected=$(curl -sI -r 0-0 "${URL[$g]}" | grep -i content-range | tr -d '\r' | sed 's:.*/::')
  actual=$(stat -c %s "$f")
  if [ -n "$expected" ] && [ "$actual" != "$expected" ]; then
    echo "[$g] size mismatch: got $actual, expected $expected" >&2
    exit 1
  fi
  echo "[$g] extracting $(date)"
  mkdir -p "$g" && tar -xzf "$f" -C "$g"
  rm "$f" && touch ".done_$g"
  echo "[$g] done $(date)"
done
echo "ALL DONE $(date)"

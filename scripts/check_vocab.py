"""Find characters in manifests that are missing from a model vocab.txt (they would silently become spaces)."""

import argparse
import unicodedata
from collections import Counter

from bhojpuri_tts import REPO_ROOT
from bhojpuri_tts.data import read_manifest
from bhojpuri_tts.text import to_char_tokens


def read_vocab(path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line[:-1] if line.endswith("\n") else line for line in f]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--extend-out", help="Write vocab + missing characters (appended, keeping existing indices)")
    args = parser.parse_args()

    vocab = read_vocab(args.vocab)
    known = set(vocab)
    missing, examples, total = Counter(), {}, 0
    for manifest in args.manifests:
        for row in read_manifest(REPO_ROOT / manifest):
            for token in to_char_tokens([row["text"]])[0]:
                total += 1
                if token not in known:
                    missing[token] += 1
                    examples.setdefault(token, row["text"][:80])

    print(f"vocab size {len(vocab)}, tokens checked {total}, missing token types {len(missing)}")
    for token, count in missing.most_common():
        names = " + ".join(unicodedata.name(c, f"U+{ord(c):04X}") for c in token)
        print(f"  {token!r:8} {count:8d}  {names}  e.g. {examples[token]}")

    if args.extend_out:
        with open(args.extend_out, "w", encoding="utf-8") as f:
            f.writelines(f"{token}\n" for token in vocab + sorted(missing))
        print(f"Wrote extended vocab ({len(vocab) + len(missing)} entries) to {args.extend_out}")


if __name__ == "__main__":
    main()

import csv
import random

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from f5_tts.model.modules import MelSpec

from bhojpuri_tts import REPO_ROOT
from bhojpuri_tts.text import to_char_tokens

MANIFEST_FIELDS = ["audio_path", "text", "duration", "speaker", "utt_id", "domain"]
SAMPLE_RATE = 24_000
HOP_LENGTH = 256
MEL_KWARGS = dict(
    n_fft=1024,
    hop_length=HOP_LENGTH,
    win_length=1024,
    n_mel_channels=100,
    target_sample_rate=SAMPLE_RATE,
    mel_spec_type="vocos",
)


def read_manifest(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="|"))
    for row in rows:
        row["duration"] = float(row["duration"])
    return rows


def write_manifest(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, delimiter="|")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{row[k]:.3f}" if k == "duration" else row[k]) for k in MANIFEST_FIELDS})


def load_wav(audio_path: str) -> torch.Tensor:
    wav, sr = sf.read(REPO_ROOT / audio_path, dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE, f"{audio_path}: expected {SAMPLE_RATE} Hz, got {sr}"
    return torch.from_numpy(wav.mean(axis=1))


class MelDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.mel_spec = MelSpec(**MEL_KWARGS)

    def frame_len(self, index: int) -> float:
        return self.rows[index]["duration"] * SAMPLE_RATE / HOP_LENGTH

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        mel = self.mel_spec(load_wav(row["audio_path"])[None]).squeeze(0)  # (n_mels, frames)
        return {"mel": mel, "text": row["text"]}


def collate(batch: list[dict]) -> dict:
    lens = torch.tensor([item["mel"].shape[-1] for item in batch])
    mel = torch.stack([F.pad(item["mel"], (0, int(lens.max()) - item["mel"].shape[-1])) for item in batch])
    return {
        "mel": mel.permute(0, 2, 1),  # (batch, frames, n_mels) as CFM expects
        "lens": lens,
        "text": to_char_tokens([item["text"] for item in batch]),
    }


class FrameBatchSampler(Sampler[list[int]]):
    """Batches of similar-length clips whose total mel frames stay under a budget; batch order reshuffled per epoch."""

    def __init__(self, frame_lens: list[float], max_frames: int, max_samples: int, seed: int, shuffle: bool = True):
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0
        self.batches, current, current_frames = [], [], 0.0
        for index in sorted(range(len(frame_lens)), key=frame_lens.__getitem__):
            frames = frame_lens[index]
            if frames > max_frames:
                continue
            if current and (current_frames + frames > max_frames or len(current) >= max_samples):
                self.batches.append(current)
                current, current_frames = [], 0.0
            current.append(index)
            current_frames += frames
        if current:
            self.batches.append(current)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        return (self.batches[i] for i in order)

    def __len__(self):
        return len(self.batches)

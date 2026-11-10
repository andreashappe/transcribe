#!/usr/bin/env python3
"""Transcribe an audio file with the local Parakeet server and label speakers
with pyannote.audio.

Usage:
    python transcribe_diarize.py "audio(2).mp3" [--num-speakers N]
                                                 [--min-speakers N] [--max-speakers N]
                                                 [--hf-token TOKEN]

Requires a Hugging Face token (env HF_TOKEN or --hf-token) and acceptance of the
model's terms at https://huggingface.co/pyannote/speaker-diarization-3.1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

PARAKEET_URL = "http://127.0.0.1:5092/v1"
PARAKEET_MODEL = "parakeet-tdt-0.6b-v3"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
SAMPLE_RATE = 16000


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str = "UNKNOWN"


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


# ---------------------------------------------------------------- Parakeet ---

def transcribe(audio_path: Path) -> tuple[str, list[Word]]:
    client = OpenAI(base_url=PARAKEET_URL, api_key="sk-no-key-required", timeout=3600)
    with audio_path.open("rb") as f:
        result = client.audio.transcriptions.create(
            model=PARAKEET_MODEL,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
    text = result.text
    tokens = [(w.word, w.start, w.end) for w in (result.words or [])]
    return text, merge_tokens_into_words(text, tokens)


def merge_tokens_into_words(text: str, tokens: list[tuple[str, float, float]]) -> list[Word]:
    """The server reports timestamps per sub-word token ("rec", "ord", "ing").
    Walk through the full text and glue tokens together until whitespace
    appears in the text, so each Word carries a real word and its time span."""
    words: list[Word] = []
    pos = 0
    current: Word | None = None
    for tok, start, end in tokens:
        if not tok:
            continue
        had_space = False
        while pos < len(text) and text[pos].isspace():
            pos += 1
            had_space = True
        if text.startswith(tok, pos):
            pos += len(tok)
        else:
            # alignment lost; resync by searching ahead, else treat as new word
            idx = text.find(tok, pos)
            if idx != -1:
                had_space = had_space or text[pos:idx].strip() == ""
                pos = idx + len(tok)
            had_space = True
        if current is None or had_space:
            current = Word(tok, start, end)
            words.append(current)
        else:
            current.text += tok
            current.end = max(current.end, end)
    return words


# ---------------------------------------------------------------- pyannote ---

def load_audio_16k(audio_path: Path) -> np.ndarray:
    """Decode anything ffmpeg understands into 16 kHz mono float32."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(audio_path),
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]
    pcm = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def diarize(audio_path: Path, hf_token: str, num_speakers: int | None,
            min_speakers: int | None, max_speakers: int | None) -> list[Turn]:
    import torch
    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
    except TypeError:  # pyannote.audio < 4 uses the old keyword
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
    if pipeline is None:
        sys.exit(
            f"Could not load {DIARIZATION_MODEL}. Accept the model terms on "
            "huggingface.co and make sure the token is valid."
        )
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    waveform = torch.from_numpy(load_audio_16k(audio_path)).unsqueeze(0)
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    if min_speakers:
        kwargs["min_speakers"] = min_speakers
    if max_speakers:
        kwargs["max_speakers"] = max_speakers

    result = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kwargs)
    annotation = getattr(result, "speaker_diarization", result)  # pyannote 4 vs 3

    turns = [
        Turn(float(seg.start), float(seg.end), str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t.start)
    return turns


# ------------------------------------------------------------------- merge ---

def assign_speakers(words: list[Word], turns: list[Turn]) -> None:
    """Give each word the speaker whose turn overlaps it most; if no turn
    overlaps (e.g. pyannote saw silence), use the nearest turn in time."""
    if not turns:
        return
    for w in words:
        best, best_overlap = None, 0.0
        for t in turns:
            overlap = min(w.end, t.end) - max(w.start, t.start)
            if overlap > best_overlap:
                best, best_overlap = t, overlap
        if best is None:
            mid = (w.start + w.end) / 2
            best = min(turns, key=lambda t: min(abs(t.start - mid), abs(t.end - mid)))
        w.speaker = best.speaker


def rename_speakers(words: list[Word]) -> None:
    """SPEAKER_03 -> Speaker 1 etc., numbered in order of first appearance."""
    mapping: dict[str, str] = {}
    for w in words:
        if w.speaker not in mapping:
            mapping[w.speaker] = f"Speaker {len(mapping) + 1}"
        w.speaker = mapping[w.speaker]


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render(words: list[Word]) -> str:
    lines = []
    para: list[Word] = []
    for w in words:
        if para and w.speaker != para[-1].speaker:
            lines.append(f"[{fmt_time(para[0].start)}] {para[0].speaker}: "
                         + " ".join(x.text for x in para))
            para = []
        para.append(w)
    if para:
        lines.append(f"[{fmt_time(para[0].start)}] {para[0].speaker}: "
                     + " ".join(x.text for x in para))
    return "\n\n".join(lines) + "\n"


# -------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="default: <audio>.speakers.txt")
    ap.add_argument("--num-speakers", type=int, help="exact number of speakers, if known")
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    if not args.hf_token:
        sys.exit("A Hugging Face token is required: set HF_TOKEN or pass --hf-token.")
    output = args.output or args.audio.with_suffix(".speakers.txt")

    t0 = time.time()
    print(f"Transcribing {args.audio} with Parakeet ...", flush=True)
    text, words = transcribe(args.audio)
    print(f"  {len(words)} words in {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    print("Running speaker diarization with pyannote (this is slow on CPU) ...", flush=True)
    turns = diarize(args.audio, args.hf_token, args.num_speakers, args.min_speakers, args.max_speakers)
    n_speakers = len({t.speaker for t in turns})
    print(f"  {len(turns)} turns, {n_speakers} speakers in {time.time() - t1:.0f}s", flush=True)

    assign_speakers(words, turns)
    rename_speakers(words)
    output.write_text(render(words), encoding="utf-8")
    # keep the plain transcript too
    args.audio.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(f"Speaker transcript written to {output}")


if __name__ == "__main__":
    main()

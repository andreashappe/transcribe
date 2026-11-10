# transcribe

Two small scripts for transcribing long audio recordings (interviews,
meetings, focus groups) fully offline on a CPU-only machine:

| Script | What it does | Output |
|---|---|---|
| `testme.py` | Plain transcription with NVIDIA Parakeet TDT 0.6B v3 | `<audio>.txt` |
| `transcribe_diarize.py` | Transcription plus speaker diarization with pyannote.audio, so every paragraph is labelled with who spoke | `<audio>.speakers.txt` and `<audio>.txt` |

Transcription runs in a local Docker container, diarization runs in-process
with PyTorch on the CPU. Audio never leaves the machine. The only network
access is the one-time download of model weights from Hugging Face.

## 1. Run the Parakeet server

Both scripts talk to the OpenAI-compatible Parakeet server from
[groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai](https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai)
on `http://127.0.0.1:5092`.

Build the CPU image once:

```bash
git clone https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai.git
cd parakeet-tdt-0.6b-v3-fastapi-openai
docker build -f Dockerfile.cpu -t parakeet-tdt:cpu .
```

Then start the container:

```bash
docker run -d --name parakeet-cpu -p 5092:5092 \
  -v parakeet-models:/app/models \
  -e PARAKEET_USE_GPU=false \
  -e PARAKEET_DEFAULT_MODEL=parakeet-tdt-0.6b-v3 \
  -e PARAKEET_BATCHED=0 \
  -e PARAKEET_MAX_AUDIO_SECONDS=10800 \
  parakeet-tdt:cpu
```

Notes:

- The first start downloads the model weights (about 2.5 GB) into the
  `parakeet-models` volume. Follow progress with `docker logs -f parakeet-cpu`
  and wait for `Service ready`.
- `PARAKEET_MAX_AUDIO_SECONDS` raises the per-file duration limit from the
  default of 2 hours to 3 hours. Without it, longer recordings are rejected
  with HTTP 413 and the message `audio duration ... exceeds limit 7200.0s`.
- The upload size limit defaults to 256 MB (`PARAKEET_MAX_UPLOAD_BYTES`).
- Check that the server is up with `curl http://127.0.0.1:5092/health`.
- After a reboot the container can be restarted with `docker start parakeet-cpu`.

## 2. Install the scripts

Requirements: Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), and
`ffmpeg` on the `PATH` (used to decode MP3 and other formats for diarization).

```bash
git clone <this repository>
cd transcribe
uv sync
```

`uv sync` creates `.venv` from the lock file. PyTorch is pinned to the
CPU-only wheel index in `pyproject.toml`, so the environment is about
1.3 GB instead of the multi-gigabyte CUDA build.

### Hugging Face token for pyannote (diarization only)

The pyannote models are gated. One-time setup:

1. Accept the terms on
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   and
   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Export it before running the script, or pass `--hf-token`:

   ```bash
   export HF_TOKEN=hf_...
   ```

The weights are downloaded on the first run and cached under
`~/.cache/huggingface`. Later runs work offline; set `HF_HUB_OFFLINE=1`
if you want to guarantee that nothing contacts Hugging Face.

## 3. Usage

### Plain transcript

```bash
uv run testme.py "audio.mp3"
```

Writes `audio.txt` next to the recording. Without an argument it
transcribes `audio(1).mp3`.

### Transcript with speaker labels

```bash
export HF_TOKEN=hf_...
uv run transcribe_diarize.py "audio.mp3" --min-speakers 5 --max-speakers 8
```

Options:

| Option | Meaning |
|---|---|
| `--num-speakers N` | Exact number of speakers, if known |
| `--min-speakers N` / `--max-speakers N` | Bound the automatic speaker count. Giving a realistic range noticeably improves results on group calls |
| `-o FILE` | Output path, default `<audio>.speakers.txt` |
| `--hf-token TOKEN` | Alternative to the `HF_TOKEN` environment variable |

Without any speaker options pyannote estimates the number of speakers on
its own.

Output looks like this:

```
[00:00:00] Speaker 1: Sorry. I will now start the recording. The first question ...

[00:00:39] Speaker 2: Okay, so the tool I got assigned to was ...
```

Speakers are numbered in order of first appearance. The script also writes
the plain `<audio>.txt` transcript.

### How it works

1. The audio is sent to the Parakeet server with word timestamps requested.
   Parakeet returns timestamps per sub-word token, which the script merges
   back into whole words using the full transcript text.
2. The audio is decoded with ffmpeg to 16 kHz mono and passed to the
   pyannote diarization pipeline, which returns speaker turns.
3. Each word is assigned to the speaker whose turn overlaps it most, and
   consecutive words from the same speaker are rendered as one paragraph.

### Performance

Everything runs on the CPU. Expect Parakeet to be several times faster than
real time, while pyannote diarization is roughly real time or slower, so a
90-minute recording can take well over an hour in total. The scripts use a
one-hour HTTP timeout for the transcription request.

## Files

- `testme.py`, `transcribe_diarize.py`: the scripts
- `pyproject.toml`, `uv.lock`, `.python-version`: dependency management
- `.gitignore` excludes the virtual environment, audio files and generated
  `.txt` transcripts, so recordings are never committed by accident

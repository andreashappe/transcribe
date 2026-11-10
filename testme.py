import sys
from pathlib import Path

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5092/v1",
    api_key="sk-no-key-required",
    timeout=3600,  # long recordings take a while on CPU
)

audio_path = Path(sys.argv[1] if len(sys.argv) > 1 else "audio(1).mp3")
output_path = audio_path.with_suffix(".txt")

with audio_path.open("rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="parakeet-tdt-0.6b-v3",
        file=audio_file,
        response_format="text",
    )

output_path.write_text(transcript, encoding="utf-8")
print(f"Transcript written to {output_path}")

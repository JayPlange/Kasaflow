"""
One-off manual check: transcribe a real voice clip via Khaya and print
the result. Not a pytest test despite living alongside dev tooling --
no assertions, just a quick eyeball check against a real audio file.

Run from the repo root:
    python scripts/manual_khaya_check.py
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from services.voice_tool import transcribe_audio

_CLIP_PATH = Path(__file__).parent / "fixtures" / "test_clip.ogg"

with open(_CLIP_PATH, "rb") as f:
    result = transcribe_audio(f.read(), language="twi")

print("TRANSCRIPT:", result)
"""
End-to-end check: Khaya ASR (transcription) -> OpenAI tool selection ->
tool execution -> customer-facing reply, using the SAME functions
whatsapp_routes.py calls for a real incoming voice note. Not a pytest
test -- no assertions, just a manual eyeball check against real audio.

Run from the repo root:
    python scripts/manual_voice_pipeline_check.py [path_to_ogg] [twi|english]
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from services.response_formatter import format_for_customer
from services.router import route_customer
from services.voice_tool import transcribe_audio

_DEFAULT_CLIP = Path(__file__).parent / "fixtures" / "test_clip.ogg"

clip_path = sys.argv[1] if len(sys.argv) > 1 else str(_DEFAULT_CLIP)
language = sys.argv[2] if len(sys.argv) > 2 else "twi"

print(f"--- Transcribing {clip_path} ({language}) via Khaya ---")
with open(clip_path, "rb") as f:
    transcript = transcribe_audio(f.read(), language=language)
print("TRANSCRIPT:", transcript)

if not transcript.strip():
    print("\nEmpty transcript -- stopping before the OpenAI call, nothing to route.")
    sys.exit(1)

print("\n--- Routing transcript through OpenAI tool selection + tool execution ---")
result = route_customer(transcript, session_id="local-test")
print("RAW RESULT:", result)

print("\n--- Customer-facing reply ---")
print(format_for_customer(result))

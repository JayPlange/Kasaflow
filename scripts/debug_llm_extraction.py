"""
One-off diagnostic: show exactly what the LLM extracted as its tool
call (tool name + arguments) for a transcript, without executing the
tool or touching the catalogue at all. Isolates the LLM tool-selection
layer from everything downstream of it (product matching, formatting),
so a wrong end-to-end result can be traced to "the LLM picked the wrong
tool/arguments" vs. "the LLM was right but the catalogue/matching let
it down."

Usage:
    python scripts/debug_llm_extraction.py <clip>.ogg <twi|english>
    python scripts/debug_llm_extraction.py --text "some raw text instead of audio"
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from services.llm import understand_customer
from services.voice_tool import transcribe_audio

if sys.argv[1] == "--text":
    transcript = " ".join(sys.argv[2:])
else:
    clip_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "twi"
    print(f"--- Transcribing {clip_path} ({language}) via Khaya ---")
    with open(clip_path, "rb") as f:
        transcript = transcribe_audio(f.read(), language=language)

print("TRANSCRIPT:", transcript)
print()
print("--- Raw LLM tool call (nothing executed yet) ---")
tool_request = understand_customer(transcript)
print("Tool:", tool_request["tool"])
print("Arguments:", tool_request["arguments"])

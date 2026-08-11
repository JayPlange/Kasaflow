"""
One-off manual check: send a real text message via send_text_message()
straight to the live WhatsApp Cloud API, bypassing the rest of the
pipeline entirely. Not a pytest test -- no assertions, just confirms
WHATSAPP_ACCESS_TOKEN / WHATSAPP_PHONE_NUMBER_ID in .env actually work
against the real API, the same role manual_vision_check.py plays for
the vision path.

The recipient must already be on this WhatsApp Business Account's
allowed test-recipient list (added via the app dashboard's API Setup
page) -- Meta rejects sends to any other number with a 400 while the
app is unpublished.

Run from the repo root:
    python scripts/manual_whatsapp_send_check.py <recipient_number> ["message text"]

<recipient_number> is E.164 without the leading +, e.g. 233509764406.
"""

import sys
from pathlib import Path

# Scripts in this folder are run as `python scripts/foo.py`, which puts
# only this folder on sys.path, not the repo root -- add it explicitly
# so `from services...`/`from app...` resolve regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from services.whatsapp_client import WhatsAppError, send_text_message

if len(sys.argv) < 2:
    print('Usage: python scripts/manual_whatsapp_send_check.py <recipient_number> ["message text"]')
    sys.exit(1)

recipient = sys.argv[1]
body = sys.argv[2] if len(sys.argv) > 2 else "KasaFlow manual send check: if you see this, the real API wiring works."

print(f"--- Sending to {recipient} via the live WhatsApp Cloud API ---")
try:
    send_text_message(recipient, body)
except WhatsAppError as e:
    print("FAILED:", e)
    sys.exit(1)

print("SENT OK -- check the recipient's phone to confirm delivery, not just this exit code.")

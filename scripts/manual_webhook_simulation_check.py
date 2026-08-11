"""
One-off manual check: POST a payload shaped exactly like Meta's real
WhatsApp webhook straight to a locally running KasaFlow instance,
skipping Meta's own delivery entirely. Not a pytest test -- no
assertions, just a way to exercise the real inbound pipeline
(app/whatsapp_routes.py -> services/router.py -> tool_executor ->
services/whatsapp_client.py) end to end during local development.

Why this exists rather than just testing against the real webhook:
Meta will not deliver real inbound messages to an app's webhook while
the app is unpublished (development mode), even from allowed test
recipients or app admins -- confirmed directly against this app's own
account, not assumed from docs. Publishing requires a real registered
phone number, billing, and business verification, a much bigger step
than local development needs. This script gets the same code path
exercised anyway: the sender number below should be an allowed test
recipient, so KasaFlow's real reply actually reaches a real phone.

Run from the repo root, with `uvicorn app.main:app --reload --port 8000`
already running in another terminal:
    python scripts/manual_webhook_simulation_check.py <sender_number> ["message text"]

<sender_number> is E.164 without the leading +, e.g. 233509764406.
"""

import sys

import requests

if len(sys.argv) < 2:
    print('Usage: python scripts/manual_webhook_simulation_check.py <sender_number> ["message text"]')
    sys.exit(1)

sender = sys.argv[1]
body = sys.argv[2] if len(sys.argv) > 2 else "how much is a gold ring"

payload = {
    "entry": [
        {
            "changes": [
                {
                    "value": {
                        "messages": [
                            {"from": sender, "type": "text", "text": {"body": body}}
                        ]
                    }
                }
            ]
        }
    ]
}

print(f"--- Simulating inbound WhatsApp message from {sender}: {body!r} ---")
response = requests.post("http://localhost:8000/webhook/whatsapp", json=payload, timeout=10)
print("Status:", response.status_code)
print("Body:", response.json())
print("\nCheck the uvicorn terminal for routing/tool logs, and the sender's phone for the real reply.")

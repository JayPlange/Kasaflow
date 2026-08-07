"""
One-off manual check: describe a real photo via describe_product_image()
and print the result, then route that description through the same
pipeline whatsapp_routes.py uses for a real incoming photo. Not a
pytest test -- no assertions, just an eyeball check against a real
image, the same role manual_khaya_check.py / manual_voice_pipeline_check.py
play for the voice path.

vision_tool.py's multimodal request shape hasn't been confirmed
against the real OpenAI API yet (see its module docstring) -- this is
that check.

Run from the repo root:
    python scripts/manual_vision_check.py <path_to_image.jpg>
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from services.response_formatter import format_for_customer
from services.router import route_customer
from services.vision_tool import describe_product_image

if len(sys.argv) < 2:
    print("Usage: python scripts/manual_vision_check.py <path_to_image.jpg>")
    sys.exit(1)

image_path = Path(sys.argv[1])
mime_type = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}.get(image_path.suffix.lower(), "image/jpeg")

print(f"--- Describing {image_path} via OpenAI vision ---")
with open(image_path, "rb") as f:
    description = describe_product_image(f.read(), mime_type=mime_type)
print("DESCRIPTION:", description or "(empty -- model said this isn't jewellery)")

if not description.strip():
    print("\nEmpty description -- stopping before routing, nothing to route.")
    sys.exit(0)

print("\n--- Routing description through tool selection + tool execution ---")
result = route_customer(description, session_id="local-test")
print("RAW RESULT:", result)

print("\n--- Customer-facing reply ---")
print(format_for_customer(result))

"""
Delivery is offered, not priced.

Adom De Jeweller delivers by rider from its Accra and Kumasi branches,
or ships internationally -- the real cost and timing for any of those
depends on the specific pickup/drop-off and isn't something this system
can compute on its own (no per-zone rate table exists, and inventing
one would violate this codebase's own no-fabrication rule -- see
response_formatter.py's module docstring). So this tool doesn't quote a
delivery price or time at all. It only ever offers the customer a
choice between the real delivery options; a human (the rider
coordinator) finalises the actual delivery once the order is placed --
see order_tool.py's confirm_order() for that handoff.
"""

# The only three real delivery arrangements this business offers.
# `key` is what propose_order()/the LLM deal with internally; `label`
# is what gets shown to a customer.
DELIVERY_OPTIONS = [
    {"key": "accra_rider", "label": "rider delivery within Accra"},
    {"key": "kumasi_rider", "label": "rider delivery within Kumasi"},
    {"key": "international", "label": "shipping outside Ghana"},
]

_VALID_KEYS = {option["key"] for option in DELIVERY_OPTIONS}


def get_delivery_information() -> dict:
    """Lists the real delivery options rather than quoting a price/time
    -- see module docstring for why."""
    return {"delivery_options": DELIVERY_OPTIONS}


def is_valid_delivery_option(key: str | None) -> bool:
    return key in _VALID_KEYS


def delivery_option_label(key: str | None) -> str | None:
    return next((option["label"] for option in DELIVERY_OPTIONS if option["key"] == key), None)


def delivery_options_phrase(options: list[dict] | None = None) -> str:
    """"rider delivery within Accra, rider delivery within Kumasi, or
    shipping outside Ghana" -- the one place this phrasing is built, so
    order_tool.py's clarifying question and response_formatter.py's
    customer-facing replies never drift out of sync with each other or
    with the real option labels above."""
    labels = [option["label"] for option in (options if options is not None else DELIVERY_OPTIONS)]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"

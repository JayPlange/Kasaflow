"""
A sandbox for replaying conversations: nothing leaves the process
except, in live mode, the model call itself.

Replaced while a replay runs:
  - WooCommerce orders API (order_tool.requests): an in-memory store that
    creates, reads, searches and cancels orders. Like WooCommerce, it
    prices an order from its own catalogue (data/products.json), not from
    anything the assistant sends, so the totals guardrail is exercised
    the way it would be in production.
  - WhatsApp sends to staff (order_tool, escalation): captured.
  - Google geocoding: off unless allow_geocoding (needs a key and network).
  - OpenAI embeddings (product search fallback, policy answers): off unless
    allow_embeddings. When off, those code paths fail the way they would
    on a network error, and the turn is marked as affected.

Settings for these modules are swapped for sandbox values (a fake store
URL and orders key, a fake staff number) so the code takes its normal
"configured" paths, then restored.
"""

from __future__ import annotations

import json
import re
import types
from dataclasses import replace
from pathlib import Path

import requests as real_requests

from services import escalation, geocoding_tool, knowledge_base, order_tool, product_search

SANDBOX_STAFF_NUMBER = "sandbox-staff"


class _Response:
    def __init__(self, status: int, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise real_requests.HTTPError(f"sandbox store returned {self.status_code}")


class SandboxStore:
    """Just enough of the WooCommerce orders API for order_tool.py."""

    def __init__(self, products_path: Path, first_id: int = 50001):
        rows = json.loads(Path(products_path).read_text(encoding="utf-8"))
        self._price = {(r.get("id"), r.get("variation_id")): float(r["price"]) for r in rows if "price" in r}
        self._name = {r.get("id"): r.get("product") for r in rows}
        self._next_id = first_id
        self.orders: dict[int, dict] = {}
        self.calls: list[tuple[str, str]] = []

    def _total(self, line_items: list[dict]) -> str:
        total = 0.0
        for item in line_items:
            price = self._price.get((item.get("product_id"), item.get("variation_id")))
            if price is None:
                # Unknown variant: WooCommerce would reject or price it at 0.
                return "0.00"
            total += price * int(item.get("quantity", 1))
        return f"{total:.2f}"

    def post(self, url, json=None, auth=None, timeout=None, **kwargs):
        self.calls.append(("POST", url))
        if not url.rstrip("/").endswith("/orders"):
            return _Response(404, {})
        order_id = self._next_id
        self._next_id += 1
        order = {
            **(json or {}),
            "id": order_id,
            "status": (json or {}).get("status", "on-hold"),
            "total": self._total((json or {}).get("line_items", [])),
        }
        for item in order.get("line_items", []):
            item["name"] = self._name.get(item.get("product_id"))
        self.orders[order_id] = order
        return _Response(201, order)

    def get(self, url, params=None, auth=None, timeout=None, **kwargs):
        self.calls.append(("GET", url))
        match = re.search(r"/orders/(\d+)$", url.rstrip("/"))
        if match:
            order = self.orders.get(int(match.group(1)))
            return _Response(200, order) if order else _Response(404, {"code": "not_found"})
        if url.rstrip("/").endswith("/orders"):
            term = (params or {}).get("search", "")
            hits = [o for o in self.orders.values() if term and term in (o.get("customer_note") or "")]
            return _Response(200, hits)
        return _Response(404, {})

    def put(self, url, json=None, auth=None, timeout=None, **kwargs):
        self.calls.append(("PUT", url))
        match = re.search(r"/orders/(\d+)$", url.rstrip("/"))
        if not match or int(match.group(1)) not in self.orders:
            return _Response(404, {})
        self.orders[int(match.group(1))].update(json or {})
        return _Response(200, self.orders[int(match.group(1))])


class _EmbeddingsDisabled(RuntimeError):
    pass


def _embeddings_disabled(*args, **kwargs):
    raise _EmbeddingsDisabled("embeddings are disabled in this replay (use --allow-embeddings)")


class Sandbox:
    def __init__(self, allow_embeddings: bool, allow_geocoding: bool):
        self.allow_embeddings = allow_embeddings
        self.allow_geocoding = allow_geocoding
        self.staff_messages: list[str] = []
        self.embedding_attempts = 0
        self.store: SandboxStore | None = None
        self._saved: list[tuple[object, str, object]] = []

    def _swap(self, target, name, value):
        self._saved.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def __enter__(self) -> "Sandbox":
        self.store = SandboxStore(order_tool.settings.products_path)
        fake_requests = types.SimpleNamespace(
            get=self.store.get, post=self.store.post, put=self.store.put, exceptions=real_requests.exceptions,
        )
        self._swap(order_tool, "requests", fake_requests)

        sandbox_settings = dict(
            woocommerce_url="https://sandbox.store",
            woocommerce_orders_consumer_key="sandbox",
            woocommerce_orders_consumer_secret="sandbox",
            staff_notification_phone=SANDBOX_STAFF_NUMBER,
        )
        for module in (order_tool, escalation):
            self._swap(module, "settings", replace(module.settings, **sandbox_settings))
        if not self.allow_geocoding:
            self._swap(geocoding_tool, "settings", replace(geocoding_tool.settings, google_maps_api_key=None))

        def capture(to, body):
            self.staff_messages.append(body)

        self._swap(order_tool, "send_text_message", capture)
        self._swap(escalation, "send_text_message", capture)

        if not self.allow_embeddings:
            def counted(*args, **kwargs):
                self.embedding_attempts += 1
                return _embeddings_disabled(*args, **kwargs)

            self._swap(product_search, "embed_texts", counted)
            self._swap(knowledge_base, "embed_texts", counted)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        while self._saved:
            target, name, value = self._saved.pop()
            setattr(target, name, value)
        return False

"""
Unit tests for services/whatsapp_client.py's send_image_message.

Scoped to the new function specifically (added alongside the WhatsApp
product-photo reply feature) -- send_text_message, send_audio_message,
and download_media are pre-existing and were already uncovered before
this file existed; this isn't re-litigating that, just not leaving the
new function in the same uncovered state.

Mocks requests.post directly rather than hitting the network, same
"never call the real API in a unit test" rule as every other test file
here.
"""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from services import whatsapp_client
from services.whatsapp_client import WhatsAppError, send_image_message


def _settings_with_whatsapp_config(monkeypatch, token="test-token", phone_id="test-phone-id"):
    monkeypatch.setattr(
        whatsapp_client,
        "settings",
        replace(
            whatsapp_client.settings,
            whatsapp_access_token=token,
            whatsapp_phone_number_id=phone_id,
        ),
    )


def _mock_post(monkeypatch):
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"messages": [{"id": "wamid.fake"}]}
    fake_post = MagicMock(return_value=fake_response)
    monkeypatch.setattr(whatsapp_client.requests, "post", fake_post)
    return fake_post


def test_send_image_message_sends_link_and_caption(monkeypatch):
    # Arrange
    _settings_with_whatsapp_config(monkeypatch)
    fake_post = _mock_post(monkeypatch)

    # Act
    send_image_message(
        "233555000111",
        "https://adomdejeweller.com/wp-content/uploads/ring.jpg",
        caption="The 18k Ring is GH₵1,200.00.",
    )

    # Assert: correct recipient, correct image payload shape (link, not
    # a media ID -- no upload step needed since it's already a public URL)
    fake_post.assert_called_once()
    _, kwargs = fake_post.call_args
    assert kwargs["json"] == {
        "messaging_product": "whatsapp",
        "to": "233555000111",
        "type": "image",
        "image": {
            "link": "https://adomdejeweller.com/wp-content/uploads/ring.jpg",
            "caption": "The 18k Ring is GH₵1,200.00.",
        },
    }


def test_send_image_message_without_caption_omits_it(monkeypatch):
    # Arrange
    _settings_with_whatsapp_config(monkeypatch)
    fake_post = _mock_post(monkeypatch)

    # Act
    send_image_message("233555000111", "https://adomdejeweller.com/photo.jpg")

    # Assert: no empty/None caption key sent to Meta's API when there isn't one
    _, kwargs = fake_post.call_args
    assert kwargs["json"]["image"] == {"link": "https://adomdejeweller.com/photo.jpg"}


def test_send_image_message_raises_when_config_missing(monkeypatch):
    # Arrange: WhatsApp credentials not yet configured (real state right
    # now -- the app runs, but this call path isn't wired up to Meta yet)
    monkeypatch.setattr(
        whatsapp_client,
        "settings",
        replace(whatsapp_client.settings, whatsapp_access_token=None, whatsapp_phone_number_id=None),
    )
    fake_post = _mock_post(monkeypatch)

    # Act / Assert
    with pytest.raises(WhatsAppError):
        send_image_message("233555000111", "https://adomdejeweller.com/photo.jpg")

    # Never even attempted the request without valid config
    fake_post.assert_not_called()

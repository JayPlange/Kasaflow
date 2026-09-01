"""
Unit tests for app/whatsapp_routes.py's _handle_message.

No test file covered this module before -- not just the new image/
vision branch, the pre-existing text and audio paths had zero coverage
either. Testing _handle_message directly (a plain function once you're
past the FastAPI routing layer) rather than going through TestClient +
a full webhook payload + BackgroundTasks, so this stays a fast, focused
unit test like everything else in this suite. Every dependency
(download_media, transcribe_audio, describe_product_image,
route_customer, format_for_customer, synthesize_speech, and the
send_*_message functions) is mocked -- never call a real API here.
"""

from unittest.mock import MagicMock

from app import whatsapp_routes
from app.whatsapp_routes import _handle_message
from services.voice_tool import VoiceServiceError
from services.whatsapp_client import WhatsAppError


def _no_image_result():
    return {"product": "Ring", "material": "gold", "price": 1200.0}


def _with_image_result():
    return {"product": "Ring", "material": "gold", "price": 1200.0, "image_url": "https://example.com/ring.jpg"}


# ---------------------------------------------------------------------
# text messages
# ---------------------------------------------------------------------

def test_text_message_routes_body_and_sends_text_reply(monkeypatch):
    # Arrange
    monkeypatch.setattr(whatsapp_routes, "route_customer", MagicMock(return_value=_no_image_result()))
    monkeypatch.setattr(whatsapp_routes, "format_for_customer", MagicMock(return_value="The gold Ring is GH₵1,200.00."))
    send_text = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_text_message", send_text)

    # Act
    _handle_message({"from": "233555000111", "type": "text", "text": {"body": "how much is a gold ring"}})

    # Assert
    whatsapp_routes.route_customer.assert_called_once_with("how much is a gold ring", session_id="233555000111")
    send_text.assert_called_once_with("233555000111", "The gold Ring is GH₵1,200.00.")


# ---------------------------------------------------------------------
# audio messages
# ---------------------------------------------------------------------

def test_audio_message_transcribes_and_sends_audio_reply(monkeypatch):
    # Arrange
    monkeypatch.setattr(whatsapp_routes, "download_media", MagicMock(return_value=b"ogg-bytes"))
    monkeypatch.setattr(whatsapp_routes, "transcribe_audio", MagicMock(return_value="how much is a gold ring"))
    monkeypatch.setattr(whatsapp_routes, "route_customer", MagicMock(return_value=_no_image_result()))
    monkeypatch.setattr(whatsapp_routes, "format_for_customer", MagicMock(return_value="The gold Ring is GH₵1,200.00."))
    monkeypatch.setattr(whatsapp_routes, "synthesize_speech", MagicMock(return_value=b"mp3-bytes"))
    send_audio = MagicMock()
    send_text = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_audio_message", send_audio)
    monkeypatch.setattr(whatsapp_routes, "send_text_message", send_text)

    # Act
    _handle_message({"from": "233555000111", "type": "audio", "audio": {"id": "media-1"}})

    # Assert: audio in, audio reply out
    send_audio.assert_called_once_with("233555000111", b"mp3-bytes")
    send_text.assert_not_called()


def test_audio_message_empty_transcript_sends_fallback_without_routing(monkeypatch):
    # Arrange
    monkeypatch.setattr(whatsapp_routes, "download_media", MagicMock(return_value=b"ogg-bytes"))
    monkeypatch.setattr(whatsapp_routes, "transcribe_audio", MagicMock(return_value="   "))
    route_mock = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "route_customer", route_mock)
    send_text = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_text_message", send_text)

    # Act
    _handle_message({"from": "233555000111", "type": "audio", "audio": {"id": "media-1"}})

    # Assert: honest fallback, never even tried to route nothing
    route_mock.assert_not_called()
    send_text.assert_called_once()
    assert "couldn't quite make that out" in send_text.call_args[0][1]


def test_audio_message_tts_failure_falls_back_to_text(monkeypatch):
    # Arrange
    monkeypatch.setattr(whatsapp_routes, "download_media", MagicMock(return_value=b"ogg-bytes"))
    monkeypatch.setattr(whatsapp_routes, "transcribe_audio", MagicMock(return_value="how much is a gold ring"))
    monkeypatch.setattr(whatsapp_routes, "route_customer", MagicMock(return_value=_no_image_result()))
    monkeypatch.setattr(whatsapp_routes, "format_for_customer", MagicMock(return_value="The gold Ring is GH₵1,200.00."))
    monkeypatch.setattr(whatsapp_routes, "synthesize_speech", MagicMock(side_effect=VoiceServiceError("TTS down")))
    send_text = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_text_message", send_text)

    # Act
    _handle_message({"from": "233555000111", "type": "audio", "audio": {"id": "media-1"}})

    # Assert: customer still gets an answer, just as text
    send_text.assert_called_once_with("233555000111", "The gold Ring is GH₵1,200.00.")


# ---------------------------------------------------------------------
# image messages (the new vision path)
# ---------------------------------------------------------------------

def test_image_message_describes_photo_and_routes_description(monkeypatch):
    # Arrange
    monkeypatch.setattr(whatsapp_routes, "download_media", MagicMock(return_value=b"jpeg-bytes"))
    monkeypatch.setattr(whatsapp_routes, "describe_product_image", MagicMock(return_value="gold twist ring"))
    route_mock = MagicMock(return_value=_no_image_result())
    monkeypatch.setattr(whatsapp_routes, "route_customer", route_mock)
    monkeypatch.setattr(whatsapp_routes, "format_for_customer", MagicMock(return_value="The gold Ring is GH₵1,200.00."))
    send_text = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_text_message", send_text)

    # Act
    _handle_message({"from": "233555000111", "type": "image", "image": {"id": "media-2"}})

    # Assert: the photo's description is what actually gets routed --
    # same pipeline a typed/transcribed message would use
    route_mock.assert_called_once_with("gold twist ring", session_id="233555000111")
    send_text.assert_called_once_with("233555000111", "The gold Ring is GH₵1,200.00.")


def test_image_message_with_matched_product_sends_photo_reply(monkeypatch):
    # Arrange: the catalogue match has its own photo -- customer sent a
    # photo, gets back a photo of the actual matching product plus price
    monkeypatch.setattr(whatsapp_routes, "download_media", MagicMock(return_value=b"jpeg-bytes"))
    monkeypatch.setattr(whatsapp_routes, "describe_product_image", MagicMock(return_value="gold twist ring"))
    monkeypatch.setattr(whatsapp_routes, "route_customer", MagicMock(return_value=_with_image_result()))
    monkeypatch.setattr(whatsapp_routes, "format_for_customer", MagicMock(return_value="The gold Ring is GH₵1,200.00."))
    send_image = MagicMock()
    send_text = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_image_message", send_image)
    monkeypatch.setattr(whatsapp_routes, "send_text_message", send_text)

    # Act
    _handle_message({"from": "233555000111", "type": "image", "image": {"id": "media-2"}})

    # Assert
    send_image.assert_called_once_with("233555000111", "https://example.com/ring.jpg", caption="The gold Ring is GH₵1,200.00.")
    send_text.assert_not_called()


def test_image_message_photo_reply_failure_falls_back_to_text(monkeypatch):
    # Arrange
    monkeypatch.setattr(whatsapp_routes, "download_media", MagicMock(return_value=b"jpeg-bytes"))
    monkeypatch.setattr(whatsapp_routes, "describe_product_image", MagicMock(return_value="gold twist ring"))
    monkeypatch.setattr(whatsapp_routes, "route_customer", MagicMock(return_value=_with_image_result()))
    monkeypatch.setattr(whatsapp_routes, "format_for_customer", MagicMock(return_value="The gold Ring is GH₵1,200.00."))
    monkeypatch.setattr(whatsapp_routes, "send_image_message", MagicMock(side_effect=WhatsAppError("Meta rejected it")))
    send_text = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_text_message", send_text)

    # Act
    _handle_message({"from": "233555000111", "type": "image", "image": {"id": "media-2"}})

    # Assert
    send_text.assert_called_once_with("233555000111", "The gold Ring is GH₵1,200.00.")


def test_image_message_with_caption_merges_caption_into_routed_text(monkeypatch):
    # Arrange: a photo sent with a caption naming the product -- WhatsApp's
    # Cloud API image object supports an optional "caption" field (see
    # whatsapp_client.py's own send_image_message, which sends one on the
    # way out); previously only message["image"]["id"] was ever read here,
    # silently dropping any caption.
    monkeypatch.setattr(whatsapp_routes, "download_media", MagicMock(return_value=b"jpeg-bytes"))
    monkeypatch.setattr(whatsapp_routes, "describe_product_image", MagicMock(return_value="gold pendant necklace"))
    route_mock = MagicMock(return_value=_no_image_result())
    monkeypatch.setattr(whatsapp_routes, "route_customer", route_mock)
    monkeypatch.setattr(whatsapp_routes, "format_for_customer", MagicMock(return_value="ok"))
    monkeypatch.setattr(whatsapp_routes, "send_text_message", MagicMock())

    # Act
    _handle_message({
        "from": "233555000111",
        "type": "image",
        "image": {"id": "media-2", "caption": "is this the Adinkra necklace"},
    })

    # Assert: caption is folded in ahead of the vision description
    route_mock.assert_called_once_with(
        "is this the Adinkra necklace gold pendant necklace", session_id="233555000111"
    )


def test_image_message_without_caption_routes_description_alone(monkeypatch):
    # Arrange: no caption field at all on the payload -- must not crash
    # on a missing key, and must not glue on an empty string
    monkeypatch.setattr(whatsapp_routes, "download_media", MagicMock(return_value=b"jpeg-bytes"))
    monkeypatch.setattr(whatsapp_routes, "describe_product_image", MagicMock(return_value="gold pendant necklace"))
    route_mock = MagicMock(return_value=_no_image_result())
    monkeypatch.setattr(whatsapp_routes, "route_customer", route_mock)
    monkeypatch.setattr(whatsapp_routes, "format_for_customer", MagicMock(return_value="ok"))
    monkeypatch.setattr(whatsapp_routes, "send_text_message", MagicMock())

    # Act
    _handle_message({"from": "233555000111", "type": "image", "image": {"id": "media-2"}})

    # Assert
    route_mock.assert_called_once_with("gold pendant necklace", session_id="233555000111")


def test_image_message_empty_description_sends_fallback_without_routing(monkeypatch):
    # Arrange: photo didn't look like jewellery at all
    monkeypatch.setattr(whatsapp_routes, "download_media", MagicMock(return_value=b"jpeg-bytes"))
    monkeypatch.setattr(whatsapp_routes, "describe_product_image", MagicMock(return_value=""))
    route_mock = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "route_customer", route_mock)
    send_text = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_text_message", send_text)

    # Act
    _handle_message({"from": "233555000111", "type": "image", "image": {"id": "media-2"}})

    # Assert: honest fallback, never routed an empty/nonsense description
    route_mock.assert_not_called()
    send_text.assert_called_once()
    assert "couldn't quite tell what that was" in send_text.call_args[0][1]


# ---------------------------------------------------------------------
# recommendation browse (multi-image) -- Webb, 2026-09-01: "visual
# browsing is part of the jewellery experience", one WhatsApp image
# message per recommended product rather than a single text reply.
# ---------------------------------------------------------------------

def _rings_recommendation():
    return {
        "recommendations": [
            {"product": "Minimal White Stone Gold Ring, 1g", "category": "Rings", "material": "12k",
             "price": 15127.20, "image_url": "https://example.com/ring1-12k.jpg"},
            {"product": "Minimal White Stone Gold Ring, 1g", "category": "Rings", "material": "18k",
             "price": 20628.00, "image_url": "https://example.com/ring1-18k.jpg"},
            {"product": "Set Multi Stone Gold Ring, 7g", "category": "Rings", "material": "18k",
             "price": 12033.00, "image_url": "https://example.com/ring2-18k.jpg"},
        ]
    }


def test_recommendations_result_sends_one_image_message_per_product(monkeypatch):
    monkeypatch.setattr(whatsapp_routes, "route_customer", MagicMock(return_value=_rings_recommendation()))
    send_image = MagicMock()
    send_text = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_image_message", send_image)
    monkeypatch.setattr(whatsapp_routes, "send_text_message", send_text)

    _handle_message({"from": "233555000111", "type": "text", "text": {"body": "show me the rings"}})

    # Two distinct products in the fixture -- two image messages, not
    # one text dump of the whole list.
    assert send_image.call_count == 2
    send_text.assert_not_called()
    first_call, second_call = send_image.call_args_list
    assert first_call.args[0] == "233555000111"
    assert first_call.args[1] == "https://example.com/ring1-12k.jpg"
    assert "*Minimal White Stone Gold Ring, 1g*" in first_call.kwargs["caption"]
    assert "12k · GH₵15,127.20" in first_call.kwargs["caption"]
    assert second_call.args[1] == "https://example.com/ring2-18k.jpg"
    assert "*Set Multi Stone Gold Ring, 7g*" in second_call.kwargs["caption"]


def test_recommendations_result_caps_at_four_products(monkeypatch):
    items = [
        {"product": f"Ring {i}", "category": "Rings", "material": "18k", "price": 100.0 * i,
         "image_url": f"https://example.com/ring{i}.jpg"}
        for i in range(1, 6)
    ]
    monkeypatch.setattr(whatsapp_routes, "route_customer", MagicMock(return_value={"recommendations": items}))
    send_image = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_image_message", send_image)
    monkeypatch.setattr(whatsapp_routes, "send_text_message", MagicMock())

    _handle_message({"from": "233555000111", "type": "text", "text": {"body": "what rings do you have"}})

    assert send_image.call_count == 4


def test_recommendations_without_photos_falls_back_to_one_text_message(monkeypatch):
    items = [
        {"product": "Ring 1", "category": "Rings", "material": "18k", "price": 100.0},
        {"product": "Ring 2", "category": "Rings", "material": "18k", "price": 200.0},
    ]
    monkeypatch.setattr(whatsapp_routes, "route_customer", MagicMock(return_value={"recommendations": items}))
    send_image = MagicMock()
    send_text = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_image_message", send_image)
    monkeypatch.setattr(whatsapp_routes, "send_text_message", send_text)

    _handle_message({"from": "233555000111", "type": "text", "text": {"body": "show me the rings"}})

    # No item had an image_url -- nothing to send a photo for, so this
    # falls back to one text message listing both, not silence.
    send_image.assert_not_called()
    send_text.assert_called_once()
    body = send_text.call_args[0][1]
    assert "Ring 1" in body and "Ring 2" in body


def test_recommendations_image_send_failure_falls_back_to_text_for_that_item(monkeypatch):
    monkeypatch.setattr(whatsapp_routes, "route_customer", MagicMock(return_value=_rings_recommendation()))
    send_image = MagicMock(side_effect=WhatsAppError("Meta rejected it"))
    send_text = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_image_message", send_image)
    monkeypatch.setattr(whatsapp_routes, "send_text_message", send_text)

    _handle_message({"from": "233555000111", "type": "text", "text": {"body": "show me the rings"}})

    # Both sends were attempted and both failed -- both captions still
    # reach the customer, as one trailing text message rather than
    # silence.
    assert send_image.call_count == 2
    send_text.assert_called_once()
    body = send_text.call_args[0][1]
    assert "Minimal White Stone Gold Ring, 1g" in body
    assert "Set Multi Stone Gold Ring, 7g" in body


def test_recommendations_audio_message_still_gets_a_single_spoken_reply(monkeypatch):
    # A voice note that resolves to a browse still gets one synthesized
    # reply, not several -- audio takes priority over the new multi-image
    # branch, unchanged from before this feature existed.
    monkeypatch.setattr(whatsapp_routes, "download_media", MagicMock(return_value=b"raw-audio"))
    monkeypatch.setattr(whatsapp_routes, "transcribe_audio", MagicMock(return_value="show me the rings"))
    monkeypatch.setattr(whatsapp_routes, "route_customer", MagicMock(return_value=_rings_recommendation()))
    monkeypatch.setattr(
        whatsapp_routes, "format_for_customer", MagicMock(return_value="Here's what I found for you: ...")
    )
    monkeypatch.setattr(whatsapp_routes, "synthesize_speech", MagicMock(return_value=b"reply-audio"))
    send_audio = MagicMock()
    send_image = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_audio_message", send_audio)
    monkeypatch.setattr(whatsapp_routes, "send_image_message", send_image)
    monkeypatch.setattr(whatsapp_routes, "send_text_message", MagicMock())

    _handle_message({"from": "233555000111", "type": "audio", "audio": {"id": "media-1"}})

    send_audio.assert_called_once()
    send_image.assert_not_called()



# ---------------------------------------------------------------------
# unhandled message types / unexpected failures
# ---------------------------------------------------------------------

def test_unhandled_message_type_is_ignored(monkeypatch):
    # Arrange
    send_text = MagicMock()
    monkeypatch.setattr(whatsapp_routes, "send_text_message", send_text)

    # Act: WhatsApp sends other event types (stickers, reactions, etc.)
    _handle_message({"from": "233555000111", "type": "sticker"})

    # Assert: silently ignored, no reply sent, no crash
    send_text.assert_not_called()


def test_unexpected_exception_does_not_propagate(monkeypatch):
    # Arrange: a genuine bug somewhere downstream
    monkeypatch.setattr(whatsapp_routes, "route_customer", MagicMock(side_effect=RuntimeError("boom")))

    # Act / Assert: swallowed and logged, never crashes the background task
    _handle_message({"from": "233555000111", "type": "text", "text": {"body": "hello"}})

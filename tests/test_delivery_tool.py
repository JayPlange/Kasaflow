from services.delivery_tool import get_delivery_information


def test_get_delivery_information_returns_expected_shape():
    # Arrange: nothing to set up, this tool takes no arguments

    # Act
    result = get_delivery_information()

    # Assert: even a "boring" hardcoded tool deserves a test -- this one
    # protects you the day someone renames a key and every caller downstream
    # silently breaks (e.g. router.py expects "delivery_time" specifically).
    assert result == {
        "delivery_time": "2-5 business days",
        "shipping_cost": 25,
    }

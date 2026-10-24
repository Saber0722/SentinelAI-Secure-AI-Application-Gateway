"""PII detection: successful detections *and* documented misses."""

import pytest

from sentinelai.config import DEFAULT_PII_ENTITIES


def entities(detector, text, entity_list=DEFAULT_PII_ENTITIES):
    return {f.entity_type for f in detector.analyze(text, entity_list, 0.4)}


@pytest.mark.parametrize(
    ("text", "entity"),
    [
        ("mail me at jane.doe@example.com", "EMAIL_ADDRESS"),
        ("call +1 415-555-0132 now", "PHONE_NUMBER"),
        ("server 192.168.1.42 is down", "IP_ADDRESS"),
        ("card 4111 1111 1111 1111", "CREDIT_CARD"),
        ("IBAN DE89 3704 0044 0532 0130 00", "IBAN_CODE"),
        ("I'm John Carter from support", "PERSON"),
        ("Ship to Sarah O'Neil", "PERSON"),
        ("My PAN card number is BNZPM2501F", "IN_PAN"),
        ("My PAN is ABCDE1234F", "IN_PAN"),
    ],
)
def test_detects_common_entities(pii_detector, text, entity):
    assert entity in entities(pii_detector, text)


def test_spans_point_at_the_value(pii_detector):
    text = "email: a.b@example.org"
    (f,) = pii_detector.analyze(text, ["EMAIL_ADDRESS"], 0.5)
    assert text[f.spans[0].start : f.spans[0].end] == "a.b@example.org"
    assert "a.b@example.org" not in f.evidence


@pytest.mark.parametrize("text", ["How do API keys differ from OAuth tokens?", "Convert 25 degrees Celsius to Fahrenheit."])
def test_person_filter_rejects_non_names(pii_detector, text):
    assert "PERSON" not in entities(pii_detector, text)


def test_location_is_opt_in(pii_detector):
    assert "LOCATION" not in entities(pii_detector, "Plan a trip to Kyoto")
    assert "LOCATION" in entities(pii_detector, "Plan a trip to Kyoto", [*DEFAULT_PII_ENTITIES, "LOCATION"])


@pytest.mark.parametrize(
    "text",
    [
        "Reach me at priya [at] example [dot] org",  # obfuscated e-mail address
        "Send it to accounts@acme.example",  # reserved .example TLD fails Presidio's TLD validation
    ],
)
def test_known_misses_are_documented(pii_detector, text):
    """These are genuine limitations (docs/limitations.md); if one starts passing, update the docs."""
    assert entities(pii_detector, text) - {"PERSON"} == set()


def test_empty_inputs(pii_detector):
    assert pii_detector.analyze("", DEFAULT_PII_ENTITIES, 0.5) == []
    assert pii_detector.analyze("john@example.com", [], 0.5) == []

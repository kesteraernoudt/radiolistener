import pytest

from utils.genai import (
    DEFAULT_JSON_RESULT,
    extract_first_json_object,
    parse_json_or_reject,
)


def test_extract_first_json_object_with_junk():
    payload = 'leading noise {"codeword":"snow","confidence":0.9,"evidence":"they said snow"} trailing junk'
    extracted = extract_first_json_object(payload)
    assert extracted == '{"codeword":"snow","confidence":0.9,"evidence":"they said snow"}'


def test_schema_rejects_extra_keys():
    raw = '{"codeword":"sun","confidence":0.9,"evidence":"word is sun","extra":1}'
    assert parse_json_or_reject(raw) == DEFAULT_JSON_RESULT


def test_invalid_json_returns_default():
    assert parse_json_or_reject("not valid json") == DEFAULT_JSON_RESULT


def test_confidence_bounds_enforced():
    raw = '{"codeword":"rain","confidence":2,"evidence":"keyword is rain"}'
    assert parse_json_or_reject(raw) == DEFAULT_JSON_RESULT


def test_valid_json_preserves_confidence():
    raw = '{"codeword":"rain","confidence":0.75,"evidence":"keyword is rain"}'
    parsed = parse_json_or_reject(raw)
    assert parsed["codeword"] == "rain"
    assert parsed["confidence"] == 0.75
    assert parsed["evidence"] == "keyword is rain"


def test_evidence_required_when_codeword_present():
    raw = '{"codeword":"cloud","confidence":0.9,"evidence":null}'
    assert parse_json_or_reject(raw) == DEFAULT_JSON_RESULT


def test_codeword_null_forces_zero_confidence():
    raw = '{"codeword":null,"confidence":0.4,"evidence":"irrelevant"}'
    parsed = parse_json_or_reject(raw)
    assert parsed == DEFAULT_JSON_RESULT


def test_codeword_null_can_carry_confidence_with_null_evidence():
    raw = '{"codeword":null,"confidence":0.6,"evidence":null}'
    parsed = parse_json_or_reject(raw)
    assert parsed == {"codeword": None, "confidence": 0.6, "evidence": None}

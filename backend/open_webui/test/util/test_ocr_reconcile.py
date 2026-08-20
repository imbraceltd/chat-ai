"""Unit tests for OCR-grounded confidence reconciliation.

Pure-Python: exercises ``open_webui.utils.ocr.reconcile`` with synthetic OcrTokens,
so it needs no OCR backend (numpy/PIL/onnxruntime). Run with:

    pytest backend/open_webui/test/util/test_ocr_reconcile.py
"""

from open_webui.utils.ocr.base import OcrToken
from open_webui.utils.ocr.reconcile import aggregate_confidence, reconcile_fields


def _tokens():
    return [
        OcrToken("Invoice", [0.0, 0.0, 0.2, 0.05], 0.99, 1),
        OcrToken("12345", [0.3, 0.0, 0.5, 0.05], 0.95, 1),
        OcrToken("John", [0.0, 0.1, 0.2, 0.15], 0.90, 1),
        OcrToken("Smith", [0.25, 0.1, 0.45, 0.15], 0.92, 1),
    ]


def _wrapped(node):
    return isinstance(node, dict) and "value" in node and "confidence" in node


def test_exact_match_uses_ocr_confidence():
    out = reconcile_fields({"invoice_no": "12345"}, _tokens())
    field = out["invoice_no"]
    assert _wrapped(field)
    assert field["value"] == "12345"
    assert field["confidence"] == 0.95  # ocr_conf(0.95) * agreement(1.0)
    assert field["box"] == [0.3, 0.0, 0.5, 0.05]


def test_multi_token_value_spans_tokens():
    out = reconcile_fields({"name": "John Smith"}, _tokens())
    field = out["name"]
    assert _wrapped(field)
    # min confidence over the matched run (0.90, 0.92) * agreement(~1.0)
    assert field["confidence"] >= 0.88
    # box is the union of the two matched word boxes
    assert field["box"] == [0.0, 0.1, 0.45, 0.15]


def test_value_is_substring_of_line_token_keeps_high_confidence():
    # Line-level OCR returns the whole line as one token; a correctly-extracted
    # sub-value must still score high (substring/partial match), not be penalized.
    tokens = [OcrToken("INVOICE 12345", [0.0, 0.0, 0.4, 0.05], 0.99, 1)]
    out = reconcile_fields({"invoice_no": "12345"}, tokens)
    assert out["invoice_no"]["confidence"] >= 0.95


def test_short_value_does_not_trivially_match():
    # 1-2 char values must not get a free substring match against unrelated tokens.
    tokens = [OcrToken("INVOICE 12345", [0.0, 0.0, 0.4, 0.05], 0.99, 1)]
    out = reconcile_fields({"x": "Q"}, tokens)
    assert out["x"]["confidence"] < 0.5


def test_hallucinated_value_scores_low():
    out = reconcile_fields({"name": "Zzzzz Nonexistent"}, _tokens())
    assert out["name"]["confidence"] < 0.3


def test_threshold_keeps_low_confidence_values_plain():
    out = reconcile_fields({"name": "Zzzzz Nonexistent"}, _tokens(), threshold=0.5)
    # Below threshold → left as a plain value, not wrapped.
    assert out["name"] == "Zzzzz Nonexistent"


def test_nested_and_prewrapped_values():
    data = {
        "header": {"invoice_no": "12345"},
        "people": ["John Smith"],
        # already wrapped by the LLM under extract_raw_data — must be re-grounded, not double-wrapped
        "ref": {"value": "12345", "box": [0, 0, 0, 0], "confidence": 0.1},
    }
    out = reconcile_fields(data, _tokens())
    assert _wrapped(out["header"]["invoice_no"])
    assert _wrapped(out["people"][0])
    assert _wrapped(out["ref"])
    assert not _wrapped(out["ref"]["value"])  # inner value is a plain string, not re-wrapped
    assert out["ref"]["confidence"] == 0.95  # re-grounded from OCR, not the stale 0.1


def test_booleans_and_none_untouched():
    out = reconcile_fields({"flag": True, "missing": None}, _tokens())
    assert out["flag"] is True
    assert out["missing"] is None


def test_no_tokens_returns_data_unchanged():
    data = {"invoice_no": "12345"}
    assert reconcile_fields(data, []) == data


def test_aggregate_confidence_is_mean():
    out = reconcile_fields({"a": "12345", "b": "John Smith"}, _tokens())
    agg = aggregate_confidence(out)
    assert agg is not None
    assert 0.88 <= agg <= 0.96


def test_aggregate_confidence_none_when_no_fields():
    assert aggregate_confidence({"flag": True}) is None


# --- Reliability improvements: CER/WER metric + numeric/date-aware matching ---

def test_numeric_formatting_does_not_penalize():
    # LLM value is formatted with separators; OCR read the raw digits — should still match.
    tokens = [OcrToken("1234.50", [0.0, 0.0, 0.2, 0.05], 0.97, 1)]
    out = reconcile_fields({"amount": "1,234.50"}, tokens)
    assert out["amount"]["confidence"] >= 0.95


def test_numeric_id_substring_of_line():
    tokens = [OcrToken("Invoice No: INV-2024-00123", [0.0, 0.0, 0.6, 0.05], 0.96, 1)]
    out = reconcile_fields({"invoice_no": "2024-00123"}, tokens)
    assert out["invoice_no"]["confidence"] >= 0.9


def test_word_order_insensitive_match():
    # OCR returned the name tokens in a different order than the LLM value.
    tokens = [OcrToken("Smith", [0.25, 0.1, 0.45, 0.15], 0.95, 1),
              OcrToken("John", [0.0, 0.1, 0.2, 0.15], 0.95, 1)]
    out = reconcile_fields({"name": "John Smith"}, tokens)
    assert out["name"]["confidence"] >= 0.85


def test_partial_match_noise_does_not_inflate_hallucination():
    # A hallucinated value that shares incidental n-grams with a line token must stay low;
    # lenient substring/word-order metrics are gated so they don't rescue non-matches.
    tokens = [OcrToken("INVOICE NO INV-2024-00123", [0.0, 0.0, 0.6, 0.05], 0.95, 1)]
    out = reconcile_fields({"vendor": "NONEXISTENT"}, tokens)
    assert out["vendor"]["confidence"] < 0.5


def test_short_numeric_value_not_trivially_matched():
    # A 1-2 digit value must not get a free numeric match against unrelated digits.
    tokens = [OcrToken("Total 987654", [0.0, 0.0, 0.4, 0.05], 0.99, 1)]
    out = reconcile_fields({"qty": "7"}, tokens)
    assert out["qty"]["confidence"] < 0.6


def test_preprocess_for_ocr_is_geometry_preserving_and_defensive():
    import numpy as np
    from open_webui.utils.ocr.base import normalize_box, preprocess_for_ocr

    # Defensive: garbage input never raises, returns the input.
    assert preprocess_for_ocr(None) is None

    # A small synthetic page; preprocessing returns an ndarray and only upscales.
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    out = preprocess_for_ocr(img)
    assert hasattr(out, "shape")
    oh, ow = out.shape[:2]
    assert oh >= 400 and ow >= 600  # never downscaled
    # Uniform upscale preserves aspect ratio, so a normalized box is invariant.
    assert abs((ow / oh) - (600 / 400)) < 0.02
    box_before = normalize_box([60, 40, 180, 120], 600, 400)
    box_after = normalize_box([60 * ow / 600, 40 * oh / 400, 180 * ow / 600, 120 * oh / 400], ow, oh)
    assert max(abs(a - b) for a, b in zip(box_before, box_after)) < 1e-6

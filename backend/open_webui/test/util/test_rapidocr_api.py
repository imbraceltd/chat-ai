"""Unit tests for the external RapidOCR API (rapidocr_api) integration.

Covers URL resolution and mapping the /ocr JSON response into OcrTokens — no server
or local ONNX model required.

Run: pytest backend/open_webui/test/util/test_rapidocr_api.py
"""

import os

from open_webui.utils.ocr.rapidocr_engine import _tokens_from_api_result, get_api_url


def test_parse_documented_response():
    # Example shape from the rapidocr_api docs (dict keyed by index).
    result = {
        "0": {"rec_txt": "8月26日！", "dt_boxes": [[333.0, 72.0], [545.0, 40.0], [552.0, 90.0], [341.0, 122.0]], "score": "0.7342"},
        "1": {"rec_txt": "hello", "dt_boxes": [[10, 10], [110, 10], [110, 40], [10, 40]], "score": "0.99"},
    }
    tokens = _tokens_from_api_result(result, width=600, height=200, page=3)
    assert len(tokens) == 2
    assert tokens[0].text == "8月26日！"
    assert abs(tokens[0].confidence - 0.7342) < 1e-6
    assert tokens[0].page == 3
    # box normalized to 0..1 of (600,200): xs 333..552 -> 0.555..0.92 ; ys 40..122 -> 0.2..0.61
    x1, y1, x2, y2 = tokens[0].box
    assert abs(x1 - 333 / 600) < 1e-6 and abs(x2 - 552 / 600) < 1e-6
    assert abs(y1 - 40 / 200) < 1e-6 and abs(y2 - 122 / 200) < 1e-6
    assert all(0.0 <= v <= 1.0 for v in tokens[0].box)


def test_empty_response():
    assert _tokens_from_api_result({}, 100, 100, 1) == []


def test_skips_items_without_text_or_bad_score():
    result = {
        "0": {"rec_txt": "", "dt_boxes": [[0, 0]], "score": "0.9"},          # empty text -> skip
        "1": {"rec_txt": "ok", "dt_boxes": None, "score": "not-a-number"},   # bad score -> 0.0, box default
    }
    tokens = _tokens_from_api_result(result, 100, 100, 1)
    assert len(tokens) == 1
    assert tokens[0].text == "ok"
    assert tokens[0].confidence == 0.0
    assert tokens[0].box == [0.0, 0.0, 0.0, 0.0]


def test_get_api_url_appends_ocr_and_respects_full_endpoint(monkeypatch):
    monkeypatch.setenv("RAPIDOCR_API_URL", "http://localhost:9003")
    # get_api_url reads the PersistentConfig; simulate via the config module value.
    import open_webui.config as cfg
    monkeypatch.setattr(cfg.RAPIDOCR_API_URL, "value", "http://localhost:9003", raising=False)
    assert get_api_url() == "http://localhost:9003/ocr"
    monkeypatch.setattr(cfg.RAPIDOCR_API_URL, "value", "http://localhost:9003/ocr", raising=False)
    assert get_api_url() == "http://localhost:9003/ocr"
    monkeypatch.setattr(cfg.RAPIDOCR_API_URL, "value", "", raising=False)
    assert get_api_url() == ""

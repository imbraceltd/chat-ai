"""Deterministic grounding of LLM-extracted fields against real OCR tokens.

Given the structured data produced by the vision LLM and a list of ``OcrToken``s
from a real OCR engine, walk every leaf value and attach a *grounded* confidence:

    confidence = ocr_token_confidence * agreement

where ``agreement`` is the text similarity (stdlib ``difflib``) between the LLM's
value and the best-matching run of OCR tokens. A value the LLM invented matches no
token (agreement ~ 0), and a value read from a blurry region inherits the engine's
low confidence — so the score flags both hallucination and illegible source.

Schema-agnostic: works on arbitrary nested dict/list output.
"""

import logging
import re
from difflib import SequenceMatcher
from typing import Any, List, Optional, Tuple

from open_webui.utils.ocr.base import OcrToken

log = logging.getLogger(__name__)

# Prefer rapidfuzz for the agreement metric: its normalized Levenshtein is the
# industry-standard CER/WER (edit-distance) similarity and is much faster than
# difflib. Fall back to difflib (Ratcliff/Obershelp) if rapidfuzz is unavailable.
try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz.distance import Levenshtein as _rf_lev

    _HAS_RAPIDFUZZ = True
except Exception:  # pragma: no cover - exercised only when rapidfuzz is absent
    _HAS_RAPIDFUZZ = False


def _sim(a: str, b: str) -> float:
    """Full-string similarity in 0..1 (normalized Levenshtein = 1 − CER)."""
    if _HAS_RAPIDFUZZ:
        return _rf_lev.normalized_similarity(a, b)
    return SequenceMatcher(None, a, b).ratio()


def _token_sort(a: str, b: str) -> float:
    """Word-order-insensitive similarity (handles OCR reordering words vs the value)."""
    if _HAS_RAPIDFUZZ:
        return _rf_fuzz.token_sort_ratio(a, b) / 100.0
    a2 = " ".join(sorted(a.split()))
    b2 = " ".join(sorted(b.split()))
    return SequenceMatcher(None, a2, b2).ratio()

_WRAPPED_KEYS = {"value", "box", "confidence"}

# Max candidate tokens evaluated per field (ranked by shared n-gram overlap). Bounds
# reconciliation cost on dense, many-page documents without dropping the true match.
_MAX_CANDIDATES = 40

# Minimum score for a lenient "rescue" metric (substring / word-order / digit substring)
# to count. Below this it's treated as noise, so hallucinated values stay low.
_RESCUE_GATE = 0.85


def _norm(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace. Keeps unicode word chars."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w]", " ", s.lower())).strip()


def _is_wrapped(node: dict) -> bool:
    """True for a ``{value, box?, confidence?}`` envelope (ours or LLM-produced)."""
    return "value" in node and set(node.keys()).issubset(_WRAPPED_KEYS)


def _digits(s: str) -> str:
    """Keep only ASCII digits — used for numeric/date-aware matching."""
    return re.sub(r"\D", "", s)


def _partial(short: str, long: str) -> float:
    """Best-substring similarity in 0..1.

    Lets a field value that is a *substring* of an OCR line keep high agreement even
    though OCR returned the whole line as one token (e.g. value "12345" vs token
    "INVOICE 12345"). Caller guards to values of length >= 3 to avoid trivial matches.
    """
    if not short or not long:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return _rf_fuzz.partial_ratio(short, long) / 100.0
    return _partial_ratio_difflib(short, long)


def _partial_ratio_difflib(short: str, long: str) -> float:
    """Stdlib difflib fallback for :func:`_partial` (best-substring alignment)."""
    if len(short) > len(long):
        short, long = long, short
    matcher = SequenceMatcher(None, short, long)
    best = 0.0
    for block in matcher.get_matching_blocks():
        start = max(0, block.b - block.a)
        window = long[start : start + len(short)]
        best = max(best, SequenceMatcher(None, short, window).ratio())
        if best >= 0.999:
            break
    return best


def _ngrams(s: str, n: int = 2) -> set:
    """Character n-grams (whitespace removed). CJK-safe: works without word spaces."""
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _build_ngram_index(norm_tokens: List[Tuple[OcrToken, str]]) -> dict:
    """Inverted index: char n-gram -> token indices that contain it.

    Used to cheaply shortlist candidate tokens per field value (instead of scanning
    every token for every field), which keeps reconciliation fast on dense, many-page
    documents. Character n-grams (not whitespace words) so it also works for CJK text.
    """
    index: dict = {}
    for i, (_t, tnorm) in enumerate(norm_tokens):
        for gram in _ngrams(tnorm):
            index.setdefault(gram, set()).add(i)
    return index


def _agreement(value_norm: str, combined: str) -> float:
    """Similarity of a value against a token run, in 0..1.

    Combines: normalized-Levenshtein (CER) full similarity, a directional substring
    bonus (only when value ⊆ combined by length, so multi-word values keep accumulating
    tokens rather than claim a full match on a fragment), word-order-insensitive
    similarity, and a numeric/date-aware digit-stream comparison so formatting
    (commas/dots/spaces) doesn't penalize IDs, amounts, and dates.
    """
    # Fast path: exact substring containment (the common "OCR read it verbatim" case).
    if len(value_norm) >= 3 and value_norm in combined:
        return 1.0

    # Strict full similarity (normalized Levenshtein) is the base. The substring/word-order
    # "rescue" metrics are lenient and inflate non-matches, so they only count when they
    # strongly indicate a real match (>= _RESCUE_GATE); otherwise the strict score stands.
    score = _sim(value_norm, combined)
    if len(value_norm) >= 3:
        ts = _token_sort(value_norm, combined)
        if ts >= _RESCUE_GATE:
            score = max(score, ts)
        if len(combined) >= len(value_norm):
            pr = _partial(value_norm, combined)
            if pr >= _RESCUE_GATE:
                score = max(score, pr)

    # Numeric/date-aware: for values that are mostly digits (IDs, amounts, Western
    # dates), also compare digit streams so separators/formatting don't lower the score.
    value_digits = _digits(value_norm)
    compact = value_norm.replace(" ", "")
    if len(value_digits) >= 3 and compact and len(value_digits) / len(compact) >= 0.5:
        combined_digits = _digits(combined)
        if combined_digits:
            if value_digits in combined_digits:
                score = max(score, 1.0)
            else:
                score = max(score, _sim(value_digits, combined_digits))
                if len(combined_digits) >= len(value_digits):
                    dp = _partial(value_digits, combined_digits)
                    if dp >= _RESCUE_GATE:
                        score = max(score, dp)
    return score


def _union_box(boxes: List[List[float]]) -> Optional[List[float]]:
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _best_match(
    value_norm: str, norm_tokens: List[Tuple[OcrToken, str]], ngram_index: dict
) -> Tuple[float, List[OcrToken]]:
    """Find the consecutive token run whose combined text best matches ``value_norm``.

    Returns ``(agreement_ratio, matched_tokens)``. Considers windows of consecutive
    tokens to handle values that span multiple OCR words/lines.
    """
    n_words = max(1, len(value_norm.split()))
    max_window = n_words + 2
    best_ratio = 0.0
    best_len = 0
    best_run: List[OcrToken] = []

    # Shortlist candidate start tokens by shared char n-grams, ranked by overlap count.
    # The token actually containing the value shares the most n-grams, so the top-K
    # ranking keeps the true match while bounding cost on dense pages where many tokens
    # share a common substring. A value absent from the page (e.g. an LLM hallucination)
    # shares nothing → no candidates → returns 0.0 (the desired low confidence).
    candidate_scores: dict = {}
    for gram in _ngrams(value_norm):
        for idx in ngram_index.get(gram, ()):
            candidate_scores[idx] = candidate_scores.get(idx, 0) + 1
    if not candidate_scores:
        return 0.0, []
    top = sorted(candidate_scores, key=lambda i: candidate_scores[i], reverse=True)[:_MAX_CANDIDATES]

    for i in sorted(top):
        combined = ""
        run: List[OcrToken] = []
        for j in range(i, min(i + max_window, len(norm_tokens))):
            token, tnorm = norm_tokens[j]
            if not tnorm:
                continue
            run.append(token)
            combined = f"{combined} {tnorm}".strip()
            ratio = _agreement(value_norm, combined)
            # Prefer higher agreement; on a (near) tie prefer the tightest run, so a
            # value matches the smallest token span that contains it rather than
            # absorbing neighbouring tokens.
            if ratio > best_ratio + 1e-6 or (
                abs(ratio - best_ratio) <= 1e-6 and (not best_run or len(combined) < best_len)
            ):
                best_ratio = ratio
                best_len = len(combined)
                best_run = list(run)
            # Stop growing once the window is far longer than the value.
            if len(combined) > len(value_norm) * 2 + 5:
                break
    return best_ratio, best_run


def _annotate(
    value: Any, norm_tokens: List[Tuple[OcrToken, str]], ngram_index: dict, threshold: float
) -> Any:
    """Wrap a single leaf value with grounded ``{value, box, confidence}``."""
    # Only ground OCR-able scalars; leave booleans/objects/None untouched.
    if value is None or isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return value

    text = str(value).strip()
    value_norm = _norm(text)
    if not value_norm:
        return value

    ratio, run = _best_match(value_norm, norm_tokens, ngram_index)
    ocr_conf = min((t.confidence for t in run), default=0.0)
    confidence = round(ocr_conf * ratio, 3)

    if confidence < threshold:
        return value  # below threshold: keep the plain value

    return {"value": value, "box": _union_box([t.box for t in run]), "confidence": confidence}


def _walk(node: Any, norm_tokens: List[Tuple[OcrToken, str]], ngram_index: dict, threshold: float) -> Any:
    if isinstance(node, dict):
        if _is_wrapped(node):
            # Unwrap an existing envelope and re-ground (prevents double-wrapping).
            return _annotate(node.get("value"), norm_tokens, ngram_index, threshold)
        return {k: _walk(v, norm_tokens, ngram_index, threshold) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v, norm_tokens, ngram_index, threshold) for v in node]
    return _annotate(node, norm_tokens, ngram_index, threshold)


def reconcile_fields(data: Any, tokens: List[OcrToken], threshold: float = 0.0) -> Any:
    """Attach grounded ``{value, box, confidence}`` to every leaf of ``data``.

    Returns ``data`` unchanged when there are no OCR tokens (OCR unavailable/empty),
    so extraction degrades gracefully.
    """
    if not tokens:
        return data
    norm_tokens = [(t, _norm(t.text)) for t in tokens]
    ngram_index = _build_ngram_index(norm_tokens)
    return _walk(data, norm_tokens, ngram_index, threshold)


def _collect_confidences(node: Any, out: List[float]) -> None:
    if isinstance(node, dict):
        if _is_wrapped(node) and isinstance(node.get("confidence"), (int, float)):
            out.append(float(node["confidence"]))
            return
        for v in node.values():
            _collect_confidences(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_confidences(v, out)


def aggregate_confidence(data: Any) -> Optional[float]:
    """Document-level confidence = mean of all attached field confidences."""
    confs: List[float] = []
    _collect_confidences(data, confs)
    return round(sum(confs) / len(confs), 3) if confs else None

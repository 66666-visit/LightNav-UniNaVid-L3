"""Python 3.8-compatible replacements for newer ``str`` methods.

LightNav upstream targets Python 3.11 and uses ``str.removeprefix``.  The
Uni-NaVid VLN-CE environment pins Python 3.8, so the vendored modules call this
helper instead.  Behaviour is byte-for-byte identical to ``str.removeprefix``.
"""

from __future__ import annotations

import re as _re


def removeprefix(value: str, prefix: str) -> str:
    """Return ``value`` without ``prefix`` when present, like Python 3.9+."""
    if not isinstance(value, str):
        raise TypeError("removeprefix requires a string")
    if value.startswith(prefix):
        return value[len(prefix):]
    return value


_SYNONYMS = {
    "sofa": "couch", "couches": "couch", "sofas": "couch",
    "tv": "television", "tvs": "television", "televisions": "television",
    "chairs": "chair", "tables": "table", "benches": "bench",
}
_TERMS = _re.compile(r"[^\W_]+", _re.UNICODE)


def normalized_terms(value: str) -> tuple[str, ...]:
    """Copy of ``memory_write_gate.normalized_terms`` for the writer hook."""
    return tuple(_SYNONYMS.get(term, term) for term in _TERMS.findall(value.casefold()))


def instruction_match(label: str, instruction: str) -> tuple[str, ...]:
    """Copy of ``memory_write_gate.instruction_match`` (used only with a gate)."""
    label_terms = set(normalized_terms(label))
    instruction_terms = set(normalized_terms(instruction))
    return tuple(sorted(label_terms)) if label_terms and label_terms <= instruction_terms else ()

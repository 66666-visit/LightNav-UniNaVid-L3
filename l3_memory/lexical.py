"""Exact lexical helpers reused by the vendored L3 modules.

The upstream L3 serializers import two small helpers from larger modules:

* ``_plain_label`` from ``lightnav.memory.local_graph_fusion`` (the L1 module);
* ``_terms`` from ``lightnav.memory.v3_retrieval`` (the V3 retrieval module).

Importing either full module would pull the L1/retrieval paths into this
L3-only project even though they are never called.  This file contains only
those two helpers, copied byte-for-byte from the frozen LightNav source.  No
behaviour was changed; the vendoring test compares their outputs with fixtures
derived from the upstream functions.

Upstream commit: ``6aefffa`` (LightNav-0-codex, experiment/l3-official-style-prompt).
"""

from __future__ import annotations

import re

# --- from src/lightnav/memory/v3_retrieval.py -------------------------------

_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_SYNONYMS = {
    "sofa": "couch", "sofas": "couch", "couches": "couch",
    "tv": "television", "tvs": "television", "televisions": "television",
}
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "at", "be", "by", "for", "in", "is", "of",
    "on", "or", "the", "this", "to", "was", "were", "with",
    "find", "go", "head", "look", "move", "navigate", "proceed", "search",
    "travel", "toward", "towards", "walk",
})


def _terms(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (tuple, list)):
        return set().union(*(_terms(part) for part in value)) if value else set()
    return {
        _SYNONYMS.get(word, word)
        for word in _WORDS.findall(str(value).casefold())
        if word not in _STOPWORDS
    }


# --- from src/lightnav/memory/local_graph_fusion.py -------------------------


def _plain_label(value: str) -> str:
    """Keep detector labels short and unable to introduce prompt markers."""
    words = [part for part in value.split() if part]
    cleaned = " ".join("".join(char for char in word if char.isalnum() or char == "-")
                       for word in words)
    return cleaned[:48].strip()

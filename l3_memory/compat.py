"""Python 3.8-compatible string helpers for the vendored L3 modules.

The vendored L3 code comes from LightNav, which targets Python 3.11 and calls
``str.removeprefix``.  Uni-NaVid's VLN-CE environment pins Python 3.8, so the
vendored module calls this helper instead.

This file is deliberately independent of ``l1_memory.compat``: an L3-only run
must not import L1 prompt modules just to get a string helper.  The behaviour
matches both ``l1_memory.compat.removeprefix`` and Python 3.9+
``str.removeprefix``.
"""

from __future__ import annotations

__all__ = ["removeprefix"]


def removeprefix(value: str, prefix: str) -> str:
    """Return ``value`` without ``prefix`` when present, like Python 3.9+."""
    if not isinstance(value, str):
        raise TypeError("removeprefix requires a string")
    if value.startswith(prefix):
        return value[len(prefix):]
    return value

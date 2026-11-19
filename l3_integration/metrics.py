"""Small pure-Python metric helpers for the paired L3 experiment report."""

from __future__ import annotations

import math
import random
from statistics import mean
from typing import Any, Iterable, Sequence


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def metric_mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [finite_float(row.get(key)) for row in rows]
    finite = [value for value in values if value is not None]
    return mean(finite) if finite else None


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean_or_none(values: Iterable[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else None


def mcnemar_exact(baseline_only_success: int, l3_only_success: int) -> dict[str, Any]:
    """Two-sided exact McNemar test for paired binary outcomes.

    ``baseline_only_success`` is A=success, B=failure; ``l3_only_success`` is
    A=failure, B=success.  Discordant pairs are the only information used.
    """
    if baseline_only_success < 0 or l3_only_success < 0:
        raise ValueError("discordant counts must be nonnegative")
    discordant = baseline_only_success + l3_only_success
    if discordant == 0:
        return {
            "b": baseline_only_success,
            "c": l3_only_success,
            "discordant": 0,
            "p_value": 1.0,
            "method": "exact_binomial_two_sided",
        }
    tail = min(baseline_only_success, l3_only_success)
    probability = sum(
        math.comb(discordant, k) for k in range(tail + 1)
    ) / (2 ** discordant)
    return {
        "b": baseline_only_success,
        "c": l3_only_success,
        "discordant": discordant,
        "p_value": min(1.0, 2.0 * probability),
        "method": "exact_binomial_two_sided",
    }


def paired_bootstrap_ci(
    deltas: Sequence[float], *, seed: int = 42, samples: int = 10_000,
) -> dict[str, Any]:
    """Percentile bootstrap CI for the mean paired delta (fixed seed)."""
    values = [float(value) for value in deltas]
    if not values:
        return {"mean": None, "low": None, "high": None, "samples": 0}
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "mean": sum(values) / n,
        "low": percentile(means, 0.025),
        "high": percentile(means, 0.975),
        "samples": samples,
        "seed": seed,
    }


def paired_rows(
    baseline: dict[tuple[str, str], dict[str, Any]],
    l3: dict[tuple[str, str], dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return paired rows in baseline order; keys must match exactly."""
    if set(baseline) != set(l3):
        missing = sorted(set(baseline) - set(l3))
        extra = sorted(set(l3) - set(baseline))
        raise ValueError(
            f"paired Episode keys differ: missing={missing[:5]}, extra={extra[:5]}"
        )
    return [(baseline[key], l3[key]) for key in baseline]


def paired_summary(
    baseline: Sequence[dict[str, Any]], l3: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Paired deltas (L3 minus baseline) for the headline metrics."""
    if len(baseline) != len(l3):
        raise ValueError("paired groups have different episode counts")
    metrics = ("success", "spl", "ndtw", "oracle_success", "distance_to_goal",
               "path_length")
    result: dict[str, Any] = {"n_pairs": len(baseline), "metrics": {}}
    for metric in metrics:
        deltas: list[float] = []
        for a_row, b_row in zip(baseline, l3):
            if metric in {"success", "oracle_success"}:
                a = 1.0 if bool(a_row.get(metric)) else 0.0
                b = 1.0 if bool(b_row.get(metric)) else 0.0
            else:
                a = finite_float(a_row.get(metric))
                b = finite_float(b_row.get(metric))
            if a is None or b is None:
                continue
            deltas.append(b - a)
        result["metrics"][metric] = {
            "n": len(deltas),
            "mean_delta": mean_or_none(deltas),
            "bootstrap": paired_bootstrap_ci(deltas),
        }
    b = sum(
        1 for a_row, b_row in zip(baseline, l3)
        if bool(a_row.get("success")) is False and bool(b_row.get("success")) is True
    )
    c = sum(
        1 for a_row, b_row in zip(baseline, l3)
        if bool(a_row.get("success")) is True and bool(b_row.get("success")) is False
    )
    result["success_flips"] = {
        "baseline_fail_l3_success": b,
        "baseline_success_l3_fail": c,
        "mcnemar": mcnemar_exact(b, c),
    }
    return result

"""Pair observed memory facts with their own source background in the task slot.

This is an independent adaptation of CONFACT's SBAdir evidence/background
ordering.  The actual Uni-NaVid prompt template is supplied by the agent and
formatted only once.  L3's original context is produced by the unchanged
7602b25 organizer; its fact sentences are never rewritten or reselected here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from l3_integration.official_prompt import build_prompt, build_task_text


def _one_line(value: Any) -> str:
    """Keep data-derived provenance in the existing task text, not a new role."""
    return " ".join(str(value).split())


def _ids(fact: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in (getattr(fact, "evidence_ids", ()) or ()) if item)


def _label(fact: Any) -> str:
    return _one_line(getattr(fact, "label", "")).casefold()


def _fact_line(fact: Any) -> str:
    line = getattr(fact, "line", None)
    if callable(line):
        return _one_line(line())
    return _one_line(getattr(fact, "text", ""))


@dataclass(frozen=True)
class Claim:
    """One source-backed attribute; entity IDs must come from actual evidence."""

    entity_id: Optional[str]
    attribute: str
    value: str
    layer: str
    evidence_ids: tuple[str, ...]


def classify_claims(first: Claim, second: Claim) -> str:
    """Classify only a proven same-entity, same-attribute comparison."""
    if not first.entity_id or not second.entity_id or first.entity_id != second.entity_id:
        return "uncertain_entity"
    if first.attribute != second.attribute:
        return "complementary"
    if _one_line(first.value).casefold() == _one_line(second.value).casefold():
        return "duplicate"
    return "conflict"


def _matched_l3(
    l1_fact: Any, l3_facts: Sequence[Any],
    entity_links: Optional[Mapping[str, str]] = None,
) -> Optional[Any]:
    """Use only the L3 map's validated, unambiguous L1 evidence links."""
    l1_ids = set(_ids(l1_fact))
    if not l1_ids:
        return None
    links = entity_links or {}
    linked = []
    for fact in l3_facts:
        verified_ids = set(getattr(fact, "linked_l1_evidence_ids", ()) or ())
        verified_ids.update(
            evidence_id for evidence_id, node_id in links.items()
            if node_id == getattr(fact, "object_id", None)
        )
        if l1_ids.issubset(verified_ids):
            linked.append(fact)
    return linked[0] if len(linked) == 1 else None


def _background(fact: Any, layer: str) -> str:
    """Pair one fact with the real source fields of its own observation.

    Only fields that exist on the selected evidence are printed.  A required
    field that is absent is written as ``unavailable`` instead of being
    silently dropped or guessed; no field is a sensor trust score.
    """
    evidence = _ids(fact)
    source_records = tuple(getattr(fact, "evidence", ()) or ())
    latest_record = source_records[0] if source_records else None
    if layer == "L1" and latest_record is not None:
        source = getattr(latest_record, "detector_name", None) or getattr(
            fact, "source", "")
        step = getattr(latest_record, "observation_step", None)
        confidence = getattr(latest_record, "detection_confidence", None)
        # L1's selected records put the current/latest observation first, so
        # the printed source, step and confidence belong to this evidence ID.
        primary_evidence = getattr(latest_record, "evidence_id", None)
    else:
        source = getattr(fact, "source", "")
        step = getattr(fact, "latest_step", None)
        confidence = getattr(fact, "confidence", None)
        # L3's evidence IDs are chronological; pair the newest one with the
        # printed latest step and latest segmentation confidence.
        primary_evidence = evidence[-1] if evidence else None

    parts = ["L3 RGB-D map" if layer == "L3" else "L1 RGB observation"]
    cleaned_source = _one_line(source) if source else ""
    parts.append("source " + (cleaned_source or "unavailable"))
    if layer == "L1" and latest_record is not None:
        rgb_source = getattr(latest_record, "rgb_observation_source", None)
        parts.append("RGB input " + (
            _one_line(rgb_source) if rgb_source else "unavailable"))
    if layer == "L3":
        node = getattr(fact, "object_id", None)
        parts.append("map node " + (
            _one_line(node) if node else "unavailable"))
        parts.append(
            "geometry observed" if getattr(fact, "geometry_verified", False)
            else "geometry unverified"
        )
    if type(step) is int and step >= 0:
        parts.append("step " + str(step))
    else:
        parts.append("step unavailable")
    if isinstance(confidence, (int, float)) and math.isfinite(float(confidence)):
        parts.append(
            ("seg" if layer == "L3" else "detector") + " conf "
            + format(float(confidence), ".2f")
        )
    else:
        parts.append(("seg" if layer == "L3" else "detector") + " conf unavailable")
    parts.append("evidence " + (
        _one_line(primary_evidence) if primary_evidence else "unavailable"))
    if layer == "L1":
        place_source = getattr(fact, "visual_place_source", None)
        place_evidence = tuple(getattr(fact, "visual_place_evidence_ids", ()) or ())
        if getattr(fact, "visual_place_id", None) or place_source or place_evidence:
            parts.append("visual place source " + (
                _one_line(place_source) if place_source else "unavailable"))
            parts.append("visual place evidence " + (
                _one_line(place_evidence[-1]) if place_evidence else "unavailable"))
    return " [" + "; ".join(parts) + "]"


def _l1_line(fact: Any, match: Optional[Any], relation: str) -> str:
    if match is None:
        return "L1 RGB fact: " + _fact_line(fact)
    if relation == "conflict":
        # Both observations are kept verbatim.  The shared map entity and the
        # L3 value are named so the model can see the disagreement; neither
        # side is declared correct and no value is rewritten.
        return (
            "L1 RGB fact: " + _fact_line(fact)
            + "; same map entity " + _one_line(match.object_id)
            + "; L3 reports " + _one_line(match.label)
            + "; both retained"
        )
    # The shared object-label assertion is stated by the L3 line already.
    # Preserve the remaining L1 observation attributes and their provenance.
    role = getattr(fact, "role", None)
    sector = getattr(fact, "sector", None)
    sightings = getattr(fact, "sightings", None)
    parts = ["L1 RGB observation also supports map node " + _one_line(match.object_id)]
    if role == "current":
        parts.append("current view")
    elif role == "history":
        parts.append("earlier view; visibility now unverified")
    if sector:
        parts.append("image sector " + _one_line(sector))
    if type(sightings) is int and sightings > 1:
        parts.append(str(sightings) + " sightings")
    if getattr(fact, "visual_place_id", None):
        parts.append(
            "near a visually similar area (possible revisit; "
            "current direction and passage unknown)"
        )
    return "; ".join(parts) + "."


def _relation(fact: Any, match: Optional[Any]) -> str:
    """Classify one retained L1 fact against its verified L3 map entity."""
    if match is None:
        return "no_match"
    l1_claim = Claim(str(match.object_id), "object_label", _label(fact), "L1", _ids(fact))
    l3_claim = Claim(str(match.object_id), "object_label", _label(match), "L3", _ids(match))
    return classify_claims(l1_claim, l3_claim)


def _fact_identities(facts: Sequence[Any]) -> list[dict[str, Any]]:
    """Serializable identity of each selected fact, without its provenance."""
    return [
        {
            "object_id": getattr(fact, "object_id", None),
            "label": _one_line(getattr(fact, "label", "")),
            "evidence_ids": list(_ids(fact)),
            "line": _fact_line(fact),
        }
        for fact in facts
    ]


@dataclass
class FusionResult:
    task_text: str
    memory_text: Optional[str]
    l3_original_text: Optional[str]
    l1_text: Optional[str]
    diagnostics: dict[str, Any]
    l3_enriched_text: Optional[str] = None


def compose_sbadir(
    *,
    template: str,
    instruction: str,
    l3_text: Optional[str],
    l3_facts: Sequence[Any],
    l1_facts: Sequence[Any],
    count_tokens: Callable[[str], int],
    max_prompt_tokens: int,
    l1_max_tokens: int = 0,
    entity_links: Optional[Mapping[str, str]] = None,
) -> FusionResult:
    """Preserve all L3 core lines, then spend remaining tokens on L1 facts.

    ``max_prompt_tokens`` is measured from the loaded checkpoint's context
    limit after reserving generation and safety tokens.  ``count_tokens`` is
    the checkpoint tokenizer.  Whole source-backed lines are retained or
    omitted; evidence is never cut mid-line.
    """
    if not isinstance(instruction, str):
        raise ValueError("instruction must be text")
    if type(max_prompt_tokens) is not int or max_prompt_tokens <= 0:
        raise ValueError("a measured positive prompt token capacity is required")
    if l1_facts:
        # C must not route its original Local Graph card through SBAdir.  The
        # per-fact L1 fusion path is retained only as historical B-module code
        # and is disabled here so it cannot be reattached accidentally.
        raise ValueError(
            "SBAdir L1 fact fusion is disabled; C appends the original L1 card"
        )
    original = l3_text if isinstance(l3_text, str) and l3_text.strip() else None
    core = build_task_text(instruction, original)
    core_tokens = int(count_tokens(build_prompt(template, core)))
    if core_tokens > max_prompt_tokens:
        raise ValueError("original L3 prompt exceeds measured prompt capacity")

    lines = original.splitlines() if original else []
    selected_l3 = list(l3_facts)
    # The existing organizer emits one header followed by one sentence for
    # each selected fact.  Refuse an unfamiliar structure rather than attaching
    # source data to the wrong sentence.
    if original and len(lines) != len(selected_l3) + 1:
        raise ValueError("L3 context/fact line alignment changed")
    if not original and selected_l3:
        raise ValueError("L3 facts exist without their official context")
    enriched = list(lines)
    current_task = core
    l3_provenance_count = 0
    omitted_l3_provenance = 0
    for index, fact in enumerate(selected_l3, 1):
        candidate_lines = list(enriched)
        candidate_lines[index] = lines[index] + _background(fact, "L3")
        candidate_memory = "\n".join(candidate_lines)
        candidate_task = build_task_text(instruction, candidate_memory)
        if count_tokens(build_prompt(template, candidate_task)) <= max_prompt_tokens:
            enriched = candidate_lines
            current_task = candidate_task
            l3_provenance_count += 1
        else:
            omitted_l3_provenance += 1

    l3_enriched_tokens = int(count_tokens(build_prompt(template, current_task)))
    l1_lines: list[str] = []
    l1_retained = 0
    l1_omitted = 0
    l1_retained_evidence: list[list[str]] = []
    retained_pairs: list[tuple[Any, Optional[Any], str]] = []
    for fact in l1_facts:
        match = _matched_l3(fact, selected_l3, entity_links)
        relation = _relation(fact, match)
        # Keep both original values if the linked evidence disagrees.
        line = _l1_line(fact, match, relation)
        line += _background(fact, "L1")
        candidate_lines = l1_lines + [line]
        candidate_memory = "\n".join([*enriched, *candidate_lines])
        candidate_task = build_task_text(instruction, candidate_memory)
        candidate_tokens = int(count_tokens(build_prompt(template, candidate_task)))
        if (candidate_tokens <= max_prompt_tokens
                and candidate_tokens - l3_enriched_tokens <= l1_max_tokens):
            l1_lines = candidate_lines
            current_task = candidate_task
            l1_retained += 1
            l1_retained_evidence.append(list(_ids(fact)))
            retained_pairs.append((fact, match, relation))
        else:
            l1_omitted += 1

    duplicate_count = 0
    conflict_records: list[dict[str, Any]] = []
    uncertain_count = 0
    for fact, match, relation in retained_pairs:
        if match is None:
            if any(_label(fact) == _label(item) for item in selected_l3):
                uncertain_count += 1
            continue
        if relation == "duplicate":
            duplicate_count += 1
        elif relation == "conflict":
            conflict_records.append({
                "type": "object_label", "entity_id": str(match.object_id),
                "l1_value": _label(fact), "l3_value": _label(match),
                "l1_evidence_ids": list(_ids(fact)),
                "l3_evidence_ids": list(_ids(match)),
                "resolution": "both_retained",
            })

    final_memory = "\n".join([*enriched, *l1_lines]) or None
    final_task = build_task_text(instruction, final_memory)
    final_tokens = int(count_tokens(build_prompt(template, final_task)))
    if final_tokens > max_prompt_tokens:
        raise AssertionError("fusion exceeded the measured prompt capacity")
    # Removing only the appended source descriptions restores the original
    # L3 organizer lines exactly; the collector is never called again.
    l3_core_unchanged = all(
        enriched[index].startswith(line) for index, line in enumerate(lines)
    )
    l3_enriched_text = "\n".join(enriched) if enriched else None
    fact_identities = _fact_identities(selected_l3)
    diagnostics = {
        "l3_fact_count": len(selected_l3),
        "l1_fact_count": len(l1_facts),
        "l3_token_count": l3_enriched_tokens - int(count_tokens(build_prompt(template, instruction))),
        "l3_original_token_count": core_tokens - int(count_tokens(build_prompt(template, instruction))),
        "l1_token_count": final_tokens - l3_enriched_tokens,
        "duplicated_facts": duplicate_count,
        "conflicting_facts": len(conflict_records),
        "uncertain_entity_matches": uncertain_count,
        "retained_facts": len(selected_l3) + l1_retained,
        "omitted_facts": l1_omitted,
        "l1_retained_facts": l1_retained,
        "l1_omitted_facts": l1_omitted,
        "l3_provenance_count": l3_provenance_count,
        "omitted_l3_provenance": omitted_l3_provenance,
        "final_prompt_tokens": final_tokens,
        "max_prompt_tokens": max_prompt_tokens,
        "l1_max_tokens": l1_max_tokens,
        "l3_original_sha256": hashlib.sha256((original or "").encode("utf-8")).hexdigest(),
        # Source background is built only from the selected fact records, so
        # equal facts produce an equal enriched L3 block (B vs C check).
        "l3_enriched_sha256": hashlib.sha256(
            (l3_enriched_text or "").encode("utf-8")
        ).hexdigest(),
        "l3_fact_signature_sha256": hashlib.sha256(json.dumps(
            fact_identities, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")).hexdigest(),
        "l3_core_unchanged": l3_core_unchanged,
        "conflicts": conflict_records,
        "l1_evidence_ids": [list(_ids(fact)) for fact in l1_facts],
        "l1_retained_evidence_ids": l1_retained_evidence,
        "l3_evidence_ids": [list(_ids(fact)) for fact in selected_l3],
    }
    return FusionResult(final_task, final_memory, original,
                        "\n".join(l1_lines) or None, diagnostics,
                        l3_enriched_text)


#: The LightNav 75% Local Graph card budget.  The L1 block is built and bounded
#: by the original Local Graph builder/``build_context_text`` with exactly this value; the
#: C group does not lift it or replace it with a token-derived L1 budget.
ORIGINAL_L1_PROMPT_MAX_CHARS = 320
ORIGINAL_L1_HEADER = "[Navigation Memory]"


@dataclass
class OriginalL1Result:
    """Result of appending a verbatim original Local Graph card to B's task text."""

    task_text: str
    memory_text: Optional[str]
    l1_text: Optional[str]
    diagnostics: dict[str, Any]


def append_original_l1(
    *,
    template: str,
    instruction: str,
    base_memory_text: Optional[str],
    l1_text: Optional[str],
    count_tokens: Optional[Callable[[str], int]] = None,
    max_prompt_tokens: Optional[int] = None,
    char_budget: Optional[int] = ORIGINAL_L1_PROMPT_MAX_CHARS,
    token_aware: bool = False,
) -> OriginalL1Result:
    """Append the already-selected original L1 card after B's task text.

    This is intentionally not a fusion.  The C adapter selects whole L1 lines
    against the measured token budget before calling this function; this
    function only inserts the resulting card byte-for-byte after the frozen
    L3+source-background block, or omits it as a whole when it does not fit the
    exact prompt capacity.  It never rewrites an L1 line, never attaches an L1
    source background, and never cuts a line.

    ``char_budget`` is the original 320-character fallback.  It is ``None``
    only when a measured token budget owns selection; it is never replaced by
    an arbitrary larger character number.
    """
    if not isinstance(instruction, str):
        raise ValueError("instruction must be text")
    if max_prompt_tokens is not None and (
        type(max_prompt_tokens) is not int or max_prompt_tokens <= 0
    ):
        raise ValueError("a measured positive prompt token capacity is required")
    if count_tokens is not None and not callable(count_tokens):
        raise ValueError("count_tokens must be callable or None")
    if char_budget is not None and (
        type(char_budget) is not int or char_budget < 1
    ):
        raise ValueError("char_budget must be a positive integer or None")
    base = base_memory_text if isinstance(base_memory_text, str) and base_memory_text.strip() else None
    if l1_text is None or not isinstance(l1_text, str) or not l1_text.strip():
        diagnostics = {
            "l1_present": False,
            "l1_injected": False,
            "l1_chars": 0,
            "l1_sha256": None,
            "l1_token_count": 0,
            "empty_reason": "no_original_l1_card",
            "char_budget": char_budget,
        }
        if token_aware:
            diagnostics.update({
                "token_overflow_drops": 0,
                "budget_source": "token_aware_context",
            })
        return OriginalL1Result(
            task_text=build_task_text(instruction, base),
            memory_text=base,
            l1_text=None,
            diagnostics=diagnostics,
        )
    if char_budget is not None and len(l1_text) > char_budget:
        raise ValueError(
            "original L1 card exceeds its {}-character budget: {}".format(
                char_budget, len(l1_text),
            )
        )
    if not l1_text.startswith(ORIGINAL_L1_HEADER + "\n"):
        raise ValueError("original L1 card lacks the Local Graph header")
    combined = l1_text if base is None else base + "\n" + l1_text
    candidate_task = build_task_text(instruction, combined)
    base_task = build_task_text(instruction, base)
    candidate_tokens = None
    base_tokens = None
    measured = count_tokens is not None and max_prompt_tokens is not None
    if measured:
        try:
            candidate_tokens = int(count_tokens(build_prompt(template, candidate_task)))
            base_tokens = int(count_tokens(build_prompt(template, base_task)))
        except Exception as exc:  # noqa: BLE001 - do not guess a capacity
            return OriginalL1Result(
                task_text=base_task,
                memory_text=base,
                l1_text=None,
                diagnostics={
                    "l1_present": True,
                    "l1_injected": False,
                    "l1_chars": len(l1_text),
                    "l1_sha256": hashlib.sha256(l1_text.encode("utf-8")).hexdigest(),
                    "l1_token_count": None,
                    "token_overflow_drops": len(l1_text.splitlines()),
                    "empty_reason": "prompt_measurement_failed: {}: {}".format(
                        type(exc).__name__, exc,
                    ),
                    "char_budget": char_budget,
                    "budget_source": (
                        "token_aware_context" if token_aware else "char_fallback_320"
                    ),
                },
            )
        if candidate_tokens > max_prompt_tokens:
            return OriginalL1Result(
                task_text=base_task,
                memory_text=base,
                l1_text=None,
                diagnostics={
                    "l1_present": True,
                    "l1_injected": False,
                    "l1_chars": len(l1_text),
                    "l1_sha256": hashlib.sha256(l1_text.encode("utf-8")).hexdigest(),
                    "l1_token_count": 0,
                    "token_overflow_drops": len(l1_text.splitlines()),
                    "empty_reason": "measured_prompt_capacity",
                    "char_budget": char_budget,
                    "budget_source": (
                        "token_aware_context" if token_aware else "char_fallback_320"
                    ),
                    "base_prompt_tokens": base_tokens,
                    "final_prompt_tokens": base_tokens,
                    "max_prompt_tokens": max_prompt_tokens,
                },
            )
    if l1_text not in candidate_task:
        raise AssertionError("original L1 card was not inserted verbatim")
    return OriginalL1Result(
        task_text=candidate_task,
        memory_text=combined,
        l1_text=l1_text,
        diagnostics={
            "l1_present": True,
            "l1_injected": True,
            "l1_chars": len(l1_text),
            "l1_sha256": hashlib.sha256(l1_text.encode("utf-8")).hexdigest(),
            "l1_token_count": (
                candidate_tokens - base_tokens if measured else None
            ),
            "token_overflow_drops": 0,
            "empty_reason": None,
            "char_budget": char_budget,
            "budget_source": (
                "token_aware_context" if token_aware else "char_fallback_320"
            ),
            "base_prompt_tokens": base_tokens,
            "final_prompt_tokens": candidate_tokens,
            "max_prompt_tokens": max_prompt_tokens,
        },
    )

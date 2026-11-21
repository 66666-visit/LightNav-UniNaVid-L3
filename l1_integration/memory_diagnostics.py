"""Read-only stage counters for explaining an empty memory prompt.

The vendored LightNav builders keep their upstream code unchanged.  This module
re-reads the same store/map/events *after* a prompt context was built and reports
why a detection or a 3D object did not become a prompt fact.  It never selects,
re-scores, writes or removes a record, and it never changes a prompt.

The numbers are deliberately split by layer so a zero-fact smoke can distinguish:

* no detector event reached the writer;
* evidence was persisted but was below the 0.5 L1 prompt floor;
* evidence was persisted but its class did not overlap the instruction;
* the 3D map has objects but none are instruction-relevant;
* an object is only observed in the current step and is not served as history;
* L3 facts exist but no L1 counterpart was available for joint alignment.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from l1_memory.l1_prompt_organizer import _MIN_CONFIDENCE as L1_PROMPT_FLOOR
from l3_memory.lexical import _plain_label, _terms
from l3_memory.schemas import LayerMemoryItem
from l3_memory.v3_schemas import canonical_evidence_key

_DETECTION_KINDS = frozenset({"object_observation", "landmark"})


def _item_has_l1(item: LayerMemoryItem) -> bool:
    contribution = item.metadata.get("layer_contribution") or {}
    return bool(
        contribution.get("l1")
        or item.metadata.get("layer_source") == "l1"
        or item.metadata.get("role") in {"current", "history"}
    )


def _item_has_l3(item: LayerMemoryItem) -> bool:
    contribution = item.metadata.get("layer_contribution") or {}
    return bool(
        contribution.get("l3")
        or item.metadata.get("layer_source") == "l3"
        or item.metadata.get("object_id") is not None
    )


def _relevant(label: str | None, instruction: str) -> bool:
    if not label:
        return False
    return bool(_terms(_plain_label(label)) & _terms(instruction))


def _records_for_step(
    store: Any, scene_id: str, episode_id: str, rollout_idx: int, step: int,
) -> list[Any]:
    return [
        record for record in store.evidence_for_scene(scene_id)
        if (record.event_type in _DETECTION_KINDS
            and record.episode_id == episode_id
            and record.rollout_idx == rollout_idx
            and record.step == step)
    ]


def _history_records(
    store: Any, scene_id: str, episode_id: str, rollout_idx: int, step: int,
) -> list[Any]:
    return [
        record for record in store.evidence_for_scene(scene_id)
        if (record.event_type in _DETECTION_KINDS
            and record.episode_id == episode_id
            and record.rollout_idx == rollout_idx
            and record.step < step)
    ]


def _object_evidence(object_map: Any, obj: Any) -> list[Any]:
    return [
        object_map.evidence[evidence_id]
        for evidence_id in obj.evidence_ids
        if evidence_id in object_map.evidence
    ]


def _object_history(
    object_map: Any, obj: Any, step: int, episode_id: str, rollout_idx: int,
    scope: str, min_confidence: float,
) -> bool:
    for edge in object_map.edges:
        if edge.relation_type != "observed_from":
            continue
        if edge.object_id != obj.object_id or edge.evidence_id not in obj.evidence_ids:
            continue
        viewpoint = object_map.viewpoints.get(edge.viewpoint_id)
        record = object_map.evidence.get(edge.evidence_id)
        if viewpoint is None or record is None:
            continue
        if (record.scene_id, record.episode_id, record.rollout_idx, record.step_id) != (
            viewpoint.scene_id, viewpoint.episode_id, viewpoint.rollout_idx,
            viewpoint.step_id,
        ):
            continue
        same_episode = (record.episode_id, record.rollout_idx) == (
            episode_id, rollout_idx,
        )
        if same_episode and record.step_id >= step:
            continue
        if not same_episode and scope != "scene_persistent":
            continue
        if (record.geometry_status == "observed"
                and record.confidence >= min_confidence):
            return True
    return False


def diagnose_prompt_pipeline(
    *,
    result: Any,
    store: Any,
    object_map: Any | None,
    scene_id: str,
    episode_id: str,
    rollout_idx: int,
    step: int,
    instruction: str,
    events: Iterable[Any],
    min_confidence: float,
    scope: str,
) -> dict[str, Any]:
    """Return diagnostic counters; raising here must never break a prompt."""
    context = getattr(result, "context", None)
    items = list(context.layer_items) if context is not None else []
    has_l1 = [_item_has_l1(item) for item in items]
    has_l3 = [_item_has_l3(item) for item in items]
    selected_l1 = sum(has_l1)
    selected_l3 = sum(has_l3)
    selected_joint = sum(l1 and l3 for l1, l3 in zip(has_l1, has_l3))

    current_events = [
        event for event in events
        if event.kind in _DETECTION_KINDS and event.step_id == step
        and event.episode_id == episode_id
    ]
    current_records = _records_for_step(
        store, scene_id, episode_id, rollout_idx, step,
    )
    current_keys = {record.key for record in current_records}
    missing_keys = [
        canonical_evidence_key(scene_id, episode_id, rollout_idx, event.event_id)
        for event in current_events
        if canonical_evidence_key(
            scene_id, episode_id, rollout_idx, event.event_id,
        ) not in current_keys
    ]
    relevant_records = [
        record for record in current_records
        if _relevant(record.semantic_label, instruction)
    ]
    above_floor = [
        record for record in relevant_records
        if (record.raw_detector_confidence or 0.0) >= L1_PROMPT_FLOOR
        and record.observation_status == "observed"
        and record.provenance_validity
    ]
    below_floor = [
        record for record in relevant_records
        if (record.raw_detector_confidence or 0.0) < L1_PROMPT_FLOOR
    ]
    invalid_records = [
        record for record in relevant_records
        if record.observation_status != "observed" or not record.provenance_validity
    ]
    historical_relevant = [
        record for record in _history_records(
            store, scene_id, episode_id, rollout_idx, step,
        )
        if _relevant(record.semantic_label, instruction)
        and (record.raw_detector_confidence or 0.0) >= L1_PROMPT_FLOOR
        and record.observation_status == "observed"
        and record.provenance_validity
    ]

    l1 = {
        "current_detection_events": len(current_events),
        "current_persisted_detection_records": len(current_records),
        "current_missing_evidence_records": len(missing_keys),
        "current_instruction_relevant_records": len(relevant_records),
        "current_rejected_irrelevant_records": max(
            0, len(current_records) - len(relevant_records)
        ),
        "current_above_prompt_floor_records": len(above_floor),
        "current_below_prompt_floor_records": len(below_floor),
        "current_invalid_provenance_records": len(invalid_records),
        "history_eligible_relevant_records": len(historical_relevant),
        "selected_l1_items": selected_l1,
        "prompt_floor": L1_PROMPT_FLOOR,
    }

    l3: dict[str, Any] = {
        "map_object_count": 0,
        "instruction_relevant_object_count": 0,
        "relevance_rejected_count": 0,
        "geometry_verified_relevant_count": 0,
        "geometry_unverified_relevant_count": 0,
        "current_only_relevant_count": 0,
        "history_capable_relevant_count": 0,
        "history_unavailable_relevant_count": 0,
        "selected_l3_items": selected_l3,
        "prompt_floor": float(min_confidence),
    }
    if object_map is not None:
        l3["map_object_count"] = len(object_map.objects)
        for obj in object_map.objects.values():
            label = _plain_label(obj.class_name)
            if not label or not _relevant(label, instruction):
                l3["relevance_rejected_count"] += 1
                continue
            l3["instruction_relevant_object_count"] += 1
            evidence = _object_evidence(object_map, obj)
            verified = bool(evidence) and all(
                record.geometry_status == "observed"
                and record.confidence >= min_confidence
                for record in evidence
            )
            if verified:
                l3["geometry_verified_relevant_count"] += 1
            else:
                l3["geometry_unverified_relevant_count"] += 1
            if _object_history(
                object_map, obj, step, episode_id, rollout_idx, scope,
                min_confidence,
            ):
                l3["history_capable_relevant_count"] += 1
            else:
                l3["history_unavailable_relevant_count"] += 1
                if any(
                    record.episode_id == episode_id
                    and record.rollout_idx == rollout_idx
                    and record.step_id >= step
                    for record in evidence
                ):
                    l3["current_only_relevant_count"] += 1

    details = getattr(result, "diagnostics", None)
    layers = details.get("layers", {}) if isinstance(details, dict) else {}
    joint_layer = layers.get("joint") or {}
    alignment = joint_layer.get("alignment") or {}
    joint = {
        "selected_joint_items": selected_joint,
        "l3_facts_considered": len(alignment.get("items") or []),
        "attempted_alignments": int(alignment.get("attempted") or 0),
        "confirmed_alignments": int(alignment.get("confirmed") or 0),
        "unmatched_alignments": int(alignment.get("unmatched") or 0),
        # Historical joint cards merge a confirmed pair into one line, while
        # the legacy ``aligned_pairs`` field only counts a separate l3_aligned
        # line.  Keep both so a zero legacy counter is not mistaken for a
        # failed alignment.
        "reported_aligned_pairs": int(joint_layer.get("aligned_pairs") or 0),
    }

    return {
        "l1": l1,
        "l3": l3,
        "joint": joint,
        "selected_items": {
            "l1": selected_l1,
            "l3": selected_l3,
            "joint": selected_joint,
            "all": len(items),
        },
    }

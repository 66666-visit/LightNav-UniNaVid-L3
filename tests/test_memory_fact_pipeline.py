"""End-to-end fact propagation from YOLO output to the prompt layers.

The tests use the real vendored backends and deterministic detector stand-ins,
but never lower a confidence threshold.  They distinguish three outcomes that
the GPU smoke conflated:

* a persisted detection that is instruction-relevant and above the prompt
  floor becomes an L1 fact and an L3 prompt fact;
* a persisted detection below the 0.5 prompt floor is *not* promoted, even
  though it is a valid MemoryEvent/EvidenceRecord;
* an instruction-irrelevant detector class never becomes a fact, even at high
  confidence, while the 3D map itself is still built.
"""

from __future__ import annotations

import numpy as np

from l1_integration.memory_backend import group_config
from l3_memory.lexical import _terms
from l3_memory.v3_schemas import canonical_evidence_key
from tests.helpers import (
    FakeEpisode,
    FakeExtractor,
    FakeL1Extractor,
    FakePlaceExtractor,
    chair_instance,
    make_memory_backend,
    synthetic_frame,
)

BOX = (1.0, 1.0, 6.0, 6.0)
RGB = np.zeros((8, 8, 3), np.uint8)


def _make_backend(
    group: str = "l1_l3_joint", *, instruction: str = "find the chair",
    l1_label: str = "chair", l1_confidence: float = 0.9,
    seg_label: str = "chair", seg_confidence: float = 0.9,
    seg_every_n: int = 1,
):
    l1 = FakeL1Extractor(
        by_step={step: [(l1_label, l1_confidence, BOX)] for step in range(8)}
    )
    seg = FakeExtractor(
        [chair_instance(label=seg_label, confidence=seg_confidence)],
        every_n_steps=seg_every_n,
    )
    backend = make_memory_backend(
        group, l1=l1, seg=seg, place=FakePlaceExtractor(),
    )
    backend.reset_episode(FakeEpisode(instruction=instruction), instruction)
    return backend, l1, seg


def _run(backend, instruction: str, steps: int, *, frame=None):
    frame = frame or synthetic_frame()
    rows = []
    for step in range(steps):
        events, observe = backend.observe({"rgb": RGB}, step)
        spatial = (
            backend.observe_spatial(
                frame, events, step, transition_record=bool(step),
            )
            if backend.plan.uses_l3 else {}
        )
        result = backend.build_context(
            instruction=instruction, step=step, rgb_width=8,
            frame=frame if backend.plan.uses_l3 else None,
        )
        rows.append((result, list(events), observe, spatial))
    return rows


def _persisted_detections(backend):
    return [
        record for record in backend.v3_store.evidence_for_scene("scene")
        if record.event_type in {"object_observation", "landmark"}
    ]


def test_relevant_detection_reaches_evidence_map_and_prompt() -> None:
    instruction = "find the chair"
    backend, _l1, _seg = _make_backend(instruction=instruction)
    rows = _run(backend, instruction, 2)
    result, _events, _observe, spatial = rows[1]

    assert spatial["frame_valid"] is True
    assert spatial["segmentation_count"] >= 1
    assert spatial["object_nodes"] >= 1
    assert backend.object_map is not None
    assert backend.object_map.objects

    detections = _persisted_detections(backend)
    assert detections
    assert any(
        record.semantic_label == "chair"
        and record.observation_status == "observed"
        and record.raw_detector_confidence == 0.9
        for record in detections
    )
    by_key = {record.key: record for record in detections}
    key = canonical_evidence_key("scene", "episode", 0, "event_0")
    assert key in by_key

    layers = result.diagnostics["layers"]
    assert layers["l1"]["item_count"] >= 1
    assert layers["l3"]["item_count"] >= 1
    assert layers["l3"]["verified_count"] >= 1
    assert layers["joint"]["alignment"]["confirmed"] >= 1
    assert layers["joint"]["joint_item_count"] >= 1
    pipeline = result.diagnostics["pipeline"]
    assert pipeline["joint"]["confirmed_alignments"] >= 1
    assert pipeline["joint"]["selected_joint_items"] >= 1
    assert "chair" in result.text


def test_high_confidence_irrelevant_label_is_persisted_but_not_retrieved() -> None:
    instruction = "find the sofa"
    backend, _l1, _seg = _make_backend(
        instruction=instruction, l1_label="chair", seg_label="chair",
    )
    rows = _run(backend, instruction, 2)
    result, _events, _observe, _spatial = rows[1]

    # The write path is intact: a real detection became evidence and the
    # optional DINOv2 place channel also wrote its observation.
    assert any(
        record.semantic_label == "chair" for record in _persisted_detections(backend)
    )
    assert any(
        record.event_type == "place_observation"
        for record in backend.v3_store.evidence_for_scene("scene")
    )
    # But the retrieval layer rejects it because the instruction has no overlap.
    assert "chair" not in _terms(instruction)
    assert result.diagnostics["layers"]["l1"]["item_count"] == 0
    pipeline = result.diagnostics["pipeline"]
    assert pipeline["l1"]["current_rejected_irrelevant_records"] >= 1
    l3 = result.diagnostics["layers"]["l3"]
    assert l3["candidate_count"] == 0
    assert pipeline["l3"]["map_object_count"] >= 1
    assert pipeline["l3"]["relevance_rejected_count"] >= 1
    assert result.layer_items == []
    assert "No instruction-related" in result.text
    assert "chair" not in result.text
    assert pipeline["joint"]["l3_facts_considered"] == 0


def test_detection_below_prompt_floor_is_evidence_but_not_an_l1_fact() -> None:
    instruction = "find the chair"
    backend, _l1, _seg = _make_backend(
        instruction=instruction, l1_confidence=0.4, seg_confidence=0.9,
    )
    rows = _run(backend, instruction, 2)
    result, _events, _observe, _spatial = rows[1]

    # 0.4 is above the detector threshold (0.25) but below the prompt floor.
    assert any(
        record.semantic_label == "chair" for record in _persisted_detections(backend)
    )
    l1 = result.diagnostics["layers"]["l1"]
    assert l1["item_count"] == 0
    pipeline = result.diagnostics["pipeline"]
    assert pipeline["l1"]["current_below_prompt_floor_records"] >= 1

    # L3 still has a valid verified fact, but no L1 counterpart to align with.
    l3 = result.diagnostics["layers"]["l3"]
    assert l3["item_count"] >= 1
    assert l3["verified_count"] >= 1
    alignment = result.diagnostics["layers"]["joint"]["alignment"]
    assert alignment["attempted"] == 0
    assert alignment["unmatched"] == 1
    assert alignment["confirmed"] == 0
    assert pipeline["joint"]["l3_facts_considered"] == 1
    assert pipeline["joint"]["selected_joint_items"] == 0


def test_l3_segmentation_step_is_not_served_as_history_until_later() -> None:
    instruction = "find the chair"
    backend, _l1, seg = _make_backend(instruction=instruction, seg_every_n=5)
    rows = _run(backend, instruction, 2)

    first, _events, _observe, first_spatial = rows[0]
    assert first_spatial["segmentation_count"] >= 1
    assert backend.object_map is not None and backend.object_map.objects
    assert first.diagnostics["layers"]["l3"]["current_only_skipped"] >= 1
    assert first.diagnostics["layers"]["l3"]["item_count"] == 0

    second, _events, _observe, second_spatial = rows[1]
    assert second_spatial["segmentation_count"] == 0
    assert seg.calls == [0, 1]
    assert second.diagnostics["layers"]["l3"]["item_count"] >= 1
    assert "map lines are history" in second.text


def test_segmentation_without_valid_depth_creates_no_object_or_fact() -> None:
    instruction = "find the chair"
    backend, _l1, _seg = _make_backend(instruction=instruction)
    bad_frame = synthetic_frame(depth=0.0)
    rows = _run(backend, instruction, 2, frame=bad_frame)
    result, _events, _observe, spatial = rows[1]

    assert spatial["frame_valid"] is True
    assert spatial["segmentation_count"] >= 1
    assert backend.object_map is not None
    assert backend.object_map.objects == {}
    assert result.diagnostics["pipeline"]["l3"]["map_object_count"] == 0
    assert result.diagnostics["layers"]["l3"]["item_count"] == 0


def test_joint_does_not_merge_unlinked_different_labels() -> None:
    instruction = "find the chair or sofa"
    backend, _l1, _seg = _make_backend(
        group="l1_l3_joint", instruction=instruction,
        l1_label="chair", seg_label="sofa",
    )
    rows = _run(backend, instruction, 2)
    result, _events, _observe, _spatial = rows[1]

    layers = result.diagnostics["layers"]
    assert layers["l1"]["item_count"] == 1
    assert layers["l3"]["item_count"] == 1
    alignment = layers["joint"]["alignment"]
    assert alignment["unmatched"] == 1
    assert alignment["attempted"] == 0
    assert alignment["confirmed"] == 0
    assert layers["joint"]["joint_item_count"] == 0
    assert result.diagnostics["pipeline"]["joint"]["selected_joint_items"] == 0
    assert "chair" in result.text and "sofa" in result.text
    assert "chair: last observed (L1" not in result.text


def test_group_plan_remains_the_official_eight_group_plan() -> None:
    assert group_config("l1_l3_joint")["l1_builder"] == "l1_prompt_organizer"
    assert group_config("l1_l3_joint")["l3_builder"] == "l3_prompt_organizer"
    assert group_config("l1_l3_joint")["combination"] == "joint"

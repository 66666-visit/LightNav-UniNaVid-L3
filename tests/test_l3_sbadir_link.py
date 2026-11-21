"""L1 evidence linkage preserves the original L3 map sentences."""

from __future__ import annotations

import numpy as np

from l1_integration.sbadir_backend import L1EvidenceBackend
from l1_memory.memory_writer import MockSceneExtractor
from l3_integration.official_prompt import REFERENCE_OFFICIAL_PROMPT_TEMPLATE
from l3_integration.sbadir_fusion import compose_sbadir
from l3_memory.schemas import DetectedObject, SceneExtraction
from tests.helpers import FakeEpisode, make_sidecar


def test_verified_bbox_link_adds_provenance_without_changing_l3_core():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    baseline = make_sidecar()
    linked = make_sidecar()
    episode = FakeEpisode()
    for sidecar in (baseline, linked):
        sidecar.reset_episode(episode)

    detector = MockSceneExtractor({
        0: SceneExtraction(
            detections=[DetectedObject("chair", 0.91, (2, 2, 6, 6))],
            source="mock_scene_extractor", confidence=0.91,
        ),
    })
    l1 = L1EvidenceBackend(detector)
    l1.reset_episode("scene", "episode", "find the chair")

    baseline.observe({"rgb": rgb}, None, 0)
    events, _ = l1.observe(rgb, 0)
    linked_diag = linked.observe(
        {"rgb": rgb}, None, 0,
        l1_events=events, l1_store=l1.v3_store,
    )
    assert linked_diag["evidence_linked"] == 1
    assert linked_diag["evidence_ambiguous"] == 0

    baseline.observe({"rgb": rgb}, None, 1)
    l1.observe(rgb, 1)
    linked.observe(
        {"rgb": rgb}, None, 1,
        l1_events=l1.last_events, l1_store=l1.v3_store,
    )
    a, _ = baseline.build_context("find the chair", 1)
    b, _ = linked.build_context("find the chair", 1)
    assert a.context.text == b.context.text
    assert a.selected_facts[0].evidence_ids == b.selected_facts[0].evidence_ids
    assert b.selected_facts[0].linked_l1_evidence_ids
    assert not a.selected_facts[0].linked_l1_evidence_ids
    assert baseline.map_snapshot() == linked.map_snapshot()


def test_unique_cross_label_rgb_box_and_l3_mask_exposes_unresolved_conflict():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    instruction = "find the chair and table"
    episode = FakeEpisode(instruction=instruction)
    baseline = make_sidecar()
    linked = make_sidecar()
    for sidecar in (baseline, linked):
        sidecar.reset_episode(episode)
    detector = MockSceneExtractor({
        step: SceneExtraction(
            detections=[DetectedObject("table", 0.91, (2, 2, 6, 6))],
            source="mock_scene_extractor", confidence=0.91,
        ) for step in (0, 1)
    })
    l1 = L1EvidenceBackend(detector)
    l1.reset_episode("scene", "episode", instruction)
    for step in (0, 1):
        baseline.observe({"rgb": rgb}, None, step)
        events, _ = l1.observe(rgb, step)
        diagnostic = linked.observe(
            {"rgb": rgb}, None, step,
            l1_events=events, l1_store=l1.v3_store,
        )
        assert diagnostic["cross_label_entity_links"] == 1
        assert diagnostic["evidence_linked"] == 0
    a, _ = baseline.build_context(instruction, 1)
    b, _ = linked.build_context(instruction, 1)
    assert a.context.text == b.context.text
    selection = l1.select(step=1, rgb_width=8)
    result = compose_sbadir(
        template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
        instruction=instruction,
        l3_text=b.context.text,
        l3_facts=b.selected_facts,
        l1_facts=selection.selected_facts,
        entity_links=linked.cross_sensor_links,
        count_tokens=lambda text: len(text.split()),
        max_prompt_tokens=10000, l1_max_tokens=160,
    )
    assert result.diagnostics["conflicting_facts"] == 1
    assert result.diagnostics["conflicts"][0]["resolution"] == "both_retained"
    assert "chair" in result.memory_text
    assert "table" in result.memory_text
    assert result.diagnostics["l3_core_unchanged"] is True


def test_cross_label_match_rejects_two_overlapping_detector_boxes():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    detector = MockSceneExtractor({
        0: SceneExtraction(
            detections=[
                DetectedObject("chair", 0.92, (2, 2, 6, 6)),
                DetectedObject("table", 0.91, (2, 2, 6, 6)),
            ],
            source="mock_scene_extractor", confidence=0.92,
        ),
    })
    l1 = L1EvidenceBackend(detector)
    l1.reset_episode("scene", "episode", "find the chair and table")
    events, _ = l1.observe(rgb, 0)
    sidecar = make_sidecar()
    sidecar.reset_episode(FakeEpisode(instruction="find the chair and table"))
    diagnostic = sidecar.observe(
        {"rgb": rgb}, None, 0, l1_events=events, l1_store=l1.v3_store,
    )
    assert diagnostic["frame_valid"] is True
    assert diagnostic["cross_label_entity_links"] == 0
    assert sidecar.cross_sensor_links == {}


def test_cross_label_matcher_error_does_not_erase_valid_l3_map():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    detector = MockSceneExtractor({
        0: SceneExtraction(
            detections=[DetectedObject("table", 0.91, (2, 2, 6, 6))],
            source="mock_scene_extractor", confidence=0.91,
        ),
    })
    l1 = L1EvidenceBackend(detector)
    l1.reset_episode("scene", "episode", "find the chair and table")
    events, _ = l1.observe(rgb, 0)
    sidecar = make_sidecar()
    sidecar.reset_episode(FakeEpisode(instruction="find the chair and table"))

    def fail(*args):
        raise RuntimeError("matcher failure")

    sidecar._link_cross_label_boxes = fail
    diagnostic = sidecar.observe(
        {"rgb": rgb}, None, 0, l1_events=events, l1_store=l1.v3_store,
    )
    assert diagnostic["frame_valid"] is True
    assert diagnostic["object_nodes"] == 1
    assert "matcher failure" in diagnostic["cross_label_entity_link_error"]
    assert sidecar.last_frame is not None

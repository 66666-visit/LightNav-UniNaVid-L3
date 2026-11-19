"""Requirements 3, 4, 5, 6, 7: real L3 map lifecycle and failure reporting."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from l3_integration.habitat_geometry import CameraSpec, frame_from_habitat
from tests.helpers import (
    FakeEpisode,
    FakeExtractor,
    chair_instance,
    fake_agent_state,
    make_sidecar,
)


def test_map_is_created_and_updated_from_rgbd() -> None:
    sidecar = make_sidecar()
    sidecar.reset_episode(FakeEpisode())
    first = sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 0)
    assert first["frame_valid"] is True
    assert first["segmentation_count"] == 1
    assert first["evidence_minted"] == 1
    assert first["object_nodes"] == 1
    assert first["viewpoints"] == 1

    second = sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 1)
    assert second["frame_valid"] is True
    assert second["object_nodes"] == 1
    assert second["viewpoints"] == 2
    assert second["observed_from_edges"] == 2
    node = next(iter(sidecar.object_map.objects.values()))
    assert node.observation_count == 2
    assert len(node.evidence_ids) == 2
    assert all(
        record.linked_l1_evidence_id is None
        for record in sidecar.object_map.evidence.values()
    )


def test_context_excludes_the_current_step_and_uses_only_history() -> None:
    sidecar = make_sidecar()
    sidecar.reset_episode(FakeEpisode())
    sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 0)
    organization, diagnostic = sidecar.build_context("find the chair", 0)
    assert organization.context is None
    assert diagnostic["current_only_skipped"] == 1
    assert diagnostic["fact_count"] == 0

    sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 1)
    organization, diagnostic = sidecar.build_context("find the chair", 1)
    assert organization.context is not None
    assert diagnostic["context_injected"] is True
    assert diagnostic["fact_count"] == 1
    assert "Map record (step 0, 1 view)" in organization.context.text
    assert "step 1" not in organization.context.text.splitlines()[0]
    assert "the chair is in front of you" in organization.context.text


def test_episode_reset_clears_every_map_structure() -> None:
    sidecar = make_sidecar()
    sidecar.reset_episode(FakeEpisode(episode_id="episode-a"))
    sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 0)
    sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 1)
    assert sidecar.map_snapshot()["objects"] == 1
    old_object_ids = set(sidecar.object_map.objects)
    old_store = sidecar.store
    old_map = sidecar.object_map

    sidecar.reset_episode(FakeEpisode(episode_id="episode-b"))
    assert sidecar.map_snapshot() == {
        "objects": 0, "viewpoints": 0, "evidence": 0, "edges": 0, "labels": {},
    }
    assert sidecar.object_map is not old_map
    assert sidecar.store is not old_store
    assert old_object_ids.isdisjoint(sidecar.object_map.objects)
    assert sidecar.episode_id == "episode-b"
    assert sidecar.store.active_episode == ("scene", "episode-b", 0)

    organization, diagnostic = sidecar.build_context("find the chair", 1)
    assert organization is None
    assert diagnostic["context_reason"] == "current RGB-D or pose unavailable"


def test_same_observation_sequence_serializes_identically() -> None:
    texts = []
    snapshots = []
    for _ in range(2):
        sidecar = make_sidecar()
        sidecar.reset_episode(FakeEpisode())
        for step in (0, 1):
            sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, step)
        organization, _ = sidecar.build_context("find the chair", 1)
        texts.append(organization.context.text)
        snapshots.append(sidecar.map_snapshot())
    assert texts[0] == texts[1]
    assert snapshots[0] == snapshots[1]


def test_missing_depth_or_pose_is_recorded_and_never_injected() -> None:
    sidecar = make_sidecar()
    sidecar.reset_episode(FakeEpisode())

    def fail_frame(observations, state, spec):
        raise ValueError("depth channel missing")

    sidecar.frame_builder = fail_frame
    diagnostic = sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 0)
    assert diagnostic["frame_valid"] is False
    assert "depth channel missing" in diagnostic["frame_error"]
    organization, context = sidecar.build_context("find the chair", 0)
    assert organization is None
    assert context["context_reason"] == "current RGB-D or pose unavailable"


def test_missing_pose_is_reported_by_the_real_habitat_adapter() -> None:
    from l3_integration.habitat_geometry import (
        CameraSpec,
        build_spatial_rgbd_metadata,
    )

    spec = CameraSpec(8, 8, 120.0, 0.0, 10.0, True)
    with pytest.raises(ValueError, match="agent state is unavailable"):
        build_spatial_rgbd_metadata(
            {"rgb": np.zeros((8, 8, 3), np.uint8),
             "depth": np.full((8, 8), 0.5, np.float32)},
            None, spec,
        )


def test_scope_must_be_episode_local() -> None:
    with pytest.raises(ValueError, match="episode_local"):
        make_sidecar(scope="scene_persistent")


def test_non_increasing_step_is_rejected_without_mutating_the_map() -> None:
    sidecar = make_sidecar()
    sidecar.reset_episode(FakeEpisode())
    sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 2)
    before = copy.deepcopy(sidecar.map_snapshot())
    diagnostic = sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 2)
    assert diagnostic["frame_valid"] is False
    assert "non-increasing" in diagnostic["frame_error"]
    assert sidecar.map_snapshot() == before


def test_segmentation_cadence_still_records_viewpoints() -> None:
    sidecar = make_sidecar(extractor=FakeExtractor(every_n_steps=2))
    sidecar.reset_episode(FakeEpisode())
    counts = []
    for step in range(3):
        diagnostic = sidecar.observe(
            {"rgb": np.zeros((8, 8, 3), np.uint8)}, None, step,
        )
        counts.append(diagnostic["segmentation_count"])
    assert counts == [1, 0, 1]
    assert sidecar.map_snapshot()["viewpoints"] == 3
    assert sidecar.map_snapshot()["objects"] == 1


def test_low_confidence_segmentation_is_not_injected() -> None:
    sidecar = make_sidecar(
        extractor=FakeExtractor([chair_instance(confidence=0.3)]),
    )
    sidecar.reset_episode(FakeEpisode())
    sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 0)
    diagnostic = sidecar.observe(
        {"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 1,
    )
    assert diagnostic["evidence_minted"] == 1
    organization, context = sidecar.build_context("find the chair", 1)
    assert organization.context is None
    assert context["context_reason"] == "no_expressible_l3_fact"


def test_map_does_not_cross_scene_identity() -> None:
    sidecar = make_sidecar()
    sidecar.reset_episode(FakeEpisode(scene_id="scene-a", episode_id="episode"))
    sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 0)
    sidecar.reset_episode(FakeEpisode(scene_id="scene-b", episode_id="episode"))
    assert sidecar.scene_id == "scene-b"
    assert sidecar.map_snapshot()["objects"] == 0
    assert sidecar.store.active_episode == ("scene-b", "episode", 0)


def test_default_habitat_frame_builder_drives_the_real_map() -> None:
    sidecar = make_sidecar()
    sidecar.frame_builder = frame_from_habitat
    sidecar.camera_spec = CameraSpec(8, 8, 120.0, 0.0, 10.0, True)
    sidecar.reset_episode(FakeEpisode())
    observations = {
        "rgb": np.zeros((8, 8, 3), np.uint8),
        "depth": np.full((8, 8), 0.5, np.float32),
    }
    diagnostic = sidecar.observe(
        observations, fake_agent_state(), 0, transition_record=False,
    )
    assert diagnostic["frame_valid"] is True
    assert diagnostic["object_nodes"] == 1
    assert diagnostic["viewpoints"] == 1
    assert sidecar.last_frame is not None
    assert np.allclose(sidecar.last_frame.depth_m[2:6, 2:6], 5.0)


def test_l3_context_respects_the_600_character_budget() -> None:
    from tests.helpers import make_sidecar

    sidecar = make_sidecar(max_items=2, max_chars=600)
    sidecar.reset_episode(FakeEpisode())
    for step in (0, 1, 2):
        sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, step)
    organization, diagnostic = sidecar.build_context("find the chair", 2)
    assert organization.context is not None
    text = organization.context.text
    assert len(text) <= 600
    assert text.splitlines()[0].startswith("[Navigation Map Memory - L3;")
    assert diagnostic["context_chars"] == len(text)
    assert diagnostic["fact_count"] <= 2


def test_injected_facts_are_l3_history_with_no_l1_provenance() -> None:
    sidecar = make_sidecar()
    sidecar.reset_episode(FakeEpisode())
    for step in (0, 1):
        sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, step)
    organization, _ = sidecar.build_context("find the chair", 1)
    assert organization.context is not None
    assert organization.context.layer_items
    for item in organization.context.layer_items:
        assert item.layer == "long"
        assert item.kind == "l3_official_map_fact"
        assert item.metadata["layer_source"] == "l3"
        assert item.metadata["role"] == "spatial_history"
        assert item.metadata["linked_l1_evidence_ids"] == []
        assert item.metadata["history_kind"] == "map_record_not_current_view"
        assert item.metadata["range_kind"] == "straight_line_not_path"
        assert item.metadata["traversability_status"] == "unknown"
        assert all(ref.event_id.startswith("l3_seg_") or "l3_seg" in ref.event_id
                   for ref in item.evidence_refs)
        assert item.source == "l3_official_prompt"
        assert all(ref.source.startswith("yolo11n-seg")
                   for ref in item.evidence_refs)

"""Identity-based execution tracking is independent of Habitat's order."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from l3_integration.episode_execution import (
    EpisodeExecutionError,
    EpisodeExecutionTracker,
    EpisodeMetadata,
    episode_identity,
    metadata_from_pilot_manifest,
)


def _episode(scene: str, episode_id: str) -> SimpleNamespace:
    return SimpleNamespace(scene_id=scene, episode_id=episode_id)


def _metadata(index: int, distance_group: str) -> EpisodeMetadata:
    return EpisodeMetadata(
        manifest_order_index=index,
        pilot_manifest_index=index,
        distance_group=distance_group,
    )


def test_episode_identity_normalizes_scene_paths_to_basenames() -> None:
    first = episode_identity(
        _episode("/autodl/scene_datasets/hm3d/SceneA.glb", "episode-1")
    )
    second = episode_identity(
        _episode("data/scene_datasets/hm3d/SceneA.glb", "episode-1")
    )
    assert first == second == ("SceneA.glb", "episode-1")


def test_metadata_is_assigned_by_identity_not_list_position() -> None:
    pilot = SimpleNamespace(rows=[
        SimpleNamespace(
            scene_id="data/scene_datasets/hm3d/SceneB.glb",
            episode_id="episode-b",
            distance_group="long",
            instruction="go to b",
        ),
        SimpleNamespace(
            scene_id="data/scene_datasets/hm3d/SceneA.glb",
            episode_id="episode-a",
            distance_group="short",
            instruction="go to a",
        ),
    ])
    metadata = metadata_from_pilot_manifest(pilot)
    assert metadata[("SceneB.glb", "episode-b")].distance_group == "long"
    assert metadata[("SceneB.glb", "episode-b")].manifest_order_index == 0
    assert metadata[("SceneA.glb", "episode-a")].distance_group == "short"
    assert metadata[("SceneA.glb", "episode-a")].manifest_order_index == 1
    assert metadata[("SceneA.glb", "episode-a")].instruction == "go to a"


def test_tracker_accepts_any_order_and_rejects_duplicates_and_outside() -> None:
    expected = [
        _episode("/autodl/scene_datasets/hm3d/SceneA.glb", "episode-a"),
        _episode("/autodl/scene_datasets/hm3d/SceneB.glb", "episode-b"),
        _episode("/autodl/scene_datasets/hm3d/SceneC.glb", "episode-c"),
    ]
    metadata = {
        ("SceneA.glb", "episode-a"): _metadata(0, "short"),
        ("SceneB.glb", "episode-b"): _metadata(1, "medium"),
        ("SceneC.glb", "episode-c"): _metadata(2, "long"),
    }
    tracker = EpisodeExecutionTracker(expected, metadata)

    key, meta = tracker.accept(expected[2])
    assert key == ("SceneC.glb", "episode-c")
    assert meta.distance_group == "long"
    tracker.accept(expected[0])
    with pytest.raises(EpisodeExecutionError, match="duplicate Episode"):
        tracker.accept(expected[2])
    with pytest.raises(EpisodeExecutionError, match="outside"):
        tracker.accept(_episode("SceneExtra.glb", "episode-extra"))
    with pytest.raises(EpisodeExecutionError, match="missing"):
        tracker.assert_complete()

    tracker.accept(expected[1])
    tracker.assert_complete()
    assert tracker.coverage()["habitat_order"][0] == {
        "habitat_execution_index": 0,
        "scene_basename": "SceneC.glb",
        "episode_id": "episode-c",
        "manifest_order_index": 2,
        "pilot_manifest_index": 2,
        "distance_group": "long",
    }


def test_tracker_requires_metadata_for_every_expected_episode() -> None:
    expected = [
        _episode("SceneA.glb", "episode-a"),
        _episode("SceneB.glb", "episode-b"),
    ]
    with pytest.raises(EpisodeExecutionError, match="no frozen-manifest"):
        EpisodeExecutionTracker(
            expected,
            {("SceneA.glb", "episode-a"): _metadata(0, "short")},
        )


def test_tracker_rejects_a_duplicated_expected_set() -> None:
    expected = [
        _episode("SceneA.glb", "episode-a"),
        _episode("SceneA.glb", "episode-a"),
    ]
    with pytest.raises(EpisodeExecutionError, match="duplicate Episode"):
        EpisodeExecutionTracker(expected)

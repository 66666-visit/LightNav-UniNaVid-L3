"""Requirement 3: the real LightNav L3 map/serializer is reused, not replaced."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from l3_memory.conceptgraphs_lite import (
    CameraIntrinsics,
    ObjectMap,
    ObjectNode,
    ObservedFromEdge,
    Pose3D,
    RGBDFrame,
    SpatialEvidence,
    ViewpointNode,
)
from l3_memory.l3_official_prompt import (
    HEADER,
    L3OfficialPromptOrganizer,
    direction_phrase,
    support_relation,
)
from l3_memory.l3_prompt_organizer import L3PromptOrganizer
from l3_memory.v3_schemas import canonical_evidence_key
from l3_memory.v3_store import V3MemoryStore

IDENTITY = Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
PACKAGE = Path(__file__).resolve().parents[1] / "l3_memory"


def _pose(*, x: float = 0.0, z: float = 0.0, yaw_deg: float = 0.0) -> Pose3D:
    half = math.radians(yaw_deg) / 2.0
    return Pose3D((x, 0.0, z), (math.cos(half), 0.0, math.sin(half), 0.0))


def _frame(*, pose: Pose3D | None = None) -> RGBDFrame:
    return RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.full((8, 8), 2.0, dtype=np.float32),
        intrinsics=CameraIntrinsics(8.0, 8.0, 3.5, 3.5, 8, 8),
        agent_pose=pose or IDENTITY,
        camera_extrinsics=IDENTITY,
    )


def _build_map(
    objects: list[dict[str, object]], *,
    scene: str = "scene", episode: str = "episode", rollout: int = 0,
) -> tuple[V3MemoryStore, ObjectMap]:
    store = V3MemoryStore()
    store.begin_episode(scene, episode, "find the chair")
    object_map = ObjectMap()
    object_map.reset_episode(scene, episode, rollout)
    for index, spec in enumerate(objects):
        step = int(spec.get("step", 7))
        label = str(spec["label"])
        center = tuple(float(value) for value in spec["center"])
        bbox = tuple(
            tuple(float(value) for value in point) for point in spec["bbox"]
        )
        confidence = float(spec.get("confidence", 0.9))
        viewpoint_id = json.dumps(
            [scene, episode, rollout, step], separators=(",", ":"),
        )
        object_map.viewpoints.setdefault(viewpoint_id, ViewpointNode(
            viewpoint_id, scene, episode, rollout, step, IDENTITY, IDENTITY,
        ))
        evidence_id = canonical_evidence_key(
            scene, episode, rollout, f"l3_seg_{step}_{index}",
        )
        object_map.evidence[evidence_id] = SpatialEvidence(
            evidence_id, scene, episode, rollout, step, label, confidence,
            "yolo11n-seg", 100, 100, "observed", None,
        )
        object_id = f"object_{index}"
        object_map.objects[object_id] = ObjectNode(
            object_id, label, center, bbox, 2, [evidence_id], "observed",
            np.empty((0, 3)),
        )
        object_map.edges.append(ObservedFromEdge(
            object_id, viewpoint_id, evidence_id,
        ))
    return store, object_map


def _official(
    store: V3MemoryStore, object_map: ObjectMap, *,
    instruction: str = "find the chair", step: int = 10,
    frame: object = None, max_chars: int = 2000, max_items: int = 2,
    min_confidence: float = 0.5, scope: str = "episode_local",
):
    return L3OfficialPromptOrganizer(max_items=max_items).build(
        store, object_map, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=step, instruction=instruction,
        frame=_frame() if frame is None else frame,
        scope=scope, max_chars=max_chars, min_confidence=min_confidence,
    )


_CHAIR_LEFT = {
    "label": "chair", "center": (-1.0, 0.5, -2.0),
    "bbox": ((-1.3, 0.0, -2.3), (-0.7, 1.0, -1.7)),
}
_CHAIR_RIGHT = {
    "label": "chair", "center": (3.5, 0.5, -3.0),
    "bbox": ((3.2, 0.0, -3.3), (3.8, 1.0, -2.7)),
}
_TABLE = {
    "label": "dining table", "center": (0.5, 0.4, -2.5),
    "bbox": ((-0.2, 0.35, -3.2), (1.2, 0.75, -1.8)),
}


# ---------------------------------------------------------------------------
# Vendoring proof
# ---------------------------------------------------------------------------


def test_vendored_files_match_recorded_hashes_and_differ_only_in_imports() -> None:
    provenance = json.loads((PACKAGE / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_commit"] == "6aefffa"
    assert len(provenance["files"]) == 9
    for entry in provenance["files"]:
        vendored = PACKAGE / entry["name"]
        assert hashlib.sha256(vendored.read_bytes()).hexdigest() == (
            entry["vendored_sha256"]
        ), entry["name"]
        source = Path(entry["source"])
        if not source.is_file():
            continue
        assert hashlib.sha256(source.read_bytes()).hexdigest() == (
            entry["source_sha256"]
        ), entry["name"]
        source_text = source.read_text(encoding="utf-8")
        vendored_text = vendored.read_text(encoding="utf-8")
        normalized = vendored_text.replace(
            "from l3_memory.lexical import _plain_label",
            "from lightnav.memory.local_graph_fusion import _plain_label",
        ).replace(
            "from l3_memory.lexical import _terms",
            "from lightnav.memory.v3_retrieval import _terms",
        ).replace(
            "from l3_memory.compat import removeprefix\n", "",
        ).replace(
            'removeprefix(event.description, "Observed ")',
            'event.description.removeprefix("Observed ")',
        ).replace("from l3_memory.", "from lightnav.memory.")
        assert normalized == source_text, entry["name"]


def test_lexical_helpers_match_upstream_fixtures() -> None:
    from l3_memory.lexical import _plain_label, _terms

    assert _plain_label("  dining-table!! x ") == "dining-table x"
    assert _plain_label("chair\nwith\tspaces") == "chair with spaces"
    assert _terms("Go to the sofa, TV!") == {"couch", "television"}
    assert _terms(None) == set()
    assert _terms(["red chair", "table"]) == {"red", "chair", "table"}


# ---------------------------------------------------------------------------
# Geometry / expression contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("bearing", "expected"), [
    (0.0, "in front of you"),
    (22.5, "in front of you"),
    (22.6, "to your front-left"),
    (60.0, "to your front-left"),
    (90.0, "to your left"),
    (120.0, "behind you on the left"),
    (160.0, "behind you"),
    (-30.0, "to your front-right"),
    (-90.0, "to your right"),
    (-120.0, "behind you on the right"),
    (-180.0, "behind you"),
])
def test_direction_sectors_are_robot_relative(bearing: float, expected: str) -> None:
    assert direction_phrase(bearing) == expected


def test_support_relation_rejects_unknown_geometry() -> None:
    table = ((-0.5, 0.7, -2.5), (0.5, 0.75, -1.5))
    on_table = ((-0.1, 0.75, -2.1), (0.1, 0.95, -1.9))
    above_table = ((-0.1, 1.5, -2.1), (0.1, 1.7, -1.9))
    beside_table = ((0.7, 0.0, -2.0), (0.9, 0.3, -1.8))
    far_away = ((3.0, 0.0, -2.0), (3.2, 0.3, -1.8))
    bigger_than_table = ((-0.6, 0.7, -2.6), (0.6, 1.4, -1.4))
    assert support_relation(on_table, table) == "on"
    assert support_relation(beside_table, table) == "next to"
    assert support_relation(above_table, table) is None
    assert support_relation(far_away, table) is None
    assert support_relation(bigger_than_table, table) is None
    assert support_relation(table, on_table) is None


# ---------------------------------------------------------------------------
# Shared candidate set and exact official-style output
# ---------------------------------------------------------------------------


def test_official_uses_the_same_candidate_facts_as_the_organized_mode() -> None:
    store, object_map = _build_map([_CHAIR_LEFT, _CHAIR_RIGHT, _TABLE])
    organized = L3PromptOrganizer(max_items=2).build(
        store, object_map, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=10, instruction="find the chair", frame=_frame(),
        scope="episode_local", max_chars=2000, min_confidence=0.5,
    )
    official = _official(store, object_map)
    assert [fact.to_dict() for fact in official.facts] == [
        fact.to_dict() for fact in organized.facts
    ]
    assert [fact.object_id for fact in official.selected_facts] == [
        fact.object_id for fact in organized.selected_facts
    ]
    assert official.candidate_count == organized.candidate_count == 2
    assert official.verified_count == organized.verified_count == 2


def test_official_style_output_is_the_documented_lightnav_text() -> None:
    store, object_map = _build_map([_CHAIR_LEFT, _CHAIR_RIGHT, _TABLE])
    result = _official(store, object_map)
    text = result.context.text
    lines = text.splitlines()
    assert lines[0] == HEADER
    assert lines[1] == (
        "Map record (step 7, 1 view): the chair next to the dining table is "
        "to your front-left, 2.3 m straight-line."
    )
    assert lines[2] == (
        "Map record (step 7, 1 view): the rightmost chair of the 2 recorded "
        "in front is to your front-right, 4.6 m straight-line."
    )
    assert result.expression["strategy_counts"] == {
        "direction": 0, "relational_anchor": 1, "extreme": 1,
        "ordinal": 0, "direction_without_pose": 0,
    }
    assert result.expression["anchors_used"][0]["label"] == "dining table"
    assert "go to" not in text.casefold()
    assert "walk to" not in text.casefold()
    assert "reachable" not in text.casefold()
    assert "walkable" not in text.casefold()


def test_unverified_geometry_is_dropped_not_dressed_up() -> None:
    low = {**_CHAIR_LEFT, "confidence": 0.3}
    store, object_map = _build_map([low])
    result = _official(store, object_map)
    assert result.context is None
    assert result.empty_reason == "no_expressible_l3_fact"
    assert result.selected_facts == []
    assert result.expression["unverified_omitted"] == 1


def test_budget_truncation_keeps_whole_header_and_lines() -> None:
    store, object_map = _build_map([_CHAIR_LEFT, _CHAIR_RIGHT])
    no_room = _official(store, object_map, max_chars=len(HEADER) - 1)
    assert no_room.context is None
    assert no_room.empty_reason == "budget_too_small_for_header"
    one_line = _official(store, object_map, max_chars=len(HEADER) + 130)
    assert one_line.context is not None
    assert one_line.context.text.splitlines()[0] == HEADER
    assert len(one_line.context.layer_items) == 1
    assert one_line.prompt_chars <= len(HEADER) + 130


def test_overlong_context_is_truncated_at_whole_line_boundaries() -> None:
    store, object_map = _build_map([_CHAIR_LEFT, _CHAIR_RIGHT])
    result = _official(store, object_map, max_chars=600)
    assert result.context is not None
    text = result.context.text
    assert len(text) <= 600
    assert text.splitlines()[0] == HEADER
    assert all(line for line in text.splitlines())
    assert result.empty_reason is None

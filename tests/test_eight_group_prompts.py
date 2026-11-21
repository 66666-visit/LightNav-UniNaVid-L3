"""The eight prompt-ablation groups: real paths, facts, isolation and configs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from l1_integration.experiment_config import load_group_config
from l1_integration.memory_backend import (
    EIGHT_ABLATION_GROUPS,
    group_config,
)
from l3_integration.official_prompt import (
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
    build_prompt,
    build_task_text,
    verify_task_text_injection,
)
from tests.helpers import (
    FakeEpisode,
    FakeL1Extractor,
    FakePlaceExtractor,
    make_memory_backend,
    synthetic_frame,
)

FRAME = synthetic_frame()
ROOT = Path(__file__).resolve().parents[1]
GROUP_DIR = ROOT / "configs" / "prompt_ablation"


class _Seg:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        self.calls.clear()

    def close_episode(self) -> None:
        pass

    def extract(self, rgb, step):
        from tests.helpers import chair_instance

        self.calls.append(step)
        return [chair_instance()]


def _episode(instruction: str = "find the chair") -> FakeEpisode:
    return FakeEpisode(instruction=instruction)


def _run_group(group: str, *, steps: int = 2):
    seg = _Seg()
    l1 = FakeL1Extractor()
    place = FakePlaceExtractor()
    backend = make_memory_backend(group, l1=l1, seg=seg, place=place)
    instruction = "find the chair"
    backend.reset_episode(_episode(instruction), instruction)
    contexts = []
    for step in range(steps):
        events, _ = backend.observe(
            {"rgb": np.zeros((8, 8, 3), np.uint8)}, step,
        )
        if backend.plan.uses_l3:
            backend.observe_spatial(FRAME, events, step, transition_record=bool(step))
        result = backend.build_context(
            instruction=instruction, step=step, rgb_width=8,
            frame=FRAME if backend.plan.uses_l3 else None,
        )
        contexts.append(result)
    return backend, seg, l1, place, contexts


def test_every_group_maps_to_its_lightnav_plan() -> None:
    assert group_config("l1_original_only")["l1_builder"] == "local_graph_fusion"
    assert group_config("l1_organized_only")["l1_builder"] == "l1_prompt_organizer"
    assert group_config("l3_original_only")["l3_builder"] == "spatial_graph_adapter"
    assert group_config("l3_organized_only")["l3_builder"] == "l3_prompt_organizer"
    assert group_config("l1_l3_original_concat")["combination"] == "concat"
    assert group_config("l1_l3_joint")["combination"] == "joint"
    assert group_config("l1_l3_joint")["l1_builder"] == "l1_prompt_organizer"
    lossless = group_config("l1_original_l3_organized_joint")
    assert lossless["l1_builder"] == "local_graph_fusion"
    assert lossless["l3_builder"] == "l3_prompt_organizer"
    assert lossless["preserves_l1_verbatim"] is True


def test_all_eight_group_configs_are_independent_and_valid() -> None:
    seen = set()
    for group in EIGHT_ABLATION_GROUPS:
        path = GROUP_DIR / f"{group}.yaml"
        assert path.is_file(), path
        config = load_group_config(str(path))
        assert config["group"] == group
        assert config["prompt_mode"] == group
        seen.add((
            config["expected"]["l1_builder"],
            config["expected"]["l3_builder"],
            config["expected"]["combination"],
        ))
    assert len(seen) == 8  # every group has a distinct builder/combination plan


def test_group_config_rejects_a_mismatched_declaration() -> None:
    import json

    data = json.loads((GROUP_DIR / "l1_original_only.json").read_text())
    data["layers"]["l1_builder"] = "l1_prompt_organizer"
    path = GROUP_DIR.parent / "bad_group.json"
    try:
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="l1_builder"):
            load_group_config(str(path))
    finally:
        if path.exists():
            path.unlink()


@pytest.mark.parametrize("group", EIGHT_ABLATION_GROUPS)
def test_each_group_builds_its_own_source_and_prompt(group: str) -> None:
    backend, seg, l1, place, contexts = _run_group(group)
    spec = group_config(group)
    step1 = contexts[1]
    assert step1 is not None
    if spec["requires_l1_model"]:
        # L1 is written every step and one place descriptor is observed.
        assert l1.calls == [0, 1]
        assert place.calls == [0, 1]
    else:
        assert l1.calls == []
        assert backend.writer is None
    if spec["requires_seg_model"]:
        assert seg.calls == [0, 1]
        assert backend.object_map is not None
    else:
        assert seg.calls == []
        assert backend.object_map is None
    # The step-1 card is non-empty for all eight groups on this fixture.
    assert step1.context is not None
    text = step1.text
    if spec["l1_prompt"] and not spec["l3_prompt"]:
        assert "[Navigation Context - L1]" in text or "[Navigation Memory]" in text
        assert "Spatial history" not in text
        assert "map facts are history" not in text
    if spec["l3_prompt"] and not spec["l1_prompt"]:
        assert "Spatial history" in text or "map facts are history" in text
        assert "Current RGB: instruction landmark" not in text
        assert "Earlier:" not in text
    if spec["combination"] == "concat":
        assert "[Navigation Memory]" in text
        assert "Spatial history" in text or "map facts are history" in text
    if spec["combination"] == "joint":
        assert "[Navigation Context - L1+L3]" in text


def test_original_and_organized_select_the_same_l1_evidence() -> None:
    _, _, _, _, original = _run_group("l1_original_only")
    _, _, _, _, organized = _run_group("l1_organized_only")
    def facts(result):
        return {
            (item.metadata.get("role"), item.label or "")
            for item in result.context.layer_items
        }

    original_facts = facts(original[1])
    organized_facts = facts(organized[1])
    assert original_facts
    assert original_facts == organized_facts


def test_original_and_organized_select_the_same_l3_objects() -> None:
    _, _, _, _, original = _run_group("l3_original_only")
    _, _, _, _, organized = _run_group("l3_organized_only")
    original_ids = {
        item.metadata.get("object_id") for item in original[1].context.layer_items
    }
    organized_ids = {
        item.metadata.get("object_id") for item in organized[1].context.layer_items
    }
    assert original_ids
    assert original_ids == organized_ids


def test_concat_and_joint_are_real_different_fusion_paths() -> None:
    _, _, _, _, concat_ctx = _run_group("l1_l3_original_concat")
    _, _, _, _, joint_ctx = _run_group("l1_l3_joint")
    concat = concat_ctx[1]
    joint = joint_ctx[1]
    assert concat.source == "concat"
    assert joint.source == "joint"
    assert concat.text != joint.text
    assert "[Navigation Context - L1+L3]" in joint.text
    assert "[Navigation Context - L1+L3]" not in concat.text
    assert concat.diagnostics["alignment"]["confirmed"] == 0
    assert joint.diagnostics["layers"]["joint"]["alignment"]["attempted"] >= 0
    assert "joint" in joint.diagnostics["layers"]


def test_supplemental_joint_preserves_original_l1_lines() -> None:
    _, _, _, _, contexts = _run_group("l1_original_l3_organized_joint")
    joint = contexts[1]
    assert joint.source == "joint"
    assert "Local history: chair was seen near a visually similar area" in joint.text
    assert joint.diagnostics["layers"]["joint"]["l1_item_count"] >= 1


def test_l1_only_never_runs_the_l3_map_or_segmentation() -> None:
    backend, seg, _, _, _ = _run_group("l1_original_only")
    assert seg.calls == []
    assert backend.object_map is None
    assert backend.map_snapshot() == {
        "objects": 0, "viewpoints": 0, "evidence": 0, "edges": 0,
    }


def test_l3_only_never_calls_the_l1_prompt_builders(monkeypatch) -> None:
    backend = make_memory_backend("l3_organized_only")

    def forbidden(*args, **kwargs):
        raise AssertionError("L1 prompt builder was called in an L3-only group")

    monkeypatch.setattr(backend.local_graph_fusion, "build", forbidden)
    monkeypatch.setattr(backend.l1_prompt_organizer, "build", forbidden)
    backend.reset_episode(_episode(), "find the chair")
    for step in (0, 1):
        events, _ = backend.observe(
            {"rgb": np.zeros((8, 8, 3), np.uint8)}, step,
        )
        backend.observe_spatial(FRAME, events, step, transition_record=bool(step))
        result = backend.build_context(
            instruction="find the chair", step=step, rgb_width=8, frame=FRAME,
        )
    assert result.context is not None
    assert result.source == "l3_organized"


def test_joint_prompts_keep_the_official_template_fixed() -> None:
    for group in ("l1_l3_joint", "l1_original_l3_organized_joint"):
        _, _, _, _, contexts = _run_group(group)
        memory_text = contexts[1].text
        task_text = build_task_text("find the chair", memory_text)
        prompt = build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, task_text)
        verify_task_text_injection(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE, "find the chair",
            memory_text, prompt,
        )
        prefix, suffix = REFERENCE_OFFICIAL_PROMPT_TEMPLATE.split("{}")
        assert prompt.startswith(prefix + "find the chair")
        assert prompt.endswith(suffix)
        assert prompt.count("[Navigation]") == 0  # token-level marker only

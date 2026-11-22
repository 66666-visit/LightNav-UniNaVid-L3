"""Launcher smoke gate: three real smokes must satisfy the A/B/C contracts."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_three_group_smoke as gate  # noqa: E402

A_GROUP = "uninavid_l3_only"
B_GROUP = "uninavid_l3_source_background"
C_GROUP = "uninavid_l3_l1_sbadir"
PROMPT_SHA = "435d79868156aaa1352afcd98f4ceae2f03a54f8cdc5fe87e41d32774d5512c1"
IDENTITY = ("scene_a.glb", "1")


def _row(group, **overrides):
    row = {
        "scene_basename": IDENTITY[0], "episode_id": IDENTITY[1], "group": group,
        "episode_seed": 11, "instruction_sha256": "instruction-1",
        "official_prompt_sha256": PROMPT_SHA,
        "l3_enabled": True, "env_steps": 4, "l3_map_steps": 4,
        "l3_fact_count_total": 2, "prompt_has_l3_steps": 3,
        "l3_segmentation_error_count": 0,
    }
    if group in (B_GROUP, C_GROUP):
        row.update({
            "source_background_enabled": True,
            "source_background_facts_total": 2,
            "l3_original_memory_unchanged": True,
            "l1_enabled": False, "l1_enabled_steps": 0,
            "l1_observation_attempt_steps": 0,
            "l1_fact_count_total": 0, "prompt_has_l1_steps": 0,
        })
    if group == C_GROUP:
        row.update({
            "l1_enabled": True, "l1_enabled_steps": 4,
            "l1_observation_attempt_steps": 4,
            "l1_fact_count_total": 2, "l1_retained_facts_total": 2,
            "l1_omitted_facts_total": 0, "l1_prompt_built_steps": 2,
            "l1_empty_steps": 2, "prompt_has_l1_steps": 2,
            "source_background_facts_total": 2,
            "l1_events_persisted_total": 4,
            "l1_detection_events_persisted_total": 2,
            "l1_place_attempt_steps": 1, "l1_place_observed_steps": 1,
            "l1_observe_error_count": 0, "l1_selection_error_count": 0,
            "l1_detector_error_count": 0,
            "l1_place_error_count": 0,
            "cross_label_entity_link_error_count": 0,
            "budget_source": "token_aware_context",
            "context_limit_tokens": 2048,
            "base_prompt_tokens_mean": 300.0,
            "available_memory_tokens_min": 723,
            "memory_tokens_injected_mean": 80.0,
            "token_overflow_drops_total": 0,
            "budget_fallback_reason": None,
            "memory_budget_violation_steps": 0,
            "text_prompt_tokens_mean": 300.0,
            "visual_tokens_mean": 382.0,
            "multimodal_prompt_tokens_mean": 762.0,
            "context_margin_tokens_min": 500,
            "multimodal_budget_violation_steps": 0,
        })
    row.update(overrides)
    return row


def _write(tmp_path, rows):
    directories = {}
    for name, row in rows.items():
        directory = tmp_path / name
        directory.mkdir()
        (directory / "results.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8")
        directories[name] = directory
    return directories


def _validate(directories):
    return gate.validate(
        directories["a"], directories["b"], directories["c"],
        A_GROUP, B_GROUP, C_GROUP,
    )


def test_three_consistent_smokes_pass_and_write_validation(tmp_path):
    directories = _write(tmp_path, {
        "a": _row(A_GROUP), "b": _row(B_GROUP), "c": _row(C_GROUP),
    })
    result = _validate(directories)
    assert result["episode"] == list(IDENTITY)
    for name, group in (("a", A_GROUP), ("b", B_GROUP), ("c", C_GROUP)):
        check = json.loads(
            (directories[name] / "smoke_validation.json").read_text())
        assert check["group"] == group
        assert check["episode"] == list(IDENTITY)
        assert check["official_prompt_sha256"] == PROMPT_SHA


def test_smokes_must_use_the_same_episode(tmp_path):
    directories = _write(tmp_path, {
        "a": _row(A_GROUP),
        "b": _row(B_GROUP, episode_id="2"),
        "c": _row(C_GROUP),
    })
    with pytest.raises(ValueError, match="different Episode"):
        _validate(directories)
    assert not (directories["a"] / "smoke_validation.json").exists()


def test_b_smoke_may_not_enable_l1(tmp_path):
    directories = _write(tmp_path, {
        "a": _row(A_GROUP),
        "b": _row(B_GROUP, l1_enabled=True, l1_fact_count_total=1,
                  prompt_has_l1_steps=1),
        "c": _row(C_GROUP),
    })
    with pytest.raises(ValueError, match="B smoke enabled L1"):
        _validate(directories)


def test_a_smoke_must_not_contain_source_background_fields(tmp_path):
    directories = _write(tmp_path, {
        "a": _row(A_GROUP, source_background_facts_total=2),
        "b": _row(B_GROUP),
        "c": _row(C_GROUP),
    })
    with pytest.raises(ValueError, match="A smoke unexpectedly contains"):
        _validate(directories)


def test_c_smoke_pairs_only_the_frozen_l3_facts_with_a_source(tmp_path):
    directories = _write(tmp_path, {
        "a": _row(A_GROUP), "b": _row(B_GROUP),
        "c": _row(C_GROUP, source_background_facts_total=4),
    })
    with pytest.raises(ValueError, match="C smoke did not pair every L3 fact"):
        _validate(directories)


def test_c_smoke_with_no_l1_fact_is_valid_and_not_an_abort(tmp_path):
    # This is the regression test for
    # "ValueError: C smoke produced no L1 fact; choose another SMOKE_SHARD".
    directories = _write(tmp_path, {
        "a": _row(A_GROUP), "b": _row(B_GROUP),
        "c": _row(
            C_GROUP,
            l1_fact_count_total=0,
            l1_retained_facts_total=0,
            l1_omitted_facts_total=0,
            l1_prompt_built_steps=0,
            l1_empty_steps=4,
            prompt_has_l1_steps=0,
        ),
    })
    result = _validate(directories)
    assert result["episode"] == list(IDENTITY)
    check = json.loads((directories["c"] / "smoke_validation.json").read_text())
    assert check["l1_fact_count_total"] == 0
    assert check["prompt_has_l1_steps"] == 0


def test_c_smoke_with_no_fact_must_not_inject_l1_text(tmp_path):
    directories = _write(tmp_path, {
        "a": _row(A_GROUP), "b": _row(B_GROUP),
        "c": _row(
            C_GROUP,
            l1_fact_count_total=0,
            l1_retained_facts_total=0,
            l1_prompt_built_steps=0,
            prompt_has_l1_steps=1,
        ),
    })
    with pytest.raises(ValueError, match="without an L1 fact"):
        _validate(directories)


def test_c_smoke_with_facts_must_build_the_original_l1_card(tmp_path):
    directories = _write(tmp_path, {
        "a": _row(A_GROUP), "b": _row(B_GROUP),
        "c": _row(C_GROUP, l1_prompt_built_steps=0),
    })
    with pytest.raises(ValueError, match="never built the original L1 card"):
        _validate(directories)


def test_c_smoke_token_aware_budget_must_cover_injected_tokens(tmp_path):
    directories = _write(tmp_path, {
        "a": _row(A_GROUP), "b": _row(B_GROUP),
        "c": _row(C_GROUP, memory_budget_violation_steps=1),
    })
    with pytest.raises(ValueError, match="injected more L1 tokens"):
        _validate(directories)


def test_c_smoke_char_fallback_requires_a_reason(tmp_path):
    directories = _write(tmp_path, {
        "a": _row(A_GROUP), "b": _row(B_GROUP),
        "c": _row(C_GROUP, budget_source="char_fallback_320",
                  base_prompt_tokens_mean=None,
                  available_memory_tokens_min=None,
                  memory_tokens_injected_mean=None,
                  budget_fallback_reason=""),
    })
    with pytest.raises(ValueError, match="char fallback lacks a reported reason"):
        _validate(directories)


def test_c_smoke_rejects_negative_multimodal_context_margin(tmp_path):
    directories = _write(tmp_path, {
        "a": _row(A_GROUP), "b": _row(B_GROUP),
        "c": _row(C_GROUP, context_margin_tokens_min=-1),
    })
    with pytest.raises(ValueError, match="exceeds context reserve"):
        _validate(directories)


def test_c_smoke_rejects_multimodal_budget_violation(tmp_path):
    directories = _write(tmp_path, {
        "a": _row(A_GROUP), "b": _row(B_GROUP),
        "c": _row(C_GROUP, multimodal_budget_violation_steps=1),
    })
    with pytest.raises(ValueError, match="multimodal context reserve"):
        _validate(directories)

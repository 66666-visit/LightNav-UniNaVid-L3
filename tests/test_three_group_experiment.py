"""Frozen A/B/C contracts: no L1 in B, identical L3 facts, original L1 in C."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from l3_integration.experiment_groups import (
    A_GROUP,
    A_NAME,
    B_GROUP,
    B_NAME,
    C_GROUP,
    C_NAME,
    spec,
    validate_model_paths,
)
from l3_integration.official_prompt import (
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
    build_prompt,
    build_task_text,
    verify_task_text_injection,
)
from l3_integration.sbadir_fusion import (
    ORIGINAL_L1_PROMPT_MAX_CHARS,
    append_original_l1,
    compose_sbadir,
)
from l3_memory.l3_official_prompt import HEADER

L3_LINE = "Map record (step 2, 1 view): the chair is in front of you, 2.0 m straight-line."
L3_TEXT = HEADER + "\n" + L3_LINE
L1_LINE = (
    "Current RGB: instruction landmark chair in the left of the current view "
    "(detected; passage unknown)."
)
L1_TEXT = "[Navigation Memory]\n" + L1_LINE
INSTRUCTION = "find the chair"


def _count(text):
    return len(text.split())


def _l3(label="chair", linked=(), evidence=("seg-1",), **overrides):
    fields = dict(
        object_id="map-1", label=label, latest_step=2, last_step=2,
        source="yolo11n-seg:seg.pt", confidence=0.87,
        geometry_verified=True, evidence_ids=tuple(evidence),
        linked_l1_evidence_ids=tuple(linked),
        line=lambda: L3_LINE, viewpoints=1,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _b_compose(l3_facts=None, text=L3_TEXT, entity_links=None):
    return compose_sbadir(
        template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
        instruction=INSTRUCTION,
        l3_text=text,
        l3_facts=[_l3()] if l3_facts is None else list(l3_facts),
        l1_facts=(),
        count_tokens=_count,
        max_prompt_tokens=10000,
        l1_max_tokens=0,
        entity_links=entity_links,
    )


def _c_compose(l1_text=L1_TEXT, l3_facts=None, capacity=10000, char_budget=320):
    b = _b_compose(l3_facts=l3_facts)
    return append_original_l1(
        template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
        instruction=INSTRUCTION,
        base_memory_text=b.memory_text,
        l1_text=l1_text,
        count_tokens=_count,
        max_prompt_tokens=capacity,
        char_budget=char_budget,
    )


def test_group_specs_freeze_the_three_required_conditions():
    a, b, c = spec(A_GROUP), spec(B_GROUP), spec(C_GROUP)
    assert (a.name, b.name, c.name) == (A_NAME, B_NAME, C_NAME)
    assert a.name == "l3_organized_baseline_7602b25"
    assert b.name == "l3_sbadir_source_background_only"
    assert c.name == "l3_l1_sbadir"
    assert (a.uses_l3, a.source_background, a.uses_l1) == (True, False, False)
    assert (b.uses_l3, b.source_background, b.uses_l1) == (True, True, False)
    assert (c.uses_l3, c.source_background, c.uses_l1) == (True, True, True)
    assert a.worktree == "baseline" and not b.worktree == "baseline"


def test_a_prompt_is_the_original_l3_text_without_any_source_background():
    assert spec(A_GROUP).source_background is False
    task_text = build_task_text(INSTRUCTION, L3_TEXT)
    prompt = build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, task_text)
    expected = REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format(
        INSTRUCTION + "\n" + L3_TEXT
    )
    assert prompt == expected
    assert "[L3 RGB-D map" not in prompt
    assert "[L1 RGB observation" not in prompt
    assert "evidence seg-1" not in prompt
    verify_task_text_injection(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, INSTRUCTION, L3_TEXT, prompt,
    )


def test_b_never_constructs_l1_and_its_prompt_has_no_l1_text():
    assert spec(B_GROUP).uses_l1 is False
    validate_model_paths(
        B_GROUP, l3_seg_model="seg.pt",
        l1_det_model="", l1_place_model="",
    )
    with pytest.raises(ValueError, match="must not receive L1 model paths"):
        validate_model_paths(
            B_GROUP, l3_seg_model="seg.pt",
            l1_det_model="detect.pt", l1_place_model="dinov2",
        )
    with pytest.raises(ValueError, match="requires --l1-det-model"):
        validate_model_paths(
            C_GROUP, l3_seg_model="seg.pt",
            l1_det_model="", l1_place_model="",
        )
    validate_model_paths(
        C_GROUP, l3_seg_model="seg.pt",
        l1_det_model="detect.pt", l1_place_model="dinov2",
    )

    result = _b_compose()
    assert result.l1_text is None
    assert result.diagnostics["l1_fact_count"] == 0
    assert result.diagnostics["l1_retained_facts"] == 0
    assert result.diagnostics["l1_retained_evidence_ids"] == []
    assert "L1 RGB" not in result.memory_text
    prompt = build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, result.task_text)
    assert "[L1 RGB" not in prompt


def test_b_keeps_a_l3_fact_lines_order_and_signature_exactly():
    original_lines = L3_TEXT.splitlines()
    result = _b_compose()
    assert result.diagnostics["l3_core_unchanged"] is True
    assert result.l3_enriched_text.splitlines()[0] == original_lines[0]
    enriched_lines = result.l3_enriched_text.splitlines()
    assert len(enriched_lines) == len(original_lines)
    for original, enriched in zip(original_lines, enriched_lines):
        assert enriched.startswith(original)
    assert result.diagnostics["l3_original_sha256"] == hashlib.sha256(
        L3_TEXT.encode("utf-8")).hexdigest()
    identity = [{
        "object_id": "map-1", "label": "chair",
        "evidence_ids": ["seg-1"], "line": L3_LINE,
    }]
    assert result.diagnostics["l3_fact_signature_sha256"] == hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def test_b_pairs_each_l3_fact_with_its_real_evidence_fields():
    result = _b_compose()
    enriched = result.l3_enriched_text
    assert "L3 RGB-D map" in enriched
    assert "source yolo11n-seg:seg.pt" in enriched
    assert "map node map-1" in enriched
    assert "geometry observed" in enriched
    assert "step 2" in enriched
    assert "seg conf 0.87" in enriched
    assert "evidence seg-1" in enriched
    assert result.diagnostics["l3_provenance_count"] == 1

    missing = _l3(source="", confidence=None, evidence=(), latest_step=-1,
                  geometry_verified=False)
    unavailable = _b_compose(l3_facts=[missing])
    text = unavailable.l3_enriched_text
    assert "source unavailable" in text
    assert "geometry unverified" in text
    assert "step unavailable" in text
    assert "seg conf unavailable" in text
    assert "evidence unavailable" in text


def test_c_keeps_b_l3_fact_text_and_source_background_identical():
    b = _b_compose()
    c = _c_compose()
    # C's L1 append must not touch B's frozen L3 block.
    assert c.memory_text.startswith(b.memory_text)
    assert c.memory_text.splitlines()[:len(b.memory_text.splitlines())] == (
        b.memory_text.splitlines()
    )
    assert c.diagnostics["l1_injected"] is True
    assert c.diagnostics["l1_chars"] == len(L1_TEXT)


def test_c_appends_the_original_l1_card_verbatim_after_the_l3_block():
    result = _c_compose()
    assert result.l1_text == L1_TEXT
    assert result.memory_text == _b_compose().memory_text + "\n" + L1_TEXT
    prompt = build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, result.task_text)
    assert result.memory_text in prompt
    assert prompt.index(_b_compose().memory_text) < prompt.index(L1_TEXT)
    assert "L1 RGB fact" not in prompt
    assert "L1 RGB observation" not in prompt
    assert result.diagnostics["l1_sha256"] == hashlib.sha256(
        L1_TEXT.encode("utf-8")).hexdigest()
    verify_task_text_injection(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, INSTRUCTION,
        result.memory_text, prompt,
    )


def test_c_with_no_l1_card_keeps_the_b_prompt_byte_identical():
    b = _b_compose()
    c = _c_compose(l1_text=None)
    assert c.memory_text == b.memory_text
    assert c.task_text == b.task_text
    assert c.l1_text is None
    assert c.diagnostics["l1_injected"] is False
    assert c.diagnostics["empty_reason"] == "no_original_l1_card"


def test_c_omits_the_whole_card_when_the_measured_capacity_is_too_small():
    b = _b_compose()
    c = _c_compose(capacity=1)
    assert c.memory_text == b.memory_text
    assert c.task_text == b.task_text
    assert c.l1_text is None
    assert c.diagnostics["l1_injected"] is False
    assert c.diagnostics["empty_reason"] == "measured_prompt_capacity"


def test_c_rejects_a_card_that_violates_the_original_320_character_budget():
    oversized = "[Navigation Memory]\n" + ("x" * ORIGINAL_L1_PROMPT_MAX_CHARS)
    with pytest.raises(ValueError, match="exceeds its 320-character budget"):
        _c_compose(l1_text=oversized)


def test_c_requires_the_original_local_graph_header():
    with pytest.raises(ValueError, match="lacks the Local Graph header"):
        _c_compose(l1_text="[Navigation Context - L1]\nNow: chair")

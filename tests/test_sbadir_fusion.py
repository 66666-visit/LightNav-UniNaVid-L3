"""B's source pairing and the verbatim original L1 append used by C."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from l3_integration.official_prompt import (
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
    build_prompt,
    split_official_template,
    verify_task_text_injection,
)
from l3_integration.sbadir_fusion import append_original_l1, compose_sbadir


HEADER = "[Navigation Map Memory - L3; history, not current view]"
L3_LINE = "Map record (step 2, 1 view): the chair is in front of you, 2.0 m straight-line."
L1_TEXT = (
    "[Navigation Memory]\n"
    "Current RGB: instruction landmark chair in the left of the current view "
    "(detected; passage unknown)."
)


def _count(text):
    # Deterministic fake tokenizer; the real agent supplies checkpoint.encode.
    return len(text.split())


def _l3(label="chair", linked=("rgb-1",)):
    return SimpleNamespace(
        object_id="map-1", label=label, latest_step=2, last_step=2,
        source="yolo11n-seg:seg.pt", confidence=0.87,
        geometry_verified=True, evidence_ids=("seg-1",),
        linked_l1_evidence_ids=linked,
    )


def _compose(l3=None, **kwargs):
    l3_facts = [_l3()] if l3 is None else l3
    return compose_sbadir(
        template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
        instruction="find the chair",
        l3_text=HEADER + "\n" + L3_LINE if l3_facts else None,
        l3_facts=l3_facts, l1_facts=(),
        count_tokens=_count, max_prompt_tokens=10000,
        l1_max_tokens=0, **kwargs,
    )


def test_b_l3_provenance_pairs_latest_step_with_latest_evidence_id():
    fact = _l3()
    fact.evidence_ids = ("seg-old", "seg-new")
    fact.latest_step = 7
    result = _compose(l3=[fact])
    assert "step 7; seg conf 0.87; evidence seg-new" in result.memory_text
    assert "evidence seg-old" not in result.memory_text
    assert result.diagnostics["l3_provenance_count"] == 1


def test_b_never_adds_an_l1_fact_or_source_suffix():
    result = _compose()
    assert result.l1_text is None
    assert result.diagnostics["l1_fact_count"] == 0
    assert result.diagnostics["l1_retained_facts"] == 0
    assert "L1 RGB" not in result.memory_text
    assert result.diagnostics["l3_provenance_count"] == 1
    assert result.diagnostics["l1_retained_facts"] == 0


def test_official_fixed_parts_and_visual_token_position_remain_exact():
    result = _compose()
    prompt = build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, result.task_text)
    prefix, suffix = split_official_template(REFERENCE_OFFICIAL_PROMPT_TEMPLATE)
    assert prompt == prefix + result.task_text + suffix
    assert prompt.count("<image>") == 1
    assert prompt.index("<image>") < prompt.index("find the chair")
    assert prompt.endswith(suffix)
    verify_task_text_injection(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, "find the chair",
        result.memory_text, prompt,
    )
    assert result.diagnostics["l3_original_sha256"] == hashlib.sha256(
        (HEADER + "\n" + L3_LINE).encode("utf-8")
    ).hexdigest()


def test_original_l1_block_is_appended_after_b_without_rewriting_it():
    b = _compose()
    result = append_original_l1(
        template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
        instruction="find the chair",
        base_memory_text=b.memory_text,
        l1_text=L1_TEXT,
        count_tokens=_count,
        max_prompt_tokens=10000,
    )
    assert result.memory_text == b.memory_text + "\n" + L1_TEXT
    assert result.l1_text == L1_TEXT
    assert result.diagnostics["l1_injected"] is True
    assert result.diagnostics["l1_chars"] == len(L1_TEXT)
    prompt = build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, result.task_text)
    assert b.memory_text in prompt
    assert prompt.index(b.memory_text) < prompt.index(L1_TEXT)
    assert "L1 RGB" not in prompt


def test_original_l1_absent_is_a_no_op():
    b = _compose()
    result = append_original_l1(
        template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
        instruction="find the chair",
        base_memory_text=b.memory_text,
        l1_text=None,
        count_tokens=_count,
        max_prompt_tokens=10000,
    )
    assert result.memory_text == b.memory_text
    assert result.task_text == b.task_text
    assert result.diagnostics["l1_injected"] is False


def test_original_l1_is_omitted_as_a_whole_when_capacity_is_small():
    b = _compose()
    result = append_original_l1(
        template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
        instruction="find the chair",
        base_memory_text=b.memory_text,
        l1_text=L1_TEXT,
        count_tokens=_count,
        max_prompt_tokens=1,
    )
    assert result.memory_text == b.memory_text
    assert result.l1_text is None
    assert result.diagnostics["l1_injected"] is False
    assert result.diagnostics["empty_reason"] == "measured_prompt_capacity"


def test_original_l1_budget_and_header_are_enforced():
    with pytest.raises(ValueError, match="exceeds its 320-character budget"):
        append_original_l1(
            template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
            instruction="find the chair",
            base_memory_text=None,
            l1_text="[Navigation Memory]\n" + "x" * 320,
            count_tokens=_count,
            max_prompt_tokens=10000,
        )
    with pytest.raises(ValueError, match="lacks the Local Graph header"):
        append_original_l1(
            template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
            instruction="find the chair",
            base_memory_text=None,
            l1_text="[Navigation Context - L1]\nNow: chair",
            count_tokens=_count,
            max_prompt_tokens=10000,
        )


def test_per_fact_l1_sbdir_fusion_is_disabled():
    # C must use append_original_l1; the old per-fact L1 source-pairing path
    # must not be reachable through compose_sbadir.
    with pytest.raises(ValueError, match="SBAdir L1 fact fusion is disabled"):
        compose_sbadir(
            template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
            instruction="find the chair",
            l3_text=HEADER + "\n" + L3_LINE,
            l3_facts=[_l3()],
            l1_facts=[SimpleNamespace(label="chair")],
            count_tokens=_count,
            max_prompt_tokens=10000,
        )

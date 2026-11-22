"""C's model-derived L1 budget: formula, whole-line selection, 320 fallback.

These are CPU tests of the adapter around the unchanged ``LocalGraphFusion``.
They do not claim the LightNav 75% SR; that requires the real checkpoint and
the formal 88-Episode run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from l1_integration.sbadir_backend import (
    ORIGINAL_L1_PROMPT_MAX_CHARS,
    L1EvidenceBackend,
)
from l1_integration.token_budget import (
    BOUNDARY_RESERVE_TOKENS,
    ORIGINAL_CHAR_FALLBACK,
    UNBOUNDED_CHAR_LIMIT,
    MemoryTokenBudget,
    TokenBudgetFailure,
    build_token_budget,
)
from l1_memory.memory_writer import MockSceneExtractor
from l3_integration.official_prompt import REFERENCE_OFFICIAL_PROMPT_TEMPLATE
from l3_integration.sbadir_fusion import append_original_l1
from l3_memory.schemas import DetectedObject, SceneExtraction

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "Find the chair and table"


def _counter(text):
    """Deterministic word counter; the production counter is the tokenizer."""
    return len(text.split())


def _rgb():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _two_fact_backend():
    detector = MockSceneExtractor({
        0: SceneExtraction(
            detections=[
                DetectedObject("chair", 0.91, (0, 0, 3, 3)),
                DetectedObject("table", 0.90, (3, 0, 6, 3)),
            ],
            source="mock_scene_extractor", confidence=0.91,
        ),
    })
    backend = L1EvidenceBackend(detector)
    backend.reset_episode("scene-A", "episode-1", INSTRUCTION)
    backend.observe(_rgb(), 0)
    return backend


def _budget(available_tokens, *, base=500, reserve=1024):
    # Keep the test object internally consistent with LightNav's formula.
    context = base + reserve + BOUNDARY_RESERVE_TOKENS + available_tokens
    return MemoryTokenBudget(
        available_tokens=available_tokens,
        context_limit_tokens=context,
        base_prompt_tokens=base,
        generation_reserve_tokens=reserve,
        safety_margin_tokens=BOUNDARY_RESERVE_TOKENS,
        context_limit_source="cpu-test",
        counter=_counter,
    )


def test_budget_formula_is_lightnav_formula_with_one_safety_token():
    budget = build_token_budget(
        context_limit_tokens=2048,
        base_prompt_tokens=300,
        generation_reserve_tokens=1024,
        counter=_counter,
        context_limit_source="cpu-test",
    )
    assert isinstance(budget, MemoryTokenBudget)
    assert budget.available_tokens == 2048 - 300 - 1024 - 1
    assert budget.available_tokens == 723
    assert budget.safety_margin_tokens == 1
    assert budget.diagnostics()["budget_source"] == "token_aware_context"


def test_budget_measurement_failure_produces_the_original_320_fallback():
    failure = build_token_budget(
        context_limit_tokens=None,
        base_prompt_tokens=None,
        generation_reserve_tokens=1024,
        counter=_counter,
        context_limit_source="cpu-test",
    )
    assert isinstance(failure, TokenBudgetFailure)
    diagnostics = failure.diagnostics()
    assert diagnostics["budget_source"] == "char_fallback_320"
    assert diagnostics["available_memory_tokens"] is None
    assert "context_limit_unavailable" in diagnostics["budget_fallback_reason"]

    backend = _two_fact_backend()
    native = backend.select(step=0, rgb_width=8)
    fallback = backend.select(
        step=0, rgb_width=8, budget_failure=failure,
    )
    assert fallback.diagnostics["budget_source"] == "char_fallback_320"
    assert fallback.diagnostics["internal_char_limit"] == "native_320"
    assert fallback.diagnostics["budget_fallback_reason"] == (
        diagnostics["budget_fallback_reason"]
    )
    assert fallback.prompt_text == native.prompt_text
    assert fallback.prompt_text.startswith("[Navigation Memory]\n")
    assert len(fallback.prompt_text) <= ORIGINAL_L1_PROMPT_MAX_CHARS
    assert fallback.fact_count == 2
    assert fallback.diagnostics["token_overflow_drops"] == 0
    backend.close_episode()


def test_insufficient_tokens_drop_whole_lines_and_never_cut_a_sentence():
    backend = _two_fact_backend()
    native = backend.select(step=0, rgb_width=8)
    lines = native.context.text.splitlines()
    assert len(lines) == 3
    capacity = _counter("\n".join(lines[:2]))
    budget = _budget(capacity)
    selection = backend.select(
        step=0, rgb_width=8, token_budget=budget,
    )

    assert selection.diagnostics["budget_source"] == "token_aware_context"
    assert selection.diagnostics["internal_char_limit"] == "token_budget_owned"
    assert selection.diagnostics["available_memory_tokens"] == capacity
    assert selection.diagnostics["fact_count"] == 2
    assert selection.fact_count == 1
    assert selection.diagnostics["retained_fact_count"] == 1
    assert selection.diagnostics["omitted_facts"] == 1
    assert selection.diagnostics["token_overflow_drops"] == 1
    assert selection.context.text == "\n".join(lines[:2])
    assert lines[2] not in selection.prompt_text
    # The kept text is exactly whole original lines; no ellipsis or truncation.
    assert selection.prompt_text.endswith("passage unknown).")
    assert "..." not in selection.prompt_text
    assert "…" not in selection.prompt_text
    assert len(selection.evidence_ids) == 1
    # The native 320-character setting is restored after the token-aware call.
    assert backend.local_graph.config.prompt_max_chars == 320
    backend.close_episode()


def test_zero_headroom_injects_no_card_and_reports_no_token_headroom():
    backend = _two_fact_backend()
    native = backend.select(step=0, rgb_width=8)
    line_count = len(native.context.text.splitlines())
    selection = backend.select(
        step=0, rgb_width=8, token_budget=_budget(0),
    )

    assert selection.context is None
    assert selection.prompt_text is None
    assert selection.diagnostics["fact_count"] == 2
    assert selection.fact_count == 0
    assert selection.diagnostics["retained_fact_count"] == 0
    assert selection.diagnostics["available_memory_tokens"] == 0
    assert selection.diagnostics["empty_reason"] == "no_token_headroom"
    assert selection.diagnostics["token_overflow_drops"] == line_count
    backend.close_episode()


def test_token_aware_append_may_exceed_320_but_never_a_half_line():
    long_card = (
        "[Navigation Memory]\n"
        "Current RGB: instruction landmark chair in the left of the current "
        "view (detected; passage unknown)."
    )
    long_card += " " + ("observed " * 50).strip()
    assert len(long_card) > 320
    result = append_original_l1(
        template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
        instruction="find the chair",
        base_memory_text=None,
        l1_text=long_card,
        count_tokens=_counter,
        max_prompt_tokens=10000,
        char_budget=None,
        token_aware=True,
    )
    assert result.l1_text == long_card
    assert result.diagnostics["l1_injected"] is True
    assert result.diagnostics["budget_source"] == "token_aware_context"
    assert result.diagnostics["char_budget"] is None

    with pytest.raises(ValueError, match="320-character budget"):
        append_original_l1(
            template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
            instruction="find the chair",
            base_memory_text=None,
            l1_text=long_card,
            count_tokens=_counter,
            max_prompt_tokens=10000,
            char_budget=ORIGINAL_L1_PROMPT_MAX_CHARS,
            token_aware=False,
        )


def test_original_320_fallback_and_token_budget_are_not_arbitrary_numbers():
    assert ORIGINAL_CHAR_FALLBACK == 320
    assert ORIGINAL_L1_PROMPT_MAX_CHARS == 320
    new_module = (ROOT / "l1_integration/token_budget.py").read_text(encoding="utf-8")
    backend = (ROOT / "l1_integration/sbadir_backend.py").read_text(encoding="utf-8")
    for text in (new_module, backend):
        assert "l1_max_tokens" not in text
        assert "65536" not in text
        assert "160" not in text
        assert "600" not in text


def test_modified_python_files_parse_as_python_38():
    for name in (
        "l1_integration/token_budget.py",
        "l1_integration/sbadir_backend.py",
        "l3_integration/sbadir_fusion.py",
        "agent_uninavid.py",
    ):
        ast.parse(
            (ROOT / name).read_text(encoding="utf-8"),
            filename=name,
            feature_version=(3, 8),
        )


def test_token_aware_path_lifts_only_the_internal_cap_then_restores_320(monkeypatch):
    backend = _two_fact_backend()
    captured = []
    original_build = backend.local_graph.build

    def recording_build(*args, **kwargs):
        captured.append(backend.local_graph.config.prompt_max_chars)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(backend.local_graph, "build", recording_build)
    selection = backend.select(
        step=0, rgb_width=8, token_budget=_budget(10000),
    )
    assert selection.diagnostics["budget_source"] == "token_aware_context"
    assert captured[-1] == UNBOUNDED_CHAR_LIMIT
    assert backend.local_graph.config.prompt_max_chars == 320

    fallback = backend.select(step=0, rgb_width=8)
    assert fallback.diagnostics["internal_char_limit"] == "native_320"
    assert captured[-1] == 320
    assert backend.local_graph.config.prompt_max_chars == 320
    backend.close_episode()


def test_full_token_budget_keeps_the_original_card_byte_identical():
    backend = _two_fact_backend()
    native = backend.select(step=0, rgb_width=8)
    token_aware = backend.select(
        step=0, rgb_width=8, token_budget=_budget(10000),
    )
    assert token_aware.prompt_text == native.prompt_text
    assert token_aware.context.text == native.context.text
    assert token_aware.diagnostics["fact_count"] == 2
    assert token_aware.diagnostics["retained_fact_count"] == 2
    backend.close_episode()


def test_header_only_selection_is_rejected_instead_of_injecting_no_fact():
    backend = _two_fact_backend()
    native = backend.select(step=0, rgb_width=8)
    header_tokens = _counter("[Navigation Memory]")
    selection = backend.select(
        step=0, rgb_width=8, token_budget=_budget(header_tokens),
    )
    assert selection.context is None
    assert selection.prompt_text is None
    assert selection.diagnostics["empty_reason"] == "token_selection_empty"
    assert selection.diagnostics["retained_fact_count"] == 0
    assert selection.diagnostics["token_overflow_drops"] == (
        len(native.context.text.splitlines()) - 1
    )
    backend.close_episode()


def test_measured_budget_wins_over_a_stale_failure_object():
    backend = _two_fact_backend()
    selection = backend.select(
        step=0,
        rgb_width=8,
        token_budget=_budget(10000),
        budget_failure=TokenBudgetFailure(reason="stale failure"),
    )
    assert selection.diagnostics["budget_source"] == "token_aware_context"
    assert selection.diagnostics["budget_fallback_reason"] is None
    backend.close_episode()


def test_token_aware_append_omits_the_whole_card_on_exact_capacity_overflow():
    card = (
        "[Navigation Memory]\n"
        "Current RGB: instruction landmark chair in the left of the current "
        "view (detected; passage unknown)."
    )
    result = append_original_l1(
        template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
        instruction="find the chair",
        base_memory_text=None,
        l1_text=card,
        count_tokens=_counter,
        max_prompt_tokens=1,
        char_budget=None,
        token_aware=True,
    )
    assert result.l1_text is None
    assert result.diagnostics["l1_injected"] is False
    assert result.diagnostics["empty_reason"] == "measured_prompt_capacity"
    assert result.diagnostics["token_overflow_drops"] == len(card.splitlines())
    assert result.diagnostics["char_budget"] is None

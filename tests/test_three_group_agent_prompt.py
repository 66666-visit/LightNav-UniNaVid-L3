"""The real agent prompt path keeps B frozen and injects C's original L1 card."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tests" / "sbadir_agent_probe.py"
OFFICIAL_PREFIX = (
    "Imagine you are a robot programmed for navigation tasks. You have been given "
    "a video of historical observations and an image of the current observation "
)
L1_HEADER = "[Navigation Memory]\n"


def _probe(mode: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(PROBE), mode],
        cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_a_prompt_has_l3_but_no_l1_and_no_source_background():
    record = _probe("a")
    prompt = record["prompt"]
    assert prompt.startswith(OFFICIAL_PREFIX)
    assert prompt.count("<image>") == 1
    assert "Navigation Map Memory - L3" in prompt
    assert "[L3 RGB-D map;" not in prompt
    assert L1_HEADER not in prompt
    assert record["summary"].get("l1_fact_count_total") in (None, 0)


def test_b_prompt_has_l3_source_background_but_no_l1():
    record = _probe("b")
    prompt = record["prompt"]
    assert "Navigation Map Memory - L3" in prompt
    assert "[L3 RGB-D map;" in prompt
    assert L1_HEADER not in prompt
    assert record["summary"]["l1_enabled"] is False
    assert record["summary"].get("l1_fact_count_total") in (None, 0)
    assert record["summary"]["source_background_facts_total"] == (
        record["summary"]["l3_fact_count_total"]
    )


def test_c_prompt_contains_the_verbatim_original_l1_card_after_b():
    record = _probe("c")
    prompts = record["prompts"]
    pairs = [
        (prompt, l1_text)
        for prompt, l1_text in zip(prompts, record["l1_context_texts"])
        if L1_HEADER in prompt and "[L3 RGB-D map;" in prompt
    ]
    assert pairs, "C never injected the original L1 card with the L3 block"
    prompt, l1_text = pairs[0]
    assert "Current RGB: instruction landmark" in prompt
    assert "L1 RGB fact" not in prompt
    assert "L1 RGB observation" not in prompt
    assert "L1 RGB observation also supports" not in prompt
    # The card is the exact text selected by the original Local Graph builder.
    assert l1_text is not None
    assert l1_text.startswith(L1_HEADER)
    assert l1_text in prompt
    budget = record["token_budget"]
    if budget["budget_source"] == "token_aware_context":
        assert budget["memory_tokens_injected"] > 0
        assert budget["memory_tokens_injected"] <= budget["available_memory_tokens"]
    else:
        assert len(l1_text) <= 320
    assert record["summary"]["l1_prompt_built_steps"] >= 1
    assert record["summary"]["prompt_has_l1_steps"] >= 1
    assert record["summary"]["l1_retained_facts_total"] == (
        record["summary"]["l1_fact_count_total"]
    )


def test_c_does_not_require_an_l1_fact_on_every_step():
    record = _probe("c")
    summary = record["summary"]
    assert summary["env_steps"] == 2
    assert summary["l1_enabled_steps"] == 2
    assert summary["l1_prompt_built_steps"] >= 1
    assert summary["l1_empty_steps"] >= 0
    assert summary["l1_observe_error_count"] == 0
    assert summary["l1_detector_error_count"] == 0


def test_c_agent_survives_an_episode_with_no_l1_fact():
    record = _probe("c-empty")
    summary = record["summary"]
    assert summary["l1_enabled"] is True
    assert summary["l1_enabled_steps"] == summary["env_steps"] == 2
    assert summary["l1_fact_count_total"] == 0
    assert summary["l1_retained_facts_total"] == 0
    assert summary["l1_prompt_built_steps"] == 0
    assert summary["prompt_has_l1_steps"] == 0
    assert summary["l1_empty_steps"] == 2
    assert summary["l1_observe_error_count"] == 0
    assert summary["l1_selection_error_count"] == 0
    assert summary["l1_detector_error_count"] == 0
    assert summary["l1_place_error_count"] == 0
    assert L1_HEADER not in "\n".join(record["prompts"])
    budget = record["token_budget"]
    assert budget["budget_source"] == "token_aware_context"
    assert budget["budget_fallback_reason"] is None
    assert budget["memory_tokens_injected"] == 0
    assert budget["base_prompt_tokens"] > 0
    assert budget["base_prompt_tokens"] == (
        budget["text_prompt_tokens"] + budget["visual_tokens"]
    )
    assert budget["multimodal_prompt_tokens"] == budget["base_prompt_tokens"]
    assert budget["context_margin_tokens"] >= 0


def test_c_token_budget_is_measured_from_the_real_uninavid_prompt():
    record = _probe("c")
    budget = record["token_budget"]
    assert budget["budget_source"] == "token_aware_context"
    assert budget["budget_fallback_reason"] is None
    assert budget["context_limit_tokens"] == 4096
    assert isinstance(budget["base_prompt_tokens"], int)
    assert budget["base_prompt_tokens"] > 0
    assert budget["base_prompt_tokens"] == (
        budget["text_prompt_tokens"] + budget["visual_tokens"]
    )
    assert budget["visual_tokens"] > 0
    assert budget["visual_token_accounting"]["visual_tokens_net"] == (
        budget["visual_tokens"]
    )
    assert budget["visual_token_accounting"]["exact"] is False
    assert budget["multimodal_prompt_tokens_is_upper_bound"] is True
    assert budget["available_memory_tokens"] == (
        budget["context_limit_tokens"]
        - budget["base_prompt_tokens"]
        - 1024
        - 1
    )
    assert budget["memory_tokens_injected"] > 0
    assert budget["memory_tokens_injected"] <= budget["available_memory_tokens"]
    assert budget["token_overflow_drops"] == 0
    limit = budget["context_limit_tokens"] - 1024 - 1
    assert budget["multimodal_prompt_tokens"] == (
        budget["base_prompt_tokens"] + budget["memory_tokens_injected"]
    )
    assert budget["multimodal_prompt_tokens"] <= limit
    assert budget["context_margin_tokens"] >= 0
    assert budget["multimodal_budget_overflow"] is False
    summary = record["summary"]
    assert summary["l1_token_budget_steps"] == summary["env_steps"]
    assert summary["l1_token_budget_fallback_steps"] == 0
    assert summary["available_memory_tokens_min"] == (
        budget["available_memory_tokens"]
    )
    assert summary["budget_source"] == "token_aware_context"
    assert summary["memory_budget_violation_steps"] == 0
    assert summary["visual_tokens_mean"] > 0
    assert summary["context_margin_tokens_min"] >= 0
    assert summary["multimodal_budget_violation_steps"] == 0


def test_c_falls_back_to_the_original_320_character_card_with_a_reason():
    record = _probe("c-fallback")
    budget = record["token_budget"]
    assert budget["budget_source"] == "char_fallback_320"
    assert "prompt_measurement_failed" in budget["budget_fallback_reason"]
    assert budget["available_memory_tokens"] is None
    assert budget["base_prompt_tokens"] is None
    assert budget["memory_tokens_injected"] is None
    assert budget["visual_tokens"] is None
    assert budget["multimodal_prompt_tokens"] is None
    assert record["summary"]["l1_token_budget_steps"] == 0
    assert record["summary"]["l1_token_budget_fallback_steps"] == 2
    assert record["summary"]["budget_source"] == "char_fallback_320"
    l1_cards = [
        text for text in record["l1_context_texts"]
        if text and text.startswith(L1_HEADER)
    ]
    assert l1_cards, "fallback did not build the original L1 card"
    assert all(len(text) <= 320 for text in l1_cards)
    assert all(text in prompt for text, prompt in zip(l1_cards, record["prompts"]))


def test_a_and_b_do_not_emit_the_l1_token_budget_diagnostics():
    for mode in ("a", "b"):
        record = _probe(mode)
        assert all(value is None for value in record["token_budget"].values())
        assert record["summary"].get("l1_token_budget_steps") is None
        assert record["summary"].get("memory_tokens_injected_mean") is None
        assert record["summary"].get("budget_source") is None


def test_c_falls_back_when_visual_token_accounting_is_unavailable():
    record = _probe("c-visual-fallback")
    budget = record["token_budget"]
    assert budget["budget_source"] == "char_fallback_320"
    assert "visual_token_accounting_failed" in budget["budget_fallback_reason"]
    assert budget["visual_tokens"] is None
    assert budget["multimodal_prompt_tokens"] is None
    l1_cards = [
        text for text in record["l1_context_texts"]
        if text and text.startswith(L1_HEADER)
    ]
    assert l1_cards
    assert all(len(text) <= 320 for text in l1_cards)
    assert record["summary"]["l1_token_budget_fallback_steps"] == 2


def test_c_no_longer_injects_when_the_multimodal_base_already_overflows():
    record = _probe("c-tight")
    budget = record["token_budget"]
    assert budget["budget_source"] == "token_aware_context"
    assert budget["available_memory_tokens"] == 0
    assert budget["memory_tokens_injected"] == 0
    assert budget["multimodal_prompt_tokens"] == budget["base_prompt_tokens"]
    assert budget["context_margin_tokens"] < 0
    assert budget["multimodal_budget_overflow"] is True
    assert record["summary"]["prompt_has_l1_steps"] == 0
    assert L1_HEADER not in "\n".join(record["prompts"])

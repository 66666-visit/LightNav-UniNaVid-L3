"""Agent-level token accounting and zero-fact diagnosis regressions.

The GPU smoke reported ``memory_tokens_injected > 0`` together with
``context_tokens == 0`` and zero facts.  These tests exercise the real
``UniNaVid_Agent.act`` path (with CPU stand-ins for the checkpoint) and pin the
two different definitions:

* ``context_tokens`` -- tokenizer count of the exact injected memory text;
* ``memory_tokens_injected`` -- measured delta of the assembled official prompt.

They also pin the diagnostic that distinguishes "nothing was detected" from
"detections were written but no instruction-relevant fact was eligible".
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from l3_integration.official_prompt import REFERENCE_OFFICIAL_PROMPT_TEMPLATE
from tests.helpers import (
    FakeEpisode,
    FakeExtractor,
    FakeL1Extractor,
    FakePlaceExtractor,
    agent_module_stubs,
    chair_instance,
    make_memory_backend,
)
from tests.test_agent_memory_contract import _build_agent

BOX = (1.0, 1.0, 6.0, 6.0)
RGB = np.zeros((8, 8, 3), np.uint8)


def _agent_with_detectors(
    agent_module, *, instruction: str, l1_label: str = "chair",
    l1_confidence: float = 0.9, seg_label: str = "chair",
    seg_confidence: float = 0.9, context_len: int = 8192,
):
    backend = make_memory_backend(
        "l1_l3_joint",
        l1=FakeL1Extractor(
            by_step={step: [(l1_label, l1_confidence, BOX)] for step in range(8)}
        ),
        seg=FakeExtractor(
            [chair_instance(label=seg_label, confidence=seg_confidence)]
        ),
        place=FakePlaceExtractor(),
    )
    agent = _build_agent(agent_module, "l1_l3_joint", context_len=context_len)
    agent.memory = backend
    agent.memory_prompt_mode = backend.plan.mode
    agent.episode_instruction = instruction
    backend.reset_episode(FakeEpisode(instruction=instruction), instruction)
    return agent


def _step(agent, instruction: str):
    observations = {
        "rgb": RGB,
        "instruction": {"text": instruction},
    }
    return agent.act(observations, {}, "episode")


def _assert_token_counts(agent, row) -> None:
    text = row["memory_context_text"]
    assert isinstance(text, str) and text
    expected_context_tokens = len(
        agent.tokenizer.encode(text, add_special_tokens=False)
    )
    assert row["context_tokens"] == expected_context_tokens
    assert row["context_tokens"] > 0
    assert row["context_chars"] == len(text)
    base_tokens = row["token_budget"]["base_prompt_tokens"]
    assert row["prompt_tokens"] - base_tokens == row["memory_tokens_injected"]


def test_empty_fact_joint_block_is_a_fixed_hint_with_real_token_counts(monkeypatch):
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        instruction = "find the sofa"
        agent = _agent_with_detectors(
            agent_module, instruction=instruction,
            l1_label="chair", seg_label="chair",
        )
        _step(agent, instruction)
        _step(agent, instruction)
        row = agent.step_diagnostics[-1]

        assert row["context_injected"] is True
        assert row["context_has_facts"] is False
        assert row["prompt_has_l1_fact"] is False
        assert row["prompt_has_l3_fact"] is False
        assert row["memory_block_kind"] == "fixed_hint"
        assert row["fact_count"] == 0
        assert row["l1_fact_count"] == 0
        assert row["l3_fact_count"] == 0
        assert row["evidence_ids"] == []
        assert row["l1_evidence_ids"] == []
        assert row["l3_evidence_ids"] == []
        assert row["memory_tokens_injected"] > 0
        assert "No instruction-related" in row["memory_context_text"]
        assert "chair" not in row["memory_context_text"]
        _assert_token_counts(agent, row)

        # The map exists, but it contributed no prompt candidate.
        assert row["retrieved_nodes"] == 0
        assert row["map_object_nodes"] >= 1
        assert row["l3_map_object_count"] >= 1
        assert row["l3_relevance_rejected"] >= 1
        assert row["l1_rejected_irrelevant"] >= 1
        assert row["joint_l3_facts_considered"] == 0
        assert row["joint_attempted_pairs"] == 0


def test_relevant_fact_propagates_and_token_definitions_are_consistent(monkeypatch):
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        instruction = "find the chair"
        agent = _agent_with_detectors(agent_module, instruction=instruction)
        _step(agent, instruction)
        _step(agent, instruction)
        row = agent.step_diagnostics[-1]

        assert row["context_injected"] is True
        assert row["context_has_facts"] is True
        assert row["prompt_has_l1_fact"] is True
        assert row["prompt_has_l3_fact"] is True
        assert row["memory_block_kind"] == "facts"
        assert row["l1_fact_count"] >= 1
        assert row["l3_fact_count"] >= 1
        assert row["fact_count"] >= 1
        assert row["retrieved_nodes"] >= 1
        assert row["map_object_nodes"] >= 1
        assert row["joint_l3_facts_considered"] >= 1
        assert row["joint_attempted_pairs"] >= 1
        assert row["joint_aligned_pairs"] >= 1
        assert "chair" in row["memory_context_text"]
        assert "chair" in agent.last_prompt_text
        assert agent.last_prompt_text.startswith(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE.split("{}")[0] + instruction
        )
        assert agent.last_prompt_text.endswith(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE.split("{}")[1]
        )
        _assert_token_counts(agent, row)


def test_l3_fact_without_l1_counterpart_has_zero_attempted_pairs(monkeypatch):
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        instruction = "find the chair"
        agent = _agent_with_detectors(
            agent_module, instruction=instruction,
            l1_confidence=0.4, seg_confidence=0.9,
        )
        _step(agent, instruction)
        _step(agent, instruction)
        row = agent.step_diagnostics[-1]

        assert row["context_has_facts"] is True
        assert row["l1_fact_count"] == 0
        assert row["l3_fact_count"] >= 1
        assert row["l1_rejected_below_confidence"] >= 1
        assert row["joint_l3_facts_considered"] == 1
        assert row["joint_candidate_pairs"] == 0
        assert row["joint_attempted_pairs"] == 0
        assert row["joint_aligned_pairs"] == 0
        assert row["joint_alignment_success_rate"] is None
        _assert_token_counts(agent, row)

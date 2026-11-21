"""The real agent path with the vendored L1/L3 backend and the official prompt."""

from __future__ import annotations

import numpy as np
import torch

from l3_integration.official_prompt import (
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
    prompt_sha256,
    update_prompt_view,
    verify_task_text_injection,
)
from tests.helpers import (
    FakeEpisode,
    agent_module_stubs,
    make_memory_backend,
    synthetic_frame,
)

OFFICIAL_PREFIX, OFFICIAL_SUFFIX = REFERENCE_OFFICIAL_PROMPT_TEMPLATE.split("{}")


def _build_agent(agent_module, group: str, *, context_len=8192):
    from tests.test_prompt_tokens import (
        IMAGE_TOKEN_INDEX,
        CharTokenizer,
        FakeConv,
        FakeModel,
        FakeStoppingCriteria,
        _tokenizer_image_token,
    )

    agent_module.DEFAULT_IMAGE_TOKEN = "<image>"
    agent_module.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
    agent_module.tokenizer_image_token = _tokenizer_image_token
    agent_module.KeywordsStoppingCriteria = FakeStoppingCriteria
    agent_module.conv_templates = {"vicuna_v1": FakeConv()}

    backend = make_memory_backend(group)
    backend.reset_episode(FakeEpisode(), "find the chair")
    agent = object.__new__(agent_module.UniNaVid_Agent)
    agent.l3 = None
    agent.memory = backend
    agent.env = None
    agent.config = None
    agent.camera_spec = None
    agent.experiment_group = group
    agent.memory_prompt_mode = backend.plan.mode
    agent.memory_token_budget_enabled = True
    agent.result_path = "/tmp"
    agent.require_map = False
    agent.require_data = False
    agent.promt_template = REFERENCE_OFFICIAL_PROMPT_TEMPLATE
    agent.prompt_template_sha256 = prompt_sha256(agent.promt_template)
    agent.conv_mode = "vicuna_v1"
    agent.rgb_list = []
    agent.topdown_map_list = []
    agent.pending_action_list = []
    agent.step_index = 0
    agent.step_diagnostics = []
    agent.inference_latencies_ms = []
    agent.prompt_token_counts = []
    agent.last_prompt_tokens = None
    agent.last_inference_ms = None
    agent.last_prompt_text = None
    agent.last_task_text = None
    agent.last_l3_context_text = None
    agent.last_memory_context_text = None
    agent.last_memory_diagnostics = None
    agent.last_memory_token_budget = None
    agent.last_executed_action = None
    agent.last_prediction = None
    agent.episode_instruction = "find the chair"
    agent.episode_id = "episode"
    agent._steps_stream = None
    agent.tokenizer = CharTokenizer()
    agent.model = FakeModel(action_text="forward")
    agent.context_len = context_len
    agent.last_memory_frame = None
    agent.memory_observe_ms = []
    agent.memory_l3_observe_ms = []
    agent.memory_retrieval_ms = []
    agent.memory_prompt_build_ms = []
    agent.memory_token_budget_ms = []
    agent.memory_injection_flags = []
    agent.memory_omitted_items = []
    agent.memory_token_overflow_drops = []
    agent.memory_duplicates_removed = []
    agent.memory_conflict_counts = []
    agent.memory_l1_fact_counts = []
    agent.memory_l3_fact_counts = []
    agent.memory_retrieved_node_counts = []
    agent.memory_evidence_ids = []
    agent.process_images = lambda rgb: ["IMAGE"]
    agent._build_frame = lambda observations: (synthetic_frame(), None)
    return agent


def _step(agent):
    observations = {
        "rgb": np.zeros((8, 8, 3), np.uint8),
        "instruction": {"text": "find the chair"},
    }
    return agent.act(observations, {}, "episode")


def test_memory_text_is_inside_the_official_task_slot(monkeypatch):
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        agent = _build_agent(agent_module, "l1_l3_joint")
        for _ in range(3):
            _step(agent)
        memory_text = agent.last_memory_context_text
        assert memory_text
        assert agent.last_prompt_text == (
            OFFICIAL_PREFIX + "find the chair\n" + memory_text + OFFICIAL_SUFFIX
        )
        verify_task_text_injection(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE, "find the chair",
            memory_text, agent.last_prompt_text,
        )
        assert agent.last_prompt_text.count(OFFICIAL_PREFIX) == 1
        assert agent.last_prompt_text.count(OFFICIAL_SUFFIX) == 1
        assert memory_text.replace("\n", "") in update_prompt_view(
            agent.last_prompt_text
        )
        # No second role, turn or navigation token is introduced in the text.
        assert "[Navigation]" not in agent.last_prompt_text
        assert "SYSTEM:" not in agent.last_prompt_text


def test_memory_budget_and_prompt_tokens_are_recorded(monkeypatch):
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        agent = _build_agent(agent_module, "l1_l3_joint")
        _step(agent)
        _step(agent)
        row = agent.step_diagnostics[-1]
        assert row["token_budget"]["budget_source"] == "token_aware_context"
        assert row["l1_evidence_ids"]
        assert row["l3_evidence_ids"]
        assert row["prompt_tokens"] > row["token_budget"]["base_prompt_tokens"]
        assert row["memory_tokens_injected"] >= 0
        assert row["context_injected"] is True
        assert row["l1_fact_count"] >= 1
        summary = agent.episode_summary()
        assert summary["prompt_injection_rate"] > 0
        assert summary["prompt_tokens_mean"] is not None
        assert summary["l1_observe_ms_mean"] is not None
        assert summary["prompt_build_ms_mean"] is not None
        assert summary["token_budget_ms_mean"] is not None
        assert summary["injected_evidence_id_count"] >= 1


def test_budget_measurement_falls_back_with_a_reason(monkeypatch):
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        agent = _build_agent(agent_module, "l1_original_only", context_len=0)
        _step(agent)
        row = agent.step_diagnostics[-1]
        assert row["token_budget"]["budget_source"] == "char_fallback_320"
        assert "context_limit_unavailable" in row["token_budget"][
            "token_budget_unavailable_reason"
        ]
        assert row["context_injected"] is True


def test_token_overflow_drops_whole_lines_without_a_partial_block(monkeypatch):
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        agent = _build_agent(agent_module, "l1_l3_joint", context_len=1500)
        _step(agent)
        _step(agent)
        row = agent.step_diagnostics[-1]
        assert row["token_budget"]["budget_source"] == "token_aware_context"
        if row["context_injected"]:
            assert row["memory_tokens_injected"] <= row["token_budget"][
                "available_memory_tokens"
            ]
        else:
            assert agent.last_prompt_text == (
                REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format("find the chair")
            )


def test_l1_only_and_l3_only_prompts_have_no_cross_layer_text(monkeypatch):
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        l1_agent = _build_agent(agent_module, "l1_organized_only")
        _step(l1_agent)
        _step(l1_agent)
        l1_prompt = l1_agent.last_prompt_text
        assert "[Navigation Context - L1]" in l1_prompt
        assert "Spatial history" not in l1_prompt

        l3_agent = _build_agent(agent_module, "l3_organized_only")
        _step(l3_agent)
        _step(l3_agent)
        _step(l3_agent)
        l3_prompt = l3_agent.last_prompt_text
        assert "map facts are history" in l3_prompt
        assert "[Navigation Context - L1]" not in l3_prompt
        assert "Now:" not in l3_prompt


def test_history_then_current_rgb_frames_keep_their_arrival_order(monkeypatch):
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        agent = _build_agent(agent_module, "l1_original_only")
        seen = []

        def process_images(rgb_list):
            seen.append([frame.copy() for frame in rgb_list])
            return ["IMAGE"]

        agent.process_images = process_images
        first = np.full((8, 8, 3), 1, dtype=np.uint8)
        second = np.full((8, 8, 3), 2, dtype=np.uint8)
        for frame in (first, second):
            agent.act(
                {"rgb": frame, "instruction": {"text": "find the chair"}},
                {}, "episode",
            )
        assert len(seen) == 2
        # Uni-NaVid keeps the historical video in the official online KV cache
        # (``initialize_online_inference_nav_feat_cache``); ``rgb_list`` carries
        # only the frames observed since the previous generation, in arrival
        # order, with the current frame last.  This path is unchanged.
        assert [frame[0, 0, 0] for frame in seen[0]] == [1]
        assert [frame[0, 0, 0] for frame in seen[1]] == [2]
        assert seen[1][-1][0, 0, 0] == 2


def test_step_log_records_prompt_facts_budget_and_timings(
    monkeypatch, tmp_path,
) -> None:
    import json

    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        agent = _build_agent(agent_module, "l1_l3_joint")
        stream = (tmp_path / "steps.jsonl").open("w", encoding="utf-8")
        agent._steps_stream = stream
        agent.require_data = True
        try:
            _step(agent)
            _step(agent)
        finally:
            stream.close()
        rows = [
            json.loads(line)
            for line in (tmp_path / "steps.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 2
        row = rows[-1]
        for key in (
            "episode_id", "experiment_group", "step", "prompt_tokens",
            "prompt_sha256", "memory_prompt_mode", "context_injected",
            "l1_fact_count", "l3_fact_count", "evidence_ids",
            "l1_evidence_ids", "l3_evidence_ids", "l1_fact_ids",
            "l3_fact_ids", "retrieved_nodes", "duplicates_removed",
            "conflict_count", "omitted_items", "token_overflow_drops",
            "l1_observe_ms", "l3_observe_ms", "prompt_build_ms",
            "token_budget_ms", "token_budget",
        ):
            assert key in row, key
        assert row["episode_id"] == "episode"
        assert row["experiment_group"] == "l1_l3_joint"
        assert row["token_budget"]["budget_source"] == "token_aware_context"
        assert row["l1_evidence_ids"]
        assert row["l3_evidence_ids"]
        assert row["l1_evidence_ids"]
        assert row["l3_evidence_ids"]

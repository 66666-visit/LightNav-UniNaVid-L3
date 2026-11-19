"""Requirement 8: official parsing/action execution stays unchanged with L3."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch

from l3_integration.official_prompt import (
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
    update_prompt_view,
)
from tests.helpers import agent_module_stubs

BLOCK = (
    "[Navigation Map Memory - L3; history, not current view; directions "
    "relative to you; straight-line ranges, not paths; passage unverified]\n"
    "Map record (step 0, 1 view): the chair is in front of you, 2.0 m straight-line."
)


def _observations(
    *, rgb=None, instruction: str = "find the chair",
) -> dict:
    return {
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8) if rgb is None else rgb,
        "instruction": {"text": instruction},
    }


class FakeL3:
    def __init__(self) -> None:
        self.observed = []
        self.context_calls = []

    def observe(self, observations, state, step, *, transition_record=False):
        self.observed.append((step, transition_record))
        return {
            "frame_valid": True, "segmentation_count": 1, "evidence_minted": 1,
            "evidence_unmatched": 1, "object_nodes": 1, "viewpoints": 1,
            "observe_ms": 1.0, "segmentation_ms": 0.5, "frame_error": None,
        }

    def build_context(self, instruction, step):
        self.context_calls.append((instruction, step))
        return SimpleNamespace(context=SimpleNamespace(text=BLOCK)), {
            "context_injected": True, "fact_count": 1,
            "context_chars": len(BLOCK), "retrieval_ms": 0.25,
            "candidate_count": 1, "verified_count": 1,
            "unverified_count": 0, "current_only_skipped": 0,
            "omitted_items": 0, "empty_reason": None,
            "expression": {"strategy_counts": {"direction": 1}},
        }

    def map_snapshot(self):
        return {"objects": 1, "viewpoints": 1, "evidence": 1, "edges": 1}

    def close_episode(self):
        pass


def _fake_agent(agent_module, *, l3=None, predictions=None):
    agent = object.__new__(agent_module.UniNaVid_Agent)
    agent.l3 = l3
    agent.env = None
    agent.config = None
    agent.experiment_group = "uninavid_l3_only" if l3 else "uninavid_baseline"
    agent.result_path = "/tmp"
    agent.require_map = False
    agent.require_data = False
    agent.promt_template = REFERENCE_OFFICIAL_PROMPT_TEMPLATE
    agent.prompt_template_sha256 = (
        "435d79868156aaa1352afcd98f4ceae2f03a54f8cdc5fe87e41d32774d5512c1"
    )
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
    agent.last_l3_context_text = None
    agent.episode_instruction = ""
    agent._steps_stream = None
    agent.tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: list(range(len(text.split())))
    )
    agent.episode_id = None
    calls = list(predictions or ["forward left stop"])
    captured = []

    def predict(prompt):
        captured.append(prompt)
        agent.last_prompt_tokens = len(prompt.split())
        return calls.pop(0) if calls else "stop"

    agent.predict_inference = predict
    agent.captured_prompts = captured
    return agent


def test_extract_result_is_unchanged() -> None:
    with agent_module_stubs() as agent_module:
        agent = _fake_agent(agent_module)
        assert agent.extract_result("stop") == (0, None)
        assert agent.extract_result("forward") == (None, None)
        assert agent.extract_result("forward 25") == (1, 25.0)
        assert agent.extract_result("left -30") == (2, -30.0)
        assert agent.extract_result("right 15") == (3, 15.0)
        assert agent.extract_result("turn around") == (None, None)
        assert agent.extract_result("forward") == (None, None)


def test_pending_actions_are_executed_without_a_new_inference() -> None:
    with agent_module_stubs() as agent_module:
        agent = _fake_agent(agent_module)
        first = agent.act(_observations(), {}, "episode")
        assert first == {"action": 1}
        assert agent.pending_action_list == [2]
        assert len(agent.captured_prompts) == 1
        second = agent.act(_observations(), {}, "episode")
        assert second == {"action": 2}
        assert agent.pending_action_list == []
        assert len(agent.captured_prompts) == 1
        assert agent.step_index == 2


def test_baseline_prompt_is_exactly_the_official_template() -> None:
    with agent_module_stubs() as agent_module:
        agent = _fake_agent(agent_module)
        agent.act(_observations(), {}, "episode")
        assert agent.captured_prompts[0] == (
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format("find the chair")
        )
        assert agent.captured_prompts[0].count("[Navigation Map Memory") == 0


def test_l3_block_is_inside_the_task_slot_and_reaches_both_prompt_views() -> None:
    with agent_module_stubs() as agent_module:
        sidecar = FakeL3()
        agent = _fake_agent(agent_module, l3=sidecar)
        agent.act(_observations(), {}, "episode")
        prompt = agent.captured_prompts[0]
        prefix, suffix = REFERENCE_OFFICIAL_PROMPT_TEMPLATE.split("{}")
        assert prompt == prefix + "find the chair\n" + BLOCK + suffix
        assert prompt.startswith(prefix + "find the chair")
        assert prompt.endswith(suffix)
        assert BLOCK in prompt
        assert not prompt.endswith(BLOCK)
        assert BLOCK.replace("\n", "") in update_prompt_view(prompt)
        assert sidecar.observed == [(0, False)]
        assert sidecar.context_calls == [("find the chair", 0)]


def test_l3_is_updated_even_when_a_pending_action_is_returned() -> None:
    with agent_module_stubs() as agent_module:
        sidecar = FakeL3()
        agent = _fake_agent(agent_module, l3=sidecar)
        agent.act(_observations(), {}, "episode")
        agent.act(_observations(), {}, "episode")
        assert [step for step, _ in sidecar.observed] == [0, 1]
        assert sidecar.observed[1][1] is True
        assert len(sidecar.context_calls) == 1


def test_episode_summary_reports_latency_tokens_and_l3_facts() -> None:
    with agent_module_stubs() as agent_module:
        sidecar = FakeL3()
        agent = _fake_agent(agent_module, l3=sidecar)
        agent.act(_observations(), {}, "episode")
        agent.act(_observations(), {}, "episode")
        summary = agent.episode_summary()
        assert summary["l3_enabled"] is True
        assert summary["env_steps"] == 2
        assert summary["inference_calls"] == 1
        assert summary["l3_context_steps"] == 1
        assert summary["l3_fact_count_total"] == 1
        assert summary["prompt_has_l3_steps"] == 1
        assert summary["inference_latency_ms_mean"] is not None
        assert summary["prompt_tokens_max"] is not None
        assert summary["l3_object_nodes_final"] == 1


def test_invalid_l3_context_fails_closed_to_the_official_prompt() -> None:
    class InvalidL3(FakeL3):
        def build_context(self, instruction, step):
            organization, diagnostic = super().build_context(instruction, step)
            organization.context.text = None
            return organization, diagnostic

    with agent_module_stubs() as agent_module:
        agent = _fake_agent(agent_module, l3=InvalidL3())
        agent.act(_observations(), {}, "episode")
        assert agent.captured_prompts[0] == (
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format("find the chair")
        )
        assert agent.last_l3_context_text is None
        assert agent.step_diagnostics[0]["prompt_has_l3"] is False


class EpisodeL3(FakeL3):
    def __init__(self) -> None:
        super().__init__()
        self.episode = None
        self.reset_calls = []
        self.block = None

    def reset_episode(self, episode, instruction):
        self.episode = episode
        self.reset_calls.append((episode.episode_id, instruction))
        self.block = (
            "[Navigation Map Memory - L3; history, not current view]\n"
            f"Map record for {episode.episode_id}."
        )

    def build_context(self, instruction, step):
        self.context_calls.append((instruction, step))
        return SimpleNamespace(
            context=SimpleNamespace(text=self.block),
        ), {
            "context_injected": True, "fact_count": 1,
            "context_chars": len(self.block), "retrieval_ms": 0.1,
            "candidate_count": 1, "verified_count": 1,
            "unverified_count": 0, "current_only_skipped": 0,
            "omitted_items": 0, "empty_reason": None, "expression": None,
        }


def test_agent_reset_isolates_the_l3_task_text_between_episodes() -> None:
    with agent_module_stubs() as agent_module:
        sidecar = EpisodeL3()
        agent = _fake_agent(agent_module, l3=sidecar)
        model = SimpleNamespace(
            config=SimpleNamespace(run_type="eval"),
            get_model=lambda: SimpleNamespace(
                initialize_online_inference_nav_feat_cache=lambda: None,
                new_frames=0,
            ),
        )
        agent.model = model
        agent.count_id = 0

        agent.reset(
            episode=SimpleNamespace(episode_id="episode-a"),
            instruction="find the chair",
        )
        agent.act(_observations(), {}, "episode-a")
        first_prompt = agent.captured_prompts[0]
        assert "Map record for episode-a" in first_prompt
        assert "episode-b" not in first_prompt

        agent.reset(
            episode=SimpleNamespace(episode_id="episode-b"),
            instruction="find the chair",
        )
        assert agent.last_l3_context_text is None
        agent.act(_observations(), {}, "episode-b")
        second_prompt = agent.captured_prompts[1]
        assert "Map record for episode-b" in second_prompt
        assert "episode-a" not in second_prompt
        assert sidecar.reset_calls == [
            ("episode-a", "find the chair"),
            ("episode-b", "find the chair"),
        ]


def test_opt_in_prompt_audit_log_records_the_exact_prompt(monkeypatch) -> None:
    import io
    import json

    class Stream(io.StringIO):
        closed = False

        def close(self):
            self.closed = True

    with agent_module_stubs() as agent_module:
        agent = _fake_agent(agent_module, l3=FakeL3())
        stream = Stream()
        agent._steps_stream = stream
        monkeypatch.setenv("L3_LOG_PROMPTS", "1")
        agent.act(_observations(), {}, "episode")
        row = json.loads(stream.getvalue().splitlines()[0])
        assert row["prompt_text"] == agent.captured_prompts[0]
        assert row["prompt_has_l3"] is True
        assert row["task_text"].startswith("find the chair\n")
        assert BLOCK in row["task_text"]
        assert row["l3_context_text"] == BLOCK


class RngConsumingL3(FakeL3):
    def observe(self, observations, state, step, *, transition_record=False):
        random.random()
        np.random.rand()
        torch.rand(1)
        return super().observe(
            observations, state, step, transition_record=transition_record,
        )

    def build_context(self, instruction, step):
        random.random()
        np.random.rand()
        torch.rand(1)
        return super().build_context(instruction, step)


def test_l3_perception_cannot_advance_the_model_rng() -> None:
    with agent_module_stubs() as agent_module:
        agent = _fake_agent(agent_module, l3=RngConsumingL3())
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        agent.act(_observations(), {}, "episode")
        assert random.getstate() == python_before
        after_numpy = np.random.get_state()
        assert after_numpy[0] == numpy_before[0]
        assert np.array_equal(after_numpy[1], numpy_before[1])
        assert after_numpy[2:] == numpy_before[2:]
        assert torch.equal(torch.get_rng_state(), torch_before)


class RaisingContextL3(FakeL3):
    def build_context(self, instruction, step):
        raise RuntimeError("sidecar exploded")


class MissingContextL3(FakeL3):
    def build_context(self, instruction, step):
        return SimpleNamespace(), {"context_injected": True}


class MalformedDiagnosticL3(FakeL3):
    def build_context(self, instruction, step):
        return SimpleNamespace(context=SimpleNamespace(text=BLOCK)), "not-a-dict"


def test_malformed_l3_contexts_fail_closed_without_crashing() -> None:
    with agent_module_stubs() as agent_module:
        for sidecar in (
            RaisingContextL3(), MissingContextL3(), MalformedDiagnosticL3(),
        ):
            agent = _fake_agent(agent_module, l3=sidecar)
            assert agent.act(_observations(), {}, "episode") == {"action": 1}
            assert agent.captured_prompts[0] == (
                REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format("find the chair")
            )
            assert agent.last_l3_context_text is None
            assert agent.step_diagnostics[0]["prompt_has_l3"] is False


class RngResetL3(FakeL3):
    def reset_episode(self, episode, instruction):
        random.random()
        np.random.rand()
        torch.rand(1)


def test_l3_reset_cannot_advance_the_model_rng() -> None:
    with agent_module_stubs() as agent_module:
        agent = _fake_agent(agent_module, l3=RngResetL3())
        agent.model = SimpleNamespace(
            config=SimpleNamespace(run_type="eval"),
            get_model=lambda: SimpleNamespace(
                initialize_online_inference_nav_feat_cache=lambda: None,
                new_frames=0,
            ),
        )
        agent.count_id = 0
        random.seed(777)
        np.random.seed(777)
        torch.manual_seed(777)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        agent.reset(
            episode=SimpleNamespace(episode_id="episode-a"),
            instruction="find the chair",
        )
        assert random.getstate() == python_before
        after_numpy = np.random.get_state()
        assert after_numpy[0] == numpy_before[0]
        assert np.array_equal(after_numpy[1], numpy_before[1])
        assert after_numpy[2:] == numpy_before[2:]
        assert torch.equal(torch.get_rng_state(), torch_before)


class BadObserveL3(FakeL3):
    def observe(self, observations, state, step, *, transition_record=False):
        return "not-a-diagnostic"


def test_invalid_observe_diagnostic_fails_closed_without_crashing() -> None:
    with agent_module_stubs() as agent_module:
        agent = _fake_agent(agent_module, l3=BadObserveL3())
        assert agent.act(_observations(), {}, "episode") == {"action": 1}
        row = agent.step_diagnostics[0]
        assert row["frame_valid"] is False
        assert row["frame_error"] == "invalid L3 observe diagnostic"

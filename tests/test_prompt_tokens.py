"""Requirement 4/5: the real token assembly keeps its visual/navigation layout.

The CPU environment has no Uni-NaVid checkpoint or Vicuna tokenizer.  These
tests therefore use a deterministic invertible character tokenizer and execute
the **real** ``UniNaVid_Agent.predict_inference`` with stand-ins for the model,
conversation template and image processor.  The official token assembly is also
reproduced verbatim as an oracle, so the baseline path is compared against the
official algorithm rather than against itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from l3_integration.official_prompt import (
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
    build_official_prompt,
    build_prompt_with_l3,
)
from tests.helpers import FakeEpisode, agent_module_stubs, make_sidecar

BASE = 1000
BOS = 1
IMAGE_TOKEN_INDEX = -200
L3_BLOCK = (
    "[Navigation Map Memory - L3; history, not current view; directions "
    "relative to you; straight-line ranges, not paths; passage unverified]\n"
    "Map record (step 0, 1 view): the chair is in front of you, 2.0 m straight-line."
)
SPECIAL_TOKENS = (
    "<video_special>",
    "<image_sep>",
    "</video_special>",
    "<image_special>",
    "</image_special>",
    "[Navigation]",
)


class CharTokenizer:
    """Invertible character tokenizer with a BOS id, like a sentencepiece model."""

    bos_token_id = BOS

    def __call__(self, text, return_tensors=None):
        ids = [BOS] + [BASE + ord(char) for char in text]
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))
        return SimpleNamespace(input_ids=ids)

    def encode(self, text, add_special_tokens=False):
        return [BASE + ord(char) for char in text]

    def batch_decode(self, sequences, skip_special_tokens=True):
        rows = sequences.tolist() if hasattr(sequences, "tolist") else sequences
        return [
            "".join(
                chr(value - BASE) for value in row
                if isinstance(value, int) and value >= BASE
            )
            for row in rows
        ]


class FakeConv:
    roles = ("USER", "ASSISTANT")
    sep = "</s>"
    sep_style = "one"

    def __init__(self, messages=None):
        self.messages = list(messages or [])

    def copy(self):
        return FakeConv(self.messages)

    def append_message(self, role, message):
        self.messages.append((role, message))

    def get_prompt(self):
        return "".join(
            f"{role}: {'' if message is None else message}\n"
            for role, message in self.messages
        )


class FakeStoppingCriteria:
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.input_ids = input_ids


class FakeModel:
    def __init__(self, action_text="forward left stop"):
        self.config = SimpleNamespace(mm_use_im_start_end=False, run_type="eval")
        self.action_ids = [BASE + ord(char) for char in action_text]
        self.prompts = None
        self.captured = None

    def get_model(self):
        return SimpleNamespace(
            initialize_online_inference_nav_feat_cache=lambda: None,
            new_frames=0,
        )

    def update_prompt(self, prompts=None):
        self.prompts = prompts

    def generate(self, input_ids, images=None, **kwargs):
        self.captured = {
            "input_ids": input_ids.clone(),
            "images": images,
            "kwargs": dict(kwargs),
        }
        action = torch.tensor([self.action_ids], dtype=torch.long)
        return torch.cat([input_ids, action], dim=1)


def _tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                           return_tensors=None):
    """Verbatim algorithm of ``uninavid.mm_utils.tokenizer_image_token``."""
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(values, separator):
        return [
            element
            for pair in zip(values, [separator] * len(values))
            for element in pair
        ][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for chunk in insert_separator(
        prompt_chunks, [image_token_index] * (offset + 1),
    ):
        input_ids.extend(chunk[offset:])
    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


def _special_ids(tokenizer):
    return {
        token: tokenizer(token, return_tensors="pt").input_ids[0][1:].tolist()
        for token in SPECIAL_TOKENS
    }


def _official_input_ids(prompt_text, tokenizer):
    """Official ``predict_inference`` token assembly, copied from upstream."""
    qs = prompt_text
    qs = "<image>\n" + qs.replace("<image>", "")
    conv = FakeConv()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    token_prompt = _tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
    )
    special = _special_ids(tokenizer)
    indices_to_replace = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
    new_list = []
    while indices_to_replace.numel() > 0:
        index = indices_to_replace[0]
        new_list.append(token_prompt[:index])
        new_list.append(torch.tensor(special["<video_special>"]))
        new_list.append(torch.tensor(special["<image_sep>"]))
        new_list.append(token_prompt[index:index + 1])
        new_list.append(torch.tensor(special["</video_special>"]))
        new_list.append(torch.tensor(special["<image_special>"]))
        new_list.append(torch.tensor(special["</image_special>"]))
        new_list.append(torch.tensor(special["[Navigation]"]))
        token_prompt = token_prompt[index + 1:]
        indices_to_replace = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
    if token_prompt.numel() > 0:
        new_list.append(token_prompt)
    return torch.cat(new_list, dim=0).unsqueeze(0)


class FakeL3:
    def __init__(self, block: str | None = L3_BLOCK):
        self.block = block

    def observe(self, observations, state, step, *, transition_record=False):
        return {"frame_valid": True, "context_injected": False}

    def build_context(self, instruction, step):
        return SimpleNamespace(
            context=None if self.block is None else SimpleNamespace(text=self.block),
        ), {
            "context_injected": self.block is not None,
            "fact_count": 1 if self.block else 0,
            "context_chars": len(self.block or ""), "retrieval_ms": 0.1,
            "candidate_count": 1, "verified_count": 1,
            "unverified_count": 0, "current_only_skipped": 0,
            "omitted_items": 0, "empty_reason": None, "expression": None,
        }

    def close_episode(self):
        pass


def _agent(agent_module, monkeypatch, *, l3=None):
    monkeypatch.setattr(agent_module, "DEFAULT_IMAGE_TOKEN", "<image>")
    monkeypatch.setattr(agent_module, "IMAGE_TOKEN_INDEX", IMAGE_TOKEN_INDEX)
    monkeypatch.setattr(agent_module, "tokenizer_image_token", _tokenizer_image_token)
    monkeypatch.setattr(agent_module, "KeywordsStoppingCriteria", FakeStoppingCriteria)
    monkeypatch.setattr(agent_module, "conv_templates", {"vicuna_v1": FakeConv()})

    agent = object.__new__(agent_module.UniNaVid_Agent)
    agent.l3 = l3
    agent.env = None
    agent.config = None
    agent.experiment_group = "uninavid_l3_only" if l3 else "uninavid_baseline"
    agent.result_path = "/tmp"
    agent.require_map = False
    agent.require_data = False
    agent.conv_mode = "vicuna_v1"
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
    agent.last_task_text = None
    agent.last_l3_context_text = None
    agent.episode_instruction = ""
    agent.episode_id = None
    agent._steps_stream = None
    agent.tokenizer = CharTokenizer()
    agent.model = FakeModel()
    agent.images_seen = []

    def process_images(rgb_list):
        agent.images_seen = list(rgb_list)
        return ["VISION_FEATURE_TENSOR"]

    agent.process_images = process_images
    return agent


def _act(agent):
    observations = {
        "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "instruction": {"text": "find the chair"},
    }
    return agent.act(observations, {}, "episode")


def _find(haystack, needle):
    for index in range(len(haystack) - len(needle) + 1):
        if haystack[index:index + len(needle)] == list(needle):
            return index
    return -1


def _occurrences(haystack, needle):
    return sum(
        1 for index in range(len(haystack) - len(needle) + 1)
        if haystack[index:index + len(needle)] == list(needle)
    )


def test_baseline_token_stream_is_exactly_the_official_assembly(monkeypatch):
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        agent = _agent(agent_module, monkeypatch)
        assert _act(agent) == {"action": 1}
        prompt = build_official_prompt(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE, "find the chair",
        )
        expected = _official_input_ids(prompt, agent.tokenizer)
        captured = agent.model.captured["input_ids"]
        assert torch.equal(captured, expected)
        kwargs = agent.model.captured["kwargs"]
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_new_tokens"] == 1024
        assert kwargs["use_cache"] is True
        assert agent.model.captured["images"] == ["VISION_FEATURE_TENSOR"]
        assert len(agent.images_seen) == 1
        assert agent.model.prompts == [[
            prompt.replace("<image>", "").replace("\n", "")
        ]]


def test_l3_changes_only_the_task_text_tokens(monkeypatch):
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        baseline_agent = _agent(agent_module, monkeypatch)
        _act(baseline_agent)
        baseline = baseline_agent.model.captured["input_ids"][0].tolist()

        l3_agent = _agent(agent_module, monkeypatch, l3=FakeL3())
        assert _act(l3_agent) == {"action": 1}
        injected = l3_agent.model.captured["input_ids"][0].tolist()

        prompt = build_prompt_with_l3(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE, "find the chair", L3_BLOCK,
        )
        expected = _official_input_ids(prompt, l3_agent.tokenizer)[0].tolist()
        assert injected == expected

        # Character tokenizer: the injected sequence is the baseline sequence
        # with exactly the L3 text inserted after the original instruction.
        l3_tokens = l3_agent.tokenizer.encode("\n" + L3_BLOCK)
        first_difference = next(
            index for index, (left, right) in enumerate(zip(baseline, injected))
            if left != right
        )
        assert injected[first_difference:first_difference + len(l3_tokens)] == l3_tokens
        assert injected[:first_difference] == baseline[:first_difference]
        assert injected[first_difference + len(l3_tokens):] == baseline[first_difference:]

        # The visual/navigation special-token block is identical in both
        # streams, appears exactly once, and keeps the image placeholder.
        special = _special_ids(l3_agent.tokenizer)
        block = (
            special["<video_special>"] + special["<image_sep>"] + [-200]
            + special["</video_special>"] + special["<image_special>"]
            + special["</image_special>"] + special["[Navigation]"]
        )
        assert _find(baseline, block) == _find(injected, block) >= 0
        assert _occurrences(baseline, block) == _occurrences(injected, block) == 1
        assert injected.count(-200) == 1

        # Vision input and generation settings are unchanged by L3.
        assert l3_agent.model.captured["images"] == (
            baseline_agent.model.captured["images"]
        )
        assert len(l3_agent.images_seen) == len(baseline_agent.images_seen) == 1
        assert np.array_equal(
            l3_agent.images_seen[0], baseline_agent.images_seen[0],
        )
        for key in ("do_sample", "temperature", "max_new_tokens", "use_cache"):
            assert l3_agent.model.captured["kwargs"][key] == (
                baseline_agent.model.captured["kwargs"][key]
            )
        # update_prompt sees the newline-normalized prompt, exactly like upstream.
        assert l3_agent.model.prompts == [[
            prompt.replace("<image>", "").replace("\n", "")
        ]]
        assert L3_BLOCK.replace("\n", "") in l3_agent.model.prompts[0][0]
        assert (
            "a video of historical observations and an image of the current observation"
            in l3_agent.model.prompts[0][0]
        )


def test_empty_or_none_l3_is_byte_identical_to_baseline(monkeypatch):
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        baseline_agent = _agent(agent_module, monkeypatch)
        _act(baseline_agent)
        baseline = baseline_agent.model.captured["input_ids"][0].tolist()

        for empty in (None, ""):
            agent = _agent(agent_module, monkeypatch, l3=FakeL3(block=empty))
            assert _act(agent) == {"action": 1}
            assert agent.model.captured["input_ids"][0].tolist() == baseline
            assert agent.last_l3_context_text is None


def test_real_l3_sidecar_reaches_the_real_agent_prompt(monkeypatch):
    """No fake L3 here: the vendored map/serializer feeds the real act path."""
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        sidecar = make_sidecar()
        agent = _agent(agent_module, monkeypatch, l3=sidecar)
        agent.count_id = 0
        agent.reset(
            episode=FakeEpisode(
                scene_id="scene-a", episode_id="episode-a",
                instruction="find the chair",
            ),
            instruction="find the chair",
        )
        assert _act(agent) == {"action": 1}
        first_prompt = agent.model.captured["input_ids"]
        first_text = agent.last_prompt_text
        assert "Navigation Map Memory" not in first_text

        # The first inference queued one pending action; consume it, then the
        # next inference sees the historical L3 map.
        assert _act(agent) == {"action": 2}
        assert _act(agent) == {"action": 1}
        second_text = agent.last_prompt_text
        assert "Navigation Map Memory - L3" in second_text
        assert "Map record (step" in second_text
        assert "the chair is in front of you" in second_text
        assert agent.last_l3_context_text in second_text
        assert first_prompt is not None
        assert agent.model.captured["input_ids"] is not None
        assert agent.model.captured["images"] == ["VISION_FEATURE_TENSOR"]


def test_l1_budget_counter_matches_the_exact_official_token_stream(monkeypatch):
    """C's budget must count the same stream predict_inference builds."""
    with agent_module_stubs() as agent_module:
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
        agent = _agent(agent_module, monkeypatch)
        prompt = build_official_prompt(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE, "find the chair",
        )
        expected = _official_input_ids(prompt, agent.tokenizer)
        assert agent._count_prompt_tokens(prompt) == expected.shape[1]

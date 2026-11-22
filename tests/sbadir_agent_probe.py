"""Standalone CPU probe of the real UniNaVid agent for the B/C groups.

Run ``python tests/sbadir_agent_probe.py b|c``.  The script installs minimal
stand-ins for Habitat, Uni-NaVid, cv2, imageio and torch, because this probe
only needs the agent's prompt-composition path.  It prints one JSON object.

The probe exists so the pytest suite can verify the *actual* agent method the
GPU run executes, without importing torch/Habitat in the test process.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_torch_stub() -> None:
    torch = types.ModuleType("torch")
    torch.get_rng_state = lambda: None
    torch.set_rng_state = lambda state: None

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    torch.cuda = _Cuda()
    sys.modules["torch"] = torch


_install_torch_stub()

import numpy as np  # noqa: E402

from l1_integration.sbadir_backend import L1EvidenceBackend  # noqa: E402
from l1_memory.memory_writer import MockSceneExtractor  # noqa: E402
from l3_integration.experiment_groups import (  # noqa: E402
    A_GROUP,
    B_GROUP,
    C_GROUP,
)
from l3_integration.official_prompt import (  # noqa: E402
    REFERENCE_OFFICIAL_PROMPT_SHA256,
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
)
from l3_memory.schemas import DetectedObject, SceneExtraction  # noqa: E402
from tests.helpers import (  # noqa: E402
    FakeEpisode,
    agent_module_stubs,
    make_sidecar,
)

INSTRUCTION = "find the chair and table"


class _ProbeTokenizer:
    """Deterministic character tokenizer with a BOS id, enough for budgeting."""

    bos_token_id = 1

    def __call__(self, text, return_tensors=None):
        return types.SimpleNamespace(
            input_ids=[[self.bos_token_id] + [1000 + ord(char) for char in text]]
        )

    def encode(self, text, add_special_tokens=False):
        return [1000 + ord(char) for char in text]


class _ProbeConv:
    roles = ("USER", "ASSISTANT")
    sep = "</s>"
    sep_style = "one"

    def __init__(self, messages=None):
        self.messages = list(messages or [])

    def copy(self):
        return _ProbeConv(self.messages)

    def append_message(self, role, message):
        self.messages.append((role, message))

    def get_prompt(self):
        return "".join(
            "{role}: {message}\n".format(
                role=role, message="" if message is None else message,
            )
            for role, message in self.messages
        )


def _probe_tokenizer_image_token(prompt, tokenizer, image_token_index=-200):
    """Verbatim upstream algorithm, returned as a flat Python list."""
    prompt_chunks = [tokenizer(chunk).input_ids[0] for chunk in prompt.split("<image>")]

    def insert_separator(values, separator):
        return [
            element for pair in zip(values, [separator] * len(values))
            for element in pair
        ][:-1]

    input_ids = []
    offset = 0
    if (
        prompt_chunks and prompt_chunks[0]
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    separator = [image_token_index] * (offset + 1)
    for chunk in insert_separator(prompt_chunks, separator):
        if isinstance(chunk, list):
            input_ids.extend(chunk[offset:])
        else:
            input_ids.append(chunk)
    return input_ids


def _fake_agent(agent_module, *, l3, l1, source_background, group):
    agent = object.__new__(agent_module.UniNaVid_Agent)
    agent.l3 = l3
    agent.l1 = l1
    agent.source_background = source_background
    agent.env = None
    agent.config = None
    agent.experiment_group = group
    agent.result_path = "/tmp"
    agent.require_map = False
    agent.require_data = False
    agent.context_len = 4096
    agent.max_new_tokens = 1024
    agent.conv_mode = "vicuna_v1"
    # Exercise the >64-frame online-merge path with a real cache shape, not a
    # 64*frames estimate.  The probe never runs the vision tower.
    probe_inner = types.SimpleNamespace(
        feat_cache=types.SimpleNamespace(shape=(64, 4, 8)),
        long_feat_cache=None,
        new_frames=0,
    )
    agent.model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            mm_use_im_start_end=False, compress_type="grid:2",
        ),
        get_model=lambda: probe_inner,
    )
    agent.promt_template = REFERENCE_OFFICIAL_PROMPT_TEMPLATE
    agent.prompt_template_sha256 = REFERENCE_OFFICIAL_PROMPT_SHA256
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
    agent.last_sbadir_memory_text = None
    agent.last_l1_context_text = None
    agent.last_l1_prompt_text = None
    agent.episode_instruction = ""
    agent._steps_stream = None
    agent.episode_id = None
    agent.last_action = None
    agent.tokenizer = _ProbeTokenizer()
    captured = []

    def predict(prompt):
        captured.append(prompt)
        agent.last_prompt_tokens = len(prompt.split())
        return "forward"

    agent.predict_inference = predict
    agent.captured_prompts = captured
    return agent


def main() -> int:
    os.environ.setdefault("L3_LOG_PROMPTS", "1")
    if len(sys.argv) != 2 or sys.argv[1] not in {"a", "b", "c", "c-empty", "c-fallback", "c-visual-fallback", "c-tight"}:
        raise SystemExit("usage: sbadir_agent_probe.py a|b|c|c-empty")
    mode = sys.argv[1]
    group_mode = "c" if mode in {"c-empty", "c-fallback", "c-visual-fallback", "c-tight"} else mode
    with agent_module_stubs() as agent_module:
        agent_module.tokenizer_image_token = _probe_tokenizer_image_token
        agent_module.conv_templates = {"vicuna_v1": _ProbeConv()}
        agent_module.IMAGE_TOKEN_INDEX = -200
        agent_module.DEFAULT_IMAGE_TOKEN = "<image>"
        agent_module.DEFAULT_IM_START_TOKEN = "<im_start>"
        agent_module.DEFAULT_IM_END_TOKEN = "<im_end>"
        sidecar = make_sidecar()
        sidecar.reset_episode(FakeEpisode(instruction=INSTRUCTION))
        l1 = None
        if mode == "c-empty":
            l1 = L1EvidenceBackend(MockSceneExtractor())
            l1.reset_episode("scene", "episode", INSTRUCTION)
        elif mode in {"c", "c-fallback", "c-visual-fallback", "c-tight"}:
            detector = MockSceneExtractor({
                step: SceneExtraction(
                    detections=[DetectedObject(
                        "table" if step == 0 else "chair",
                        0.91, (2, 2, 6, 6),
                    )],
                    source="mock_scene_extractor", confidence=0.91,
                )
                for step in (0, 1)
            })
            l1 = L1EvidenceBackend(detector)
            l1.reset_episode("scene", "episode", INSTRUCTION)
        agent = _fake_agent(
            agent_module, l3=sidecar, l1=l1,
            source_background=group_mode in {"b", "c"},
            group={"a": A_GROUP, "b": B_GROUP, "c": C_GROUP}[group_mode],
        )
        if mode == "c-fallback":
            def _fail_budget(text):
                raise RuntimeError("probe tokenizer unavailable")
            agent._count_prompt_tokens = _fail_budget
        elif mode == "c-visual-fallback":
            agent.model.config.compress_type = None
        elif mode == "c-tight":
            # Text + visual base already exceeds context - generation - 1 on
            # both steps; C must not force an L1 card into this prompt.
            agent.context_len = 2000
            agent.model.get_model().feat_cache = types.SimpleNamespace(
                shape=(150, 4, 8)
            )
            agent.model.get_model().long_feat_cache = types.SimpleNamespace(
                shape=(86, 8)
            )
        observations = {
            "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "instruction": {"text": INSTRUCTION},
        }
        action = None
        for _ in range(2):
            action = agent.act(observations, {}, "episode")
        row = agent.step_diagnostics[-1]
        summary = agent.episode_summary()
        keys = (
            "source_background_enabled", "l1_enabled", "l1_enabled_steps",
            "l1_fact_count_total", "l1_retained_facts_total",
            "l1_omitted_facts_total", "l1_prompt_built_steps",
            "l1_empty_steps", "l1_events_persisted_total",
            "l1_detection_events_persisted_total",
            "l1_observe_error_count", "l1_selection_error_count", "l1_detector_error_count",
            "l1_place_error_count",
            "l1_observation_attempt_steps", "prompt_has_l1_steps",
            "source_background_facts_total", "l3_provenance_count_total",
            "l3_fact_count_total", "l3_original_memory_unchanged",
            "duplicated_facts_total", "conflicting_facts_total",
            "conflict_type_counts", "conflict_evidence_ids",
            "l3_segmentation_error_count", "env_steps",
            "budget_source", "context_limit_tokens", "base_prompt_tokens",
            "available_memory_tokens", "memory_tokens_injected",
            "token_overflow_drops", "budget_fallback_reason",
            "l1_token_budget_steps", "l1_token_budget_fallback_steps",
            "token_overflow_drops_total", "memory_tokens_injected_mean",
            "base_prompt_tokens_mean", "available_memory_tokens_min",
            "memory_budget_violation_steps",
            "text_prompt_tokens_mean", "visual_tokens_mean",
            "multimodal_prompt_tokens_mean",
            "context_margin_tokens_min",
            "multimodal_budget_violation_steps",
        )
        record = {
            "mode": mode,
            "action": action,
            "prompt": agent.captured_prompts[-1],
            "prompts": agent.captured_prompts,
            "l1_context_texts": [
                step_row.get("l1_context_text")
                for step_row in agent.step_diagnostics
            ],
            "memory": agent.last_sbadir_memory_text,
            "l3_original_text": agent.last_l3_context_text,
            "l3_enriched_sha256": row.get("l3_enriched_sha256"),
            "l3_fact_signature_sha256": row.get("l3_fact_signature_sha256"),
            "l3_original_sha256": row.get("l3_original_sha256"),
            "prompt_has_l1": row.get("prompt_has_l1"),
            "step_source_background_facts": row.get("source_background_facts"),
            "l3_core_unchanged": row.get("l3_original_memory_unchanged"),
            "l3_states": [
                {key: step_row.get(key) for key in (
                    "frame_valid", "segmentation_count", "segmentation_error",
                    "evidence_minted", "observed_objects", "object_nodes",
                    "viewpoints", "observed_from_edges", "traversed_to_edges",
                    "context_available", "candidate_count",
                    "verified_count", "unverified_count", "current_only_skipped",
                    "omitted_items", "fact_count",
                )}
                for step_row in agent.step_diagnostics
            ],
            "token_budget": {
                key: row.get(key) for key in (
                    "budget_source", "context_limit_tokens",
                    "base_prompt_tokens", "text_prompt_tokens",
                    "visual_tokens", "visual_token_accounting",
                    "available_memory_tokens", "memory_tokens_injected",
                    "token_overflow_drops", "multimodal_prompt_tokens",
                    "context_margin_tokens",
                    "multimodal_prompt_tokens_is_upper_bound",
                    "multimodal_budget_overflow",
                    "budget_fallback_reason",
                )
            },
            "summary": {key: summary.get(key) for key in keys},
        }
        print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

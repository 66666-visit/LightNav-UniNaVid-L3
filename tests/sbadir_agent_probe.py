"""Standalone CPU probe of the real UniNaVid agent for the B/C groups.

Run ``python tests/sbadir_agent_probe.py b|c``.  The script installs minimal
stand-ins for Habitat, Uni-NaVid, cv2, imageio and torch, because this probe
only needs the agent's prompt-composition path.  It prints one JSON object.

The probe exists so the pytest suite can verify the *actual* agent method the
GPU run executes, without importing torch/Habitat in the test process.
"""

from __future__ import annotations

import json
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


def _fake_agent(agent_module, *, l3, l1, source_background, group):
    agent = object.__new__(agent_module.UniNaVid_Agent)
    agent.l3 = l3
    agent.l1 = l1
    agent.l1_max_tokens = 160
    agent.source_background = source_background
    agent.env = None
    agent.config = None
    agent.experiment_group = group
    agent.result_path = "/tmp"
    agent.require_map = False
    agent.require_data = False
    agent.context_len = 4096
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
    agent.episode_instruction = ""
    agent._steps_stream = None
    agent.episode_id = None
    agent.last_action = None
    agent.tokenizer = types.SimpleNamespace(
        encode=lambda text, add_special_tokens=False: list(range(len(text.split())))
    )
    captured = []

    def predict(prompt):
        captured.append(prompt)
        agent.last_prompt_tokens = len(prompt.split())
        return "forward"

    agent.predict_inference = predict
    agent.captured_prompts = captured
    return agent


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"a", "b", "c"}:
        raise SystemExit("usage: sbadir_agent_probe.py a|b|c")
    mode = sys.argv[1]
    with agent_module_stubs() as agent_module:
        sidecar = make_sidecar()
        sidecar.reset_episode(FakeEpisode(instruction=INSTRUCTION))
        l1 = None
        if mode == "c":
            detector = MockSceneExtractor({
                step: SceneExtraction(
                    detections=[DetectedObject("table", 0.91, (2, 2, 6, 6))],
                    source="mock_scene_extractor", confidence=0.91,
                )
                for step in (0, 1)
            })
            l1 = L1EvidenceBackend(detector)
            l1.reset_episode("scene", "episode", INSTRUCTION)
        agent = _fake_agent(
            agent_module, l3=sidecar, l1=l1,
            source_background=mode in {"b", "c"},
            group={"a": A_GROUP, "b": B_GROUP, "c": C_GROUP}[mode],
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
            "l1_observation_attempt_steps", "prompt_has_l1_steps",
            "source_background_facts_total", "l3_provenance_count_total",
            "l3_fact_count_total", "l3_original_memory_unchanged",
            "duplicated_facts_total", "conflicting_facts_total",
            "conflict_type_counts", "conflict_evidence_ids",
            "l3_segmentation_error_count", "env_steps",
        )
        record = {
            "mode": mode,
            "action": action,
            "prompt": agent.captured_prompts[-1],
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
            "summary": {key: summary.get(key) for key in keys},
        }
        print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

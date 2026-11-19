#!/usr/bin/env python3
"""CPU-only audit of the *real* Uni-NaVid prompt construction path.

The audit runs the production ``UniNaVid_Agent.act`` -> ``_augment_task_text``
-> ``official_prompt.build_prompt`` -> ``predict_inference`` ->
``model.update_prompt`` -> ``model.generate`` chain.  It does **not** build a
display prompt by hand.

Because this host has no Habitat scene, no Uni-NaVid checkpoint and no Vicuna
tokenizer, the run uses a deterministic **synthetic fixture**:

* the repository's test RGB-D frame and deterministic fake segmenter
  (``tests.helpers.make_sidecar``) feed the real ``L3MapSidecar`` and the real
  ``L3OfficialPromptOrganizer``;
* the real official ``conv_vicuna_v1`` template and the real
  ``uninavid.mm_utils.tokenizer_image_token`` build the model input;
* only the checkpoint/tokenizer/image-processor/model are stand-ins.

Consequences, recorded in ``prompt_trace.json``:

* the prompt text is real for this fixture, but it is **not** a real Habitat
  navigation prompt (marked ``synthetic_fixture``);
* the character tokenizer is structural only.  It is never reported as the
  Uni-NaVid token count and no token count is invented.

Usage:
  PYTHONPATH=. python3 scripts/audit_prompt.py
  PYTHONPATH=. python3 scripts/audit_prompt.py --output output/prompt_audit
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import difflib
import hashlib
import importlib
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from l3_integration.official_prompt import (  # noqa: E402
    L3_SEPARATOR,
    NAVIGATION_IDENTIFIER,
    REFERENCE_OFFICIAL_PROMPT_SHA256,
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
    build_prompt,
    build_task_text,
    prompt_sha256,
    split_official_template,
    update_prompt_view,
    verify_task_text_injection,
)
from tests.helpers import FakeEpisode, make_sidecar  # noqa: E402

AUDIT_VERSION = "prompt_audit_v1"
FIXTURE_INSTRUCTION = "find the chair"
FIXTURE_SCENE = "scene-a"
FIXTURE_EPISODE = "episode-a"
FIXTURE_STEPS = 3
BASELINE_FILENAME = "baseline_prompt.txt"
L3_FILENAME = "l3_prompt.txt"
BLOCK_FILENAME = "l3_memory_block.txt"
DIFF_FILENAME = "prompt_diff.txt"
TRACE_FILENAME = "prompt_trace.json"
EXPECTED_TEMPLATE_SHA256 = (
    "435d79868156aaa1352afcd98f4ceae2f03a54f8cdc5fe87e41d32774d5512c1"
)


class PromptAuditError(RuntimeError):
    """The real prompt path violated a frozen prompt contract."""


# ---------------------------------------------------------------------------
# Stand-ins.  These are deliberately labelled in the trace; none of them is
# allowed to fabricate a real tokenizer id or a real multimodal token count.
# ---------------------------------------------------------------------------


class StructuralCharTokenizer:
    """Invertible character tokenizer used only to inspect token *structure*.

    This is **not** the Uni-NaVid Vicuna tokenizer.  It has the same small
    interface surface (``__call__``, ``encode``, ``decode``, ``bos_token_id``)
    that the official ``tokenizer_image_token`` and ``predict_inference``
    consume.  Its ids are never reported as real model tokens.
    """

    bos_token_id = 1
    _BASE = 1000

    def __call__(self, text: str, return_tensors: str | None = None):
        ids = [self.bos_token_id] + [self._BASE + ord(char) for char in text]
        if return_tensors == "pt":
            return SimpleNamespace(
                input_ids=torch.tensor([ids], dtype=torch.long),
            )
        if return_tensors is not None:
            raise ValueError(f"unsupported return_tensors: {return_tensors}")
        return SimpleNamespace(input_ids=ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._BASE + ord(char) for char in text]

    def batch_decode(self, sequences, skip_special_tokens: bool = True):
        rows = sequences.tolist() if hasattr(sequences, "tolist") else sequences
        return [
            "".join(
                chr(value - self._BASE)
                for value in row
                if isinstance(value, int) and value >= self._BASE
            )
            for row in rows
        ]


class RecordingModel:
    """Minimal model that records exactly what the official code passes it."""

    def __init__(self, action_text: str = "forward left stop") -> None:
        self.config = SimpleNamespace(mm_use_im_start_end=False, run_type="eval")
        self._action_ids = [
            StructuralCharTokenizer._BASE + ord(char) for char in action_text
        ]
        self.prompts = None
        self.generate_calls: list[dict[str, Any]] = []

    def get_model(self):
        return SimpleNamespace(
            initialize_online_inference_nav_feat_cache=lambda: None,
            new_frames=0,
        )

    def update_prompt(self, prompts=None) -> None:
        self.prompts = prompts

    def generate(self, input_ids, images=None, **kwargs):
        self.generate_calls.append({
            "input_ids": input_ids.clone(),
            "images": images,
            "kwargs": dict(kwargs),
            # update_prompt is called immediately before generate; this is the
            # exact text the official model receives for this generation.
            "prompts": self.prompts,
        })
        action = torch.tensor([self._action_ids], dtype=torch.long)
        return torch.cat([input_ids, action], dim=1)


@contextlib.contextmanager
def _cpu_tensors() -> Iterator[None]:
    """Let the official ``.cuda()`` calls run on this CPU-only host."""
    original = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.Tensor.cuda = original  # type: ignore[assignment]


@contextlib.contextmanager
def _official_imports() -> Iterator[Any]:
    """Import ``agent_uninavid`` with the *real* official text modules.

    Habitat/cv2/imageio and the checkpoint loader are unavailable; the official
    ``uninavid.conversation`` and ``uninavid.mm_utils`` modules are kept real.
    A package shim prevents ``uninavid/__init__.py`` from importing the heavy
    model architecture.
    """
    names = [
        "habitat", "habitat.core", "habitat.core.agent", "habitat.utils",
        "habitat.utils.visualizations", "habitat.utils.visualizations.maps",
        "cv2", "imageio", "uninavid", "uninavid.model",
        "uninavid.model.builder", "uninavid.mm_utils", "uninavid.constants",
        "uninavid.conversation", "agent_uninavid",
    ]
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name in ("cv2", "imageio"):
            sys.modules[name] = types.ModuleType(name)
        habitat = types.ModuleType("habitat")
        habitat.__path__ = []
        sys.modules["habitat"] = habitat
        core = types.ModuleType("habitat.core")
        core.__path__ = []
        sys.modules["habitat.core"] = core
        agent_module = types.ModuleType("habitat.core.agent")
        agent_module.Agent = type("Agent", (), {})
        sys.modules["habitat.core.agent"] = agent_module
        utils = types.ModuleType("habitat.utils")
        utils.__path__ = []
        sys.modules["habitat.utils"] = utils
        visualizations = types.ModuleType("habitat.utils.visualizations")
        visualizations.__path__ = []
        sys.modules["habitat.utils.visualizations"] = visualizations
        maps = types.ModuleType("habitat.utils.visualizations.maps")
        visualizations.maps = maps
        sys.modules["habitat.utils.visualizations.maps"] = maps

        package = types.ModuleType("uninavid")
        package.__path__ = [str(REPO_ROOT / "uninavid")]
        sys.modules["uninavid"] = package
        # Real official text-processing modules.
        importlib.import_module("uninavid.conversation")
        importlib.import_module("uninavid.mm_utils")
        # Only the checkpoint loader is replaced; it is never called here.
        builder = types.ModuleType("uninavid.model.builder")
        builder.__path__ = []
        builder.load_pretrained_model = lambda *args, **kwargs: None
        model_package = types.ModuleType("uninavid.model")
        model_package.__path__ = []
        sys.modules["uninavid.model"] = model_package
        sys.modules["uninavid.model.builder"] = builder

        sys.modules.pop("agent_uninavid", None)
        import agent_uninavid  # noqa: F401

        yield agent_uninavid
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


# ---------------------------------------------------------------------------
# Real production path
# ---------------------------------------------------------------------------


def _make_agent(agent_module: Any, *, l3: Any = None) -> Any:
    """Create the production agent object without loading a checkpoint."""
    agent = object.__new__(agent_module.UniNaVid_Agent)
    agent.l3 = l3
    agent.env = None
    agent.config = None
    agent.experiment_group = (
        "uninavid_l3_only" if l3 is not None else "uninavid_baseline"
    )
    agent.result_path = "/tmp/prompt_audit"
    agent.require_map = False
    agent.require_data = False
    agent.conv_mode = "vicuna_v1"
    agent.promt_template = REFERENCE_OFFICIAL_PROMPT_TEMPLATE
    agent.prompt_template_sha256 = prompt_sha256(agent.promt_template)
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
    agent.count_id = 0
    agent.count_stop = 0
    agent.tokenizer = StructuralCharTokenizer()
    agent.model = RecordingModel()
    agent.images_seen = []

    def process_images(rgb_list):
        agent.images_seen = list(rgb_list)
        return ["VISION_FEATURE_TENSOR"]

    agent.process_images = process_images
    return agent


def _run_mode(agent_module: Any, *, use_l3: bool) -> dict[str, Any]:
    """Execute three real ``act`` calls with or without the L3 sidecar.

    Step 0 and step 1 build map history; step 2 is the first inference whose
    official task slot carries a nonempty L3 block.  The second action is
    served from ``pending_action_list`` without a model call, exactly as in the
    production loop.
    """
    sidecar = make_sidecar() if use_l3 else None
    observations = {
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "instruction": {"text": FIXTURE_INSTRUCTION},
    }
    build_context_calls: list[dict[str, Any]] = []
    inference_calls: list[dict[str, Any]] = []
    l1_event_lists: list[list[Any]] = []

    import l3_integration.map_sidecar as sidecar_module

    original_evidence = sidecar_module.make_spatial_evidence

    def recording_evidence(instances, events, store, **kwargs):
        l1_event_lists.append(list(events))
        return original_evidence(instances, events, store, **kwargs)

    if sidecar is not None:
        sidecar_module.make_spatial_evidence = recording_evidence
        original_build_context = sidecar.build_context

        def recording_build_context(instruction, step):
            organization, diagnostic = original_build_context(instruction, step)
            build_context_calls.append({
                "instruction": instruction,
                "step": step,
                "organization": organization,
                "diagnostic": diagnostic,
            })
            return organization, diagnostic

        sidecar.build_context = recording_build_context

    agent = _make_agent(agent_module, l3=sidecar)
    original_predict = agent.predict_inference

    def recording_predict(prompt):
        step = agent.step_index
        before = len(agent.model.generate_calls)
        outputs = original_predict(prompt)
        call = (
            agent.model.generate_calls[before]
            if len(agent.model.generate_calls) > before else None
        )
        inference_calls.append({
            "step": step,
            "prompt": prompt,
            "task_text": agent.last_task_text,
            "l3_block": agent.last_l3_context_text,
            "model_call": call,
            "outputs": outputs,
        })
        return outputs

    agent.predict_inference = recording_predict
    try:
        if sidecar is not None:
            agent.reset(
                episode=FakeEpisode(
                    scene_id=FIXTURE_SCENE, episode_id=FIXTURE_EPISODE,
                    instruction=FIXTURE_INSTRUCTION,
                ),
                instruction=FIXTURE_INSTRUCTION,
            )
        else:
            agent.reset()
        step_results = []
        for _ in range(FIXTURE_STEPS):
            step_results.append(agent.act(observations, {}, FIXTURE_EPISODE))
    finally:
        if sidecar is not None:
            sidecar_module.make_spatial_evidence = original_evidence
            sidecar.build_context = original_build_context

    return {
        "agent": agent,
        "sidecar": sidecar,
        "step_results": step_results,
        "inference_calls": inference_calls,
        "build_context_calls": build_context_calls,
        "l1_event_lists": l1_event_lists,
        "observations": observations,
    }


# ---------------------------------------------------------------------------
# Official token assembly oracle (verbatim copy of the production algorithm)
# ---------------------------------------------------------------------------


def _special_ids(tokenizer, agent_module: Any) -> dict[str, list[int]]:
    names = {
        "video_start": "<video_special>",
        "image_sep": "<image_sep>",
        "video_end": "</video_special>",
        "image_start": "<image_special>",
        "image_end": "</image_special>",
        "navigation": "[Navigation]",
    }
    return {
        key: tokenizer(token, return_tensors="pt").input_ids[0][1:].tolist()
        for key, token in names.items()
    }


def _official_input_ids_oracle(
    prompt_text: str, tokenizer, agent_module: Any,
) -> dict[str, Any]:
    """Reproduce ``predict_inference``'s token assembly from the official code."""
    qs = prompt_text
    if agent_module.DEFAULT_IMAGE_TOKEN != "<image>":
        raise PromptAuditError("DEFAULT_IMAGE_TOKEN is not the official <image>")
    qs = agent_module.DEFAULT_IMAGE_TOKEN + "\n" + qs.replace("<image>", "")
    conv = agent_module.conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    conv_prompt = conv.get_prompt()
    token_prompt = agent_module.tokenizer_image_token(
        conv_prompt, tokenizer, agent_module.IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    special = _special_ids(tokenizer, agent_module)
    new_list = []
    indices = torch.where(token_prompt == agent_module.IMAGE_TOKEN_INDEX)[0]
    while indices.numel() > 0:
        index = indices[0]
        new_list.append(token_prompt[:index])
        new_list.append(torch.tensor(special["video_start"]))
        new_list.append(torch.tensor(special["image_sep"]))
        new_list.append(token_prompt[index:index + 1])
        new_list.append(torch.tensor(special["video_end"]))
        new_list.append(torch.tensor(special["image_start"]))
        new_list.append(torch.tensor(special["image_end"]))
        new_list.append(torch.tensor(special["navigation"]))
        token_prompt = token_prompt[index + 1:]
        indices = torch.where(token_prompt == agent_module.IMAGE_TOKEN_INDEX)[0]
    if token_prompt.numel() > 0:
        new_list.append(token_prompt)
    input_ids = torch.cat(new_list, dim=0).unsqueeze(0)
    return {
        "conv_prompt": conv_prompt,
        "input_ids": input_ids[0].tolist(),
        "special": special,
        "image_token_index": agent_module.IMAGE_TOKEN_INDEX,
    }


# ---------------------------------------------------------------------------
# Source evidence
# ---------------------------------------------------------------------------


def _line_of(path: Path, needle: str, *, start: int = 1) -> int | None:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if number >= start and needle in line:
            return number
    return None


def _template_from_agent_source(path: Path) -> tuple[str, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "promt_template"
        ):
            return ast.literal_eval(node.value), int(node.lineno)
    raise PromptAuditError("self.promt_template assignment not found")


def _call_chain() -> list[dict[str, Any]]:
    agent_path = REPO_ROOT / "agent_uninavid.py"
    official_path = REPO_ROOT / "l3_integration" / "official_prompt.py"
    sidecar_path = REPO_ROOT / "l3_integration" / "map_sidecar.py"
    serializer_path = REPO_ROOT / "l3_memory" / "l3_official_prompt.py"
    arch_path = REPO_ROOT / "uninavid" / "model" / "uninavid_arch.py"
    conv_path = REPO_ROOT / "uninavid" / "conversation.py"

    return [
        {
            "stage": "instruction",
            "file": "agent_uninavid.py",
            "function": "UniNaVid_Agent.act",
            "line": _line_of(
                agent_path, 'instruction = observations["instruction"]["text"]',
            ),
            "detail": "Habitat instruction text read from the observation",
        },
        {
            "stage": "l3_observe",
            "file": "agent_uninavid.py",
            "function": "UniNaVid_Agent.act -> _observe_l3 -> L3MapSidecar.observe",
            "line": _line_of(
                agent_path, "l3_diagnostic = self._observe_l3(observations, step)",
            ),
            "detail": (
                "current RGB-D/pose updates the real ObjectMap; "
                "make_spatial_evidence receives an empty L1 event list"
            ),
        },
        {
            "stage": "l3_retrieve_serialize",
            "file": "l3_integration/map_sidecar.py",
            "function": "L3MapSidecar.build_context -> L3OfficialPromptOrganizer.build",
            "line": _line_of(sidecar_path, "organization = self.organizer.build("),
            "detail": (
                "real retrieval + official-style serializer; returns "
                "organization.context.text"
            ),
        },
        {
            "stage": "task_text_builder",
            "file": "l3_integration/official_prompt.py",
            "function": "build_task_text",
            "line": _line_of(official_path, "def build_task_text"),
            "detail": (
                "returns instruction + '\n' + l3_memory, or the unchanged "
                "instruction when L3 is empty"
            ),
        },
        {
            "stage": "l3_serialize_context",
            "file": "l3_memory/l3_official_prompt.py",
            "function": "L3OfficialPromptOrganizer._serialize",
            "line": _line_of(
                serializer_path, "organization.context = MemoryContext(",
            ),
            "detail": (
                "the LightNav official-style serializer emits the exact "
                "map-memory text placed in the task slot"
            ),
        },
        {
            "stage": "task_text",
            "file": "agent_uninavid.py",
            "function": "UniNaVid_Agent._augment_task_text -> official_prompt.build_task_text",
            "line": _line_of(
                agent_path,
                "task_text, l3_context, l3_diagnostic = self._augment_task_text(",
            ),
            "detail": (
                "task_text = instruction + '\\n' + l3_context; the original "
                "instruction is always the prefix"
            ),
        },
        {
            "stage": "official_format",
            "file": "agent_uninavid.py",
            "function": "UniNaVid_Agent.act -> official_prompt.build_prompt",
            "line": _line_of(
                agent_path, "navigation_qs = build_prompt(self.promt_template, task_text)",
            ),
            "detail": (
                "the official template is filled once with "
                "self.promt_template.format(task_text); L3 is inside the {} slot"
            ),
        },
        {
            "stage": "official_format_call",
            "file": "l3_integration/official_prompt.py",
            "function": "build_prompt",
            "line": _line_of(official_path, "return template.format(task_text)"),
            "detail": "the exact official str.format call",
        },
        {
            "stage": "prompt_guard",
            "file": "agent_uninavid.py",
            "function": "UniNaVid_Agent.act -> verify_task_text_injection",
            "line": _line_of(agent_path, "verify_task_text_injection("),
            "detail": (
                "fails closed unless the prompt equals prefix + task_text + suffix "
                "and L3 is not appended after the complete prompt"
            ),
        },
        {
            "stage": "predict_inference",
            "file": "agent_uninavid.py",
            "function": "UniNaVid_Agent.act -> predict_inference",
            "line": _line_of(agent_path, "navigation = self.predict_inference(navigation_qs)"),
            "detail": "one prompt argument feeds both model-input paths",
        },
        {
            "stage": "question_view",
            "file": "agent_uninavid.py",
            "function": "UniNaVid_Agent.predict_inference",
            "line": _line_of(
                agent_path,
                "question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\\n', '')",
            ),
            "detail": (
                "update_prompt view: remove the literal <image> and every newline; "
                "L3 text remains"
            ),
        },
        {
            "stage": "conv_template",
            "file": "uninavid/conversation.py",
            "function": "conv_vicuna_v1.get_prompt",
            "line": _line_of(conv_path, "def get_prompt(self):"),
            "detail": (
                "official vicuna_v1 USER/ASSISTANT template; the single <image> "
                "is prefixed by predict_inference"
            ),
        },
        {
            "stage": "update_prompt",
            "file": "agent_uninavid.py",
            "function": "UniNaVid_Agent.predict_inference",
            "line": _line_of(agent_path, "self.model.update_prompt([[cur_prompt]])"),
            "detail": "the model's navigation prompt is question (L3 included)",
        },
        {
            "stage": "generate",
            "file": "agent_uninavid.py",
            "function": "UniNaVid_Agent.predict_inference",
            "line": _line_of(agent_path, "output_ids = self.model.generate("),
            "detail": (
                "input_ids carry the L3 token span; images/kwargs are unchanged "
                "by L3"
            ),
        },
        {
            "stage": "model_prompt_use",
            "file": "uninavid/model/uninavid_arch.py",
            "function": "LlavaLlamaAttForCausalLM.update_prompt",
            "line": _line_of(arch_path, "def update_prompt(self, prompts=None):"),
            "detail": "self.prompts = prompts",
        },
        {
            "stage": "model_navigation_check",
            "file": "uninavid/model/uninavid_arch.py",
            "function": "vlm_attention",
            "line": _line_of(
                arch_path, "is_navigation = NAVIGATION_IDENTIFIER in prompt[0]",
            ),
            "detail": (
                "the official navigation identifier must survive in the "
                "update_prompt view"
            ),
        },
        {
            "stage": "model_text_embeddings",
            "file": "uninavid/model/uninavid_arch.py",
            "function": "prepare_inputs_labels_for_multimodal",
            "line": _line_of(
                arch_path,
                "cur_new_input_embeds.append(self.get_model().embed_tokens(",
            ),
            "detail": (
                "the language model embeds the full input_ids text, including "
                "the L3 task-slot span; image features are spliced at -200"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Audit construction
# ---------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PromptAuditError(message)


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    for index in range(len(haystack) - len(needle) + 1):
        if haystack[index:index + len(needle)] == needle:
            return index
    return -1


def build_audit() -> dict[str, Any]:
    """Run both conditions and return exact texts plus the audit trace."""
    with _cpu_tensors(), _official_imports() as agent_module:
        template, template_line = _template_from_agent_source(
            REPO_ROOT / "agent_uninavid.py",
        )
        prefix, suffix = split_official_template(template)
        _require(
            template == REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
            "agent template differs from the reference official template",
        )
        _require(
            prompt_sha256(template) == EXPECTED_TEMPLATE_SHA256,
            "official template SHA256 changed",
        )
        _require(
            template.count("{}") == 1,
            "official template must have exactly one task slot",
        )

        baseline = _run_mode(agent_module, use_l3=False)
        l3 = _run_mode(agent_module, use_l3=True)

        baseline_record = baseline["inference_calls"][-1]
        l3_record = l3["inference_calls"][-1]
        baseline_prompt = baseline_record["prompt"]
        l3_prompt = l3_record["prompt"]
        block = l3_record["l3_block"]
        task_text = l3_record["task_text"]

        _require(
            isinstance(block, str) and bool(block.strip()),
            "the real L3 path produced no memory block at the audited step",
        )
        _require(
            block == l3["sidecar"].last_context_text,
            "the audited block is not the sidecar's exact serialized text",
        )
        _require(
            baseline_record["l3_block"] is None,
            "baseline unexpectedly carried an L3 block",
        )
        _require(
            baseline_prompt == build_prompt(template, FIXTURE_INSTRUCTION),
            "baseline prompt is not the official template with the instruction",
        )
        _require(
            baseline_prompt == template.format(FIXTURE_INSTRUCTION),
            "baseline prompt is not the official str.format result",
        )
        _require(
            l3_prompt == build_prompt(template, task_text),
            "L3 prompt is not the official template with the task text",
        )
        _require(
            l3_prompt == template.format(FIXTURE_INSTRUCTION + L3_SEPARATOR + block),
            "L3 prompt is not official_template.format(instruction + L3)",
        )
        _require(
            l3_prompt != template.format(FIXTURE_INSTRUCTION) + L3_SEPARATOR + block,
            "L3 was appended after the complete official prompt (mode A)",
        )
        _require(
            task_text == FIXTURE_INSTRUCTION + L3_SEPARATOR + block,
            "task text does not preserve instruction + newline + L3",
        )
        verify_task_text_injection(template, FIXTURE_INSTRUCTION, block, l3_prompt)

        injection_index = l3_prompt.find(block)
        expected_index = len(prefix) + len(FIXTURE_INSTRUCTION) + len(L3_SEPARATOR)
        _require(
            injection_index == expected_index,
            f"L3 block starts at {injection_index}, expected {expected_index}",
        )
        _require(
            l3_prompt.endswith(suffix),
            "official prompt suffix is not preserved",
        )
        _require(
            l3_prompt.startswith(prefix + FIXTURE_INSTRUCTION),
            "original instruction is not the task-slot prefix",
        )
        _require(
            l3_prompt != baseline_prompt and block not in baseline_prompt,
            "L3 changed the baseline prompt or leaked into it",
        )
        _require(
            NAVIGATION_IDENTIFIER in l3_prompt,
            "official navigation identifier was lost",
        )

        l3_call = l3_record["model_call"]
        baseline_call = baseline_record["model_call"]
        _require(l3_call is not None, "L3 inference did not call model.generate")
        _require(baseline_call is not None, "baseline inference did not call generate")
        question = l3_call["prompts"][0][0]
        cur_prompt = question
        _require(
            question == update_prompt_view(l3_prompt),
            "update_prompt text is not prompt minus <image> and newlines",
        )
        _require(
            l3_call["prompts"] == [[cur_prompt]],
            "update_prompt did not receive the single navigation question",
        )
        _require(
            block.replace("\n", "") in question,
            "L3 text is missing from the update_prompt view",
        )
        _require(
            NAVIGATION_IDENTIFIER in question,
            "official navigation identifier is missing from update_prompt view",
        )
        baseline_question = baseline_call["prompts"][0][0]
        _require(
            baseline_question == update_prompt_view(baseline_prompt),
            "baseline update_prompt view is not derived from its prompt",
        )

        tokenizer = l3["agent"].tokenizer
        # Official oracle: same real conv template + real tokenizer_image_token.
        l3_oracle = _official_input_ids_oracle(l3_prompt, tokenizer, agent_module)
        baseline_oracle = _official_input_ids_oracle(
            baseline_prompt, tokenizer, agent_module,
        )
        l3_ids = l3_call["input_ids"][0].tolist()
        baseline_ids = baseline_call["input_ids"][0].tolist()
        _require(
            l3_ids == l3_oracle["input_ids"],
            "model input_ids differ from the official token assembly oracle",
        )
        _require(
            baseline_ids == baseline_oracle["input_ids"],
            "baseline input_ids differ from the official token assembly oracle",
        )
        _require(
            l3_ids.count(agent_module.IMAGE_TOKEN_INDEX) == 1,
            "the audited input must contain exactly one image placeholder",
        )
        _require(
            baseline_ids.count(agent_module.IMAGE_TOKEN_INDEX) == 1,
            "baseline input must contain exactly one image placeholder",
        )

        special = l3_oracle["special"]
        block_ids = tokenizer.encode(L3_SEPARATOR + block)
        l3_span = _find_subsequence(l3_ids, block_ids)
        _require(l3_span >= 0, "L3 text tokens are absent from input_ids")
        _require(
            _find_subsequence(baseline_ids, block_ids) < 0,
            "L3 text tokens leaked into baseline input_ids",
        )
        _require(
            l3_ids[:l3_span] + l3_ids[l3_span + len(block_ids):] == baseline_ids,
            "L3 input_ids are not exactly baseline plus the inserted block",
        )
        special_block = (
            special["video_start"] + special["image_sep"]
            + [agent_module.IMAGE_TOKEN_INDEX] + special["video_end"]
            + special["image_start"] + special["image_end"]
            + special["navigation"]
        )
        special_start = _find_subsequence(l3_ids, special_block)
        _require(special_start >= 0, "official visual/navigation block is missing")
        baseline_special_start = _find_subsequence(baseline_ids, special_block)
        _require(
            baseline_special_start == special_start,
            "the visual/navigation special-token block moved with L3",
        )
        _require(
            l3_ids[special_start:special_start + len(special_block)] == special_block,
            "the visual/navigation special-token block changed with L3",
        )
        _require(
            l3_ids[:special_start] == baseline_ids[:baseline_special_start],
            "the text before the visual/navigation block changed with L3",
        )
        _require(
            {
                key: value for key, value in l3_call["kwargs"].items()
                if key != "stopping_criteria"
            } == {
                key: value for key, value in baseline_call["kwargs"].items()
                if key != "stopping_criteria"
            },
            "generation parameters changed with L3",
        )
        for key, expected_value in {
            "do_sample": True, "temperature": 0.2,
            "max_new_tokens": 1024, "use_cache": True,
        }.items():
            _require(
                l3_call["kwargs"].get(key) == expected_value,
                f"official generation parameter {key} changed",
            )
        _require(
            len(l3_call["kwargs"].get("stopping_criteria", [])) == 1,
            "the official stopping criteria are missing",
        )
        _require(
            l3_call["images"] == baseline_call["images"] == ["VISION_FEATURE_TENSOR"],
            "vision input changed between A and B",
        )
        _require(
            len(l3["agent"].images_seen) == len(baseline["agent"].images_seen) > 0,
            "the RGB history length changed between A and B",
        )
        _require(
            np.array_equal(
                l3["agent"].images_seen[-1], baseline["agent"].images_seen[-1],
            ),
            "the current RGB frame changed between A and B",
        )

        l3_diagnostic = next(
            row for row in l3["agent"].step_diagnostics
            if row["step"] == l3_record["step"]
        )
        _require(
            l3_diagnostic["context_injected"] is True
            and l3_diagnostic["prompt_has_l3"] is True,
            "L3 step diagnostic did not record a prompt injection",
        )
        build_context_call = l3["build_context_calls"][-1]
        organization = build_context_call["organization"]
        _require(
            organization is not None and organization.context is not None,
            "the real serializer returned no context",
        )
        _require(
            organization.context.text == block,
            "the audited block differs from the organization context",
        )
        _require(
            len(organization.selected_facts) == l3_diagnostic["fact_count"],
            "fact count differs between organization and diagnostic",
        )
        _require(
            all(
                fact.geometry_verified for fact in organization.selected_facts
            ),
            "the serializer selected an unverified L3 fact",
        )

        l1_lists = l3["l1_event_lists"]
        _require(
            bool(l1_lists) and all(events == [] for events in l1_lists),
            "L1 events reached the L3 evidence minting path",
        )
        evidence = l3["sidecar"].object_map.evidence.values()
        _require(
            all(record.linked_l1_evidence_id is None for record in evidence),
            "L3 evidence was linked to L1",
        )
        _require(
            all(
                not fact.linked_l1_evidence_ids
                for fact in organization.selected_facts
            ),
            "an injected L3 fact carries L1 evidence",
        )

        baseline_question_view = baseline_call["prompts"][0][0]
        baseline_sha = prompt_sha256(baseline_prompt)
        l3_sha = prompt_sha256(l3_prompt)
        block_sha = prompt_sha256(block)
        diff = "\n".join(difflib.unified_diff(
            baseline_prompt.splitlines(),
            l3_prompt.splitlines(),
            fromfile=BASELINE_FILENAME,
            tofile=L3_FILENAME,
            lineterm="",
        )) + "\n"
        _require(
            diff.count("\n+") >= 1 and "@@" in diff,
            "line diff did not record the L3 insertion",
        )

        template_sha = prompt_sha256(template)
        prefix_sha = prompt_sha256(prefix)
        suffix_sha = prompt_sha256(suffix)
        decoded_l3 = tokenizer.batch_decode(
            torch.tensor([l3_ids], dtype=torch.long),
        )[0]
        _require(
            block in decoded_l3,
            "decoded structural model input does not contain the L3 block",
        )

        facts = []
        for fact in organization.selected_facts:
            facts.append({
                "object_id": fact.object_id,
                "label": fact.label,
                "distance_m": fact.distance_m,
                "bearing_deg": fact.bearing_deg,
                "latest_step": fact.latest_step,
                "viewpoints": fact.viewpoints,
                "confidence": fact.confidence,
                "geometry_verified": fact.geometry_verified,
                "past_episode": fact.past_episode,
                "evidence_ids": list(fact.evidence_ids),
                "source": fact.source,
            })

        call_chain = _call_chain()
        _require(
            all(stage.get("line") is not None for stage in call_chain),
            "a production call-chain line could not be located",
        )

        trace = {
            "audit_version": AUDIT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fixture": {
                "kind": "synthetic_fixture",
                "is_real_habitat_navigation_prompt": False,
                "description": (
                    "Deterministic 8x8 synthetic RGB-D frame + deterministic "
                    "fake segmenter through the real L3MapSidecar and real "
                    "L3OfficialPromptOrganizer. No Habitat scene, checkpoint, "
                    "image processor or Vicuna tokenizer is used."
                ),
                "instruction": FIXTURE_INSTRUCTION,
                "scene_id": FIXTURE_SCENE,
                "episode_id": FIXTURE_EPISODE,
                "steps_executed": FIXTURE_STEPS,
                "inference_step": l3_record["step"],
                "step_results": {
                    "baseline": baseline["step_results"],
                    "l3": l3["step_results"],
                },
            },
            "official_template": {
                "source_file": "agent_uninavid.py",
                "source_line": template_line,
                "attribute": "self.promt_template",
                "reference_copy": "l3_integration/official_prompt.py",
                "length_chars": len(template),
                "sha256": template_sha,
                "reference_sha256": REFERENCE_OFFICIAL_PROMPT_SHA256,
                "matches_reference": template == REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
                "task_slot_count": template.count("{}"),
                "navigation_identifier": NAVIGATION_IDENTIFIER,
                "prefix_sha256": prefix_sha,
                "suffix_sha256": suffix_sha,
            },
            "call_chain": call_chain,
            "injection": {
                "placement": "inside_official_task_slot",
                "mode": "B: official_template.format(instruction + L3)",
                "is_mode_a_append_after_prompt": False,
                "separator": repr(L3_SEPARATOR),
                "task_text": task_text,
                "task_text_sha256": prompt_sha256(task_text),
                "l3_block_start_in_prompt": injection_index,
                "l3_block_end_in_prompt": injection_index + len(block),
                "expected_start": expected_index,
                "instruction_is_prefix": l3_prompt.startswith(prefix + FIXTURE_INSTRUCTION),
                "official_suffix_preserved": l3_prompt.endswith(suffix),
                "official_navigation_identifier_present": (
                    NAVIGATION_IDENTIFIER in l3_prompt
                ),
                "baseline_prompt_sha256": baseline_sha,
                "l3_prompt_sha256": l3_sha,
            },
            "prompts": {
                "baseline": {
                    "path": BASELINE_FILENAME,
                    "chars": len(baseline_prompt),
                    "lines": len(baseline_prompt.splitlines()),
                    "sha256": baseline_sha,
                },
                "l3": {
                    "path": L3_FILENAME,
                    "chars": len(l3_prompt),
                    "lines": len(l3_prompt.splitlines()),
                    "sha256": l3_sha,
                },
                "l3_memory_block": {
                    "path": BLOCK_FILENAME,
                    "chars": len(block),
                    "lines": len(block.splitlines()),
                    "sha256": block_sha,
                },
                "diff": {
                    "path": DIFF_FILENAME,
                    "format": "unified line diff",
                    "chars": len(diff),
                },
            },
            "facts": {
                "fact_count": len(organization.selected_facts),
                "candidate_count": organization.candidate_count,
                "verified_count": organization.verified_count,
                "unverified_count": organization.unverified_count,
                "current_only_skipped": organization.current_only_skipped,
                "omitted_items": organization.omitted_items,
                "empty_reason": organization.empty_reason,
                "expression": organization.expression,
                "selected_facts": facts,
                "map_snapshot": l3["sidecar"].map_snapshot(),
                "history_only": all(
                    fact.latest_step < l3_record["step"]
                    for fact in organization.selected_facts
                ),
            },
            "model_input": {
                "navigation_qs_sha256": prompt_sha256(
                    l3_record["prompt"],
                ),
                "navigation_qs_is_l3_prompt": l3_record["prompt"] == l3_prompt,
                "task_text_passed_to_builder": task_text,
                "question": question,
                "question_sha256": prompt_sha256(question),
                "cur_prompt": cur_prompt,
                "cur_prompt_is_question": cur_prompt == question,
                "update_prompt_argument": l3_call["prompts"],
                "update_prompt_equals_question": (
                    l3_call["prompts"] == [[question]]
                ),
                "l3_in_navigation_qs": block in l3_prompt,
                "l3_in_question": block.replace("\n", "") in question,
                "l3_in_cur_prompt": block.replace("\n", "") in cur_prompt,
                "l3_in_input_ids": l3_span >= 0,
                "l3_token_span_structural": {
                    "start": l3_span,
                    "end": l3_span + len(block_ids),
                    "length": len(block_ids),
                    "kind": "character_tokenizer_structural_not_real",
                },
                "official_navigation_identifier_in_question": (
                    NAVIGATION_IDENTIFIER in question
                ),
                "mm_use_im_start_end": bool(
                    l3["agent"].model.config.mm_use_im_start_end
                ),
                "generate_called": True,
                "generate_kwargs": {
                    key: value for key, value in l3_call["kwargs"].items()
                    if key != "stopping_criteria"
                },
                "generate_has_stopping_criteria": (
                    len(l3_call["kwargs"].get("stopping_criteria", [])) == 1
                ),
                "vision_input_marker": l3_call["images"],
                "vision_input_unchanged": (
                    l3_call["images"] == baseline_call["images"]
                ),
                "conv_prompt": l3_oracle["conv_prompt"],
                "conv_prompt_image_literal_count": (
                    l3_oracle["conv_prompt"].count("<image>")
                ),
                "navigation_qs_image_literal_count": l3_prompt.count("<image>"),
                "question_image_literal_count": question.count("<image>"),
                "l3_prompt_newline_count": l3_prompt.count("\n"),
                "question_newline_count": question.count("\n"),
                "conv_prompt_source": (
                    "reconstructed by the official token-assembly oracle; the "
                    "captured input_ids match it exactly"
                ),
                "conv_prompt_sha256": prompt_sha256(l3_oracle["conv_prompt"]),
                "official_token_oracle_matches": True,
                "image_token_index": agent_module.IMAGE_TOKEN_INDEX,
                "image_token_positions": [
                    index for index, value in enumerate(l3_ids)
                    if value == agent_module.IMAGE_TOKEN_INDEX
                ],
                "special_block_start": special_start,
                "special_block_length": len(special_block),
                "special_block_unchanged": True,
                "baseline_image_token_positions": [
                    index for index, value in enumerate(baseline_ids)
                    if value == agent_module.IMAGE_TOKEN_INDEX
                ],
                "structural_input_ids": {
                    "length": len(l3_ids),
                    "sha256": hashlib.sha256(
                        json.dumps(l3_ids).encode("utf-8")
                    ).hexdigest(),
                    "baseline_length": len(baseline_ids),
                    "baseline_sha256": hashlib.sha256(
                        json.dumps(baseline_ids).encode("utf-8")
                    ).hexdigest(),
                    "kind": "character_tokenizer_structural_not_real",
                    "ids_redacted": True,
                    "note": (
                        "These are not Uni-NaVid token ids and are not a real "
                        "multimodal token count."
                    ),
                },
                "baseline_question": baseline_question_view,
                "baseline_question_sha256": prompt_sha256(
                    baseline_question_view,
                ),
            },
            "token_accounting": {
                "real_tokenizer_available": False,
                "token_count": None,
                "l3_block_token_count": None,
                "reason": (
                    "No local Uni-NaVid checkpoint/Vicuna tokenizer is "
                    "available. Only text-level results and structural token "
                    "positions are reported; no real token id or multimodal "
                    "token count is invented."
                ),
                "char_counts": {
                    "baseline_prompt": len(baseline_prompt),
                    "l3_prompt": len(l3_prompt),
                    "l3_memory_block": len(block),
                    "inserted_chars": len(l3_prompt) - len(baseline_prompt),
                },
                "utf8_bytes": {
                    "baseline_prompt": len(baseline_prompt.encode("utf-8")),
                    "l3_prompt": len(l3_prompt.encode("utf-8")),
                    "l3_memory_block": len(block.encode("utf-8")),
                },
                "diagnostic_context_tokens_structural": l3_diagnostic.get(
                    "context_tokens",
                ),
            },
            "l1": {
                "enabled": False,
                "evidence_event_lists": l1_lists,
                "all_event_lists_empty": all(
                    events == [] for events in l1_lists
                ),
                "linked_l1_evidence_count": 0,
                "selected_facts_with_l1_links": 0,
                "statement": (
                    "L3MapSidecar.observe passes an empty L1 event list; no L1 "
                    "text, event or fusion enters the map or the prompt."
                ),
            },
            "invariants": {
                "baseline_is_official_template_with_instruction": True,
                "l3_is_official_template_with_instruction_plus_l3": True,
                "mode_b_task_slot_injection": True,
                "not_mode_a_append_after_prompt": True,
                "official_template_byte_identical": True,
                "original_instruction_preserved": True,
                "official_suffix_preserved": True,
                "official_navigation_identifier_preserved": True,
                "l3_in_predict_inference_prompt": True,
                "l3_in_update_prompt_question": True,
                "l3_in_model_input_ids": True,
                "visual_token_layout_unchanged": True,
                "generation_parameters_unchanged": True,
                "l1_not_injected": True,
                "facts_are_history_only": all(
                    fact.latest_step < l3_record["step"]
                    for fact in organization.selected_facts
                ),
            },
            "limitations": [
                "synthetic_fixture: no real Habitat RGB-D/pose observation.",
                "No Uni-NaVid checkpoint or Vicuna tokenizer is available; "
                "token ids/counts are structural or null, never real.",
                "The image features are a marker string; visual encoding is "
                "not exercised.",
                "This audit does not run the 88-Episode evaluation.",
            ],
        }

        return {
            "baseline_prompt": baseline_prompt,
            "l3_prompt": l3_prompt,
            "l3_memory_block": block,
            "prompt_diff": diff,
            "trace": trace,
        }


def write_audit(output_dir: Path, audit: dict[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "baseline": output_dir / BASELINE_FILENAME,
        "l3": output_dir / L3_FILENAME,
        "block": output_dir / BLOCK_FILENAME,
        "diff": output_dir / DIFF_FILENAME,
        "trace": output_dir / TRACE_FILENAME,
    }
    # Exact bytes: no added newline, so the recorded SHA256 is the prompt's.
    paths["baseline"].write_text(audit["baseline_prompt"], encoding="utf-8")
    paths["l3"].write_text(audit["l3_prompt"], encoding="utf-8")
    paths["block"].write_text(audit["l3_memory_block"], encoding="utf-8")
    paths["diff"].write_text(audit["prompt_diff"], encoding="utf-8")
    paths["trace"].write_text(
        json.dumps(audit["trace"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "output" / "prompt_audit"),
        help="directory for the five audit artifacts",
    )
    args = parser.parse_args(argv)
    audit = build_audit()
    paths = write_audit(Path(args.output), audit)
    trace = audit["trace"]
    print(f"wrote {paths['baseline']}")
    print(f"wrote {paths['l3']}")
    print(f"wrote {paths['block']}")
    print(f"wrote {paths['diff']}")
    print(f"wrote {paths['trace']}")
    print(
        "mode={mode}; facts={facts}; chars={chars}; sha256={sha}".format(
            mode=trace["injection"]["mode"],
            facts=trace["facts"]["fact_count"],
            chars=trace["prompts"]["l3"]["chars"],
            sha=trace["prompts"]["l3"]["sha256"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

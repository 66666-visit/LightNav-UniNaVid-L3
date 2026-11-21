"""Official Uni-NaVid prompt snapshot and visual/navigation token order."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest
import torch

from l3_integration.official_prompt import (
    NAVIGATION_IDENTIFIER,
    REFERENCE_OFFICIAL_PROMPT_SHA256,
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
    build_official_prompt,
    build_prompt_with_l3,
    update_prompt_view,
)

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT.parent / "references" / "Uni-NaVid" / "offline_eval_uninavid.py"
AGENT = ROOT / "agent_uninavid.py"


def _literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} assignment not found in {path}")


def test_official_template_is_the_literal_from_both_sources() -> None:
    agent_template = _literal(AGENT, "promt_template")
    assert agent_template == REFERENCE_OFFICIAL_PROMPT_TEMPLATE
    assert len(agent_template.encode("utf-8")) == 340
    assert hashlib.sha256(agent_template.encode("utf-8")).hexdigest() == (
        REFERENCE_OFFICIAL_PROMPT_SHA256
    )
    assert agent_template.count("{}") == 1
    assert NAVIGATION_IDENTIFIER in agent_template
    assert "determine your next four actions" in agent_template
    assert "forward, left, right, or stop" in agent_template
    if OFFICIAL.is_file():
        official_template = _literal(OFFICIAL, "promt_template")
        assert official_template == agent_template


def test_memory_insertion_preserves_every_fixed_character() -> None:
    instruction = "Go past the sofa and stop at the table"
    memory = (
        "[Navigation Context - L1]\n"
        "Now: chair (center), conf 0.90, 1 sighting since step 0, last step 0."
    )
    base = build_official_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction)
    injected = build_prompt_with_l3(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, memory,
    )
    prefix, suffix = REFERENCE_OFFICIAL_PROMPT_TEMPLATE.split("{}")
    assert base == prefix + instruction + suffix
    assert injected == prefix + instruction + "\n" + memory + suffix
    assert injected.startswith(prefix + instruction)
    assert injected.endswith(suffix)
    assert injected.count(NAVIGATION_IDENTIFIER) == 1
    assert injected.count("<image>") == 1
    assert "[Navigation]" not in injected


def test_instruction_is_unchanged_and_is_the_task_text_prefix() -> None:
    instruction = "  Walk to the kitchen, then find the green cup. "
    memory = "Map record (step 2, 1 view): the cup is to your front-left, 2.4 m."
    prompt = build_prompt_with_l3(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, memory,
    )
    prefix, suffix = REFERENCE_OFFICIAL_PROMPT_TEMPLATE.split("{}")
    assert prompt.startswith(prefix + instruction)
    assert prompt == prefix + instruction + "\n" + memory + suffix
    assert instruction in update_prompt_view(prompt)


def test_empty_memory_is_byte_identical_to_the_official_prompt() -> None:
    instruction = "find the chair"
    base = build_official_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction)
    for memory in (None, "", "   \n\t"):
        assert build_prompt_with_l3(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, memory,
        ) == base


@pytest.mark.parametrize("prompt", [
    build_official_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, "find the chair"),
    build_prompt_with_l3(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, "find the chair",
        "[Navigation Context - L1+L3]\nStep 1; RGB is the current view.",
    ),
])
def test_visual_and_navigation_tokens_keep_the_official_order(prompt, monkeypatch):
    from tests.test_prompt_tokens import (
        CharTokenizer,
        _find,
        _occurrences,
        _official_input_ids,
        _special_ids,
    )

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    tokenizer = CharTokenizer()
    ids = _official_input_ids(prompt, tokenizer)[0].tolist()
    special = _special_ids(tokenizer)
    block = (
        special["<video_special>"] + special["<image_sep>"] + [-200]
        + special["</video_special>"] + special["<image_special>"]
        + special["</image_special>"] + special["[Navigation]"]
    )
    assert _occurrences(ids, block) == 1
    image_at = ids.index(-200)
    block_at = _find(ids, special["<video_special>"])
    assert block_at >= 0
    assert block_at < image_at < block_at + len(block)
    instruction_ids = tokenizer.encode("find the chair")
    instruction_at = next(
        index for index in range(len(ids) - len(instruction_ids) + 1)
        if ids[index:index + len(instruction_ids)] == instruction_ids
        and index > block_at + len(block)
    )
    assert block_at < instruction_at
    assert "forward, left, right, or stop" in tokenizer.batch_decode([ids])[0]

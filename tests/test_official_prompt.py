"""Requirement 1: the official Uni-NaVid prompt stays byte-identical."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from l3_integration.official_prompt import (
    NAVIGATION_IDENTIFIER,
    REFERENCE_OFFICIAL_PROMPT_SHA256,
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
    build_official_prompt,
    build_prompt_with_l3,
    build_task_text,
    prompt_sha256,
    split_official_template,
    update_prompt_view,
    verify_task_text_injection,
)

AGENT_PATH = Path(__file__).resolve().parents[1] / "agent_uninavid.py"
L3_HEADER = (
    "[Navigation Map Memory - L3; history, not current view; directions "
    "relative to you; straight-line ranges, not paths; passage unverified]"
)


def _agent_template() -> str:
    tree = ast.parse(AGENT_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "promt_template"
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("self.promt_template assignment not found")


def test_agent_template_is_byte_identical_to_the_official_oracle() -> None:
    actual = _agent_template()
    assert actual == REFERENCE_OFFICIAL_PROMPT_TEMPLATE
    assert hashlib.sha256(actual.encode("utf-8")).hexdigest() == (
        REFERENCE_OFFICIAL_PROMPT_SHA256
    )
    assert len(actual) == 340
    assert actual.count("{}") == 1


def test_official_prompt_build_does_not_rewrite_the_template() -> None:
    prompt = build_official_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, "go to the chair")
    assert prompt == REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format("go to the chair")
    assert prompt.startswith(
        "Imagine you are a robot programmed for navigation tasks. "
        "You have been given a video of historical observations"
    )
    assert NAVIGATION_IDENTIFIER in prompt
    assert "'go to the chair'" in prompt
    assert prompt.endswith(
        "The predicted action should be one of the following: forward, left, "
        "right, or stop."
    )


def test_l3_is_inside_the_official_task_text_slot() -> None:
    instruction = "find the chair"
    context = L3_HEADER + "\nMap record (step 3, 1 view): the chair is in front of you, 2.0 m straight-line."
    prefix, suffix = split_official_template(REFERENCE_OFFICIAL_PROMPT_TEMPLATE)
    task_text = build_task_text(instruction, context)
    prompt = build_prompt_with_l3(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, context,
    )
    assert task_text == instruction + "\n" + context
    assert prompt == prefix + task_text + suffix
    assert prompt.startswith(prefix + instruction)
    assert prompt.endswith(suffix)
    assert context in prompt
    assert not prompt.endswith(context)
    assert prompt.index(context) < prompt.index(suffix)
    assert prompt != build_official_prompt(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction,
    ) + "\n" + context
    verify_task_text_injection(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, context, prompt,
    )

    update_view = update_prompt_view(prompt)
    assert context.replace("\n", "") in update_view
    assert update_view.startswith(update_prompt_view(prefix + instruction))
    assert update_view.endswith(update_prompt_view(suffix))
    assert NAVIGATION_IDENTIFIER in update_view


def test_empty_l3_context_leaves_the_prompt_unchanged() -> None:
    instruction = "stop at the door"
    official = build_official_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction)
    assert build_prompt_with_l3(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, "",
    ) == official
    assert build_prompt_with_l3(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, None,
    ) == official


def test_prompt_guard_rejects_template_rewrites_and_wrong_l3_position() -> None:
    instruction = "find the chair"
    context = L3_HEADER + "\nMap record (step 1, 1 view): the chair is to your left, 2.0 m straight-line."
    official = build_official_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction)
    prefix, suffix = split_official_template(REFERENCE_OFFICIAL_PROMPT_TEMPLATE)
    with pytest.raises(ValueError):
        verify_task_text_injection(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, context,
            official.replace("Imagine", "Act as"),
        )
    # Old, wrong placement: appended after the complete prompt.
    with pytest.raises(ValueError):
        verify_task_text_injection(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, context,
            official + "\n" + context,
        )
    # Wrong placement: before the original instruction.
    with pytest.raises(ValueError):
        verify_task_text_injection(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, context,
            prefix + context + "\n" + instruction + suffix,
        )
    # Missing L3.
    with pytest.raises(ValueError):
        verify_task_text_injection(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, context,
            official,
        )


def test_prompt_sha_is_stable() -> None:
    assert prompt_sha256("abc") == hashlib.sha256(b"abc").hexdigest()
    with pytest.raises(ValueError):
        prompt_sha256(None)  # type: ignore[arg-type]


def test_predict_inference_derives_both_paths_from_one_prompt() -> None:
    """The generation path and update_prompt path share the same ``prompt``.

    This is a source-level contract check: the L3 code must not introduce a
    second prompt construction path.
    """
    source = AGENT_PATH.read_text(encoding="utf-8")
    assert "question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\\n', '')" in source
    assert "qs = prompt" in source
    assert "cur_prompt = question" in source
    assert "self.model.update_prompt([[cur_prompt]])" in source
    assert "navigation = self.predict_inference(navigation_qs)" in source
    assert "navigation_qs = build_prompt(self.promt_template, task_text)" in source
    assert "task_text, l3_context, l3_diagnostic = self._augment_task_text(" in source
    assert "verify_task_text_injection(" in source
    assert "append_l3_context" not in source


def test_build_prompt_uses_the_official_format_operation() -> None:
    instruction = "find the chair"
    assert build_official_prompt(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction,
    ) == REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format(instruction)
    context = "Map record: a {literal} brace in a label"
    assert build_prompt_with_l3(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, context,
    ) == REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format(
        instruction + "\n" + context,
    )


def test_invalid_l3_type_is_rejected_by_the_task_text_builder() -> None:
    with pytest.raises(ValueError, match="l3_memory"):
        build_task_text("find the chair", 123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="instruction"):
        build_task_text(None)  # type: ignore[arg-type]


def test_verify_rejects_a_prompt_with_a_rewritten_official_suffix() -> None:
    instruction = "find the chair"
    context = "Map record (step 1): the chair is in front of you, 2.0 m straight-line."
    prompt = REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format(instruction + "\n" + context)
    with pytest.raises(ValueError):
        verify_task_text_injection(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, context,
            prompt.replace("forward, left, right, or stop", "forward or left"),
        )


def test_whitespace_only_l3_is_treated_as_empty() -> None:
    instruction = "find the chair"
    official = build_official_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction)
    for blank in ("", "   ", "\n", "\n\t "):
        assert build_prompt_with_l3(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE, instruction, blank,
        ) == official

"""Official Uni-NaVid prompt preservation and task-text L3 injection.

The official template is kept as a literal in ``agent_uninavid.py``.  This
module contains a verified reference copy for tests and the *task-text only*
injection helpers:

    task_text = original_instruction
    if l3_memory:
        task_text = original_instruction + "\\n" + l3_memory
    navigation_qs = official_template.format(task_text)

This follows Uni-NaVid issue #19: the prepared text replaces the navigation
instruction tokens inside the official task slot.  The L3 block is **not**
appended after the complete official prompt.  No role, chain-of-thought,
extra navigation command or new map format is added here.
"""

from __future__ import annotations

import hashlib

#: Exact template from ``agent_uninavid.py`` at commit ``ce2f804`` (340 bytes).
#: This copy is a test reference, not a second production template.
REFERENCE_OFFICIAL_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. You have been given "
    "a video of historical observations and an image of the current observation "
    "<image>. Your assigned task is: '{}'. Analyze this series of images to "
    "determine your next four actions. The predicted action should be one of the "
    "following: forward, left, right, or stop."
)

REFERENCE_OFFICIAL_PROMPT_SHA256 = (
    "435d79868156aaa1352afcd98f4ceae2f03a54f8cdc5fe87e41d32774d5512c1"
)

#: The exact phrase Uni-NaVid uses to recognise a navigation sample.
NAVIGATION_IDENTIFIER = (
    "a video of historical observations and an image of the current observation"
)

#: The official template has exactly one task-text slot.
TASK_SLOT = "{}"
L3_SEPARATOR = "\n"


def split_official_template(template: str) -> tuple[str, str]:
    """Return the immutable prefix and suffix around the single ``{}`` slot."""
    if not isinstance(template, str) or not template:
        raise ValueError("official prompt template must be a nonempty string")
    if template.count(TASK_SLOT) != 1:
        raise ValueError("official prompt template must contain exactly one {} slot")
    prefix, suffix = template.split(TASK_SLOT)
    if not prefix or not suffix:
        raise ValueError("official prompt template slot must be surrounded by text")
    return prefix, suffix


def build_task_text(instruction: str, l3_memory: str | None = None) -> str:
    """Build the exact text that goes inside the official ``{}`` slot.

    The original instruction is always the prefix.  L3 is appended only to the
    task text, never to the complete prompt.  Empty/None L3 returns the original
    instruction unchanged (byte-identical baseline).
    """
    if not isinstance(instruction, str):
        raise ValueError("instruction must be a string")
    if l3_memory is None:
        return instruction
    if not isinstance(l3_memory, str):
        raise ValueError("l3_memory must be a string or None")
    if not l3_memory.strip():
        return instruction
    return instruction + L3_SEPARATOR + l3_memory


def build_prompt(template: str, task_text: str) -> str:
    """Fill the official template with ``str.format``, exactly as upstream does.

    ``split_official_template`` validates the single ``{}`` slot first.  The
    call itself is the official ``self.promt_template.format(task_text)``;
    concatenating prefix/task/suffix would be equivalent only for brace-free
    task text, so the official formatting operation is kept.
    """
    split_official_template(template)
    if not isinstance(task_text, str):
        raise ValueError("task_text must be a string")
    return template.format(task_text)


def build_official_prompt(template: str, instruction: str) -> str:
    """Format the official template with the original instruction only."""
    return build_prompt(template, build_task_text(instruction))


def build_prompt_with_l3(
    template: str, instruction: str, l3_memory: str | None = None,
) -> str:
    """Build the official prompt with L3 inside the task-text slot."""
    return build_prompt(template, build_task_text(instruction, l3_memory))


def update_prompt_view(generation_prompt: str) -> str:
    """Reproduce exactly what ``predict_inference`` passes to ``update_prompt``.

    The official method derives ``question`` from the same ``prompt`` argument:

    ``prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\\n', '')``

    There is no second prompt builder.
    """
    if not isinstance(generation_prompt, str):
        raise ValueError("generation prompt must be a string")
    # ``DEFAULT_IMAGE_TOKEN`` is the literal "<image>" in uninavid.constants.
    return generation_prompt.replace("<image>", "").replace("\n", "")


def verify_task_text_injection(
    template: str,
    instruction: str,
    l3_memory: str | None,
    generation_prompt: str,
) -> None:
    """Fail closed unless L3 is present only inside the official task slot.

    The comparison is exact: the generated prompt must equal
    ``prefix + (instruction [+ newline + L3]) + suffix``.  This rejects both
    template edits and the old, wrong placement that appended L3 after the
    complete prompt.
    """
    prefix, suffix = split_official_template(template)
    if not isinstance(instruction, str):
        raise ValueError("instruction must be a string")
    if not isinstance(generation_prompt, str):
        raise ValueError("generation prompt must be a string")
    task_text = build_task_text(instruction, l3_memory)
    expected = prefix + task_text + suffix
    if generation_prompt != expected:
        raise ValueError("generated prompt differs from the official task-slot format")
    if not generation_prompt.startswith(prefix + instruction):
        raise ValueError("original instruction is not preserved at the task-slot start")
    if not generation_prompt.endswith(suffix):
        raise ValueError("official template suffix is not preserved")
    if NAVIGATION_IDENTIFIER not in generation_prompt:
        raise ValueError("official navigation identifier was lost")
    if isinstance(l3_memory, str) and l3_memory.strip():
        if l3_memory not in generation_prompt:
            raise ValueError("L3 memory is missing from the task text")
        if generation_prompt.endswith(L3_SEPARATOR + l3_memory):
            raise ValueError("L3 was appended after the complete official prompt")
        normalized = update_prompt_view(generation_prompt)
        if l3_memory.replace("\n", "") not in normalized:
            raise ValueError("L3 memory is missing from update_prompt's view")


def prompt_sha256(text: str) -> str:
    """Stable digest used in per-step diagnostics and result rows."""
    if not isinstance(text, str):
        raise ValueError("prompt must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

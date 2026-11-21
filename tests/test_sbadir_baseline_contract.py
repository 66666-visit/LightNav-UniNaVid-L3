"""Verify the frozen 7602b25 L3 prompt and visual token code remain intact."""

from __future__ import annotations

import ast
import copy
import hashlib
import subprocess
from pathlib import Path

from l3_integration.official_prompt import REFERENCE_OFFICIAL_PROMPT_TEMPLATE
from l3_memory.l3_official_prompt import HEADER


BASE = "7602b254ea514aa71b0057e41b5db3bdfedf9ffa"
ROOT = Path(__file__).resolve().parents[1]


def _frozen(path):
    return subprocess.check_output(
        ["git", "show", BASE + ":" + path], cwd=str(ROOT),
    )


def _method(source, name):
    tree = ast.parse(source)
    agent = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "UniNaVid_Agent")
    return next(node for node in agent.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def _without_exact_budget_guard(method):
    result = copy.deepcopy(method)
    guards = [node for node in result.body if isinstance(node, ast.If)
              and "SBAdir exact input tokens exceed checkpoint context reserve" in ast.dump(node)]
    assert len(guards) == 1
    result.body.remove(guards[0])
    return result


def test_l3_organizer_source_is_exactly_the_7602b25_baseline():
    for path in ("l3_memory/l3_official_prompt.py", "l3_memory/l3_prompt_organizer.py"):
        assert (ROOT / path).read_bytes() == _frozen(path)


def test_visual_token_assembly_and_generation_parameters_are_exact_baseline():
    baseline = _frozen("agent_uninavid.py").decode("utf-8")
    current = (ROOT / "agent_uninavid.py").read_text(encoding="utf-8")
    for name in ("process_images", "extract_result"):
        assert ast.dump(_method(current, name), include_attributes=False) == ast.dump(
            _method(baseline, name), include_attributes=False,
        )
    assert ast.dump(
        _without_exact_budget_guard(_method(current, "predict_inference")),
        include_attributes=False,
    ) == ast.dump(_method(baseline, "predict_inference"), include_attributes=False)


def test_frozen_l3_prompt_snapshot():
    memory = (
        HEADER + "\nMap record (step 0, 1 view): the chair is in front "
        "of you, 2.0 m straight-line."
    )
    prompt = REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format("find the chair\n" + memory)
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == (
        "4b63d2969209037a5a94d9fff5ab040288689b8b3bac9856ddc2fddf70e7665d"
    )
    assert prompt.count("<image>") == 1

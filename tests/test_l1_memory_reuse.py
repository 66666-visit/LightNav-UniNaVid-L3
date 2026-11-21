"""The L1 modules are vendored LightNav code, not a reimplementation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "l1_memory"


def _normalise(vendored: str) -> str:
    text = vendored
    text = text.replace(
        "from l3_memory.lexical import _terms",
        "from lightnav.memory.v3_retrieval import _terms",
    )
    text = text.replace(
        "from l1_memory.compat import instruction_match",
        "from lightnav.memory.memory_write_gate import instruction_match",
    )
    text = text.replace("from l1_memory.compat import removeprefix\n", "")
    text = re.sub(
        r"removeprefix\(([^,\n]+), (\"[^\"]*\")\)",
        r"\1.removeprefix(\2)",
        text,
    )
    text = text.replace("l3_memory.", "lightnav.memory.")
    text = text.replace("l1_memory.", "lightnav.memory.")
    return text


def test_vendored_l1_files_match_recorded_hashes_and_differ_only_in_adapters() -> None:
    provenance = json.loads((PACKAGE / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_commit"] == "6aefffa"
    assert len(provenance["files"]) == 13
    for entry in provenance["files"]:
        vendored = PACKAGE / entry["name"]
        assert hashlib.sha256(vendored.read_bytes()).hexdigest() == (
            entry["vendored_sha256"]
        ), entry["name"]
        source = Path(entry["source"])
        if not source.is_file():
            continue
        assert hashlib.sha256(source.read_bytes()).hexdigest() == (
            entry["source_sha256"]
        ), entry["name"]
        assert _normalise(vendored.read_text(encoding="utf-8")) == (
            source.read_text(encoding="utf-8")
        ), entry["name"]


def test_the_real_lightnav_functions_are_imported_by_the_backend() -> None:
    from l1_integration.memory_backend import PromptMemoryBackend
    from l1_memory.joint_prompt import JointPromptBuilder
    from l1_memory.l1_prompt_organizer import L1PromptOrganizer
    from l1_memory.local_graph_fusion import LocalGraphFusion
    from l1_memory.memory_writer import MemoryWriter
    from l1_memory.prompt_assembly import build_prompt_context

    assert PromptMemoryBackend is not None
    assert JointPromptBuilder.__module__ == "l1_memory.joint_prompt"
    assert L1PromptOrganizer.__module__ == "l1_memory.l1_prompt_organizer"
    assert LocalGraphFusion.__module__ == "l1_memory.local_graph_fusion"
    assert MemoryWriter.__module__ == "l1_memory.memory_writer"
    assert build_prompt_context.__module__ == "l1_memory.prompt_assembly"

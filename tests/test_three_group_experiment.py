"""Frozen A/B/C contracts: no L1 in B, identical L3 facts, real sources."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from l1_integration.sbadir_backend import L1SelectedFact, L1SourceEvidence
from l3_integration.experiment_groups import (
    A_GROUP,
    A_NAME,
    B_GROUP,
    B_NAME,
    C_GROUP,
    C_NAME,
    spec,
    validate_model_paths,
)
from l3_integration.official_prompt import (
    REFERENCE_OFFICIAL_PROMPT_SHA256,
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
    build_prompt,
    build_task_text,
    verify_task_text_injection,
)
from l3_integration.sbadir_fusion import compose_sbadir
from l3_memory.l3_official_prompt import HEADER

ROOT = Path(__file__).resolve().parents[1]
L3_LINE = "Map record (step 2, 1 view): the chair is in front of you, 2.0 m straight-line."
L3_TEXT = HEADER + "\n" + L3_LINE
INSTRUCTION = "find the chair"


def _count(text):
    return len(text.split())


def _l3(label="chair", linked=(), evidence=("seg-1",), **overrides):
    fields = dict(
        object_id="map-1", label=label, latest_step=2, last_step=2,
        source="yolo11n-seg:seg.pt", confidence=0.87,
        geometry_verified=True, evidence_ids=tuple(evidence),
        linked_l1_evidence_ids=tuple(linked),
        line=lambda: L3_LINE, viewpoints=1,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _l1(label="chair", evidence_id="rgb-1", sector="left"):
    source = L1SourceEvidence(
        evidence_id=evidence_id, observation_step=2, object_label=label,
        detection_confidence=0.91, detector_name="ultralytics_yolo:detect.pt",
        rgb_observation_source="observation.rgb",
        bbox_xyxy=(1.0, 2.0, 10.0, 12.0), source_evidence_ids=(),
    )
    return L1SelectedFact(
        text="Now: %s (%s), conf 0.91, 1 sighting since step 2, last step 2."
             % (label, sector),
        label=label, role="current", sector=sector, first_step=2,
        last_step=2, sightings=1, evidence_ids=(evidence_id,),
        evidence=(source,), selection_source="l1_prompt_organizer",
    )


def _compose(l3_facts=None, l1_facts=(), text=L3_TEXT, entity_links=None):
    return compose_sbadir(
        template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
        instruction=INSTRUCTION,
        l3_text=text,
        l3_facts=[_l3()] if l3_facts is None else list(l3_facts),
        l1_facts=list(l1_facts),
        count_tokens=_count,
        max_prompt_tokens=10000,
        l1_max_tokens=160,
        entity_links=entity_links,
    )


def test_group_specs_freeze_the_three_required_conditions():
    a, b, c = spec(A_GROUP), spec(B_GROUP), spec(C_GROUP)
    assert (a.name, b.name, c.name) == (A_NAME, B_NAME, C_NAME)
    assert a.name == "l3_organized_baseline_7602b25"
    assert b.name == "l3_sbadir_source_background_only"
    assert c.name == "l3_l1_sbadir"
    assert (a.uses_l3, a.source_background, a.uses_l1) == (True, False, False)
    assert (b.uses_l3, b.source_background, b.uses_l1) == (True, True, False)
    assert (c.uses_l3, c.source_background, c.uses_l1) == (True, True, True)
    assert a.worktree == "baseline" and not b.worktree == "baseline"


def test_a_prompt_is_the_original_l3_text_without_any_source_background():
    assert spec(A_GROUP).source_background is False
    task_text = build_task_text(INSTRUCTION, L3_TEXT)
    prompt = build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, task_text)
    expected = REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format(
        INSTRUCTION + "\n" + L3_TEXT
    )
    assert prompt == expected
    assert "[L3 RGB-D map" not in prompt
    assert "[L1 RGB observation" not in prompt
    assert "evidence seg-1" not in prompt
    verify_task_text_injection(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, INSTRUCTION, L3_TEXT, prompt,
    )


def test_b_never_constructs_l1_and_its_prompt_has_no_l1_text():
    assert spec(B_GROUP).uses_l1 is False
    validate_model_paths(
        B_GROUP, l3_seg_model="seg.pt",
        l1_det_model="", l1_place_model="",
    )
    with pytest.raises(ValueError, match="must not receive L1 model paths"):
        validate_model_paths(
            B_GROUP, l3_seg_model="seg.pt",
            l1_det_model="detect.pt", l1_place_model="dinov2",
        )
    with pytest.raises(ValueError, match="requires --l1-det-model"):
        validate_model_paths(
            C_GROUP, l3_seg_model="seg.pt",
            l1_det_model="", l1_place_model="",
        )
    validate_model_paths(
        C_GROUP, l3_seg_model="seg.pt",
        l1_det_model="detect.pt", l1_place_model="dinov2",
    )

    result = _compose(l1_facts=())
    assert result.l1_text is None
    assert result.diagnostics["l1_fact_count"] == 0
    assert result.diagnostics["l1_retained_facts"] == 0
    assert result.diagnostics["l1_retained_evidence_ids"] == []
    assert result.diagnostics["duplicated_facts"] == 0
    assert result.diagnostics["conflicting_facts"] == 0
    assert "L1 RGB" not in result.memory_text
    prompt = build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, result.task_text)
    assert "[L1 RGB" not in prompt


def test_b_keeps_a_l3_fact_lines_order_and_signature_exactly():
    original_lines = L3_TEXT.splitlines()
    result = _compose(l1_facts=())
    assert result.diagnostics["l3_core_unchanged"] is True
    assert result.l3_enriched_text.splitlines()[0] == original_lines[0]
    enriched_lines = result.l3_enriched_text.splitlines()
    assert len(enriched_lines) == len(original_lines)
    for original, enriched in zip(original_lines, enriched_lines):
        assert enriched.startswith(original)
    assert result.diagnostics["l3_original_sha256"] == hashlib.sha256(
        L3_TEXT.encode("utf-8")).hexdigest()
    identity = [{
        "object_id": "map-1", "label": "chair",
        "evidence_ids": ["seg-1"], "line": L3_LINE,
    }]
    assert result.diagnostics["l3_fact_signature_sha256"] == hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def test_b_pairs_each_l3_fact_with_its_real_evidence_fields():
    result = _compose(l1_facts=())
    enriched = result.l3_enriched_text
    assert "L3 RGB-D map" in enriched
    assert "source yolo11n-seg:seg.pt" in enriched
    assert "map node map-1" in enriched
    assert "geometry observed" in enriched
    assert "step 2" in enriched
    assert "seg conf 0.87" in enriched
    assert "evidence seg-1" in enriched
    assert result.diagnostics["l3_provenance_count"] == 1

    missing = _l3(source="", confidence=None, evidence=(), latest_step=-1,
                  geometry_verified=False)
    unavailable = _compose(l3_facts=[missing], l1_facts=())
    text = unavailable.l3_enriched_text
    assert "source unavailable" in text
    assert "geometry unverified" in text
    assert "step unavailable" in text
    assert "seg conf unavailable" in text
    assert "evidence unavailable" in text


def test_c_keeps_b_l3_fact_text_and_source_background_identical():
    b = _compose(l1_facts=())
    c = _compose(
        l3_facts=[_l3(linked=("rgb-1",))], l1_facts=[_l1()],
        entity_links={"rgb-1": "map-1"},
    )
    # C's L3 fact carries an L1 entity link; the fact text and the source
    # background must still be byte-identical to B.
    assert c.l3_enriched_text == b.l3_enriched_text
    assert c.diagnostics["l3_enriched_sha256"] == b.diagnostics["l3_enriched_sha256"]
    assert c.diagnostics["l3_original_sha256"] == b.diagnostics["l3_original_sha256"]
    assert (c.diagnostics["l3_fact_signature_sha256"]
            == b.diagnostics["l3_fact_signature_sha256"])
    assert c.memory_text.splitlines()[:2] == b.memory_text.splitlines()
    assert len(c.memory_text.splitlines()) == 3


def test_c_adds_l1_facts_and_their_real_sources_after_the_l3_block():
    result = _compose(
        l3_facts=[_l3(linked=("rgb-1",))], l1_facts=[_l1()],
        entity_links={"rgb-1": "map-1"},
    )
    assert result.diagnostics["l1_fact_count"] == 1
    assert result.diagnostics["l1_retained_facts"] == 1
    assert result.diagnostics["l1_retained_evidence_ids"] == [["rgb-1"]]
    assert "L1 RGB observation also supports map node map-1" in result.l1_text
    assert "source ultralytics_yolo:detect.pt" in result.l1_text
    assert "RGB input observation.rgb" in result.l1_text
    assert "detector conf 0.91" in result.l1_text
    assert "evidence rgb-1" in result.l1_text
    prompt = build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, result.task_text)
    assert result.l3_enriched_text in prompt
    assert result.l1_text in prompt
    assert prompt.index(result.l3_enriched_text) < prompt.index(result.l1_text)


def test_same_entity_conflict_keeps_both_facts_values_and_evidence():
    result = _compose(
        l3_facts=[_l3(linked=("rgb-1",))],
        l1_facts=[_l1(label="table")],
        entity_links={"rgb-1": "map-1"},
    )
    assert result.diagnostics["conflicting_facts"] == 1
    assert result.diagnostics["duplicated_facts"] == 0
    record = result.diagnostics["conflicts"][0]
    assert record["entity_id"] == "map-1"
    assert record["l1_value"] == "table"
    assert record["l3_value"] == "chair"
    assert record["l1_evidence_ids"] == ["rgb-1"]
    assert record["l3_evidence_ids"] == ["seg-1"]
    assert record["resolution"] == "both_retained"
    assert "the chair" in result.memory_text
    assert "table" in result.memory_text
    assert "L3 RGB-D map" in result.memory_text
    assert "L1 RGB observation" in result.memory_text


def test_conflict_entity_id_and_both_evidence_ids_enter_the_official_prompt():
    result = _compose(
        l3_facts=[_l3(linked=("rgb-1",))],
        l1_facts=[_l1(label="table")],
        entity_links={"rgb-1": "map-1"},
    )
    assert "same map entity map-1" in result.memory_text
    assert "L3 reports chair" in result.memory_text
    assert "both retained" in result.memory_text
    assert "both retained" in result.l1_text
    prompt = build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, result.task_text)
    assert "same map entity map-1" in prompt
    assert "evidence rgb-1" in prompt
    assert "evidence seg-1" in prompt
    # Neither value is rewritten and no winner is declared.
    assert "table" in prompt and "chair" in prompt
    assert "correct" not in prompt.casefold()


def test_official_prompt_snapshot_is_identical_across_a_b_c():
    assert REFERENCE_OFFICIAL_PROMPT_SHA256 == hashlib.sha256(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    a_task = build_task_text(INSTRUCTION, L3_TEXT)
    b = _compose(l1_facts=())
    c = _compose(
        l3_facts=[_l3(linked=("rgb-1",))], l1_facts=[_l1()],
        entity_links={"rgb-1": "map-1"},
    )
    prompts = [
        build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, a_task),
        build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, b.task_text),
        build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, c.task_text),
    ]
    for prompt in prompts:
        assert prompt.count("<image>") == 1
        assert prompt.startswith(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE.split("{}")[0] + INSTRUCTION)
        assert prompt.endswith(REFERENCE_OFFICIAL_PROMPT_TEMPLATE.split("{}")[1])
    verify_task_text_injection(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, INSTRUCTION, b.memory_text, prompts[1])
    verify_task_text_injection(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, INSTRUCTION, c.memory_text, prompts[2])
    # Frozen snapshot of the A prompt with the exact 7602b25 L3 block.
    frozen_memory = (
        HEADER + "\nMap record (step 0, 1 view): the chair is in front "
        "of you, 2.0 m straight-line."
    )
    frozen_prompt = REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format(
        "find the chair\n" + frozen_memory)
    assert hashlib.sha256(frozen_prompt.encode("utf-8")).hexdigest() == (
        "4b63d2969209037a5a94d9fff5ab040288689b8b3bac9856ddc2fddf70e7665d"
    )


def test_python38_syntax_and_api_compatibility_for_changed_modules():
    paths = [
        "l3_integration/experiment_groups.py",
        "l3_integration/sbadir_fusion.py",
        "agent_uninavid.py",
        "run.py",
        "scripts/build_l3_l1_report.py",
        "scripts/run_l3_vs_l3l1.sh",
    ]
    for relative in paths[:-1]:
        source = (ROOT / relative).read_text(encoding="utf-8")
        ast.parse(source, filename=relative, feature_version=(3, 8))
        compile(source, relative, "exec")
    script = (ROOT / paths[-1]).read_text(encoding="utf-8")
    assert "CONFIRM=YES" in script
    for relative in paths[:-1]:
        source = (ROOT / relative).read_text(encoding="utf-8")
        for forbidden in ("removeprefix(", "removesuffix(", "functools.cache(",
                          "zoneinfo", "math.lcm("):
            assert forbidden not in source, "{} uses {}".format(relative, forbidden)
    # The only 3.9 API shim is the explicit map-sidecar compatibility class.
    assert "class _Py38Description(str)" in (
        ROOT / "l3_integration/map_sidecar.py").read_text(encoding="utf-8")


def _run_agent_probe(mode):
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "sbadir_agent_probe.py"), mode],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, completed.stderr
    return json.loads(lines[-1])


def test_real_agent_b_emits_l3_source_background_and_never_l1():
    result = _run_agent_probe("b")
    prompt = result["prompt"]
    summary = result["summary"]
    assert "L3 RGB-D map" in prompt
    assert "source yolo11n-seg:fake-yolo11n-seg.pt" in prompt
    assert "evidence " in prompt
    assert "[L1 RGB" not in prompt
    assert result["prompt_has_l1"] is False
    assert summary["source_background_enabled"] is True
    assert summary["l1_enabled"] is False
    assert summary["l1_enabled_steps"] == 0
    assert summary["l1_observation_attempt_steps"] == 0
    assert summary["l1_fact_count_total"] == 0
    assert summary["prompt_has_l1_steps"] == 0
    assert summary["l3_fact_count_total"] > 0
    assert (summary["source_background_facts_total"]
            == summary["l3_fact_count_total"])
    assert summary["l3_original_memory_unchanged"] is True
    assert result["l3_core_unchanged"] is True
    # The original organizer block is retained byte-for-byte at the start.
    assert result["memory"].startswith(result["l3_original_text"].splitlines()[0])
    for original in result["l3_original_text"].splitlines():
        assert any(line.startswith(original)
                   for line in result["memory"].splitlines())


def test_real_agent_c_keeps_b_l3_block_and_adds_l1_evidence():
    b = _run_agent_probe("b")
    c = _run_agent_probe("c")
    # C's L3 facts, order, text and source background are exactly B's.
    assert c["l3_original_text"] == b["l3_original_text"]
    assert c["l3_original_sha256"] == b["l3_original_sha256"]
    assert c["l3_enriched_sha256"] == b["l3_enriched_sha256"]
    assert c["l3_fact_signature_sha256"] == b["l3_fact_signature_sha256"]
    assert c["memory"].startswith(b["memory"])
    # The L3 map/fact-selection state is identical step by step.
    assert c["l3_states"] == b["l3_states"]
    # C adds the L1 RGB memory on top.
    assert "[L1 RGB observation" in c["prompt"]
    assert c["summary"]["l1_enabled"] is True
    assert c["summary"]["l1_enabled_steps"] == c["summary"]["env_steps"]
    assert c["summary"]["l1_fact_count_total"] > 0
    assert c["summary"]["prompt_has_l1_steps"] > 0
    # Same-entity conflict keeps both facts and both evidence sources.
    conflicts = c["summary"]["conflict_evidence_ids"]
    assert c["summary"]["conflicting_facts_total"] == 1
    assert conflicts and conflicts[0]["l1"] and conflicts[0]["l3"]
    assert "same map entity" in c["prompt"]
    assert "L3 reports chair" in c["prompt"]
    assert "both retained" in c["prompt"]
    assert "correct" not in c["prompt"].casefold()
    # Every retained fact (L3 and L1) has a source background.
    assert c["summary"]["source_background_facts_total"] == (
        c["summary"]["l3_fact_count_total"]
        + c["summary"]["l1_fact_count_total"]
    )


def test_l1_modules_are_imported_only_by_the_c_backend_builder():
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []
            self.found = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_ImportFrom(self, node):
            if (node.module or "").startswith(("l1_memory", "l1_integration")):
                self.found.append((node.lineno, tuple(self.stack), node.module))

    visitor = Visitor()
    visitor.visit(ast.parse((ROOT / "run.py").read_text(encoding="utf-8")))
    assert visitor.found, "run.py must still build the real L1 backend"
    assert all(stack and stack[-1] == "_build_l1_backend"
               for _line, stack, _module in visitor.found), visitor.found
    for relative in ("agent_uninavid.py", "l3_integration/sbadir_fusion.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        modules = [node.module or "" for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom)]
        modules += [alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for alias in node.names]
        assert not any(name.startswith(("l1_memory", "l1_integration"))
                       for name in modules), relative


def test_b_pairs_every_l3_fact_with_its_own_source_not_only_the_first():
    second_line = (
        "Map record (step 5, 2 views): the table is to your right, 3.0 m straight-line."
    )
    second = _l3(
        label="table", evidence=("seg-2",), object_id="map-2",
        latest_step=5, confidence=0.73, line=lambda: second_line,
    )
    text = HEADER + "\n" + L3_LINE + "\n" + second_line
    result = _compose(l3_facts=[_l3(), second], l1_facts=(), text=text)
    assert result.diagnostics["l3_fact_count"] == 2
    assert result.diagnostics["l3_provenance_count"] == 2
    assert result.diagnostics["omitted_l3_provenance"] == 0
    lines = result.l3_enriched_text.splitlines()
    assert lines[0] == HEADER
    assert lines[1].startswith(L3_LINE) and "evidence seg-1" in lines[1]
    assert "map node map-1" in lines[1]
    assert lines[2].startswith(second_line)
    assert "evidence seg-2" in lines[2]
    assert "map node map-2" in lines[2]
    assert "step 5" in lines[2]
    assert "seg conf 0.73" in lines[2]


def test_real_agent_a_uses_only_the_original_l3_task_text():
    result = _run_agent_probe("a")
    prompt = result["prompt"]
    assert "L3 RGB-D map" not in prompt
    assert "[L1 RGB" not in prompt
    assert "evidence " not in prompt
    assert result["l3_original_text"] in prompt
    assert result["memory"] is None
    assert result["prompt_has_l1"] is None
    assert result["summary"]["l3_fact_count_total"] > 0
    assert result["summary"]["source_background_facts_total"] is None
    assert result["summary"]["l1_fact_count_total"] is None

"""Requirements 2, 10: L1 is absent and no GT/oracle/future input enters L3."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from l3_integration import map_sidecar as sidecar_module
from tests.helpers import FakeEpisode, make_sidecar

ROOT = Path(__file__).resolve().parents[1]
L3_CODE = [ROOT / "l3_integration", ROOT / "l3_memory"]

FORBIDDEN_L1_PATTERNS = [
    r"\bL1PromptOrganizer\b",
    r"\bLocalGraphFusion\b",
    r"\bJointPromptBuilder\b",
    r"\bl1_prompt_organizer\b",
    r"\bjoint_prompt\b",
    r"from\s+lightnav\.memory\.l1",
]

FORBIDDEN_INPUT_PATTERNS = [
    r"top_down_map",
    r"top_down",
    r"oracle",
    r"shortest_path",
    r"distance_to_goal",
    r"gt_path",
    r"future_frame",
    r"future_observation",
]


def _python_files(paths):
    for path in paths:
        yield from path.rglob("*.py")


def test_l1_builders_are_not_imported_or_called() -> None:
    for path in _python_files(L3_CODE):
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_L1_PATTERNS:
            assert not re.search(pattern, text), f"{pattern} found in {path}"
    agent_text = (ROOT / "agent_uninavid.py").read_text(encoding="utf-8")
    for pattern in FORBIDDEN_L1_PATTERNS:
        assert not re.search(pattern, agent_text), pattern


def test_sidecar_passes_an_empty_l1_event_list(monkeypatch) -> None:
    captured = []
    original = sidecar_module.make_spatial_evidence

    def wrapper(instances, events, store, **kwargs):
        captured.append(list(events))
        return original(instances, events, store, **kwargs)

    monkeypatch.setattr(sidecar_module, "make_spatial_evidence", wrapper)
    sidecar = make_sidecar()
    sidecar.reset_episode(FakeEpisode())
    sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 0)
    assert captured == [[]]
    assert all(
        record.linked_l1_evidence_id is None
        for record in sidecar.object_map.evidence.values()
    )


def test_l3_input_code_does_not_reference_evaluator_oracle_fields() -> None:
    runtime_modules = [
        ROOT / "l3_integration" / "habitat_geometry.py",
        ROOT / "l3_integration" / "map_sidecar.py",
        ROOT / "l3_integration" / "official_prompt.py",
        ROOT / "l3_integration" / "pilot_manifest.py",
    ]
    for path in runtime_modules:
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_INPUT_PATTERNS:
            assert not re.search(pattern, text), f"{pattern} found in {path}"
    # The vendored L3 serializer is allowed to mention "traversal" but must not
    # read the evaluator's top-down/oracle fields.
    for path in _python_files([ROOT / "l3_memory"]):
        text = path.read_text(encoding="utf-8")
        for pattern in ("top_down_map", "oracle_success", "gt_path"):
            assert pattern not in text, f"{pattern} found in {path}"


def test_oracle_metrics_in_observations_never_reach_the_prompt() -> None:
    sidecar = make_sidecar()
    sidecar.reset_episode(FakeEpisode())
    observations = {
        "rgb": np.zeros((8, 8, 3), np.uint8),
        "top_down_map_vlnce": "ORACLE_TOPDOWN_SENTINEL",
        "oracle_success": True,
        "distance_to_goal": 0.123,
        "shortest_path": "SHORTEST_PATH_SENTINEL",
    }
    sidecar.observe(observations, None, 0)
    sidecar.observe(observations, None, 1)
    organization, diagnostic = sidecar.build_context("find the chair", 1)
    text = organization.context.text
    for sentinel in (
        "ORACLE_TOPDOWN_SENTINEL", "SHORTEST_PATH_SENTINEL", "0.123",
        "oracle", "top_down", "shortest_path",
    ):
        assert sentinel not in text
    assert diagnostic["context_injected"] is True
    assert "Map record (step 0" in text


def test_future_observations_cannot_appear_in_the_current_prompt() -> None:
    sidecar = make_sidecar()
    sidecar.reset_episode(FakeEpisode())
    for step in (0, 1):
        sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, step)
    organization, _ = sidecar.build_context("find the chair", 1)
    text = organization.context.text
    assert "Map record (step 0" in text
    assert "Map record (step 1" not in text

    sidecar.observe({"rgb": np.zeros((8, 8, 3), np.uint8)}, None, 2)
    organization, _ = sidecar.build_context("find the chair", 2)
    text = organization.context.text
    assert "Map record (step 0" in text or "Map record (step 1, 2 views)" in text
    assert "Map record (step 2" not in text


def test_extractor_requires_a_real_local_segment_model() -> None:
    from l3_memory.yolo_seg_extractor import YOLOSegExtractor

    with __import__("pytest").raises(FileNotFoundError):
        YOLOSegExtractor("/does/not/exist.pt", device="cpu")
    with __import__("pytest").raises(ValueError):
        YOLOSegExtractor("weights.bin", device="cpu")


def test_no_runtime_module_imports_lightnav_or_an_l1_builder() -> None:
    import ast

    for path in list(_python_files(L3_CODE)) + [ROOT / "agent_uninavid.py", ROOT / "run.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith("lightnav"), f"{path} imports {name}"
                assert "L1PromptOrganizer" not in name
                assert "LocalGraphFusion" not in name
                assert "JointPromptBuilder" not in name

#!/usr/bin/env python3
"""CPU code-level smoke test for the eight prompt-ablation groups.

This is **not** a navigation-result run.  It exercises the real vendored L1/L3
backends, the real official-template composition and the real group configs
with deterministic synthetic RGB-D frames and fake detectors.  It explicitly
marks the checkpoint/Habitat/GPU inference as pending on AutoDL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l1_integration.experiment_config import load_group_config
from l1_integration.memory_backend import (
    EIGHT_ABLATION_GROUPS,
    PromptMemoryBackend,
    group_config,
)
from l3_integration.official_prompt import (
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
    build_prompt,
    build_task_text,
    prompt_sha256,
    verify_task_text_injection,
)
from l3_memory.conceptgraphs_lite import (
    CameraIntrinsics,
    Pose3D,
    RGBDFrame,
)
from l3_memory.schemas import DetectedObject, SceneExtraction
from l3_memory.spatial_graph_adapter import make_spatial_evidence  # noqa: F401
from l3_memory.spatial_l3_config import SpatialL3Config  # noqa: F401
from l3_memory.yolo_seg_extractor import SegmentationInstance

IDENTITY = Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
INSTRUCTION = "find the chair"


def frame() -> RGBDFrame:
    return RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.full((8, 8), 2.0, dtype=np.float32),
        intrinsics=CameraIntrinsics(8.0, 8.0, 3.5, 3.5, 8, 8),
        agent_pose=IDENTITY,
        camera_extrinsics=IDENTITY,
    )


class FakeL1:
    source = "cpu_smoke_yolo11n"
    def __init__(self) -> None:
        self.calls: list[int] = []

    def reset_episode(self, episode_id, instruction) -> None:
        self.calls = []

    def close_episode(self) -> None:
        pass

    def extract(self, packet):
        self.calls.append(packet.step_id)
        if packet.step_id != 0:
            return SceneExtraction()
        return SceneExtraction(
            detections=[DetectedObject("chair", 0.9, (1.0, 1.0, 6.0, 6.0))],
            objects=["chair"], confidence=0.9, source=self.source,
        )


class FakePlace:
    source = "cpu_smoke_dinov2"
    def __init__(self) -> None:
        self.calls: list[int] = []

    def reset_episode(self, episode_id, instruction) -> None:
        self.calls = []

    def close_episode(self) -> None:
        pass

    def extract(self, packet):
        self.calls.append(packet.step_id)
        return (1.0, 0.0)


class FakeSeg:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def reset_episode(self, episode_id, instruction) -> None:
        self.calls = []

    def close_episode(self) -> None:
        pass

    def extract(self, rgb, step):
        self.calls.append(step)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        return [SegmentationInstance("chair", 0.9, mask, (2.0, 2.0, 6.0, 6.0))]


def _episode() -> SimpleNamespace:
    return SimpleNamespace(
        scene_id="cpu_smoke_scene", episode_id="cpu_smoke_episode",
        instruction=SimpleNamespace(instruction_text=INSTRUCTION),
    )


def _run_group(group: str) -> dict:
    spec = group_config(group)
    config = load_group_config(str(ROOT / "configs" / "prompt_ablation" / f"{group}.yaml"))
    l1, place, seg = FakeL1(), FakePlace(), FakeSeg()
    backend = PromptMemoryBackend(
        spec["prompt_mode"],
        l1_extractor=l1 if spec["requires_l1_model"] else None,
        seg_extractor=seg if spec["requires_seg_model"] else None,
        place_extractor=place if spec["requires_l1_model"] else None,
        max_landmarks=config["memory"]["max_landmarks"],
        max_context_chars=config["memory"]["max_context_chars"],
        prompt_max_chars=config["memory"]["prompt_max_chars"],
        scope=config["memory"]["scope"],
        place_every_n_steps=config["memory"]["place_every_n_steps"],
        seg_model_path="cpu_smoke_yolo11n-seg.pt",
        seg_source="yolo11n-seg:cpu_smoke_yolo11n-seg.pt",
    )
    backend.reset_episode(_episode(), INSTRUCTION)
    steps = []
    for step in range(3):
        events, observe = backend.observe(
            {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}, step,
        )
        spatial = backend.observe_spatial(
            frame(), events, step, transition_record=bool(step),
        ) if backend.plan.uses_l3 else {}
        result = backend.build_context(
            instruction=INSTRUCTION, step=step, rgb_width=8,
            frame=frame() if backend.plan.uses_l3 else None,
        )
        memory_text = result.text if result.context is not None else None
        task_text = build_task_text(INSTRUCTION, memory_text) if memory_text else INSTRUCTION
        prompt = build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, task_text)
        verify_task_text_injection(
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE, INSTRUCTION, memory_text, prompt,
        )
        evidence_ids = sorted({
            ref.event_id for item in (result.context.layer_items if result.context else [])
            for ref in item.evidence_refs
        })
        object_ids = sorted({
            str(item.metadata.get("object_id"))
            for item in (result.context.layer_items if result.context else [])
            if item.metadata.get("object_id") is not None
        })
        l1_facts = sorted({
            (str(item.metadata.get("role")), str(item.label or ""))
            for item in (result.context.layer_items if result.context else [])
            if item.metadata.get("layer_source") == "l1"
            or item.metadata.get("role") in {"current", "history"}
        })
        steps.append({
            "step": step,
            "events": len(events),
            "memory_source": result.source,
            "context_injected": result.context is not None,
            "context_chars": len(memory_text) if memory_text else 0,
            "context_tokens_estimate": len(memory_text.split()) if memory_text else 0,
            "prompt_chars": len(prompt),
            "prompt_tokens_estimate": len(prompt.split()),
            "prompt_sha256": prompt_sha256(prompt),
            "memory_text": memory_text,
            "evidence_ids": evidence_ids,
            "object_ids": object_ids,
            "l1_facts": l1_facts,
            "alignment": result.diagnostics.get("alignment"),
            "l1_item_count": int((result.diagnostics.get("layers", {}).get("l1") or {}).get("item_count", 0)),
            "l3_item_count": int((result.diagnostics.get("layers", {}).get("l3") or {}).get("item_count", 0)),
            "duplicates_removed": int(result.diagnostics.get("duplicates_removed", 0) or 0),
            "conflict_count": len(result.diagnostics.get("conflicts") or []),
            "omitted_items": int(result.diagnostics.get("omitted_items", 0) or 0),
            "observe": observe,
            "spatial": {k: spatial.get(k) for k in (
                "frame_valid", "segmentation_count", "evidence_minted",
                "evidence_linked", "evidence_unmatched", "object_nodes",
            )} if spatial else None,
            "prompt": prompt,
        })
    report = {
        "group": group,
        "prompt_mode": spec["prompt_mode"],
        "combination": spec["combination"],
        "l1_builder": spec["l1_builder"],
        "l3_builder": spec["l3_builder"],
        "config_source": config["source"],
        "l1_calls": list(l1.calls),
        "place_calls": list(place.calls),
        "seg_calls": list(seg.calls),
        "object_map": backend.map_snapshot(),
        "steps": steps,
    }
    backend.close_episode()
    return report


def _prompt_injection_rate(report: dict) -> float:
    steps = report["steps"]
    return sum(step["context_injected"] for step in steps) / len(steps)


def main() -> int:
    output = ROOT / "output" / "prompt_ablation_cpu_smoke"
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    checks = []

    for group in EIGHT_ABLATION_GROUPS:
        try:
            reports[group] = _run_group(group)
            checks.append((group, "build", True, ""))
        except Exception as exc:  # noqa: BLE001
            checks.append((group, "build", False, f"{type(exc).__name__}: {exc}"))
            reports[group] = {"group": group, "error": f"{type(exc).__name__}: {exc}"}

    def check(name, condition, detail=""):
        checks.append(("global", name, bool(condition), detail))

    for group, report in reports.items():
        if "error" in report:
            continue
        spec = group_config(group)
        steps = report["steps"]
        last_prompt = steps[-1]["prompt"]
        prefix, suffix = REFERENCE_OFFICIAL_PROMPT_TEMPLATE.split("{}")
        check(f"{group}:fixed-prefix", last_prompt.startswith(prefix + INSTRUCTION))
        check(f"{group}:fixed-suffix", last_prompt.endswith(suffix))
        if spec["requires_l1_model"]:
            check(f"{group}:l1-called", bool(report["l1_calls"]))
            check(f"{group}:place-called", bool(report["place_calls"]))
        else:
            check(f"{group}:l1-not-called", not report["l1_calls"])
            check(f"{group}:place-not-called", not report["place_calls"])
        if spec["requires_seg_model"]:
            check(f"{group}:seg-called", bool(report["seg_calls"]))
        else:
            check(f"{group}:seg-not-called", not report["seg_calls"])
            check(f"{group}:no-object-map", report["object_map"]["objects"] == 0)
        if spec["l3_prompt"]:
            check(f"{group}:l3-context", any(
                step["context_injected"] for step in steps
            ))
        else:
            check(f"{group}:no-l3-context", all(
                "Spatial history" not in (step["memory_text"] or "")
                and "map facts are history" not in (step["memory_text"] or "")
                for step in steps
            ))
        if spec["l1_prompt"] and not spec["l3_prompt"]:
            check(f"{group}:no-l3-prompt", all(
                "Spatial history" not in (step["memory_text"] or "")
                for step in steps
            ))

    l1_original = reports.get("l1_original_only", {})
    l1_organized = reports.get("l1_organized_only", {})
    if "steps" in l1_original and "steps" in l1_organized:
        original_ids = set(l1_original["steps"][-1]["l1_facts"])
        organized_ids = set(l1_organized["steps"][-1]["l1_facts"])
        check(
            "l1:original-vs-organized-same-facts",
            original_ids == organized_ids and bool(original_ids),
            f"original={sorted(original_ids)} organized={sorted(organized_ids)}",
        )
    l3_original = reports.get("l3_original_only", {})
    l3_organized = reports.get("l3_organized_only", {})
    if "steps" in l3_original and "steps" in l3_organized:
        original_ids = set(l3_original["steps"][-1]["object_ids"])
        organized_ids = set(l3_organized["steps"][-1]["object_ids"])
        check(
            "l3:original-vs-organized-same-facts",
            original_ids == organized_ids and bool(original_ids),
            f"original={sorted(original_ids)} organized={sorted(organized_ids)}",
        )
    concat = reports.get("l1_l3_original_concat", {})
    joint = reports.get("l1_l3_joint", {})
    if "steps" in concat and "steps" in joint:
        check(
            "fusion:concat-differs-from-joint",
            concat["steps"][-1]["prompt_sha256"] != joint["steps"][-1]["prompt_sha256"],
        )
        check(
            "fusion:joint-has-alignment",
            joint["steps"][-1].get("alignment") is not None,
        )

    summary = {
        "artifact": "cpu_prompt_ablation_smoke",
        "real_inference_run": False,
        "real_inference_pending": (
            "Habitat + Uni-NaVid checkpoint + YOLO/DINOv2 weights + GPU are not "
            "available on this host; run scripts/run_prompt_ablation.sh on AutoDL"
        ),
        "official_template_sha256": prompt_sha256(REFERENCE_OFFICIAL_PROMPT_TEMPLATE),
        "groups": reports,
        "checks": [
            {"group": group, "check": name, "passed": passed, "detail": detail}
            for group, name, passed, detail in checks
        ],
        "passed": sum(1 for _, _, passed, _ in checks if passed),
        "failed": sum(1 for _, _, passed, _ in checks if not passed),
        "total": len(checks),
    }
    (output / "smoke_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    for group, report in reports.items():
        if "steps" in report:
            (output / f"{group}_prompt.txt").write_text(
                report["steps"][-1]["prompt"] + "\n", encoding="utf-8",
            )
    print(f"CPU smoke: {summary['passed']} passed, {summary['failed']} failed, "
          f"{summary['total']} checks")
    for group, name, passed, detail in checks:
        if not passed:
            print(f"FAIL {group} {name}: {detail}")
    print(f"report: {output / 'smoke_report.json'}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

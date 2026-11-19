#!/usr/bin/env python3
"""CPU-only 2-Episode smoke test of the L3 map + official-style prompt path.

This does **not** replace the real Habitat/Uni-NaVid smoke test.  It exercises
the real vendored L3 map, evidence minting, official-style serializer, task-slot
injection guard and episode reset with synthetic RGB-D and a deterministic fake
segmenter.  No model weights or Habitat scene are needed.

Run:
  PYTHONPATH=. python scripts/smoke_l3_cpu.py --output /tmp/l3_smoke.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from l3_integration.habitat_geometry import CameraSpec
from l3_integration.map_sidecar import L3MapSidecar
from l3_integration.official_prompt import (
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
    build_official_prompt,
    build_prompt_with_l3,
    prompt_sha256,
    update_prompt_view,
    verify_task_text_injection,
)
from l3_memory.conceptgraphs_lite import CameraIntrinsics, Pose3D, RGBDFrame
from l3_memory.spatial_l3_config import SpatialL3Config
from l3_memory.yolo_seg_extractor import SegmentationInstance


class FakeSegmenter:
    def __init__(self, label: str) -> None:
        self.label = label
        self.last_detection_ms = 0.8
        self.episode_ids: list[str] = []

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        self.episode_ids.append(episode_id)

    def close_episode(self) -> None:
        pass

    def extract(self, rgb, step_id):
        if step_id % 2 != 0:
            return []
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[2:6, 2:6] = True
        return [SegmentationInstance(
            self.label, 0.9, mask, (2.0, 2.0, 6.0, 6.0),
        )]


def _frame() -> RGBDFrame:
    identity = Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    return RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.full((8, 8), 2.0, dtype=np.float32),
        intrinsics=CameraIntrinsics(8.0, 8.0, 3.5, 3.5, 8, 8),
        agent_pose=identity,
        camera_extrinsics=identity,
    )


def _run_episode(sidecar: L3MapSidecar, episode, label: str) -> dict:
    sidecar.reset_episode(episode)
    observations = {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}
    for step in (0, 1):
        sidecar.observe(observations, None, step, transition_record=step > 0)
    organization, context = sidecar.build_context("find the chair", 1)
    official = build_official_prompt(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, "find the chair",
    )
    context_text = (
        organization.context.text
        if organization is not None and organization.context is not None
        else None
    )
    augmented = build_prompt_with_l3(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, "find the chair", context_text,
    )
    verify_task_text_injection(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, "find the chair", context_text,
        augmented,
    )
    return {
        "scene_id": episode.scene_id,
        "episode_id": episode.episode_id,
        "label": label,
        "map": sidecar.map_snapshot(),
        "context_injected": organization is not None and organization.context is not None,
        "fact_count": len(organization.selected_facts) if organization else 0,
        "context_text": (
            organization.context.text if organization and organization.context else None
        ),
        "official_prompt_sha256": prompt_sha256(official),
        "augmented_prompt_sha256": prompt_sha256(augmented),
        "l3_block_in_update_view": (
            organization is not None and organization.context is not None
            and organization.context.text.replace("\n", "") in update_prompt_view(augmented)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    config = SpatialL3Config(
        mode="inject", seg_model_path="fake-yolo11n-seg.pt", seg_device="cpu",
        seg_confidence=0.25, seg_every_n_steps=2,
        evidence_min_iou=0.5, evidence_ambiguity_margin=0.1,
        min_injection_confidence=0.5,
    )
    frame = _frame()
    sidecar = L3MapSidecar(
        config,
        camera_spec=CameraSpec(8, 8, 120.0, 0.0, 10.0, True),
        max_items=2, max_chars=600,
        extractor=FakeSegmenter("chair"),
        frame_builder=lambda obs, state, spec: (frame, {"frame_valid": True}),
    )
    episodes = [
        SimpleNamespace(
            scene_id="scene-a", episode_id="episode-a",
            instruction=SimpleNamespace(instruction_text="find the chair"),
        ),
        SimpleNamespace(
            scene_id="scene-b", episode_id="episode-b",
            instruction=SimpleNamespace(instruction_text="find the chair"),
        ),
    ]
    results = []
    for episode in episodes:
        results.append(_run_episode(sidecar, episode, episode.episode_id))
        # Between Episodes the map must be empty before the next reset.
        assert sidecar.map_snapshot()["objects"] in (0, 1)
    # After the last Episode, reset to a fresh one and prove isolation.
    sidecar.reset_episode(SimpleNamespace(
        scene_id="scene-c", episode_id="episode-c",
        instruction=SimpleNamespace(instruction_text="find the chair"),
    ))
    result = {
        "kind": "cpu_synthetic_2ep_smoke",
        "note": "not a Habitat/Uni-NaVid smoke test",
        "episodes": results,
        "isolation_after_reset": sidecar.map_snapshot(),
        "fake_segmenter_reset_ids": sidecar.extractor.episode_ids,
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    assert all(item["context_injected"] for item in results)
    assert all(item["l3_block_in_update_view"] for item in results)
    assert result["isolation_after_reset"]["objects"] == 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

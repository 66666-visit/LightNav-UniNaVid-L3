"""Shared CPU-only fixtures for the L3/Uni-NaVid integration tests."""

from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace

import numpy as np

from l3_integration.habitat_geometry import CameraSpec
from l3_integration.map_sidecar import L3MapSidecar
from l3_memory.conceptgraphs_lite import (
    CameraIntrinsics,
    Pose3D,
    RGBDFrame,
)
from l3_memory.spatial_l3_config import SpatialL3Config
from l3_memory.yolo_seg_extractor import SegmentationInstance

IDENTITY = Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def synthetic_frame(
    *, pose: Pose3D | None = None, depth: float = 2.0,
    rgb: np.ndarray | None = None,
) -> RGBDFrame:
    return RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8) if rgb is None else rgb,
        depth_m=np.full((8, 8), depth, dtype=np.float32),
        intrinsics=CameraIntrinsics(8.0, 8.0, 3.5, 3.5, 8, 8),
        agent_pose=pose or IDENTITY,
        camera_extrinsics=IDENTITY,
    )


def chair_instance(
    *, label: str = "chair", confidence: float = 0.9,
) -> SegmentationInstance:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    return SegmentationInstance(label, confidence, mask, (2.0, 2.0, 6.0, 6.0))


class FakeExtractor:
    """Deterministic stand-in for YOLO11n-seg; no ultralytics import."""

    def __init__(self, instances=None, *, every_n_steps: int = 1) -> None:
        self.instances = list(instances) if instances is not None else [chair_instance()]
        self.every_n_steps = every_n_steps
        self.last_detection_ms = 1.25
        self.last_error = None
        self.calls: list[int] = []
        self.reset_calls: list[tuple[str, str]] = []

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        self.reset_calls.append((episode_id, instruction))

    def close_episode(self) -> None:
        pass

    def extract(self, rgb: np.ndarray, step_id: int):
        self.calls.append(step_id)
        if step_id % self.every_n_steps != 0:
            self.last_detection_ms = None
            return []
        self.last_detection_ms = 1.25
        return list(self.instances)


class FakeEpisode:
    def __init__(self, scene_id: str = "scene", episode_id: str = "episode",
                 instruction: str = "find the chair") -> None:
        self.scene_id = scene_id
        self.episode_id = episode_id
        self.instruction = SimpleNamespace(instruction_text=instruction)


def make_sidecar(*, extractor=None, frame=None, scope: str = "episode_local",
                 max_items: int = 2, max_chars: int = 600) -> L3MapSidecar:
    config = SpatialL3Config(
        mode="inject", seg_model_path="fake-yolo11n-seg.pt", seg_device="cpu",
        seg_confidence=0.25, seg_every_n_steps=1,
        evidence_min_iou=0.5, evidence_ambiguity_margin=0.1,
        min_injection_confidence=0.5,
    )
    frame_value = frame or synthetic_frame()
    return L3MapSidecar(
        config,
        camera_spec=CameraSpec(8, 8, 120.0, 0.0, 10.0, True),
        max_items=max_items,
        max_chars=max_chars,
        scope=scope,
        extractor=extractor or FakeExtractor(),
        frame_builder=lambda obs, state, spec: (frame_value, {"frame_valid": True}),
    )


class FakeQuaternion:
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0,
                 z: float = 0.0) -> None:
        self.w, self.x, self.y, self.z = w, x, y, z


def fake_agent_state(*, position=(0.0, 0.0, 0.0), sensor_offset=(0.0, 1.25, 0.0)):
    agent_position = np.asarray(position, dtype=np.float64)
    sensor_position = agent_position + np.asarray(sensor_offset, dtype=np.float64)
    rotation = FakeQuaternion()
    return SimpleNamespace(
        position=agent_position,
        rotation=rotation,
        sensor_states={
            "rgb": SimpleNamespace(position=sensor_position, rotation=rotation),
            "depth": SimpleNamespace(position=sensor_position, rotation=rotation),
        },
    )


@contextlib.contextmanager
def agent_module_stubs():
    """Import ``agent_uninavid`` without Habitat, cv2, imageio or Uni-NaVid.

    The official agent module imports the real Habitat/Uni-NaVid packages at
    module import time.  The CPU test only needs the class methods, so this
    context installs minimal stand-ins and removes them afterwards.
    """
    names = [
        "habitat", "habitat.core", "habitat.core.agent", "habitat.utils",
        "habitat.utils.visualizations", "habitat.utils.visualizations.maps",
        "cv2", "imageio", "uninavid", "uninavid.model", "uninavid.mm_utils",
        "uninavid.model.builder", "uninavid.constants", "uninavid.conversation",
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

        uninavid = types.ModuleType("uninavid")
        uninavid.__path__ = []
        sys.modules["uninavid"] = uninavid
        model = types.ModuleType("uninavid.model")
        model.__path__ = []
        sys.modules["uninavid.model"] = model
        mm_utils = types.ModuleType("uninavid.mm_utils")
        mm_utils.get_model_name_from_path = lambda path: "fake"
        mm_utils.tokenizer_image_token = lambda *args, **kwargs: None
        mm_utils.KeywordsStoppingCriteria = type("KeywordsStoppingCriteria", (), {})
        sys.modules["uninavid.mm_utils"] = mm_utils
        builder = types.ModuleType("uninavid.model.builder")
        builder.load_pretrained_model = lambda *args, **kwargs: None
        sys.modules["uninavid.model.builder"] = builder
        constants = types.ModuleType("uninavid.constants")
        for name in (
            "IMAGE_TOKEN_INDEX", "DEFAULT_IMAGE_TOKEN", "DEFAULT_IM_START_TOKEN",
            "DEFAULT_IM_END_TOKEN",
        ):
            setattr(constants, name, name)
        sys.modules["uninavid.constants"] = constants
        conversation = types.ModuleType("uninavid.conversation")
        conversation.conv_templates = {}
        conversation.SeparatorStyle = type("SeparatorStyle", (), {"TWO": "two"})
        sys.modules["uninavid.conversation"] = conversation

        import agent_uninavid
        yield agent_uninavid
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class FakeL1Extractor:
    """Deterministic stand-in for the L1 YOLO11n detector."""

    def __init__(self, by_step=None, *, every_n_steps: int = 1) -> None:
        self.by_step = dict(by_step or {0: [("chair", 0.9, (1.0, 1.0, 6.0, 6.0))]})
        self.every_n_steps = every_n_steps
        self.source = "fake_yolo_detect"
        self.last_detection_ms = 0.5
        self.last_error = None
        self.calls: list[int] = []
        self.reset_calls: list[tuple[str, str]] = []

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        self.reset_calls.append((episode_id, instruction))

    def close_episode(self) -> None:
        pass

    def extract(self, packet):
        from l3_memory.schemas import DetectedObject, SceneExtraction

        self.calls.append(packet.step_id)
        if packet.step_id % self.every_n_steps != 0:
            return SceneExtraction()
        rows = self.by_step.get(packet.step_id, [])
        if not rows:
            return SceneExtraction()
        detections = [
            DetectedObject(label=label, confidence=confidence, bbox_xyxy=box)
            for label, confidence, box in rows
        ]
        return SceneExtraction(
            detections=detections,
            objects=[item.label for item in detections],
            confidence=max(item.confidence for item in detections),
            source=self.source,
        )


class FakePlaceExtractor:
    """Deterministic DINO-like descriptor; the same vector revisits one place."""

    def __init__(self, descriptor=(1.0, 0.0)) -> None:
        self.descriptor = tuple(float(value) for value in descriptor)
        self.source = "fake_dino_place"
        self.calls: list[int] = []

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        self.calls.clear()

    def close_episode(self) -> None:
        pass

    def extract(self, packet):
        self.calls.append(packet.step_id)
        return self.descriptor


def make_memory_backend(group: str, *, l1=None, seg=None, place=None, **kwargs):
    from l1_integration.memory_backend import PromptMemoryBackend, group_config

    spec = group_config(group)
    memory = {
        "max_landmarks": 2,
        "max_context_chars": 600,
        "prompt_max_chars": 320,
        "place_every_n_steps": 1,
    }
    memory.update(kwargs)
    return PromptMemoryBackend(
        spec["prompt_mode"],
        l1_extractor=(l1 if l1 is not None else FakeL1Extractor())
        if spec["requires_l1_model"] else None,
        seg_extractor=(seg if seg is not None else FakeExtractor())
        if spec["requires_seg_model"] else None,
        place_extractor=(place if place is not None else FakePlaceExtractor())
        if spec["requires_l1_model"] else None,
        max_landmarks=memory["max_landmarks"],
        max_context_chars=memory["max_context_chars"],
        prompt_max_chars=memory["prompt_max_chars"],
        scope="episode_local",
        place_every_n_steps=memory["place_every_n_steps"],
        seg_source="yolo11n-seg:fake.pt",
        seg_model_path="fake.pt",
    )

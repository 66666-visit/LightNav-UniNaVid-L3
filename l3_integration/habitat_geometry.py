"""Habitat RGB-D / pose -> the exact ``RGBDFrame`` contract of the L3 map.

This is the same metadata contract LightNav's Habitat server exposed as
``info["spatial_rgbd"]``.  It uses only the current RGB, aligned depth, current
sensor poses and the sensor's own pinhole calibration.  It never reads a
top-down map, shortest path, goal distance, evaluator success state or a future frame.

The output dictionary is passed to the vendored, unmodified
``RGBDFrame.from_habitat`` so depth decoding and validation stay in the
original L3 implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from l3_memory.conceptgraphs_lite import RGBDFrame


@dataclass(frozen=True)
class CameraSpec:
    """Pinhole RGB-D sensor calibration and depth encoding."""

    width: int
    height: int
    hfov_deg: float
    min_depth_m: float
    max_depth_m: float
    normalize_depth: bool
    rgb_key: str = "rgb"
    depth_key: str = "depth"

    def __post_init__(self) -> None:
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("camera width and height must be integers")
        if self.width < 1 or self.height < 1:
            raise ValueError("camera width and height must be positive")
        if not math.isfinite(self.hfov_deg) or not 0.0 < self.hfov_deg < 180.0:
            raise ValueError("camera hfov must lie in (0, 180) degrees")
        if (
            not math.isfinite(self.min_depth_m)
            or not math.isfinite(self.max_depth_m)
            or self.min_depth_m < 0.0
            or self.max_depth_m <= self.min_depth_m
        ):
            raise ValueError("camera depth range is invalid")
        if not isinstance(self.normalize_depth, bool):
            raise ValueError("camera normalize_depth must be an explicit bool")
        if not isinstance(self.rgb_key, str) or not self.rgb_key:
            raise ValueError("rgb_key must be nonempty")
        if not isinstance(self.depth_key, str) or not self.depth_key:
            raise ValueError("depth_key must be nonempty")


def _get(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    # Habitat config nodes, observations and sensor-state mappings are all
    # dict-like but not necessarily ``collections.abc.Mapping``.
    for key in (name, name.lower(), name.upper()):
        try:
            return value[key]
        except (KeyError, IndexError, TypeError):
            pass
    for key in (name, name.lower(), name.upper()):
        if hasattr(value, key):
            return getattr(value, key)
    return default


def camera_spec_from_config(config: Any) -> CameraSpec:
    """Read the pinhole calibration from a Habitat/CN task config.

    The L3 task config adds a depth sensor with the same width/height/HFOV as
    the official RGB sensor.  Missing calibration is an error: this function
    never guesses a focal length or a sensor height.
    """
    task = _get(config, "TASK_CONFIG", config)
    simulator = _get(task, "SIMULATOR")
    if simulator is None:
        raise ValueError("task config has no SIMULATOR")
    rgb = _get(simulator, "RGB_SENSOR")
    depth = _get(simulator, "DEPTH_SENSOR")
    if rgb is None:
        raise ValueError("task config has no RGB_SENSOR")
    width = _get(rgb, "WIDTH")
    height = _get(rgb, "HEIGHT")
    hfov = _get(rgb, "HFOV")
    if width is None or height is None or hfov is None:
        raise ValueError("RGB_SENSOR lacks WIDTH/HEIGHT/HFOV")
    width, height = int(width), int(height)
    hfov = float(hfov)
    if depth is None:
        raise ValueError("task config has no DEPTH_SENSOR; L3 needs aligned depth")
    depth_width = _get(depth, "WIDTH", width)
    depth_height = _get(depth, "HEIGHT", height)
    depth_hfov = _get(depth, "HFOV", hfov)
    if int(depth_width) != width or int(depth_height) != height:
        raise ValueError("RGB and depth sensor sizes differ")
    if not math.isclose(float(depth_hfov), hfov, abs_tol=1e-5):
        raise ValueError("RGB and depth sensor HFOV differ")
    min_depth = float(_get(depth, "MIN_DEPTH", 0.0))
    max_depth = float(_get(depth, "MAX_DEPTH", 10.0))
    normalize = _get(depth, "NORMALIZE_DEPTH", True)
    if not isinstance(normalize, bool):
        raise ValueError("DEPTH_SENSOR.NORMALIZE_DEPTH must be a bool")
    rgb_uuid = _get(rgb, "UUID", "rgb")
    depth_uuid = _get(depth, "UUID", "depth")
    return CameraSpec(
        width=width, height=height, hfov_deg=hfov,
        min_depth_m=min_depth, max_depth_m=max_depth,
        normalize_depth=normalize,
        rgb_key=str(rgb_uuid or "rgb").lower(),
        depth_key=str(depth_uuid or "depth").lower(),
    )


def _quaternion_wxyz(rotation: Any) -> np.ndarray:
    values = np.asarray(
        [rotation.w, rotation.x, rotation.y, rotation.z], dtype=np.float64,
    )
    norm = float(np.linalg.norm(values))
    if not np.isfinite(values).all() or not 0.99 <= norm <= 1.01:
        raise ValueError("invalid sensor or agent quaternion")
    return values / norm


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _quaternion_rotate(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    pure = np.array([0.0, *vector], dtype=np.float64)
    inverse = rotation * np.array([1.0, -1.0, -1.0, -1.0])
    return _quaternion_product(_quaternion_product(rotation, pure), inverse)[1:]


def _sensor_state(sensor_states: Any, key: str) -> Any:
    """Find a sensor state by UUID, allowing Habitat's *_sensor spellings."""
    direct = _get(sensor_states, key)
    if direct is not None:
        return direct
    try:
        names = list(sensor_states.keys())
    except (AttributeError, TypeError):
        names = []
    lowered = key.casefold()
    for name in names:
        candidate = str(name).casefold()
        if candidate == lowered or candidate.endswith("_" + lowered) or (
            lowered in candidate
        ):
            return _get(sensor_states, str(name))
    return None


def _observation_image(observations: Mapping[str, Any], key: str) -> np.ndarray:
    try:
        present = key in observations
    except TypeError:
        present = False
    if not present:
        raise ValueError(f"observation has no {key!r} channel")
    return np.asarray(observations[key])


def build_spatial_rgbd_metadata(
    observations: Mapping[str, Any], agent_state: Any, spec: CameraSpec,
) -> dict[str, Any]:
    """Build the exact ``spatial_rgbd`` metadata consumed by ``RGBDFrame``."""
    if agent_state is None:
        raise ValueError("current simulator agent state is unavailable")
    rgb = _observation_image(observations, spec.rgb_key)
    depth = _observation_image(observations, spec.depth_key)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB must be HWC with three channels")
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            raise ValueError("RGB must be uint8 or float in [0, 1]")
    if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        raise ValueError("RGB and depth image sizes differ")
    if not np.issubdtype(depth.dtype, np.floating):
        raise ValueError("depth must be a floating-point image")
    height, width = depth.shape
    if (width, height) != (spec.width, spec.height):
        raise ValueError("sensor config size differs from the RGB-D sample")

    sensor_states = getattr(agent_state, "sensor_states", None)
    if sensor_states is None:
        raise ValueError("current simulator sensor poses are unavailable")
    rgb_state = _sensor_state(sensor_states, spec.rgb_key)
    depth_state = _sensor_state(sensor_states, spec.depth_key)
    if rgb_state is None or depth_state is None:
        raise ValueError("current RGB/depth sensor poses are unavailable")
    agent_position = np.asarray(agent_state.position, dtype=np.float64)
    rgb_position = np.asarray(rgb_state.position, dtype=np.float64)
    depth_position = np.asarray(depth_state.position, dtype=np.float64)
    if (
        agent_position.shape != (3,)
        or rgb_position.shape != (3,)
        or depth_position.shape != (3,)
        or not np.isfinite(agent_position).all()
        or not np.isfinite(rgb_position).all()
        or not np.isfinite(depth_position).all()
    ):
        raise ValueError("invalid agent or sensor position")
    agent_rotation = _quaternion_wxyz(agent_state.rotation)
    rgb_rotation = _quaternion_wxyz(rgb_state.rotation)
    depth_rotation = _quaternion_wxyz(depth_state.rotation)
    if (
        np.linalg.norm(rgb_position - depth_position) > 1e-4
        or abs(float(np.dot(rgb_rotation, depth_rotation))) < 1.0 - 1e-8
    ):
        raise ValueError("RGB/depth sensor poses are not aligned")

    agent_inverse = agent_rotation * np.array([1.0, -1.0, -1.0, -1.0])
    camera_position = _quaternion_rotate(agent_inverse, rgb_position - agent_position)
    camera_rotation = _quaternion_product(agent_inverse, rgb_rotation)
    camera_rotation /= np.linalg.norm(camera_rotation)
    focal = (width / 2.0) / math.tan(math.radians(spec.hfov_deg) / 2.0)
    return {
        "status": "ok",
        "image_size_hw": [height, width],
        "K": [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        "agent_pose": {
            "position": agent_position.tolist(),
            "rotation": agent_rotation.tolist(),
        },
        "camera_extrinsics": {
            "position": camera_position.tolist(),
            "rotation": camera_rotation.tolist(),
        },
        "depth_encoding": "normalized_0_1" if spec.normalize_depth else "meters",
        "min_depth_m": spec.min_depth_m,
        "max_depth_m": spec.max_depth_m,
        "camera_frame": "habitat_right_up_back",
        "aligned": True,
    }


def frame_from_habitat(
    observations: Mapping[str, Any], agent_state: Any, spec: CameraSpec,
) -> tuple[RGBDFrame, dict[str, Any]]:
    """Return a validated ``RGBDFrame`` and the metadata used to build it."""
    metadata = build_spatial_rgbd_metadata(observations, agent_state, spec)
    depth = _observation_image(observations, spec.depth_key)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    obs = {
        "rgb": np.asarray(observations[spec.rgb_key]),
        "depth": np.asarray(depth),
    }
    frame = RGBDFrame.from_habitat(obs, {"spatial_rgbd": metadata})
    return frame, metadata

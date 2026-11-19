"""Requirement 7: exact RGB-D/pose metadata and explicit unavailable states."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from l3_integration.habitat_geometry import (
    CameraSpec,
    build_spatial_rgbd_metadata,
    camera_spec_from_config,
    frame_from_habitat,
)
from tests.helpers import FakeQuaternion, fake_agent_state


def _observations(depth_value: float = 0.5) -> dict:
    return {
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "depth": np.full((8, 8), depth_value, dtype=np.float32),
    }


def test_camera_spec_reads_rgb_and_depth_from_task_config() -> None:
    config = {
        "TASK_CONFIG": {
            "SIMULATOR": {
                "RGB_SENSOR": {
                    "WIDTH": 640, "HEIGHT": 480, "HFOV": 120,
                    "UUID": "rgb",
                },
                "DEPTH_SENSOR": {
                    "WIDTH": 640, "HEIGHT": 480, "HFOV": 120,
                    "MIN_DEPTH": 0.0, "MAX_DEPTH": 10.0,
                    "NORMALIZE_DEPTH": True, "UUID": "depth",
                },
            },
        },
    }
    spec = camera_spec_from_config(config)
    assert spec == CameraSpec(640, 480, 120.0, 0.0, 10.0, True, "rgb", "depth")


def test_camera_spec_rejects_missing_or_mismatched_depth() -> None:
    base = {
        "RGB_SENSOR": {"WIDTH": 8, "HEIGHT": 8, "HFOV": 90},
        "DEPTH_SENSOR": {
            "WIDTH": 8, "HEIGHT": 8, "HFOV": 90,
            "MIN_DEPTH": 0.0, "MAX_DEPTH": 10.0, "NORMALIZE_DEPTH": True,
        },
    }
    with pytest.raises(ValueError, match="no DEPTH_SENSOR"):
        camera_spec_from_config({"SIMULATOR": {"RGB_SENSOR": base["RGB_SENSOR"]}})
    mismatched = {
        "RGB_SENSOR": base["RGB_SENSOR"],
        "DEPTH_SENSOR": {**base["DEPTH_SENSOR"], "WIDTH": 9},
    }
    with pytest.raises(ValueError, match="sizes differ"):
        camera_spec_from_config({"SIMULATOR": mismatched})
    mismatched_fov = {
        "RGB_SENSOR": base["RGB_SENSOR"],
        "DEPTH_SENSOR": {**base["DEPTH_SENSOR"], "HFOV": 100},
    }
    with pytest.raises(ValueError, match="HFOV differ"):
        camera_spec_from_config({"SIMULATOR": mismatched_fov})


def test_metadata_uses_current_sensor_pose_and_exact_pinhole_k() -> None:
    spec = CameraSpec(8, 8, 120.0, 0.0, 10.0, True)
    state = fake_agent_state(position=(1.0, 2.0, 3.0), sensor_offset=(0.0, 1.25, 0.0))
    metadata = build_spatial_rgbd_metadata(_observations(), state, spec)
    assert metadata["status"] == "ok"
    assert metadata["aligned"] is True
    assert metadata["camera_frame"] == "habitat_right_up_back"
    assert metadata["depth_encoding"] == "normalized_0_1"
    focal = 4.0 / math.tan(math.radians(120.0) / 2.0)
    assert metadata["K"][0][0] == pytest.approx(focal)
    assert metadata["K"][0][2] == 4.0
    assert metadata["K"][1][2] == 4.0
    assert metadata["agent_pose"]["position"] == [1.0, 2.0, 3.0]
    assert metadata["camera_extrinsics"]["position"] == [0.0, 1.25, 0.0]
    assert metadata["camera_extrinsics"]["rotation"] == [1.0, 0.0, 0.0, 0.0]


def test_normalized_depth_is_decoded_to_metric_meters() -> None:
    spec = CameraSpec(8, 8, 120.0, 0.0, 10.0, True)
    frame, metadata = frame_from_habitat(_observations(0.5), fake_agent_state(), spec)
    assert metadata["depth_encoding"] == "normalized_0_1"
    assert np.allclose(frame.depth_m[2:6, 2:6], 5.0)
    assert frame.agent_pose.position == (0.0, 0.0, 0.0)
    assert frame.camera_pose.position == pytest.approx((0.0, 1.25, 0.0))


def test_metric_depth_encoding_is_explicit() -> None:
    spec = CameraSpec(8, 8, 120.0, 0.0, 10.0, False)
    frame, metadata = frame_from_habitat(_observations(2.0), fake_agent_state(), spec)
    assert metadata["depth_encoding"] == "meters"
    assert np.allclose(frame.depth_m, 2.0)


def test_channel_depth_is_squeezed_and_alignment_is_checked() -> None:
    spec = CameraSpec(8, 8, 120.0, 0.0, 10.0, True)
    observations = _observations()
    observations["depth"] = observations["depth"][:, :, None]
    metadata = build_spatial_rgbd_metadata(observations, fake_agent_state(), spec)
    assert metadata["image_size_hw"] == [8, 8]

    state = fake_agent_state()
    state.sensor_states["depth"] = SimpleNamespace(
        position=np.array([0.0, 0.0, 0.0]), rotation=FakeQuaternion(),
    )
    with pytest.raises(ValueError, match="not aligned"):
        build_spatial_rgbd_metadata(_observations(), state, spec)


def test_missing_depth_and_bad_size_fail_closed() -> None:
    spec = CameraSpec(8, 8, 120.0, 0.0, 10.0, True)
    with pytest.raises(ValueError, match="no 'depth' channel"):
        build_spatial_rgbd_metadata({"rgb": np.zeros((8, 8, 3), np.uint8)},
                                   fake_agent_state(), spec)
    bad = {"rgb": np.zeros((4, 4, 3), np.uint8),
           "depth": np.full((4, 4), 0.5, np.float32)}
    with pytest.raises(ValueError, match="size differs"):
        build_spatial_rgbd_metadata(bad, fake_agent_state(), spec)


def test_non_pinhole_or_missing_calibration_is_rejected() -> None:
    with pytest.raises(ValueError):
        CameraSpec(0, 8, 120.0, 0.0, 10.0, True)
    with pytest.raises(ValueError):
        CameraSpec(8, 8, 0.0, 0.0, 10.0, True)
    with pytest.raises(ValueError):
        CameraSpec(8, 8, 120.0, 5.0, 5.0, True)
    with pytest.raises(ValueError):
        CameraSpec(8, 8, 120.0, 0.0, 10.0, "yes")  # type: ignore[arg-type]

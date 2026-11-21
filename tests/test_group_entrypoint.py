"""Independent-start validation for the eight prompt-ablation groups."""

from __future__ import annotations

from pathlib import Path

import pytest

from l1_integration.experiment_config import load_group_config
from l1_integration.memory_backend import EIGHT_ABLATION_GROUPS
from tests.test_run_entrypoint import run_module as _run_module_fixture  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(name="run_module")
def run_module_fixture(_run_module_fixture):  # noqa: F811
    return _run_module_fixture


def _config(group: str) -> str:
    return str(ROOT / "configs" / "prompt_ablation" / f"{group}.yaml")


def _call(run, group, **overrides):
    kwargs = {
        "exp_config": "fake.yaml", "split_num": 1, "split_id": 0,
        "model_path": "model", "result_path": "/tmp/unused",
        "model_name": "uni-navid", "exp_save": "data",
        "allow_nonpilot_eval": True, "experiment_group": group,
        "group_config": _config(group),
    }
    kwargs.update(overrides)
    run.run_exp(**kwargs)


def test_l1_group_passes_its_own_config_and_l1_weights(run_module) -> None:
    run, _manifest, _digest, calls = run_module
    _call(
        run, "l1_original_only",
        memory_l1_model="fake-detect.pt", memory_place_model="fake-place",
    )
    assert len(calls) == 1
    kwargs = calls[0][1]
    assert kwargs["experiment_group"] == "l1_original_only"
    assert kwargs["group_config_spec"]["group"] == "l1_original_only"
    assert kwargs["memory_options"]["l1_model"] == "fake-detect.pt"
    assert kwargs["memory_options"]["place_model"] == "fake-place"
    assert kwargs["memory_options"]["seg_model"] == ""


def test_l3_group_runs_the_l1_writer_but_not_the_l1_prompt(run_module) -> None:
    run, _manifest, _digest, calls = run_module
    _call(
        run, "l3_organized_only", l3_seg_model="fake-seg.pt",
        memory_l1_model="fake-detect.pt", memory_place_model="fake-place",
    )
    assert len(calls) == 1
    kwargs = calls[0][1]
    assert kwargs["memory_options"]["l1_model"] == "fake-detect.pt"
    assert kwargs["memory_options"]["seg_model"] == "fake-seg.pt"
    assert kwargs["group_config_spec"]["expected"]["l1_prompt"] is False
    assert kwargs["group_config_spec"]["expected"]["l1_evidence_writer"] is True


def test_joint_group_needs_both_models(run_module) -> None:
    run, _manifest, _digest, calls = run_module
    with pytest.raises(ValueError, match="memory-l1-model"):
        _call(run, "l1_l3_joint", l3_seg_model="fake-seg.pt",
              memory_place_model="fake-place")
    with pytest.raises(ValueError, match="l3-seg-model"):
        _call(run, "l1_l3_joint", memory_l1_model="fake-detect.pt",
              memory_place_model="fake-place")
    assert calls == []
    _call(
        run, "l1_l3_joint", memory_l1_model="fake-detect.pt",
        l3_seg_model="fake-seg.pt", memory_place_model="fake-place",
    )
    assert len(calls) == 1


def test_every_eight_group_requires_its_own_config(run_module) -> None:
    run, _manifest, _digest, calls = run_module
    for group in EIGHT_ABLATION_GROUPS:
        with pytest.raises(ValueError, match="group-config"):
            _call(run, group, group_config="", memory_l1_model="fake-detect.pt",
                  l3_seg_model="fake-seg.pt", memory_place_model="fake-place")
    assert calls == []


def test_group_config_and_group_name_must_agree(run_module) -> None:
    run, _manifest, _digest, calls = run_module
    with pytest.raises(ValueError, match="names"):
        _call(
            run, "l3_original_only",
            group_config=_config("l1_original_only"),
            l3_seg_model="fake-seg.pt",
        )
    assert calls == []


def test_l1_only_group_rejects_an_l3_model(run_module) -> None:
    run, _manifest, _digest, calls = run_module
    with pytest.raises(ValueError, match="no L3 layer"):
        _call(run, "l1_organized_only", memory_l1_model="fake-detect.pt",
              memory_place_model="fake-place", l3_seg_model="unexpected.pt")
    assert calls == []


def test_formal_l1_group_requires_the_place_model(run_module) -> None:
    run, _manifest, _digest, calls = run_module
    with pytest.raises(ValueError, match="memory-place-model"):
        _call(
            run, "l1_organized_only", memory_l1_model="fake-detect.pt",
            allow_nonpilot_eval=False,
        )
    assert calls == []


def test_no_force_stop_is_forwarded(run_module) -> None:
    run, _manifest, _digest, calls = run_module
    _call(
        run, "l3_original_only", l3_seg_model="fake-seg.pt",
        memory_l1_model="fake-detect.pt", no_force_stop=True,
    )
    assert calls[0][1]["no_force_stop"] is True


def test_formal_launcher_keeps_the_frozen_conditions() -> None:
    script = (ROOT / "scripts" / "run_prompt_ablation.sh").read_text(encoding="utf-8")
    for group in EIGHT_ABLATION_GROUPS:
        assert group in script
    assert "30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98" in script
    assert "NO_FORCE_STOP" in script
    assert "--no-force-stop" in script
    assert "--expected-episodes 88" in script
    assert "--seed 42" in script
    assert "--memory-token-budget 1" in script
    assert "CONFIRM=YES" in script or "CONFIRM" in script
    assert "--expected-episodes 704" not in script
    assert "--expected-episodes 792" not in script
    assert "run_exp" not in script


def test_yaml_and_json_group_configs_are_equivalent() -> None:
    import json

    for group in EIGHT_ABLATION_GROUPS:
        yaml_path = ROOT / "configs" / "prompt_ablation" / f"{group}.yaml"
        json_path = ROOT / "configs" / "prompt_ablation" / f"{group}.json"
        assert yaml_path.is_file() and json_path.is_file()
        data = json.loads(json_path.read_text())
        assert data["group"] == group
        assert data["prompt_mode"] == group
        assert data["layers"]["combination"] == load_group_config(
            str(yaml_path)
        )["expected"]["combination"]


def test_smoke_episodes_truncates_only_non_formal_runs(run_module) -> None:
    run, _manifest, _digest, calls = run_module
    run.run_exp(
        exp_config="fake.yaml", split_num=1, split_id=0, model_path="model",
        result_path="/tmp/unused", model_name="uni-navid", exp_save="data",
        allow_nonpilot_eval=True, smoke_episodes=1, expected_episodes=1,
        experiment_group="l1_original_only",
        group_config=_config("l1_original_only"),
        memory_l1_model="fake-detect.pt", memory_place_model="fake-place",
    )
    assert len(calls) == 1
    dataset = calls[0][0]
    assert len(dataset.episodes) == 1


def test_build_memory_backend_wires_the_real_extractors(run_module, monkeypatch) -> None:
    from l3_integration.habitat_geometry import CameraSpec

    run, _manifest, _digest, _calls = run_module
    created = {}

    class FakeL1:
        def __init__(self, path, **kwargs):
            created["l1"] = (path, kwargs)

    class FakeSeg:
        def __init__(self, path, **kwargs):
            created["seg"] = (path, kwargs)

    class FakePlace:
        def __init__(self, path, **kwargs):
            created["place"] = (path, kwargs)

    monkeypatch.setattr(
        "l1_memory.yolo_scene_extractor.YOLOSceneExtractor", FakeL1,
    )
    monkeypatch.setattr(
        "l3_memory.yolo_seg_extractor.YOLOSegExtractor", FakeSeg,
    )
    monkeypatch.setattr(
        "l1_memory.place_recognition.DINOv2PlaceDescriptorExtractor", FakePlace,
    )
    monkeypatch.setattr(
        "l3_integration.habitat_geometry.camera_spec_from_config",
        lambda config: CameraSpec(8, 8, 120.0, 0.0, 10.0, True),
    )
    options = {
        "l1_model": "fake-l1.pt", "l1_device": "cpu",
        "l1_confidence": 0.25, "l1_every_n_steps": 1,
        "place_model": "fake-place", "place_device": "cpu",
        "place_every_n_steps": 5, "token_budget": True,
        "max_landmarks": 2, "max_context_chars": 600,
        "prompt_max_chars": 320, "seg_model": "fake-seg.pt",
        "seg_device": "cpu", "seg_confidence": 0.25,
        "seg_every_n_steps": 5, "evidence_min_iou": 0.5,
        "evidence_ambiguity_margin": 0.1, "min_confidence": 0.5,
    }
    backend, camera = run._build_memory_backend(
        object(), "l1_l3_joint", options,
    )
    assert backend.plan.mode == "l1_l3_joint"
    assert camera.width == 8
    assert created["l1"][0] == "fake-l1.pt"
    assert created["seg"][0] == "fake-seg.pt"
    assert created["place"][0] == "fake-place"
    from tests.helpers import FakeEpisode

    backend.reset_episode(FakeEpisode(), "find the chair")
    assert backend.object_map is not None
    assert backend.plan.l1_source == "organized"
    assert backend.plan.l3_source == "organized"

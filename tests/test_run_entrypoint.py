"""run.py must give both groups the same manifest-ordered Episode set."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from l3_integration.pilot_manifest import (
    EXPECTED_EPISODE_MANIFEST_SHA256,
    EXPECTED_PILOT_MANIFEST_SHA256,
    canonical_episode_manifest_sha256,
)


@pytest.fixture()
def run_module(tmp_path, monkeypatch):
    calls = []

    class FakeDataset:
        def __init__(self, episodes):
            self.episodes = episodes

        def get_splits(self, num):
            return [copy.copy(self)]

    episodes = []
    rows = []
    for scene in range(11):
        scene_id = f"/autodl/scene_datasets/hm3d/Scene{scene}.glb"
        for index in range(8):
            group = "short" if index < 3 else "medium" if index < 6 else "long"
            episode_id = f"ep-{scene}-{index}"
            instruction = f"go to the chair in scene {scene} number {index}"
            episodes.append(SimpleNamespace(
                scene_id=scene_id, episode_id=episode_id,
                instruction=SimpleNamespace(instruction_text=instruction),
            ))
            rows.append({
                "scene_id": f"data/scene_datasets/hm3d/Scene{scene}.glb",
                "habitat_episode_id": episode_id,
                "distance_group": group,
                "instruction": instruction,
            })
    dataset = FakeDataset(list(reversed(episodes)))
    manifest = {
        "protocol_version": "rgm_pilot_88_v1",
        "dataset_split": "val_unseen",
        "sample_seed": 42,
        "expected_count": 88,
        "per_scene": 8,
        "parent_manifest_sha256": "b" * 64,
        "episodes": rows,
    }
    path = tmp_path / "pilot_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    config = SimpleNamespace(
        TASK_CONFIG=SimpleNamespace(DATASET=SimpleNamespace(TYPE="fake")),
        EVAL=SimpleNamespace(
            IDENTIFICATION="fake", EARLY_STOP_ROTATION=25, EARLY_STOP_STEPS=400,
        ),
    )

    habitat = types.ModuleType("habitat")
    habitat.__path__ = []
    habitat.Env = object
    sys.modules["habitat"] = habitat
    datasets = types.ModuleType("habitat.datasets")
    datasets.make_dataset = lambda **kwargs: dataset
    sys.modules["habitat.datasets"] = datasets
    tqdm = types.ModuleType("tqdm")
    tqdm.trange = lambda count, **kwargs: range(count)
    sys.modules["tqdm"] = tqdm
    for name in ("VLN_CE", "VLN_CE.vlnce_baselines", "VLN_CE.vlnce_baselines.config"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    default = types.ModuleType("VLN_CE.vlnce_baselines.config.default")
    default.get_config = lambda path, opts=None: config
    sys.modules["VLN_CE.vlnce_baselines.config.default"] = default

    sys.modules.pop("run", None)
    import run
    run._original_evaluate_agent = run.evaluate_agent
    run.evaluate_agent = lambda *args, **kwargs: calls.append(
        (args[2], kwargs)
    )
    yield run, path, digest, calls

    for name in (
        "run", "habitat", "habitat.datasets", "tqdm", "VLN_CE",
        "VLN_CE.vlnce_baselines", "VLN_CE.vlnce_baselines.config",
        "VLN_CE.vlnce_baselines.config.default",
    ):
        sys.modules.pop(name, None)


def test_run_exp_gives_both_groups_the_same_episode_sequence(run_module, tmp_path) -> None:
    run, manifest_path, digest, calls = run_module
    common = dict(
        exp_config="fake.yaml", split_num=1, split_id=0, model_path="model",
        result_path=str(tmp_path / "unused"), model_name="uni-navid",
        exp_save="data", pilot_manifest=str(manifest_path),
        pilot_sha256=digest, expected_episodes=88,
        allow_nonpilot_eval=True,
    )
    run.run_exp(**common, experiment_group="uninavid_baseline")
    run.run_exp(
        **common, experiment_group="uninavid_l3_only",
        l3_seg_model="fake-yolo11n-seg.pt",
    )
    assert len(calls) == 2
    baseline_dataset, baseline_kwargs = calls[0]
    l3_dataset, l3_kwargs = calls[1]
    baseline = [(e.scene_id, e.episode_id) for e in baseline_dataset.episodes]
    l3 = [(e.scene_id, e.episode_id) for e in l3_dataset.episodes]
    assert baseline == l3
    assert len(baseline) == 88
    assert baseline_kwargs["manifest_audit"]["expected_count"] == 88
    assert l3_kwargs["manifest_audit"]["expected_count"] == 88
    assert baseline_kwargs["manifest_audit"]["file_sha256"] == digest
    assert l3_kwargs["manifest_audit"]["file_sha256"] == digest
    assert all(
        row[0] == row[1] for row in zip(
            [e.episode_id for e in baseline_dataset.episodes],
            [f"ep-{scene}-{index}" for scene in range(11) for index in range(8)],
        )
    )


def test_formal_uninavid_run_requires_the_frozen_manifest_and_digest(
    run_module, tmp_path,
) -> None:
    run, manifest_path, _digest, calls = run_module
    common = dict(
        exp_config="fake.yaml", split_num=1, split_id=0, model_path="model",
        result_path=str(tmp_path / "unused"), model_name="uni-navid",
        exp_save="data",
    )
    with pytest.raises(ValueError, match="requires --pilot-manifest"):
        run.run_exp(**common, experiment_group="uninavid_baseline")
    with pytest.raises(ValueError, match="requires --pilot-sha256"):
        run.run_exp(
            **common, experiment_group="uninavid_baseline",
            pilot_manifest=str(manifest_path), pilot_sha256="",
            expected_episodes=88,
        )
    with pytest.raises(ValueError, match="expected-episodes 88"):
        run.run_exp(
            **common, experiment_group="uninavid_baseline",
            pilot_manifest=str(manifest_path),
            pilot_sha256=(
                "cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a"
            ),
            expected_episodes=87,
        )
    assert calls == []
    run.run_exp(
        **common, experiment_group="uninavid_baseline",
        allow_nonpilot_eval=True,
    )
    assert len(calls) == 1


def test_episode_seed_is_deterministic_and_episode_specific(run_module) -> None:
    run, _manifest_path, _digest, _calls = run_module
    first = run._stable_episode_seed(42, "scene-a", "episode-1")
    assert first == run._stable_episode_seed(42, "scene-a", "episode-1")
    assert first != run._stable_episode_seed(42, "scene-a", "episode-2")
    assert first != run._stable_episode_seed(43, "scene-a", "episode-1")
    assert 0 <= first < 2 ** 31 - 1


def test_evaluate_agent_writes_formal_pilot_metadata_without_name_error(
    run_module, tmp_path, monkeypatch,
) -> None:
    run, _manifest_path, _digest, _calls = run_module
    import types

    class FakeEnv:
        def __init__(self, task_config, dataset):
            self.episodes = []
            self.current_episode = None

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    fake_agent_module = types.ModuleType("agent_uninavid")
    fake_agent_module.UniNaVid_Agent = FakeAgent
    monkeypatch.setitem(sys.modules, "agent_uninavid", fake_agent_module)
    monkeypatch.setattr(run, "Env", FakeEnv)
    config = SimpleNamespace(
        TASK_CONFIG=SimpleNamespace(),
        EVAL=SimpleNamespace(
            IDENTIFICATION="fake", EARLY_STOP_ROTATION=25,
            EARLY_STOP_STEPS=400,
        ),
    )
    result_path = tmp_path / "result"
    run._original_evaluate_agent(
        config, 0, SimpleNamespace(episodes=[]), "model", str(result_path),
        "uni-navid", "data", split_num=1,
        experiment_group="uninavid_baseline", seed=42,
        l3_options={}, manifest_audit={
            "path": "fake.json",
            "file_sha256": (
                "cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a"
            ),
            "expected_count": 88,
            "parent_manifest": None,
            "episode_manifest": {"verified": False, "canonical_sha256": None},
        }, episode_order={},
        exp_config="fake.yaml", allow_nonpilot_eval=False,
    )
    metadata = json.loads((result_path / "run_config.json").read_text())
    assert metadata["formal_pilot"] is True
    assert metadata["seed_protocol"].startswith("sha256(seed,scene,episode)")


def test_evaluate_agent_loop_seeds_before_reset_and_keeps_order(
    run_module, tmp_path, monkeypatch,
) -> None:
    import random as py_random
    import types

    import torch

    run, _manifest_path, _digest, _calls = run_module
    episodes = [
        SimpleNamespace(scene_id=f"scene-{i}", episode_id=f"episode-{i}",
                        instruction=SimpleNamespace(instruction_text=f"go {i}"))
        for i in range(2)
    ]

    class FakeEnv:
        instances = []

        def __init__(self, task_config, dataset):
            type(self).instances.append(self)
            self.episodes = list(dataset.episodes)
            self.current_episode = None
            self.episode_over = False
            self.index = 0
            self.reset_rng_draws = []
            self.actions = []

        def reset(self):
            # This draw must already reflect the per-Episode seed.
            self.reset_rng_draws.append((
                py_random.random(), float(np.random.rand()),
                float(torch.rand(1).item()),
            ))
            self.current_episode = self.episodes[self.index]
            self.index += 1
            self.episode_over = False
            return {"rgb": np.zeros((2, 2, 3), np.uint8)}

        def get_metrics(self):
            return {
                "distance_to_goal": 2.0, "success": False, "spl": 0.0,
                "ndtw": 0.0, "path_length": 1.0, "oracle_success": False,
                "steps_taken": 1, "top_down_map_vlnce": None,
            }

        def step(self, action):
            self.actions.append(action)
            self.episode_over = True

    class FakeAgent:
        reset_calls = []

        def __init__(self, *args, **kwargs):
            pass

        def reset(self, episode=None, instruction=""):
            type(self).reset_calls.append((episode.episode_id, instruction))

        def act(self, observations, info, episode_id):
            return {"action": 0}

        def close(self):
            pass

    fake_agent_module = types.ModuleType("agent_uninavid")
    fake_agent_module.UniNaVid_Agent = FakeAgent
    monkeypatch.setitem(sys.modules, "agent_uninavid", fake_agent_module)
    monkeypatch.setattr(run, "Env", FakeEnv)
    config = SimpleNamespace(
        TASK_CONFIG=SimpleNamespace(),
        EVAL=SimpleNamespace(
            IDENTIFICATION="fake", EARLY_STOP_ROTATION=25,
            EARLY_STOP_STEPS=400,
        ),
    )
    result_path = tmp_path / "result"
    run._original_evaluate_agent(
        config, 0, SimpleNamespace(episodes=episodes), "model", str(result_path),
        "uni-navid", "", split_num=1, experiment_group="uninavid_baseline",
        seed=42, l3_options={}, manifest_audit=None, episode_order={},
        exp_config="fake.yaml", allow_nonpilot_eval=False,
    )
    assert FakeAgent.reset_calls == [
        ("episode-0", "go 0"), ("episode-1", "go 1"),
    ]
    rows = [
        json.loads(line) for line in (result_path / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [row["episode_id"] for row in rows] == ["episode-0", "episode-1"]
    for row in rows:
        assert row["episode_seed"] == run._stable_episode_seed(
            42, row["scene_id"], row["episode_id"],
        )

    # Replay the seed protocol and compare the first RNG draw consumed inside
    # env.reset(); this proves seeding happened before reset.
    py_random.seed(run._stable_episode_seed(42, "scene-0", "episode-0"))
    np.random.seed(run._stable_episode_seed(42, "scene-0", "episode-0"))
    torch.manual_seed(run._stable_episode_seed(42, "scene-0", "episode-0"))
    expected = (py_random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    assert rows  # results written
    assert FakeEnv.instances[0].reset_rng_draws[0] == expected


def test_formal_path_accepts_exact_frozen_manifest(run_module, monkeypatch) -> None:
    run, _manifest_path, _digest, calls = run_module
    frozen = "cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a"

    class FakeManifest:
        rows = [object()] * 88

        def order_digest(self, instructions=None):
            return "fake-order-digest"

    monkeypatch.setattr(
        run, "load_pilot_manifest", lambda *a, **k: FakeManifest(),
    )
    monkeypatch.setattr(
        run, "select_pilot_episodes", lambda episodes, manifest: list(episodes),
    )
    monkeypatch.setattr(
        run, "audit_manifest", lambda *a, **k: {
            "path": "fake.json", "file_sha256": frozen, "expected_count": 88,
            "dataset_order_digest_with_instructions": "fake-order-digest",
            "parent_manifest": None,
            "episode_manifest": {"verified": False, "canonical_sha256": None},
        },
    )
    run.run_exp(
        exp_config="fake.yaml", split_num=1, split_id=0, model_path="model",
        result_path="/tmp/unused", model_name="uni-navid", exp_save="data",
        pilot_manifest="fake.json", pilot_sha256=frozen,
        expected_episodes=88, experiment_group="uninavid_baseline",
    )
    assert len(calls) == 1
    assert calls[0][1]["manifest_audit"]["file_sha256"] == frozen


def test_evaluate_agent_rejects_a_reset_order_mismatch(
    run_module, tmp_path, monkeypatch,
) -> None:
    import types

    run, _manifest_path, _digest, _calls = run_module
    episodes = [
        SimpleNamespace(scene_id=f"scene-{i}", episode_id=f"episode-{i}",
                        instruction=SimpleNamespace(instruction_text=f"go {i}"))
        for i in range(2)
    ]

    class WrongOrderEnv:
        def __init__(self, task_config, dataset):
            self.episodes = list(dataset.episodes)
            self.current_episode = None
            self.episode_over = False

        def reset(self):
            self.current_episode = self.episodes[1]
            return {}

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, episode=None, instruction=""):
            pass

        def close(self):
            pass

    fake_agent_module = types.ModuleType("agent_uninavid")
    fake_agent_module.UniNaVid_Agent = FakeAgent
    monkeypatch.setitem(sys.modules, "agent_uninavid", fake_agent_module)
    monkeypatch.setattr(run, "Env", WrongOrderEnv)
    config = SimpleNamespace(
        TASK_CONFIG=SimpleNamespace(),
        EVAL=SimpleNamespace(
            IDENTIFICATION="fake", EARLY_STOP_ROTATION=25,
            EARLY_STOP_STEPS=400,
        ),
    )
    with pytest.raises(RuntimeError, match="did not return"):
        run._original_evaluate_agent(
            config, 0, SimpleNamespace(episodes=episodes), "model",
            str(tmp_path / "result"), "uni-navid", "", split_num=1,
            experiment_group="uninavid_baseline", seed=42, l3_options={},
            manifest_audit=None, episode_order={}, exp_config="fake.yaml",
            allow_nonpilot_eval=False,
        )


def test_evaluate_agent_rejects_l3_config_for_baseline(
    run_module, tmp_path,
) -> None:
    run, _manifest_path, _digest, _calls = run_module
    config = SimpleNamespace(
        TASK_CONFIG=SimpleNamespace(),
        EVAL=SimpleNamespace(
            IDENTIFICATION="fake", EARLY_STOP_ROTATION=25,
            EARLY_STOP_STEPS=400,
        ),
    )
    with pytest.raises(ValueError, match="baseline group"):
        run._original_evaluate_agent(
            config, 0, SimpleNamespace(episodes=[]), "model",
            str(tmp_path / "result"), "uni-navid", "data", split_num=1,
            experiment_group="uninavid_baseline", seed=42,
            l3_options={"seg_model": "unexpected.pt"}, manifest_audit=None,
            episode_order={}, exp_config="fake.yaml",
            allow_nonpilot_eval=False,
        )


def test_run_exp_verifies_the_episode_manifest_canonical_digest_and_order(
    run_module, tmp_path,
) -> None:
    run, manifest_path, digest, calls = run_module
    pilot = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_rows = [
        {
            "scene_id": row["scene_id"],
            "episode_id": row["habitat_episode_id"],
        }
        for row in pilot["episodes"]
    ]
    episode_data = {
        "split": "val_unseen", "expected_count": 88,
        "episodes": episode_rows,
    }
    episode_path = tmp_path / "episode_manifest.json"
    episode_path.write_text(json.dumps(episode_data), encoding="utf-8")
    canonical = canonical_episode_manifest_sha256(episode_data)
    assert canonical != hashlib.sha256(episode_path.read_bytes()).hexdigest()
    assert canonical != EXPECTED_EPISODE_MANIFEST_SHA256
    assert EXPECTED_PILOT_MANIFEST_SHA256 != EXPECTED_EPISODE_MANIFEST_SHA256

    run.run_exp(
        exp_config="fake.yaml", split_num=1, split_id=0, model_path="model",
        result_path=str(tmp_path / "unused"), model_name="uni-navid",
        exp_save="data", pilot_manifest=str(manifest_path),
        pilot_sha256=digest, expected_episodes=88,
        episode_manifest=str(episode_path),
        episode_manifest_sha256=canonical, allow_nonpilot_eval=True,
    )
    assert len(calls) == 1
    audit = calls[0][1]["manifest_audit"]
    assert audit["file_sha256"] == digest
    assert audit["episode_manifest"]["verified"] is True
    assert audit["episode_manifest"]["canonical_sha256"] == canonical
    assert audit["episode_manifest"]["file_sha256"] == hashlib.sha256(
        episode_path.read_bytes()
    ).hexdigest()
    assert audit["episode_manifest"]["pilot_comparison"]["match"] is True


def test_run_exp_rejects_a_wrong_episode_manifest_canonical_digest(
    run_module, tmp_path,
) -> None:
    run, manifest_path, digest, calls = run_module
    pilot = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_path = tmp_path / "episode_manifest.json"
    episode_path.write_text(json.dumps({
        "split": "val_unseen", "expected_count": 88,
        "episodes": [
            {"scene_id": row["scene_id"],
             "episode_id": row["habitat_episode_id"]}
            for row in pilot["episodes"]
        ],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical SHA256 mismatch"):
        run.run_exp(
            exp_config="fake.yaml", split_num=1, split_id=0, model_path="model",
            result_path=str(tmp_path / "unused"), model_name="uni-navid",
            exp_save="data", pilot_manifest=str(manifest_path),
            pilot_sha256=digest, expected_episodes=88,
            episode_manifest=str(episode_path),
            episode_manifest_sha256=EXPECTED_EPISODE_MANIFEST_SHA256,
            allow_nonpilot_eval=True,
        )
    assert calls == []


def test_run_exp_require_episode_manifest_fails_closed(run_module) -> None:
    run, manifest_path, digest, calls = run_module
    with pytest.raises(ValueError, match="require-episode-manifest"):
        run.run_exp(
            exp_config="fake.yaml", split_num=1, split_id=0, model_path="model",
            result_path="/tmp/unused", model_name="uni-navid", exp_save="data",
            pilot_manifest=str(manifest_path), pilot_sha256=digest,
            expected_episodes=88, require_episode_manifest=True,
            allow_nonpilot_eval=True,
        )
    assert calls == []

"""run.py must give both groups the same manifest-ordered Episode set."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import types
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from l3_integration.episode_execution import (
    EpisodeExecutionError,
    metadata_from_execution_manifest,
)
from l3_integration.pilot_manifest import (
    EXPECTED_EPISODE_MANIFEST_SHA256,
    EXPECTED_PILOT_MANIFEST_SHA256,
    build_execution_manifest,
    canonical_episode_manifest_sha256,
    load_episode_manifest,
    load_pilot_manifest,
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


@pytest.fixture()
def formal_episode_set(run_module, tmp_path):
    """The 88 frozen tasks in a historical order distinct from the Pilot."""
    run, manifest_path, digest, _calls = run_module
    pilot = load_pilot_manifest(manifest_path, expected_sha256=digest)
    order = list(range(88))
    random.Random(20261120).shuffle(order)
    rows = [
        {
            "scene_id": pilot.rows[index].scene_id,
            "episode_id": pilot.rows[index].episode_id,
        }
        for index in order
    ]
    data = {"split": "val_unseen", "expected_count": 88, "episodes": rows}
    episode_path = tmp_path / "episode_manifest.json"
    episode_path.write_text(json.dumps(data), encoding="utf-8")
    canonical = canonical_episode_manifest_sha256(data)
    episode = load_episode_manifest(
        episode_path, expected_canonical_sha256=canonical,
    )
    execution = build_execution_manifest(episode, pilot)
    metadata = metadata_from_execution_manifest(execution)
    episodes = []
    for row in execution.rows:
        scene_id = row.scene_id.replace(
            "data/scene_datasets", "/autodl/scene_datasets",
        )
        episodes.append(SimpleNamespace(
            scene_id=scene_id,
            episode_id=row.episode_id,
            instruction=SimpleNamespace(instruction_text=row.instruction),
        ))
    return SimpleNamespace(
        run=run, pilot=pilot, execution=execution, metadata=metadata,
        episodes=episodes, canonical=canonical,
    )


def _formal_audit(formal) -> dict:
    return {
        "path": str(formal.pilot.path),
        "file_sha256": formal.pilot.file_sha256,
        "expected_count": 88,
        "parent_manifest": {
            "declared_sha256_in_pilot": formal.pilot.parent_manifest_sha256,
        },
        "episode_manifest": {
            "verified": True,
            "canonical_sha256": formal.execution.canonical_sha256,
        },
        "execution_order": formal.execution.order_source,
    }


def _episode_env_class(sequence):
    sequence = list(sequence)

    class FakeEnv:
        def __init__(self, task_config, dataset):
            self.episodes = list(dataset.episodes)
            self.current_episode = None
            self.episode_over = False
            self._sequence = list(sequence)
            self._position = 0

        def reset(self):
            if self._position >= len(self._sequence):
                raise StopIteration("fake Habitat iterator exhausted")
            self.current_episode = self._sequence[self._position]
            self._position += 1
            self.episode_over = False
            return {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}

        def get_metrics(self):
            return {
                "distance_to_goal": 2.0, "success": True, "spl": 1.0,
                "ndtw": 1.0, "path_length": 1.0, "oracle_success": True,
                "steps_taken": 1,
            }

        def step(self, action):
            self.episode_over = True
            return {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}

    return FakeEnv


def _fake_agent_module():
    class FakeAgent:
        reset_calls: ClassVar[list] = []
        reset_rng_draws: ClassVar[list] = []

        def __init__(self, *args, **kwargs):
            pass

        def reset(self, episode=None, instruction=""):
            type(self).reset_calls.append((str(episode.episode_id), instruction))
            type(self).reset_rng_draws.append(
                (random.random(), float(np.random.rand()))
            )

        def act(self, observations, info, episode_id):
            return {"action": 0}

        def close(self):
            pass

    module = types.ModuleType("agent_uninavid")
    module.UniNaVid_Agent = FakeAgent
    return module, FakeAgent


def _run_formal_evaluate(
    formal, sequence, result_path, monkeypatch,
):
    run = formal.run
    config = SimpleNamespace(
        TASK_CONFIG=SimpleNamespace(),
        EVAL=SimpleNamespace(
            IDENTIFICATION="fake", EARLY_STOP_ROTATION=25,
            EARLY_STOP_STEPS=400,
        ),
    )
    monkeypatch.setattr(run, "Env", _episode_env_class(sequence))
    module, agent_cls = _fake_agent_module()
    monkeypatch.setitem(sys.modules, "agent_uninavid", module)
    run._original_evaluate_agent(
        config, 0, SimpleNamespace(episodes=list(formal.episodes)), "model",
        str(result_path), "uni-navid", "", split_num=1,
        experiment_group="uninavid_baseline", seed=42, l3_options={},
        manifest_audit=_formal_audit(formal), episode_order={},
        episode_metadata=formal.metadata,
        execution_manifest=formal.execution,
        exp_config="fake.yaml", allow_nonpilot_eval=False,
    )
    return agent_cls


def _read_results(result_path):
    path = result_path / "results.jsonl"
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    assert len(baseline_kwargs["episode_metadata"]) == 88
    assert len(l3_kwargs["episode_metadata"]) == 88
    first_key = (
        baseline_dataset.episodes[0].scene_id.split("/")[-1],
        baseline_dataset.episodes[0].episode_id,
    )
    assert baseline_kwargs["episode_metadata"][first_key].manifest_order_index == 0
    assert baseline_kwargs["episode_metadata"][first_key].pilot_manifest_index == 0
    assert all(
        row[0] == row[1] for row in zip(
            [e.episode_id for e in baseline_dataset.episodes],
            [f"ep-{scene}-{index}" for scene in range(11) for index in range(8)],
        )
    )


def test_run_exp_uses_historical_episode_manifest_order(
    run_module, tmp_path,
) -> None:
    run, manifest_path, digest, calls = run_module
    pilot = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        {"scene_id": row["scene_id"], "episode_id": row["habitat_episode_id"]}
        for row in pilot["episodes"]
    ][::-1]
    data = {"split": "val_unseen", "expected_count": 88, "episodes": rows}
    episode_path = tmp_path / "episode_manifest.json"
    episode_path.write_text(json.dumps(data), encoding="utf-8")
    canonical = canonical_episode_manifest_sha256(data)
    result_path = tmp_path / "result"

    run_kwargs = dict(
        exp_config="fake.yaml", split_num=1, split_id=0, model_path="model",
        result_path=str(result_path), model_name="uni-navid",
        exp_save="data", pilot_manifest=str(manifest_path),
        pilot_sha256=digest, expected_episodes=88,
        episode_manifest=str(episode_path),
        episode_manifest_sha256=canonical,
        require_episode_manifest=True,
        allow_nonpilot_eval=True,
        experiment_group="uninavid_baseline",
    )
    run.run_exp(**run_kwargs)
    assert len(calls) == 1
    dataset, kwargs = calls[0]
    assert [episode.episode_id for episode in dataset.episodes] == [
        row["episode_id"] for row in rows
    ]
    assert [episode.episode_id for episode in dataset.episodes] != [
        row["habitat_episode_id"] for row in pilot["episodes"]
    ]
    first_episode = dataset.episodes[0]
    first_key = (first_episode.scene_id.split("/")[-1], first_episode.episode_id)
    first_metadata = kwargs["episode_metadata"][first_key]
    assert first_metadata.manifest_order_index == 0
    assert first_metadata.pilot_manifest_index == 87
    manifest_audit = kwargs["manifest_audit"]
    comparison = manifest_audit["episode_manifest"]["pilot_comparison"]
    assert comparison["sets_equal"] is True
    assert comparison["order_matches"] is False
    assert manifest_audit["execution_manifest"]["aligned"] is True
    assert manifest_audit["execution_order"] == "episode_manifest.json"
    assert (
        manifest_audit["pilot_order_digest_with_instructions"]
        != manifest_audit["dataset_order_digest_with_instructions"]
    )
    assert kwargs["execution_manifest"] is not None

    generated = json.loads(
        (result_path / "execution_manifest.json").read_text(encoding="utf-8")
    )
    assert canonical_episode_manifest_sha256(generated) == canonical
    assert [row["episode_id"] for row in generated["episodes"]] == [
        row["episode_id"] for row in rows
    ]
    pilot_by_id = {
        row["habitat_episode_id"]: row["distance_group"]
        for row in pilot["episodes"]
    }
    assert [row["distance_group"] for row in generated["episodes"]] == [
        pilot_by_id[row["episode_id"]] for row in rows
    ]
    alignment = json.loads(
        (result_path / "manifest_alignment.json").read_text(encoding="utf-8")
    )
    assert alignment["aligned"] is True
    assert alignment["pilot_comparison"]["order_matches"] is False

    # A second shard must verify the same execution manifest and alignment
    # file instead of rewriting or rejecting them.
    run.run_exp(**run_kwargs)
    assert len(calls) == 2
    assert canonical_episode_manifest_sha256(
        json.loads(
            (result_path / "execution_manifest.json").read_text(encoding="utf-8")
        )
    ) == canonical


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
    with pytest.raises(ValueError, match="episode-manifest-sha256"):
        run.run_exp(
            **common, experiment_group="uninavid_baseline",
            pilot_manifest=str(manifest_path),
            pilot_sha256=(
                "cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a"
            ),
            expected_episodes=88, episode_manifest="fake-episode-manifest.json",
            episode_manifest_sha256="0" * 64,
        )
    assert calls == []
    run.run_exp(
        **common, experiment_group="uninavid_baseline",
        allow_nonpilot_eval=True,
    )
    assert len(calls) == 1


def test_write_run_metadata_roundtrips_tuples_idempotently(
    run_module, tmp_path,
) -> None:
    run, *_ = run_module
    path = tmp_path / "manifest_alignment.json"
    metadata = {"comparison": {"first_key": ("Scene0.glb", "ep-0")}}
    run._write_run_metadata(str(path), metadata)
    # A second shard must compare equal after the first write's JSON round-trip.
    run._write_run_metadata(str(path), metadata)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "comparison": {"first_key": ["Scene0.glb", "ep-0"]},
    }


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


def test_evaluate_agent_seeds_each_actual_episode_before_agent_reset(
    run_module, tmp_path, monkeypatch,
) -> None:
    import random as py_random

    run, _manifest_path, _digest, _calls = run_module
    episodes = [
        SimpleNamespace(scene_id=f"scene-{i}", episode_id=f"episode-{i}",
                        instruction=SimpleNamespace(instruction_text=f"go {i}"))
        for i in range(2)
    ]

    class FakeEnv:
        def __init__(self, task_config, dataset):
            self.episodes = list(dataset.episodes)
            self.current_episode = None
            self.episode_over = False
            self.index = 0

        def reset(self):
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
            self.episode_over = True

    class FakeAgent:
        reset_calls: ClassVar[list] = []
        reset_rng_draws: ClassVar[list] = []

        def __init__(self, *args, **kwargs):
            pass

        def reset(self, episode=None, instruction=""):
            type(self).reset_calls.append((episode.episode_id, instruction))
            type(self).reset_rng_draws.append(
                (py_random.random(), float(np.random.rand()))
            )

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
    rows = _read_results(result_path)
    assert [row["episode_id"] for row in rows] == ["episode-0", "episode-1"]

    expected_draws = []
    for row in rows:
        expected_seed = run._stable_episode_seed(
            42, row["scene_id"], row["episode_id"],
        )
        assert row["episode_seed"] == expected_seed
        py_random.seed(expected_seed)
        np.random.seed(expected_seed)
        expected_draws.append((py_random.random(), float(np.random.rand())))
    # The first draw observed inside agent.reset() must already reflect the
    # actual Episode's seed.
    assert FakeAgent.reset_rng_draws == expected_draws


def test_formal_path_accepts_exact_frozen_manifest(run_module, monkeypatch) -> None:
    run, _manifest_path, _digest, calls = run_module
    frozen = "cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a"

    class FakeManifest:
        rows = [
            SimpleNamespace(
                scene_basename=f"Scene{scene}.glb",
                episode_id=f"ep-{scene}-{index}",
            )
            for scene in range(11) for index in range(8)
        ]

        def order_digest(self, instructions=None):
            return "fake-order-digest"

    monkeypatch.setattr(
        run, "load_pilot_manifest", lambda *a, **k: FakeManifest(),
    )
    monkeypatch.setattr(
        run, "select_pilot_episodes",
        lambda episodes, manifest, **kwargs: list(episodes),
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


def test_evaluate_agent_accepts_a_shuffled_reset_order(
    run_module, tmp_path, monkeypatch,
) -> None:
    run, _manifest_path, _digest, _calls = run_module
    episodes = [
        SimpleNamespace(scene_id=f"scene-{i}", episode_id=f"episode-{i}",
                        instruction=SimpleNamespace(instruction_text=f"go {i}"))
        for i in range(2)
    ]
    monkeypatch.setattr(
        run, "Env", _episode_env_class([episodes[1], episodes[0]]),
    )
    module, FakeAgent = _fake_agent_module()
    monkeypatch.setitem(sys.modules, "agent_uninavid", module)
    config = SimpleNamespace(
        TASK_CONFIG=SimpleNamespace(),
        EVAL=SimpleNamespace(
            IDENTIFICATION="fake", EARLY_STOP_ROTATION=25,
            EARLY_STOP_STEPS=400,
        ),
    )
    result_path = tmp_path / "result"
    run._original_evaluate_agent(
        config, 0, SimpleNamespace(episodes=episodes), "model",
        str(result_path), "uni-navid", "", split_num=1,
        experiment_group="uninavid_baseline", seed=42, l3_options={},
        manifest_audit=None, episode_order={}, exp_config="fake.yaml",
        allow_nonpilot_eval=False,
    )
    rows = _read_results(result_path)
    assert [row["episode_id"] for row in rows] == ["episode-1", "episode-0"]
    assert [row["habitat_execution_index"] for row in rows] == [0, 1]
    assert FakeAgent.reset_calls == [
        ("episode-1", "go 1"), ("episode-0", "go 0"),
    ]


def test_shuffled_habitat_order_completes_all_88_with_identity_metadata(
    formal_episode_set, tmp_path, monkeypatch,
) -> None:
    formal = formal_episode_set
    order = list(range(88))
    random.Random(7).shuffle(order)
    sequence = [formal.episodes[index] for index in order]
    result_path = tmp_path / "shuffled"
    agent_cls = _run_formal_evaluate(
        formal, sequence, result_path, monkeypatch,
    )
    rows = _read_results(result_path)
    assert len(rows) == 88
    assert len(agent_cls.reset_calls) == 88
    assert [row["episode_id"] for row in rows] == [
        episode.episode_id for episode in sequence
    ]
    assert [row["scene_id"] for row in rows] == [
        episode.scene_id for episode in sequence
    ]
    assert [row["habitat_execution_index"] for row in rows] == list(range(88))
    assert [row["manifest_order_index"] for row in rows] == order

    by_key = {
        (row["scene_basename"], row["episode_id"]): row for row in rows
    }
    assert set(by_key) == set(formal.metadata)
    episode_by_key = {
        (episode.scene_id.split("/")[-1], str(episode.episode_id)): episode
        for episode in formal.episodes
    }
    for key, row in by_key.items():
        metadata = formal.metadata[key]
        assert metadata.manifest_order_index == row["manifest_order_index"]
        assert metadata.pilot_manifest_index == row["pilot_manifest_index"]
        assert metadata.distance_group == row["distance_group"]
        assert row["scene_id"] == episode_by_key[key].scene_id
        assert row["episode_id"] == key[1]
        assert row["episode_seed"] == formal.run._stable_episode_seed(
            42, row["scene_id"], row["episode_id"],
        )
    # The reset position and the manifest position are independent.
    assert any(
        row["manifest_order_index"] != row["habitat_execution_index"]
        for row in rows
    )


def test_duplicate_episode_is_rejected(
    formal_episode_set, tmp_path, monkeypatch,
) -> None:
    formal = formal_episode_set
    sequence = formal.episodes[:3] + [formal.episodes[0]]
    with pytest.raises(EpisodeExecutionError, match="duplicate Episode"):
        _run_formal_evaluate(
            formal, sequence, tmp_path / "duplicate", monkeypatch,
        )


def test_out_of_set_episode_is_rejected(
    formal_episode_set, tmp_path, monkeypatch,
) -> None:
    formal = formal_episode_set
    extra = SimpleNamespace(
        scene_id="/autodl/scene_datasets/hm3d/SceneExtra.glb",
        episode_id="episode-extra",
        instruction=SimpleNamespace(instruction_text="not in the pilot"),
    )
    sequence = formal.episodes[:3] + [extra]
    with pytest.raises(EpisodeExecutionError, match="outside"):
        _run_formal_evaluate(
            formal, sequence, tmp_path / "outside", monkeypatch,
        )


def test_missing_episode_is_detected(
    formal_episode_set, tmp_path, monkeypatch,
) -> None:
    formal = formal_episode_set
    with pytest.raises(EpisodeExecutionError, match="missing"):
        _run_formal_evaluate(
            formal, formal.episodes[:87], tmp_path / "missing", monkeypatch,
        )


def test_ab_groups_pair_by_episode_identity_across_habitat_orders(
    formal_episode_set, tmp_path, monkeypatch,
) -> None:
    formal = formal_episode_set
    order_a = list(range(88))
    order_b = list(range(88))
    random.Random(11).shuffle(order_a)
    random.Random(29).shuffle(order_b)
    assert order_a != order_b

    result_a = tmp_path / "baseline"
    result_b = tmp_path / "l3"
    _run_formal_evaluate(
        formal, [formal.episodes[index] for index in order_a],
        result_a, monkeypatch,
    )
    _run_formal_evaluate(
        formal, [formal.episodes[index] for index in order_b],
        result_b, monkeypatch,
    )
    rows_a = _read_results(result_a)
    rows_b = _read_results(result_b)
    assert len(rows_a) == len(rows_b) == 88
    assert [row["episode_id"] for row in rows_a] != [
        row["episode_id"] for row in rows_b
    ]
    by_key_a = {
        (row["scene_basename"], row["episode_id"]): row for row in rows_a
    }
    by_key_b = {
        (row["scene_basename"], row["episode_id"]): row for row in rows_b
    }
    assert set(by_key_a) == set(by_key_b) == set(formal.metadata)
    for key in formal.metadata:
        for field in (
            "scene_id", "episode_id", "manifest_order_index",
            "pilot_manifest_index", "distance_group", "episode_seed",
            "instruction_sha256",
        ):
            assert by_key_a[key][field] == by_key_b[key][field], (key, field)


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

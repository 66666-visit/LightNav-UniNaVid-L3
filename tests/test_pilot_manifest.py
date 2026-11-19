"""The frozen 88-Episode manifests are loaded, never regenerated.

The three digests are tested separately:

* Pilot file raw SHA256 (``cf1755...``)
* parent manifest raw SHA256 (declared inside the Pilot)
* evaluation episode-manifest canonical SHA256 (``30260e...``)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pytest

from l3_integration.pilot_manifest import (
    EXPECTED_EPISODE_MANIFEST_SHA256,
    EXPECTED_EPISODES,
    EXPECTED_PILOT_MANIFEST_SHA256,
    ExecutionManifest,
    PilotManifestError,
    audit_manifest,
    build_execution_manifest,
    canonical_episode_manifest_sha256,
    canonicalize_episode_manifest,
    compare_episode_manifest_to_pilot,
    load_completion_marker,
    load_episode_manifest,
    load_pilot_manifest,
    order_digest,
    scan_episode_manifests,
    scene_basename,
    select_pilot_episodes,
    sha256_file,
    validate_pilot_manifest,
    write_execution_manifest,
)

FIXED_CANONICAL_DIGEST = (
    "380f6eefa4d55e4d4eff5cf6a4924662f0d6a21bdde519c46957e493bd0d5316"
)


@dataclass
class FakeEpisode:
    scene_id: str
    episode_id: str
    instruction: str

    @property
    def instruction_text(self) -> str:
        return self.instruction


def _manifest_rows() -> list[dict]:
    rows = []
    for scene_index in range(11):
        scene = (
            "data/scene_datasets/hm3d/val/"
            f"{scene_index:05d}-SceneName/Scene{scene_index}.glb"
        )
        for index in range(8):
            group = "short" if index < 3 else "medium" if index < 6 else "long"
            rows.append({
                "scene_id": scene,
                "dataset_scene_id": (
                    f"hm3d/val/{scene_index:05d}-SceneName/Scene{scene_index}.glb"
                ),
                "habitat_episode_id": f"episode-{scene_index}-{index}",
                "episode_id": f"episode-{scene_index}-{index}",
                "distance_group": group,
                "instruction": f"go to the chair in scene {scene_index} number {index}",
                "route_length": 5.0 + index,
                "trajectory_id": f"traj-{scene_index}-{index // 2}",
            })
    return rows


def _episode_rows(rows=None) -> list[dict]:
    rows = _manifest_rows() if rows is None else rows
    return [
        {
            "scene_id": (
                "/autodl/scene_datasets/hm3d/"
                f"Scene{scene_basename(row['scene_id']).removeprefix('Scene').removesuffix('.glb')}.glb"
            ),
            "episode_id": row["habitat_episode_id"],
        }
        for row in rows
    ]


def _write_manifest(
    tmp_path, rows=None, *, parent_sha: str = "a" * 64, extra=None,
) -> tuple[object, str]:
    data = {
        "protocol_version": "rgm_pilot_88_v1",
        "dataset_split": "val_unseen",
        "sample_seed": 42,
        "expected_count": 88,
        "per_scene": 8,
        "parent_manifest_sha256": parent_sha,
        "scene_quotas": {f"scene-{index}": 8 for index in range(11)},
        "episodes": _manifest_rows() if rows is None else rows,
    }
    if extra:
        data.update(extra)
    path = tmp_path / "pilot_manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _write_episode_manifest(
    tmp_path, rows=None, *, name: str = "episode_manifest.json", indent: int = 2,
) -> tuple[object, str, str]:
    rows = _episode_rows() if rows is None else rows
    data = {"split": "val_unseen", "expected_count": len(rows), "episodes": rows}
    path = tmp_path / name
    path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    return (
        path,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        canonical_episode_manifest_sha256(data),
    )


def _dataset_episodes(rows=None) -> list[FakeEpisode]:
    rows = _manifest_rows() if rows is None else rows
    episodes = []
    for row in reversed(rows):
        scene = row["scene_id"].replace(
            "data/scene_datasets", "/autodl/scene_datasets",
        )
        episodes.append(FakeEpisode(scene, row["habitat_episode_id"], row["instruction"]))
    return episodes


def test_frozen_digests_are_distinct_and_have_explicit_meanings() -> None:
    assert EXPECTED_EPISODES == 88
    assert EXPECTED_PILOT_MANIFEST_SHA256 == (
        "cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a"
    )
    assert EXPECTED_EPISODE_MANIFEST_SHA256 == (
        "30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98"
    )
    assert len(EXPECTED_PILOT_MANIFEST_SHA256) == 64
    assert len(EXPECTED_EPISODE_MANIFEST_SHA256) == 64
    assert EXPECTED_PILOT_MANIFEST_SHA256 != EXPECTED_EPISODE_MANIFEST_SHA256


def test_manifest_loads_exactly_88_and_preserves_order(tmp_path) -> None:
    path, digest = _write_manifest(tmp_path)
    manifest = load_pilot_manifest(path, expected_sha256=digest)
    assert manifest.expected_count == 88
    assert manifest.file_sha256 == digest
    assert manifest.parent_manifest_sha256 == "a" * 64
    assert len(manifest.scene_counts()) == 11
    assert set(manifest.scene_counts().values()) == {8}
    assert all(counts == {"short": 3, "medium": 3, "long": 2}
               for counts in manifest.strata_counts().values())
    selected = select_pilot_episodes(_dataset_episodes(), manifest)
    assert [str(episode.episode_id) for episode in selected] == [
        row.episode_id for row in manifest.rows
    ]
    assert [str(episode.scene_id) for episode in selected] != [
        row.scene_id for row in manifest.rows
    ]


def test_both_groups_select_the_same_episode_sequence(tmp_path) -> None:
    path, digest = _write_manifest(tmp_path)
    manifest = load_pilot_manifest(path, expected_sha256=digest)
    baseline = select_pilot_episodes(_dataset_episodes(), manifest)
    l3 = select_pilot_episodes(_dataset_episodes(), manifest)
    assert [(e.scene_id, e.episode_id) for e in baseline] == [
        (e.scene_id, e.episode_id) for e in l3
    ]
    assert [e.instruction for e in baseline] == [e.instruction for e in l3]


def test_pilot_sha_mismatch_fails_closed(tmp_path) -> None:
    path, _digest = _write_manifest(tmp_path)
    with pytest.raises(PilotManifestError, match="file SHA256 mismatch"):
        load_pilot_manifest(path, expected_sha256="0" * 64)


def test_historical_episode_digest_is_not_accepted_as_the_pilot_file_sha(
    tmp_path,
) -> None:
    path, _digest = _write_manifest(tmp_path)
    with pytest.raises(PilotManifestError, match="pilot manifest file SHA256"):
        load_pilot_manifest(
            path, expected_sha256=EXPECTED_EPISODE_MANIFEST_SHA256,
        )


def test_missing_file_requests_the_original_instead_of_sampling(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="do not regenerate"):
        load_pilot_manifest(tmp_path / "missing.json")


def test_wrong_count_duplicate_and_strata_are_rejected() -> None:
    rows = _manifest_rows()
    with pytest.raises(PilotManifestError, match="exactly 88"):
        validate_pilot_manifest({"expected_count": 87, "episodes": rows[:87]})
    duplicate = rows[:-1] + [dict(rows[0])]
    with pytest.raises(PilotManifestError, match="duplicate"):
        validate_pilot_manifest({"expected_count": 88, "episodes": duplicate})
    bad_strata = [dict(row) for row in rows]
    for row in bad_strata[:8]:
        row["distance_group"] = "short"
    with pytest.raises(PilotManifestError, match="3/3/2"):
        validate_pilot_manifest({"expected_count": 88, "episodes": bad_strata})


def test_instruction_mismatch_is_rejected(tmp_path) -> None:
    path, digest = _write_manifest(tmp_path)
    manifest = load_pilot_manifest(path, expected_sha256=digest)
    episodes = _dataset_episodes()
    episodes[0].instruction = "a different instruction"
    with pytest.raises(PilotManifestError, match="instruction differs"):
        select_pilot_episodes(episodes, manifest)


def test_missing_dataset_episode_is_reported(tmp_path) -> None:
    path, digest = _write_manifest(tmp_path)
    manifest = load_pilot_manifest(path, expected_sha256=digest)
    episodes = _dataset_episodes()[:-1]
    with pytest.raises(PilotManifestError, match="missing pilot Episodes"):
        select_pilot_episodes(episodes, manifest)


def test_order_digest_changes_when_order_or_instruction_changes(tmp_path) -> None:
    path, digest = _write_manifest(tmp_path)
    manifest = load_pilot_manifest(path, expected_sha256=digest)
    instructions = [row.instruction for row in manifest.rows]
    first = order_digest(manifest.rows, instructions=instructions)
    assert first == manifest.order_digest(instructions=instructions)
    changed = list(instructions)
    changed[0] = "changed"
    assert order_digest(manifest.rows, instructions=changed) != first
    reordered = list(manifest.rows)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert order_digest(reordered, instructions=instructions) != first


def test_audit_reports_dataset_match_and_instruction_hashes(tmp_path) -> None:
    path, digest = _write_manifest(tmp_path)
    report = audit_manifest(
        path, expected_sha256=digest, dataset_episodes=_dataset_episodes(),
    )
    assert report["dataset_match"] is True
    assert report["dataset_count"] == 88
    assert len(report["dataset_episode_keys"]) == 88
    assert report["dataset_episode_keys"][0] == {
        "scene_basename": "Scene0.glb", "episode_id": "episode-0-0",
    }
    assert report["dataset_episode_keys"][-1] == {
        "scene_basename": "Scene10.glb", "episode_id": "episode-10-7",
    }
    assert report["file_sha256"] == digest
    assert report["digest_kind"].startswith("pilot_manifest.json raw")
    assert report["parent_manifest_sha256"] == "a" * 64
    assert report["parent_manifest"]["declared_sha256_in_pilot"] == "a" * 64
    assert report["parent_manifest"]["verified"] is False
    assert report["episode_manifest"]["verified"] is False
    assert len(report["instruction_sha256"]) == 88
    assert report["dataset_order_digest"] == order_digest(
        load_pilot_manifest(path, expected_sha256=digest).rows,
        instructions=[row["instruction"] for row in _manifest_rows()],
    )
    assert scene_basename("a/b/c.glb") == "c.glb"


def test_canonical_digest_is_not_the_file_sha(tmp_path) -> None:
    rows = [
        {"scene_id": "/data/scene_datasets/hm3d/Scene0.glb",
         "episode_id": "ep-0", "ignored": "x"},
        {"scene_id": "/data/scene_datasets/hm3d/Scene1.glb",
         "episode_id": "ep-1"},
        {"scene_id": "/data/scene_datasets/hm3d/Scene2.glb",
         "episode_id": "ep-2"},
    ]
    data = {"split": "val_unseen", "expected_count": 3, "episodes": rows}
    first = canonical_episode_manifest_sha256(data)
    assert first == FIXED_CANONICAL_DIGEST
    # Unknown fields are stripped; whitespace/indent is not part of the digest.
    assert canonicalize_episode_manifest(data) == {
        "split": "val_unseen",
        "expected_count": 3,
        "episodes": [
            {"scene_id": row["scene_id"], "episode_id": row["episode_id"]}
            for row in rows
        ],
    }
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(json.dumps(data, indent=2), encoding="utf-8")
    path_b.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    file_a = hashlib.sha256(path_a.read_bytes()).hexdigest()
    file_b = hashlib.sha256(path_b.read_bytes()).hexdigest()
    assert file_a != file_b
    assert first != file_a and first != file_b
    assert canonical_episode_manifest_sha256(
        json.loads(path_a.read_text(encoding="utf-8"))
    ) == canonical_episode_manifest_sha256(
        json.loads(path_b.read_text(encoding="utf-8"))
    ) == first


def test_episode_manifest_verification_and_order_match(tmp_path) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    episode_path, episode_file, episode_canonical = _write_episode_manifest(
        tmp_path,
    )
    assert episode_file != episode_canonical
    report = audit_manifest(
        pilot_path,
        expected_sha256=pilot_digest,
        episode_manifest=episode_path,
        expected_episode_manifest_sha256=episode_canonical,
    )
    episode = report["episode_manifest"]
    assert episode["verified"] is True
    assert episode["file_sha256"] == episode_file
    assert episode["canonical_sha256"] == episode_canonical
    assert episode["expected_canonical_sha256"] == episode_canonical
    assert episode["pilot_comparison"]["match"] is True
    assert episode["pilot_comparison"]["count"] == 88
    assert episode["digest_kind"].startswith("canonical JSON SHA256")
    manifest = load_pilot_manifest(pilot_path, expected_sha256=pilot_digest)
    loaded = load_episode_manifest(
        episode_path, expected_canonical_sha256=episode_canonical,
    )
    assert compare_episode_manifest_to_pilot(loaded, manifest)["match"] is True


def test_episode_manifest_bad_canonical_digest_is_rejected(tmp_path) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    episode_path, _file_digest, _canonical = _write_episode_manifest(tmp_path)
    with pytest.raises(PilotManifestError, match="canonical SHA256 mismatch"):
        audit_manifest(
            pilot_path,
            expected_sha256=pilot_digest,
            episode_manifest=episode_path,
            expected_episode_manifest_sha256=EXPECTED_EPISODE_MANIFEST_SHA256,
        )
    # The completion-marker digest is not a raw file digest.
    with pytest.raises(PilotManifestError, match="canonical"):
        load_episode_manifest(
            episode_path,
            expected_canonical_sha256=EXPECTED_EPISODE_MANIFEST_SHA256,
        )


def test_same_set_different_order_passes_and_uses_historical_order(
    tmp_path,
) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    load_pilot_manifest(pilot_path, expected_sha256=pilot_digest)
    episode_rows = list(reversed(_episode_rows()))
    episode_path, episode_file, canonical = _write_episode_manifest(
        tmp_path, rows=episode_rows,
    )
    report = audit_manifest(
        pilot_path,
        expected_sha256=pilot_digest,
        dataset_episodes=_dataset_episodes(),
        episode_manifest=episode_path,
        expected_episode_manifest_sha256=canonical,
    )
    episode = report["episode_manifest"]
    comparison = episode["pilot_comparison"]
    assert episode["verified"] is True
    assert episode["canonical_sha256"] == canonical
    assert episode["file_sha256"] == episode_file
    assert comparison["match"] is True
    assert comparison["sets_equal"] is True
    assert comparison["order_matches"] is False
    assert comparison["order_relation"] == "same_set_different_order"
    assert comparison["order_source"] == "episode_manifest.json"
    assert comparison["first_key"] == (
        scene_basename(episode_rows[0]["scene_id"]),
        episode_rows[0]["episode_id"],
    )
    assert report["dataset_order_source"] == "episode_manifest.json"
    assert [row["episode_id"] for row in report["dataset_episode_keys"]] == [
        row["episode_id"] for row in episode_rows
    ]
    execution = report["execution_manifest"]
    assert execution["aligned"] is True
    assert execution["canonical_sha256"] == canonical
    assert execution["historical_episode_manifest_canonical_sha256"] == canonical
    assert execution["count"] == 88
    assert execution["order_source"] == "episode_manifest.json"
    assert execution["order_matches_pilot"] is False
    assert [row["episode_id"] for row in execution["episodes"]] == [
        row["episode_id"] for row in episode_rows
    ]


def test_execution_order_is_the_historical_manifest_order(tmp_path) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    pilot = load_pilot_manifest(pilot_path, expected_sha256=pilot_digest)
    episode_rows = list(reversed(_episode_rows()))
    episode_path, _file_digest, canonical = _write_episode_manifest(
        tmp_path, rows=episode_rows,
    )
    episode = load_episode_manifest(
        episode_path, expected_canonical_sha256=canonical,
    )
    execution = build_execution_manifest(episode, pilot)
    assert execution.keys() == tuple(
        (scene_basename(row["scene_id"]), row["episode_id"])
        for row in episode_rows
    )
    assert execution.keys() == tuple(reversed(pilot.keys()))
    assert execution.canonical_sha256 == canonical
    assert [row.execution_index for row in execution.rows] == list(range(88))


def test_distance_group_is_associated_by_episode_key_not_position(
    tmp_path,
) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    pilot = load_pilot_manifest(pilot_path, expected_sha256=pilot_digest)
    episode_rows = list(reversed(_episode_rows()))
    episode_path, _file_digest, canonical = _write_episode_manifest(
        tmp_path, rows=episode_rows,
    )
    episode = load_episode_manifest(
        episode_path, expected_canonical_sha256=canonical,
    )
    execution = build_execution_manifest(episode, pilot)
    pilot_by_key = {
        (row.scene_basename, row.episode_id): row for row in pilot.rows
    }
    assert [row.distance_group for row in execution.rows] == [
        pilot_by_key[row.key].distance_group for row in execution.rows
    ]
    first = execution.rows[0]
    assert first.distance_group == pilot_by_key[first.key].distance_group
    assert first.distance_group == "long"
    assert first.distance_group != pilot.rows[0].distance_group
    assert first.pilot_index == 87
    assert first.execution_index == 0
    assert execution.strata_counts() == pilot.strata_counts()


def test_dataset_selection_uses_historical_order_and_keyed_instructions(
    tmp_path,
) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    pilot = load_pilot_manifest(pilot_path, expected_sha256=pilot_digest)
    episode_rows = list(reversed(_episode_rows()))
    episode_path, _file_digest, canonical = _write_episode_manifest(
        tmp_path, rows=episode_rows,
    )
    episode = load_episode_manifest(
        episode_path, expected_canonical_sha256=canonical,
    )
    execution = build_execution_manifest(episode, pilot)
    # Give the dataset the Pilot's order, which is distinct from both the
    # historical order and the reversed helper order. Selection must still
    # follow the ExecutionManifest, not the dataset list.
    dataset = list(reversed(_dataset_episodes()))
    selected = select_pilot_episodes(
        dataset, pilot, execution_manifest=execution,
    )
    assert [str(item.episode_id) for item in selected] == [
        row["episode_id"] for row in episode_rows
    ]
    assert [str(item.episode_id) for item in dataset] == [
        row.episode_id for row in pilot.rows
    ]
    assert [str(item.episode_id) for item in selected] != [
        row.episode_id for row in pilot.rows
    ]

    # Instruction validation must follow the Episode key, not list position.
    target = next(
        item for item in dataset
        if item.episode_id == episode_rows[0]["episode_id"]
    )
    target.instruction = "a different instruction"
    with pytest.raises(PilotManifestError, match="instruction differs"):
        select_pilot_episodes(
            dataset, pilot, execution_manifest=execution,
        )


def test_missing_episode_fails_set_consistency(tmp_path) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    episode_path, _file_digest, canonical = _write_episode_manifest(
        tmp_path, rows=_episode_rows()[:-1],
    )
    with pytest.raises(PilotManifestError, match="has 87 Episodes"):
        audit_manifest(
            pilot_path,
            expected_sha256=pilot_digest,
            episode_manifest=episode_path,
            expected_episode_manifest_sha256=canonical,
        )


def test_equal_count_missing_episode_fails_set_consistency(tmp_path) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    rows = _episode_rows()[:-1] + [{
        "scene_id": "/autodl/scene_datasets/hm3d/SceneExtra.glb",
        "episode_id": "episode-extra",
    }]
    episode_path, _file_digest, canonical = _write_episode_manifest(
        tmp_path, rows=rows,
    )
    with pytest.raises(
        PilotManifestError, match="Episode set differs from the Pilot",
    ) as error:
        audit_manifest(
            pilot_path,
            expected_sha256=pilot_digest,
            episode_manifest=episode_path,
            expected_episode_manifest_sha256=canonical,
        )
    assert "episode-10-7" in str(error.value)
    assert "episode-extra" in str(error.value)


def test_extra_episode_fails_set_consistency(tmp_path) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    rows = _episode_rows()
    rows[-1] = {
        "scene_id": "/autodl/scene_datasets/hm3d/SceneExtra.glb",
        "episode_id": "episode-extra",
    }
    episode_path, _file_digest, canonical = _write_episode_manifest(
        tmp_path, rows=rows,
    )
    with pytest.raises(
        PilotManifestError, match="Episode set differs from the Pilot",
    ) as error:
        audit_manifest(
            pilot_path,
            expected_sha256=pilot_digest,
            episode_manifest=episode_path,
            expected_episode_manifest_sha256=canonical,
        )
    assert "missing=" in str(error.value)
    assert "episode-extra" in str(error.value)
    # 89 rows also fails the explicit 88-Episode count gate.
    episode_path, _file_digest, canonical = _write_episode_manifest(
        tmp_path,
        rows=_episode_rows() + [{
            "scene_id": "/autodl/scene_datasets/hm3d/SceneExtra.glb",
            "episode_id": "episode-extra",
        }],
    )
    with pytest.raises(PilotManifestError, match="has 89 Episodes"):
        audit_manifest(
            pilot_path,
            expected_sha256=pilot_digest,
            episode_manifest=episode_path,
            expected_episode_manifest_sha256=canonical,
        )


def test_duplicate_episode_fails_uniqueness_check(tmp_path) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    rows = _episode_rows()
    rows[-1] = dict(rows[0])
    data = {"split": "val_unseen", "expected_count": 88, "episodes": rows}
    episode_path = tmp_path / "episode_manifest.json"
    episode_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PilotManifestError, match="duplicate episode in manifest"):
        load_episode_manifest(episode_path, expected_canonical_sha256=None)
    with pytest.raises(PilotManifestError, match="duplicate episode in manifest"):
        audit_manifest(
            pilot_path,
            expected_sha256=pilot_digest,
            episode_manifest=episode_path,
            expected_episode_manifest_sha256=None,
        )


def test_generated_execution_manifest_aligns_with_historical_digest(
    tmp_path,
) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    pilot = load_pilot_manifest(pilot_path, expected_sha256=pilot_digest)
    episode_rows = list(reversed(_episode_rows()))
    episode_path, _file_digest, canonical = _write_episode_manifest(
        tmp_path, rows=episode_rows,
    )
    episode = load_episode_manifest(
        episode_path, expected_canonical_sha256=canonical,
    )
    execution = build_execution_manifest(episode, pilot)
    target = tmp_path / "output" / "execution_manifest.json"

    report = write_execution_manifest(
        target, execution, expected_canonical_sha256=canonical,
    )
    assert report["verified"] is True
    assert report["aligned"] is True
    assert report["created"] is True
    assert report["canonical_sha256"] == canonical
    assert report["historical_episode_manifest_canonical_sha256"] == canonical
    assert report["file_sha256"] == sha256_file(target)
    assert report["path"] == str(target)

    written = json.loads(target.read_text(encoding="utf-8"))
    assert canonical_episode_manifest_sha256(written) == canonical
    assert written["execution_order"] == "episode_manifest.json"
    assert len(written["episodes"]) == 88
    assert [row["episode_id"] for row in written["episodes"]] == [
        row["episode_id"] for row in episode_rows
    ]
    assert written["episodes"][0]["distance_group"] == "long"
    assert "instruction" in written["episodes"][0]
    loaded = load_episode_manifest(
        target, expected_canonical_sha256=canonical,
    )
    assert loaded.keys() == execution.keys()

    before = target.read_bytes()
    again = write_execution_manifest(
        target, execution, expected_canonical_sha256=canonical,
    )
    assert again["created"] is False
    assert again["verified"] is True
    assert again["canonical_sha256"] == canonical
    assert target.read_bytes() == before

    # Canonicalisation ignores Pilot metadata, so a tampered distance_group
    # keeps the digest but must not be accepted as an already-valid artifact.
    tampered = json.loads(target.read_text(encoding="utf-8"))
    tampered["episodes"][0]["distance_group"] = "short"
    target.write_text(json.dumps(tampered, indent=2), encoding="utf-8")
    with pytest.raises(PilotManifestError, match="metadata differs"):
        write_execution_manifest(
            target, execution, expected_canonical_sha256=canonical,
        )


def test_manually_built_execution_manifest_with_wrong_set_is_rejected(
    tmp_path,
) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    pilot = load_pilot_manifest(pilot_path, expected_sha256=pilot_digest)
    episode_rows = list(reversed(_episode_rows()))
    episode_path, _file_digest, canonical = _write_episode_manifest(
        tmp_path, rows=episode_rows,
    )
    episode = load_episode_manifest(
        episode_path, expected_canonical_sha256=canonical,
    )
    execution = build_execution_manifest(episode, pilot)
    broken = ExecutionManifest(
        episode=episode,
        pilot=pilot,
        rows=execution.rows[:-1],
        canonical_sha256=execution.canonical_sha256,
    )
    assert len(broken.rows) == 87
    with pytest.raises(PilotManifestError, match="has 87 Episodes"):
        broken.comparison()
    with pytest.raises(PilotManifestError, match="has 87 Episodes"):
        audit_manifest(
            pilot_path,
            expected_sha256=pilot_digest,
            episode_manifest=episode_path,
            expected_episode_manifest_sha256=canonical,
            execution_manifest=broken,
        )


def test_execution_manifest_rejects_a_wrong_expected_digest(tmp_path) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    pilot = load_pilot_manifest(pilot_path, expected_sha256=pilot_digest)
    episode_rows = list(reversed(_episode_rows()))
    episode_path, _file_digest, canonical = _write_episode_manifest(
        tmp_path, rows=episode_rows,
    )
    episode = load_episode_manifest(
        episode_path, expected_canonical_sha256=canonical,
    )
    execution = build_execution_manifest(episode, pilot)
    with pytest.raises(PilotManifestError, match="canonical SHA256 mismatch"):
        write_execution_manifest(
            tmp_path / "execution_manifest.json",
            execution,
            expected_canonical_sha256="0" * 64,
        )


def test_write_execution_manifest_refuses_to_overwrite_input_files(
    tmp_path,
) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    pilot = load_pilot_manifest(pilot_path, expected_sha256=pilot_digest)
    episode_rows = list(reversed(_episode_rows()))
    episode_path, _file_digest, canonical = _write_episode_manifest(
        tmp_path, rows=episode_rows,
    )
    episode = load_episode_manifest(
        episode_path, expected_canonical_sha256=canonical,
    )
    execution = build_execution_manifest(episode, pilot)
    pilot_before = pilot_path.read_bytes()
    episode_before = episode_path.read_bytes()

    with pytest.raises(
        PilotManifestError, match="refusing to overwrite the historical",
    ):
        write_execution_manifest(
            episode_path, execution,
            expected_canonical_sha256=canonical,
        )
    with pytest.raises(
        PilotManifestError, match="refusing to overwrite the pilot_manifest",
    ):
        write_execution_manifest(
            pilot_path, execution,
            expected_canonical_sha256=canonical,
        )
    assert pilot_path.read_bytes() == pilot_before
    assert episode_path.read_bytes() == episode_before


def test_original_manifest_files_and_their_fixed_hashes_are_unchanged(
    tmp_path,
) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    episode_rows = list(reversed(_episode_rows()))
    episode_path, episode_file, canonical = _write_episode_manifest(
        tmp_path, rows=episode_rows,
    )
    pilot_bytes = pilot_path.read_bytes()
    episode_bytes = episode_path.read_bytes()
    assert sha256_file(pilot_path) == pilot_digest
    assert sha256_file(episode_path) == episode_file
    assert canonical != episode_file

    # Each file is verified with its own fixed digest algorithm.
    pilot = load_pilot_manifest(pilot_path, expected_sha256=pilot_digest)
    episode = load_episode_manifest(
        episode_path, expected_canonical_sha256=canonical,
    )
    with pytest.raises(PilotManifestError, match="pilot manifest file SHA256"):
        load_pilot_manifest(pilot_path, expected_sha256=canonical)
    with pytest.raises(PilotManifestError, match="canonical SHA256 mismatch"):
        load_episode_manifest(
            episode_path, expected_canonical_sha256=pilot_digest,
        )

    execution = build_execution_manifest(episode, pilot)
    write_execution_manifest(
        tmp_path / "generated" / "execution_manifest.json",
        execution,
        expected_canonical_sha256=canonical,
    )
    assert pilot_path.read_bytes() == pilot_bytes
    assert episode_path.read_bytes() == episode_bytes
    assert sha256_file(pilot_path) == pilot_digest
    assert sha256_file(episode_path) == episode_file
    assert canonical_episode_manifest_sha256(
        json.loads(episode_path.read_text(encoding="utf-8"))
    ) == canonical


def test_parent_manifest_verification_is_separate(tmp_path) -> None:
    parent = tmp_path / "parent_manifest.json"
    parent.write_text('{"parent": true}\n', encoding="utf-8")
    parent_digest = hashlib.sha256(parent.read_bytes()).hexdigest()
    pilot_path, pilot_digest = _write_manifest(tmp_path, parent_sha=parent_digest)
    report = audit_manifest(
        pilot_path,
        expected_sha256=pilot_digest,
        parent_manifest=parent,
    )
    parent_report = report["parent_manifest"]
    assert parent_report["verified"] is True
    assert parent_report["file_sha256"] == parent_digest
    assert parent_report["declared_sha256_in_pilot"] == parent_digest
    assert parent_digest != pilot_digest
    with pytest.raises(PilotManifestError, match="parent manifest file SHA256"):
        audit_manifest(
            pilot_path,
            expected_sha256=pilot_digest,
            parent_manifest=parent,
            expected_parent_manifest_sha256="0" * 64,
        )


def test_scan_recomputes_canonical_digests_not_raw_file_hashes(tmp_path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    episode_path, episode_file, episode_canonical = _write_episode_manifest(
        tmp_path, name="one.json",
    )
    target = root / "episode_manifest.json"
    target.write_bytes(episode_path.read_bytes())
    other = root / "other" / "episode_manifest.json"
    other.parent.mkdir()
    other.write_text(
        json.dumps({
            "split": "val_unseen", "expected_count": 1,
            "episodes": [{"scene_id": "scene.glb", "episode_id": "ep"}],
        }),
        encoding="utf-8",
    )
    scanned = scan_episode_manifests(
        root, expected_canonical_sha256=episode_canonical,
    )
    assert len(scanned) == 2
    match = next(entry for entry in scanned if entry["path"] == str(target))
    assert match["matches_expected"] is True
    assert match["canonical_sha256"] == episode_canonical
    assert match["file_sha256"] == episode_file
    assert match["file_sha256"] != match["canonical_sha256"]


def test_unreadable_manifest_is_reported_not_regenerated(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from l3_integration.pilot_manifest import sha256_file

    path = tmp_path / "pilot_manifest.json"
    path.write_text("{}", encoding="utf-8")

    def deny(self):
        raise PermissionError("denied by test")

    monkeypatch.setattr(Path, "read_bytes", deny)
    with pytest.raises(PilotManifestError, match="not readable"):
        sha256_file(path)


def test_episode_manifest_must_stay_at_88_episodes(tmp_path) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    rows = _episode_rows()[:-1]
    episode_path, _file_digest, canonical = _write_episode_manifest(
        tmp_path, rows=rows,
    )
    with pytest.raises(PilotManifestError, match="has 87 Episodes"):
        audit_manifest(
            pilot_path,
            expected_sha256=pilot_digest,
            episode_manifest=episode_path,
            expected_episode_manifest_sha256=canonical,
        )


def test_episode_manifest_split_must_be_val_unseen(tmp_path) -> None:
    pilot_path, pilot_digest = _write_manifest(tmp_path)
    data = {
        "split": "train",
        "expected_count": 88,
        "episodes": _episode_rows(),
    }
    episode_path = tmp_path / "episode_manifest.json"
    episode_path.write_text(json.dumps(data), encoding="utf-8")
    canonical = canonical_episode_manifest_sha256(data)
    with pytest.raises(PilotManifestError, match="split must be val_unseen"):
        audit_manifest(
            pilot_path,
            expected_sha256=pilot_digest,
            episode_manifest=episode_path,
            expected_episode_manifest_sha256=canonical,
        )


def test_completion_marker_is_verified_separately_from_files(tmp_path) -> None:
    marker = tmp_path / "pilot_complete.json"
    marker.write_text(json.dumps({
        "expected_count": 88,
        "manifest_sha256": EXPECTED_EPISODE_MANIFEST_SHA256,
    }), encoding="utf-8")
    report = load_completion_marker(marker)
    assert report["verified"] is True
    assert report["expected_count"] == 88
    assert report["manifest_sha256"] == EXPECTED_EPISODE_MANIFEST_SHA256
    with pytest.raises(PilotManifestError, match="expected_count mismatch"):
        load_completion_marker(marker, expected_count=87)
    with pytest.raises(PilotManifestError, match="canonical digest mismatch"):
        load_completion_marker(marker, expected_sha256="0" * 64)

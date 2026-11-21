"""Paired report contracts: identity, immutable L3 text, and history status."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_l3_l1_report as report


KEYS = (("scene_a.glb", "1"), ("scene_b.glb", "2"))


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _config(group):
    return {
        "group": group,
        "seed": 42,
        "formal_pilot": True,
        "episode_manifest_verified": True,
        "episode_manifest_aligned": True,
        "pilot_manifest_file_sha256": report.EXPECTED_PILOT_MANIFEST_SHA256,
        "execution_manifest_canonical_sha256": report.EXPECTED_EPISODE_MANIFEST_SHA256,
        "model_name": "uni-navid",
        "model_path": "/checkpoint",
        "exp_config": "VLN_CE/vlnce_baselines/config/r2r_baselines/uninavid_l3_r2r.yaml",
        "split_num": 1,
        "exp_save": "data",
        "pilot_order_digest": "same-pilot-order",
        "execution_order_digest": "same-episode-order",
        "l3": {"max_items": 2, "max_chars": 600},
    }


def _row(key, group, success):
    row = {
        "scene_basename": key[0], "episode_id": key[1], "group": group,
        "instruction_sha256": "instruction-" + key[1], "episode_seed": int(key[1]),
        "success": success, "spl": 0.5, "ndtw": 0.4,
        "distance_to_goal": 1.0, "oracle_success": True, "path_length": 3.0,
        "inference_latency_ms_mean": 30.0, "prompt_tokens_mean": 200.0,
        "official_prompt_sha256": report.PROMPT_SHA,
        "pilot_manifest_file_sha256": report.EXPECTED_PILOT_MANIFEST_SHA256,
        "episode_manifest_canonical_sha256": report.EXPECTED_EPISODE_MANIFEST_SHA256,
        "episode_manifest_verified": True,
        "l3_fact_count_total": 2, "l3_context_tokens_mean": 10.0,
        "prompt_has_l3_steps": 1,
    }
    if group == report.B_GROUP:
        row.update({
            "l1_fact_count_total": 1, "l1_context_tokens_mean": 5.0,
            "duplicated_facts_total": 0, "conflicting_facts_total": 1,
            "retained_facts_total": 3, "omitted_facts_total": 0,
            "prompt_has_l1_steps": 1, "l3_original_memory_unchanged": True,
            "source_background_facts_total": 3,
            "l1_observation_attempt_steps": 2, "l1_observe_error_count": 0,
            "l1_detector_error_count": 0, "l1_place_attempt_steps": 1,
            "l1_place_observed_steps": 1, "l1_place_error_count": 0,
            "l3_segmentation_error_count": 0,
            "cross_label_entity_link_error_count": 0,
            "cross_label_entity_links_total": 1,
            "conflict_type_counts": {"different_value": 1},
            "conflict_evidence_ids": [{
                "l1": ["l1-evidence-" + key[1]],
                "l3": ["l3-evidence-" + key[1]],
            }],
        })
    return row


@pytest.fixture
def run_dirs(tmp_path, monkeypatch):
    baseline, fused = tmp_path / "a", tmp_path / "b"
    baseline.mkdir()
    fused.mkdir()
    _write_json(baseline / "run_config.json", _config(report.A_GROUP))
    _write_json(fused / "run_config.json", _config(report.B_GROUP))
    _write_jsonl(baseline / "results.jsonl", [
        _row(KEYS[0], report.A_GROUP, False),
        _row(KEYS[1], report.A_GROUP, True),
    ])
    _write_jsonl(fused / "results.jsonl", [
        _row(KEYS[0], report.B_GROUP, True),
        _row(KEYS[1], report.B_GROUP, True),
    ])
    (baseline / "execution_manifest.json").write_text("{}\n")
    (fused / "execution_manifest.json").write_text("{}\n")
    core = "[Navigation Map Memory - L3]\nchair: seen at step 2."
    _write_jsonl(fused / "steps.jsonl", [{
        "prompt_has_l3": True,
        "l3_context_text": core,
        "l3_original_sha256": hashlib.sha256(core.encode()).hexdigest(),
        "prompt_text": "Official task text\n[Navigation Map Memory - L3]\n"
                       "chair: seen at step 2. [RGB-D evidence=e1] Official suffix",
    }])
    source_a, source_b = tmp_path / "source_a", tmp_path / "source_b"
    dev_root = Path(__file__).resolve().parents[1]
    baseline_agent = subprocess.check_output(
        ["git", "show", report.BASELINE_SHA + ":agent_uninavid.py"], cwd=str(dev_root)
    )
    config_rel = _config(report.A_GROUP)["exp_config"]
    task_rel = "VLN_CE/habitat_extensions/config/vlnce_task_uninavid_l3_r2r.yaml"
    l3_sources = [
        "l3_integration/official_prompt.py",
        "l3_integration/habitat_geometry.py",
    ] + [str(path.relative_to(dev_root))
         for path in (dev_root / "l3_memory").glob("*.py")]
    for source in (source_a, source_b):
        for name in l3_sources:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original L3 source\n", encoding="utf-8")
        (source / "agent_uninavid.py").write_bytes(
            baseline_agent if source == source_a else (dev_root / "agent_uninavid.py").read_bytes()
        )
        for name in (config_rel, task_rel):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("same Habitat config\n", encoding="utf-8")
    runtime_sources = [
        "run.py", "l3_integration/map_sidecar.py",
        "l3_integration/sbadir_fusion.py",
    ]
    runtime_sources += [str(path.relative_to(dev_root))
                        for path in (dev_root / "l1_memory").glob("*.py")]
    runtime_sources += [str(path.relative_to(dev_root))
                        for path in (dev_root / "l1_integration").glob("*.py")]
    for name in runtime_sources:
        path = source_b / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((dev_root / name).read_bytes())
    source_hashes = report._b_source_hashes(source_b)
    source_digest = report._b_source_digest(source_hashes)
    b_config = _config(report.B_GROUP)
    b_config["l1"] = {"source_tree_sha256": source_digest}
    _write_json(fused / "run_config.json", b_config)
    launch = {
        "baseline_worktree": str(source_a), "dev_worktree": str(source_b),
        "baseline_commit": report.BASELINE_SHA,
        "official_prompt_sha256": report.PROMPT_SHA,
        "pilot_manifest_file_sha256": report.EXPECTED_PILOT_MANIFEST_SHA256,
        "episode_manifest_canonical_sha256": report.EXPECTED_EPISODE_MANIFEST_SHA256,
        "model_path": "/checkpoint", "temperature": 0.2,
        "max_new_tokens": 1024, "chunks": 1,
        "config_sha256": hashlib.sha256(b"same Habitat config\n").hexdigest(),
        "task_config_sha256": hashlib.sha256(b"same Habitat config\n").hexdigest(),
        "b_source_files_sha256": source_hashes,
        "b_source_tree_sha256": source_digest,
    }
    monkeypatch.setattr(report, "load_episode_manifest", lambda path: SimpleNamespace(keys=lambda: KEYS))
    return baseline, fused, launch, tmp_path / "report"


def _build(run_dirs):
    baseline, fused, launch, output = run_dirs
    return report.build_report(
        baseline, fused, output, KEYS, {"file_sha256": report.EXPECTED_PILOT_MANIFEST_SHA256},
        {"canonical_sha256": report.EXPECTED_EPISODE_MANIFEST_SHA256},
        launch, expected_count=2,
    )


def test_report_pairs_by_identity_and_marks_missing_history(run_dirs):
    summary = _build(run_dirs)
    output = run_dirs[3]
    assert summary["n_pairs"] == 2
    assert summary["groups"][report.A_NAME]["successes"] == 1
    assert summary["groups"][report.B_NAME]["successes"] == 2
    assert summary["paired"]["success_flips"]["baseline_fail_l3_success"] == 1
    assert summary["history"]["historical_artifacts_verified"] is False
    assert summary["integrity"]["b_step_audit"]["l3_prompt_steps_checked"] == 1
    assert summary["conflict_evidence_ids"] == [
        "l1-evidence-1", "l1-evidence-2", "l3-evidence-1", "l3-evidence-2",
    ]
    assert len((output / "paired_episode_deltas.csv").read_text().splitlines()) == 3
    assert "Historical baseline not reproduced" in (output / "final_report.md").read_text()
    with pytest.raises(FileExistsError):
        _build(run_dirs)


def test_report_rejects_duplicate_episode_identity(run_dirs):
    baseline, _fused, _launch, output = run_dirs
    with (baseline / "results.jsonl").open("a") as stream:
        stream.write(json.dumps(_row(KEYS[0], report.A_GROUP, False)) + "\n")
    with pytest.raises(ValueError, match="duplicate Episode identity"):
        _build(run_dirs)
    assert not output.exists()


def test_report_rejects_modified_l3_line_even_when_episode_flag_is_true(run_dirs):
    _baseline, fused, _launch, output = run_dirs
    steps = json.loads((fused / "steps.jsonl").read_text().strip())
    steps["prompt_text"] = steps["prompt_text"].replace("chair: seen", "table: seen")
    _write_jsonl(fused / "steps.jsonl", [steps])
    with pytest.raises(ValueError, match="original L3 line"):
        _build(run_dirs)
    assert not output.exists()


def test_paired_l3_step_audit_uses_episode_identity_when_run_order_differs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_jsonl(a / "results.jsonl", [
        {"scene_basename": "scene-a", "episode_id": "1"},
        {"scene_basename": "scene-b", "episode_id": "2"},
    ])
    _write_jsonl(b / "results.jsonl", [
        {"scene_basename": "scene-b", "episode_id": "2"},
        {"scene_basename": "scene-a", "episode_id": "1"},
    ])
    def step(text):
        return {"step": 0, "action": 1, "prompt_text": "official prompt",
                "l3_context_text": text}
    _write_jsonl(a / "steps.jsonl", [step("map-a"), step("map-b")])
    _write_jsonl(b / "steps.jsonl", [step("map-b"), step("map-a")])
    audit = report._paired_l3_step_audit(a, b)
    assert audit["status"] == "verified_on_common_action_prefix"
    assert audit["comparable_nonempty_l3_steps"] == 2
    assert audit["all_comparable_l3_text_equal"] is True
    _write_jsonl(b / "steps.jsonl", [step("map-b"), step("changed")])
    with pytest.raises(ValueError, match="original L3 text differs"):
        report._paired_l3_step_audit(a, b)


def test_report_rejects_changed_uncommitted_b_source_after_launch(run_dirs):
    _baseline, _fused, launch, output = run_dirs
    source = Path(launch["dev_worktree"]) / "l3_integration/sbadir_fusion.py"
    source.write_text(source.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="B source files changed after launch"):
        _build(run_dirs)
    assert not output.exists()

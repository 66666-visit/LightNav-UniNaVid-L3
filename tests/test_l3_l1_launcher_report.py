"""Three-group report contracts: identity, no-L1 B, and unchanged L3 content."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_l3_l1_report as report

KEYS = (("scene_a.glb", "1"), ("scene_b.glb", "2"))
L3_TEXT = "[Navigation Map Memory - L3]\nchair: seen at step 2."
L3_ENRICHED = (
    "[Navigation Map Memory - L3]\n"
    "chair: seen at step 2. [L3 RGB-D map; source yolo11n-seg:seg.pt; "
    "map node obj-1; geometry observed; step 2; seg conf 0.87; evidence seg-1]"
)


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
        "steps_taken": 5, "env_steps": 5,
        "inference_latency_ms_mean": 30.0, "prompt_tokens_mean": 200.0,
        "official_prompt_sha256": report.PROMPT_SHA,
        "pilot_manifest_file_sha256": report.EXPECTED_PILOT_MANIFEST_SHA256,
        "episode_manifest_canonical_sha256": report.EXPECTED_EPISODE_MANIFEST_SHA256,
        "episode_manifest_verified": True,
        "l3_fact_count_total": 2, "l3_context_tokens_mean": 10.0,
        "prompt_has_l3_steps": 1, "l3_map_steps": 2,
        "l3_segmentation_error_count": 0,
    }
    if group in (report.B_GROUP, report.C_GROUP):
        row.update({
            "source_background_enabled": True,
            "source_background_facts_total": 2,
            "l3_provenance_count_total": 2,
            "l3_original_memory_unchanged": True,
            "prompt_has_l1_steps": 0,
            "l1_fact_count_total": 0,
            "l1_retained_facts_total": 0,
            "l1_enabled": False,
            "l1_enabled_steps": 0,
            "l1_observation_attempt_steps": 0,
            "l1_context_tokens_mean": 0.0,
            "l1_observe_error_count": 0,
            "l1_selection_error_count": 0,
            "l1_detector_error_count": 0,
            "l1_place_attempt_steps": 0,
            "l1_place_observed_steps": 0,
            "l1_place_error_count": 0,
            "duplicated_facts_total": 0,
            "conflicting_facts_total": 0,
            "retained_facts_total": 2,
            "omitted_facts_total": 0,
            "cross_label_entity_links_total": 0,
            "cross_label_entity_link_error_count": 0,
            "conflict_type_counts": {"object_label": 0},
            "conflict_evidence_ids": [],
        })
    if group == report.C_GROUP:
        row.update({
            "source_background_facts_total": 2,
            "l1_fact_count_total": 1,
            "l1_retained_facts_total": 1,
            "l1_omitted_facts_total": 0,
            "l1_prompt_built_steps": 1,
            "l1_empty_steps": 4,
            "l1_enabled": True,
            "l1_enabled_steps": 5,
            "l1_observation_attempt_steps": 5,
            "l1_context_tokens_mean": 5.0,
            "prompt_has_l1_steps": 1,
            "l1_events_persisted_total": 5,
            "l1_detection_events_persisted_total": 2,
            "l1_place_attempt_steps": 1,
            "l1_place_observed_steps": 1,
            "l1_token_budget_steps": 5,
            "l1_token_budget_fallback_steps": 0,
            "token_overflow_drops_total": 0,
            "memory_budget_violation_steps": 0,
            "text_prompt_tokens_mean": 90.0,
            "visual_tokens_mean": 382.0,
            "multimodal_prompt_tokens_mean": 477.0,
            "context_margin_tokens_min": 1000,
            "multimodal_budget_violation_steps": 0,
            "retained_facts_total": 3,
            "conflicting_facts_total": 0,
            "conflict_type_counts": {"object_label": 0},
            "conflict_evidence_ids": [],
            "cross_label_entity_links_total": 1,
        })
    return row


def _step(action, *, enriched=None):
    memory = enriched if enriched is not None else L3_TEXT
    row = {
        "step": 0, "action": action,
        "prompt_text": "instruction\n" + memory,
        "l3_context_text": L3_TEXT,
        "prompt_has_l3": True,
        "l3_original_sha256": hashlib.sha256(L3_TEXT.encode()).hexdigest(),
        "frame_valid": True, "segmentation_count": 1,
        "segmentation_error": None, "evidence_minted": 1,
        "observed_objects": 1, "object_nodes": 1, "viewpoints": 1,
        "observed_from_edges": 1, "traversed_to_edges": 0,
        "context_available": True, "context_injected": True,
        "candidate_count": 1, "verified_count": 1, "unverified_count": 0,
        "current_only_skipped": 0, "omitted_items": 0, "fact_count": 1,
    }
    if enriched is not None:
        row["l3_enriched_sha256"] = hashlib.sha256(enriched.encode()).hexdigest()
    return row


def _smoke(directory, group, row):
    directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(directory / "results.jsonl", [row])
    _write_json(directory / "smoke_validation.json", {
        "group": group,
        "group_name": row.get("group_name"),
        "episode": [row["scene_basename"], row["episode_id"]],
        "episode_seed": row["episode_seed"],
        "instruction_sha256": row["instruction_sha256"],
        "official_prompt_sha256": row["official_prompt_sha256"],
    })


@pytest.fixture
def run_dirs(tmp_path):
    baseline = tmp_path / "a"
    source = tmp_path / "b"
    fused = tmp_path / "c"
    output = tmp_path / "report"
    for path in (baseline, source, fused):
        path.mkdir()
    _write_json(baseline / "run_config.json", _config(report.A_GROUP))
    _write_jsonl(baseline / "results.jsonl", [
        _row(KEYS[0], report.A_GROUP, False),
        _row(KEYS[1], report.A_GROUP, True),
    ])
    _write_jsonl(source / "results.jsonl", [
        _row(KEYS[0], report.B_GROUP, True),
        _row(KEYS[1], report.B_GROUP, True),
    ])
    _write_jsonl(fused / "results.jsonl", [
        _row(KEYS[0], report.C_GROUP, True),
        _row(KEYS[1], report.C_GROUP, True),
    ])
    for path in (baseline, source, fused):
        (path / "execution_manifest.json").write_text("{}\n")

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
    for root in (source_a, source_b):
        for name in l3_sources:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original L3 source\n", encoding="utf-8")
        (root / "agent_uninavid.py").write_bytes(
            baseline_agent if root == source_a else (dev_root / "agent_uninavid.py").read_bytes()
        )
        for name in (config_rel, task_rel):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("same Habitat config\n", encoding="utf-8")
    runtime_sources = [
        "run.py", "l3_integration/map_sidecar.py",
        "l3_integration/sbadir_fusion.py", "l3_integration/experiment_groups.py",
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
    background = {
        "enabled": True, "fusion": "evidence_source_pairing",
        "l3_organized_facts_preserved": True,
        "source_tree_sha256": source_digest,
    }
    b_config = _config(report.B_GROUP)
    b_config.update({"group_name": report.B_NAME, "source_background": background,
                     "l1": None})
    _write_json(source / "run_config.json", b_config)
    c_config = _config(report.C_GROUP)
    c_config.update({
        "group_name": report.C_NAME,
        "source_background": dict(background),
        "l1": {
            "det_model": "/models/detect.pt", "det_device": "cuda:0",
            "det_confidence": 0.25, "every_n_steps": 5,
            "place_model": "/models/dinov2-small", "place_device": "cuda:0",
            "place_every_n_steps": 5, "max_items": 2,
            "prompt_max_chars": 320,
            "budget": {
                "strategy": "token_aware_context", "token_aware": True,
                "original_char_budget_chars": 320,
                "generation_reserve": "max_new_tokens",
                "safety_margin_tokens": 1, "whole_lines_only": True,
                "visual_token_accounting": (
                    "online_merge_upper_bound_from_actual_cache"
                ),
                "max_landmarks": 2,
                "fallback": "original_char_budget_with_reported_reason",
            },
            "builder": "local_graph_fusion",
            "fusion": "original_l1_plus_l3_sbadir",
            "source_tree_sha256": source_digest,
        },
    })
    _write_json(fused / "run_config.json", c_config)

    for path in (baseline, source, fused):
        _write_jsonl(path / "steps.jsonl", [_step(1), _step(1)])
    for path in (source, fused):
        _write_jsonl(path / "steps.jsonl", [
            _step(1, enriched=L3_ENRICHED), _step(1, enriched=L3_ENRICHED),
        ])

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
        "l1_det_model": "/models/detect.pt",
        "l1_place_model": "/models/dinov2-small",
        "b_source_files_sha256": source_hashes,
        "b_source_tree_sha256": source_digest,
        "groups": [report.A_GROUP, report.B_GROUP, report.C_GROUP],
        "smoke_required": True,
        "smoke_directories": {
            report.A_GROUP: "smoke_a",
            report.B_GROUP: "smoke_b",
            report.C_GROUP: "smoke_c",
        },
    }
    _smoke(output / "smoke_a", report.A_GROUP,
           _row(KEYS[0], report.A_GROUP, False))
    _smoke(output / "smoke_b", report.B_GROUP,
           _row(KEYS[0], report.B_GROUP, True))
    _smoke(output / "smoke_c", report.C_GROUP,
           _row(KEYS[0], report.C_GROUP, True))
    return baseline, source, fused, launch, output


def _build(run_dirs):
    baseline, source, fused, launch, output = run_dirs
    return report.build_report(
        baseline, source, fused, output, KEYS,
        {"file_sha256": report.EXPECTED_PILOT_MANIFEST_SHA256},
        {"canonical_sha256": report.EXPECTED_EPISODE_MANIFEST_SHA256},
        launch, expected_count=2,
    )


@pytest.fixture(autouse=True)
def _manifest(monkeypatch):
    monkeypatch.setattr(
        report, "load_episode_manifest",
        lambda path: SimpleNamespace(keys=lambda: KEYS),
    )


def test_report_pairs_three_groups_by_episode_identity(run_dirs):
    summary = _build(run_dirs)
    output = run_dirs[4]
    assert summary["n_pairs"] == 2
    assert summary["groups"][report.A_NAME]["successes"] == 1
    assert summary["groups"][report.B_NAME]["successes"] == 2
    assert summary["groups"][report.C_NAME]["successes"] == 2
    assert set(summary["comparisons"]) == {"a_vs_b", "b_vs_c", "a_vs_c"}
    assert summary["paired"]["a_vs_b"]["success_flips"]["baseline_fail_l3_success"] == 1
    assert summary["paired"]["b_vs_c"]["n_pairs"] == 2
    assert summary["integrity"]["b_has_no_l1"] is True
    assert summary["integrity"]["b_source_background_present"] is True
    assert summary["integrity"]["c_adds_l1_to_b"] is True
    assert summary["integrity"]["c_keeps_b_source_background"] is True
    assert summary["integrity"]["b_c_l3_background_equal_on_common_prefix"] is True
    # C no longer runs cross-layer L1/L3 conflict fusion: the original L1
    # card is appended verbatim and SBAdir source pairing is L3-only.
    assert summary["conflict_evidence_ids"] == []
    assert summary["groups"][report.C_NAME]["conflicting_facts_total"] == 0
    assert len((output / "paired_episode_deltas.csv").read_text().splitlines()) == 3
    assert len((output / "results.jsonl").read_text().splitlines()) == 6
    assert len((output / "comparison.csv").read_text().splitlines()) == (
        3 * len(report.METRICS) + 1
    )
    text = (output / "final_report.md").read_text()
    assert "A L3 Organized" in text and "C L3+Source+L1" in text
    assert report.BASELINE_SHA in text
    assert "{}" not in text
    assert "B has no L1" in text
    assert "Historical baseline not reproduced" in text
    with pytest.raises(FileExistsError):
        _build(run_dirs)


def test_report_rejects_duplicate_episode_identity(run_dirs):
    baseline, _source, _fused, _launch, output = run_dirs
    with (baseline / "results.jsonl").open("a") as stream:
        stream.write(json.dumps(_row(KEYS[0], report.A_GROUP, False)) + "\n")
    with pytest.raises(ValueError, match="duplicate Episode identity"):
        _build(run_dirs)
    assert not (output / "summary.json").exists()


def test_report_rejects_modified_l3_line_even_when_episode_flag_is_true(run_dirs):
    _baseline, source, _fused, _launch, output = run_dirs
    steps = [json.loads(line) for line in (source / "steps.jsonl").read_text().splitlines()]
    steps[0]["prompt_text"] = steps[0]["prompt_text"].replace("chair: seen", "table: seen")
    _write_jsonl(source / "steps.jsonl", steps)
    with pytest.raises(ValueError, match="original L3 line"):
        _build(run_dirs)
    assert not (output / "summary.json").exists()


def test_report_rejects_b_that_injected_an_l1_fact(run_dirs):
    _baseline, source, _fused, _launch, output = run_dirs
    rows = [json.loads(line) for line in (source / "results.jsonl").read_text().splitlines()]
    rows[0]["l1_fact_count_total"] = 1
    _write_jsonl(source / "results.jsonl", rows)
    with pytest.raises(ValueError, match="B injected an L1 fact"):
        _build(run_dirs)
    assert not (output / "summary.json").exists()


def test_report_rejects_b_run_config_with_an_l1_block(run_dirs):
    _baseline, source, _fused, _launch, output = run_dirs
    config = json.loads((source / "run_config.json").read_text())
    config["l1"] = {"det_model": "/models/detect.pt"}
    _write_json(source / "run_config.json", config)
    with pytest.raises(ValueError, match="B run_config contains an L1 block"):
        _build(run_dirs)
    assert not (output / "summary.json").exists()


def test_report_rejects_b_run_config_recording_an_l1_weight_path(run_dirs):
    _baseline, source, _fused, _launch, output = run_dirs
    config = json.loads((source / "run_config.json").read_text())
    config["l1_det_model"] = "/models/detect.pt"
    _write_json(source / "run_config.json", config)
    with pytest.raises(ValueError, match="records L1 weight path"):
        _build(run_dirs)
    assert not (output / "summary.json").exists()


def test_report_rejects_a_result_with_source_background_fields(run_dirs):
    baseline, _source, _fused, _launch, output = run_dirs
    rows = [json.loads(line) for line in (baseline / "results.jsonl").read_text().splitlines()]
    rows[0]["source_background_facts_total"] = 2
    _write_jsonl(baseline / "results.jsonl", rows)
    with pytest.raises(ValueError, match="not the 7602b25 runtime"):
        _build(run_dirs)
    assert not (output / "summary.json").exists()


def test_report_rejects_changed_l3_map_state_before_action_divergence(run_dirs):
    _baseline, source, _fused, _launch, output = run_dirs
    steps = [json.loads(line) for line in (source / "steps.jsonl").read_text().splitlines()]
    steps[0]["object_nodes"] = 99
    _write_jsonl(source / "steps.jsonl", steps)
    with pytest.raises(ValueError, match="L3 map/fact-selection state differs"):
        _build(run_dirs)
    assert not (output / "summary.json").exists()


def test_report_rejects_changed_b_l3_source_background(run_dirs):
    _baseline, _source, fused, _launch, output = run_dirs
    changed = L3_ENRICHED.replace("seg conf 0.87", "seg conf 0.11")
    _write_jsonl(fused / "steps.jsonl", [
        _step(1, enriched=changed), _step(1, enriched=L3_ENRICHED),
    ])
    with pytest.raises(ValueError, match="L3 source background differs"):
        _build(run_dirs)
    assert not (output / "summary.json").exists()


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
    def step(text, sha):
        return {"step": 0, "action": 1, "prompt_text": "official prompt",
                "l3_context_text": text, "l3_enriched_sha256": sha,
                "frame_valid": True, "context_injected": True,
                "object_nodes": 1, "viewpoints": 1, "fact_count": 1}
    _write_jsonl(a / "steps.jsonl", [step("map-a", "sha-a"), step("map-b", "sha-b")])
    _write_jsonl(b / "steps.jsonl", [step("map-b", "sha-b"), step("map-a", "sha-a")])
    audit = report._paired_l3_step_audit(a, b, right_name="B")
    assert audit["status"] == "verified_on_common_action_prefix"
    assert audit["comparable_nonempty_l3_steps"] == 2
    assert audit["all_comparable_l3_text_equal"] is True
    assert audit["all_comparable_l3_enriched_equal"] is True
    _write_jsonl(b / "steps.jsonl", [step("map-b", "sha-b"), step("changed", "sha-a")])
    with pytest.raises(ValueError, match="original L3 text differs"):
        report._paired_l3_step_audit(a, b, right_name="B")
    _write_jsonl(b / "steps.jsonl", [step("map-b", "sha-b"), step("map-a", "changed")])
    with pytest.raises(ValueError, match="L3 source background differs"):
        report._paired_l3_step_audit(a, b, right_name="B")


def test_report_rejects_changed_uncommitted_b_source_after_launch(run_dirs):
    _baseline, _source, _fused, launch, output = run_dirs
    path = Path(launch["dev_worktree"]) / "l3_integration/sbadir_fusion.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="B/C source files changed after launch"):
        _build(run_dirs)
    assert not (output / "summary.json").exists()


def test_report_requires_three_real_smokes(run_dirs):
    _baseline, _source, _fused, _launch, output = run_dirs
    (output / "smoke_b" / "smoke_validation.json").unlink()
    with pytest.raises(FileNotFoundError):
        _build(run_dirs)
    assert not (output / "summary.json").exists()


def test_l3_state_ignores_l1_dependent_fields_only():
    base = {
        "frame_valid": True, "segmentation_count": 1,
        "segmentation_error": None, "evidence_minted": 1,
        "observed_objects": 1, "object_nodes": 1, "viewpoints": 1,
        "observed_from_edges": 1, "traversed_to_edges": 0,
        "context_available": True, "candidate_count": 1,
        "verified_count": 1, "unverified_count": 0,
        "current_only_skipped": 0, "omitted_items": 0, "fact_count": 1,
        "evidence_unmatched": 1, "context_injected": True,
    }
    other = dict(base, evidence_unmatched=0, context_injected=False)
    assert report._l3_state(base) == report._l3_state(other)
    assert report._l3_state(base)["fact_count"] == 1
    changed = dict(base, object_nodes=2)
    assert report._l3_state(base) != report._l3_state(changed)


def test_report_accepts_a_c_episode_with_no_l1_fact(run_dirs):
    # A single C Episode may legitimately have no original L1 card.  The
    # report must not turn that into a run-wide failure; the other Episode
    # still proves that C adds L1 somewhere.
    _baseline, _source, fused, _launch, _output = run_dirs
    rows = [json.loads(line) for line in
            (fused / "results.jsonl").read_text().splitlines()]
    for row in rows:
        if row.get("group") == report.C_GROUP:
            row.update({
                "l1_fact_count_total": 0,
                "l1_retained_facts_total": 0,
                "l1_omitted_facts_total": 0,
                "l1_prompt_built_steps": 0,
                "l1_empty_steps": row.get("env_steps", 5),
                "prompt_has_l1_steps": 0,
                "l1_context_tokens_mean": 0.0,
                "retained_facts_total": row.get("l3_fact_count_total", 0),
            })
            break
    _write_jsonl(fused / "results.jsonl", rows)
    summary = _build(run_dirs)
    assert summary["integrity"]["c_adds_l1_to_b"] is True


def test_step_audit_accepts_token_aware_l1_above_320_but_not_fallback(tmp_path):
    l1 = "[Navigation Memory]\n" + ("Current RGB: instruction landmark chair. " * 12)
    assert len(l1) > 320
    row = {
        "step": 0,
        "prompt_has_l1": True,
        "l1_context_text": l1,
        "prompt_text": "instruction\n" + l1,
        "l1_prompt_chars": len(l1),
        "l1_prompt_sha256": hashlib.sha256(l1.encode("utf-8")).hexdigest(),
        "l1_budget_chars": 320,
        "budget_source": "token_aware_context",
        "available_memory_tokens": 500,
        "memory_tokens_injected": 250,
    }
    path = tmp_path / "steps.jsonl"
    _write_jsonl(path, [row])
    audit = report._step_audit(path)
    assert audit["l1_prompt_steps_checked"] == 1

    row["budget_source"] = "char_fallback_320"
    _write_jsonl(path, [row])
    with pytest.raises(ValueError, match="fallback card exceeds"):
        report._step_audit(path)


def test_report_rejects_c_without_the_token_budget_protocol(run_dirs):
    _baseline, _source, fused, _launch, output = run_dirs
    config = json.loads((fused / "run_config.json").read_text())
    config["l1"].pop("budget")
    _write_json(fused / "run_config.json", config)
    with pytest.raises(ValueError, match="lacks the token-budget protocol"):
        _build(run_dirs)
    assert not (output / "summary.json").exists()


def test_step_audit_rejects_multimodal_prompt_over_context_reserve(tmp_path):
    l1 = "[Navigation Memory]\nCurrent RGB: instruction landmark chair."
    row = {
        "step": 0,
        "prompt_has_l1": True,
        "l1_context_text": l1,
        "prompt_text": "instruction\n" + l1,
        "l1_prompt_chars": len(l1),
        "l1_prompt_sha256": hashlib.sha256(l1.encode("utf-8")).hexdigest(),
        "l1_budget_chars": 320,
        "budget_source": "token_aware_context",
        "available_memory_tokens": 500,
        "memory_tokens_injected": 10,
        "text_prompt_tokens": 100,
        "visual_tokens": 400,
        "base_prompt_tokens": 500,
        "multimodal_prompt_limit_tokens": 505,
        "multimodal_prompt_tokens": 510,
        "context_margin_tokens": -5,
        "multimodal_budget_overflow": True,
    }
    path = tmp_path / "steps.jsonl"
    _write_jsonl(path, [row])
    with pytest.raises(ValueError, match="multimodal prompt exceeds|overflow flag"):
        report._step_audit(path)

"""The paired report consumes only real result rows and refuses incompleteness."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_l3_report  # noqa: E402
from l3_integration.metrics import (  # noqa: E402
    mcnemar_exact,
    paired_bootstrap_ci,
)


def _rows(*, l3: bool) -> list[dict]:
    rows = []
    for scene in range(11):
        for index in range(8):
            episode = f"ep-{scene}-{index}"
            success = (index + scene) % 2 == 0
            rows.append({
                "scene_basename": f"Scene{scene}.glb",
                "episode_id": episode,
                "instruction_sha256": f"sha-{scene}-{index}",
                "success": success,
                "spl": 0.7 if success else 0.1,
                "ndtw": 0.8 if success else 0.2,
                "oracle_success": True if index != 0 else False,
                "distance_to_goal": 2.0 if success else 5.0,
                "path_length": 7.0 + index,
                "inference_latency_ms_mean": 100.0 + index,
                "prompt_tokens_mean": 200.0 + index,
                "l3_build_ms_mean": 12.0 if l3 else 0.0,
                "l3_fact_count_total": 20 if l3 else 0,
                "l3_context_steps": 30 if l3 else 0,
                "l3_map_steps": 40 if l3 else 0,
                "prompt_has_l3_steps": 30 if l3 else 0,
                "official_prompt_sha256": "prompt-sha",
                "pilot_manifest_sha256": "manifest-sha",
                "episode_manifest_canonical_sha256": "episode-canonical-sha",
            })
    return rows


def _write_group(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )


def test_report_writes_all_required_artifacts(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    l3 = tmp_path / "l3"
    output = tmp_path / "report"
    baseline_rows = _rows(l3=False)
    l3_rows = _rows(l3=True)
    for a_row, b_row in zip(baseline_rows, l3_rows):
        if a_row["episode_id"].endswith("-0"):
            b_row["success"] = True
            b_row["spl"] = 0.9
            b_row["distance_to_goal"] = 1.0
    _write_group(baseline, baseline_rows)
    _write_group(l3, l3_rows)

    assert build_l3_report.main([
        "--baseline", str(baseline), "--l3", str(l3), "--output", str(output),
    ]) == 0

    for name in (
        "summary.json", "results.jsonl", "comparison.csv",
        "paired_episode_deltas.csv", "final_report.md",
    ):
        assert (output / name).is_file(), name
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_pairs"] == 88
    assert summary["integrity"]["same_episode_ids"] is True
    assert summary["groups"]["uninavid_baseline"]["SR"] is not None
    assert summary["groups"]["uninavid_baseline"]["pilot_manifest_sha256"] == [
        "manifest-sha"
    ]
    assert summary["episode_manifest_canonical_sha256"] == [
        "episode-canonical-sha"
    ]
    assert summary["groups"]["uninavid_l3_only"]["l3_context_steps"] > 0
    assert summary["paired"]["success_flips"]["baseline_fail_l3_success"] == 5
    with (output / "paired_episode_deltas.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 88
    report = (output / "final_report.md").read_text()
    assert "Uni-NaVid L3-only paired experiment report" in report
    assert "Exact McNemar" in report
    assert "A fail → B success: 5" in report


def test_report_rejects_mismatched_episode_sets(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    l3 = tmp_path / "l3"
    _write_group(baseline, _rows(l3=False))
    l3_rows = _rows(l3=True)[:-1]
    _write_group(l3, l3_rows)
    with pytest.raises(ValueError, match="Episode keys differ"):
        build_l3_report.main([
            "--baseline", str(baseline), "--l3", str(l3),
            "--output", str(tmp_path / "out"),
        ])


def test_mcnemar_and_bootstrap_are_deterministic() -> None:
    first = mcnemar_exact(1, 9)
    second = mcnemar_exact(1, 9)
    assert first == second
    assert first["discordant"] == 10
    assert 0.0 < first["p_value"] <= 1.0
    assert mcnemar_exact(0, 0)["p_value"] == 1.0
    ci_one = paired_bootstrap_ci([1.0, -1.0, 0.5, 0.25], samples=500)
    ci_two = paired_bootstrap_ci([1.0, -1.0, 0.5, 0.25], samples=500)
    assert ci_one == ci_two
    assert ci_one["low"] <= ci_one["mean"] <= ci_one["high"]


def test_report_rejects_a_baseline_with_l3_or_a_no_injection_l3_group(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    l3 = tmp_path / "l3"
    baseline_rows = _rows(l3=False)
    l3_rows = _rows(l3=True)
    baseline_rows[0]["prompt_has_l3_steps"] = 1
    _write_group(baseline, baseline_rows)
    _write_group(l3, l3_rows)
    with pytest.raises(ValueError, match="baseline group contains L3"):
        build_l3_report.main([
            "--baseline", str(baseline), "--l3", str(l3),
            "--output", str(tmp_path / "out1"),
        ])

    baseline_rows[0]["prompt_has_l3_steps"] = 0
    l3_rows = _rows(l3=True)
    for row in l3_rows:
        row["prompt_has_l3_steps"] = 0
        row["l3_context_steps"] = 0
    _write_group(baseline, baseline_rows)
    _write_group(l3, l3_rows)
    with pytest.raises(ValueError, match="never injected L3"):
        build_l3_report.main([
            "--baseline", str(baseline), "--l3", str(l3),
            "--output", str(tmp_path / "out2"),
        ])

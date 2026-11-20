#!/usr/bin/env python3
"""Build the paired A/B report from the two groups' results.jsonl files.

No metric is invented: every number is read from the per-Episode rows written
by run.py.  Pairing is mandatory; different or incomplete Episode sets fail.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from l3_integration.metrics import (  # noqa: E402
    finite_float,
    mean_or_none,
    paired_summary,
)
from l3_integration.pilot_manifest import (  # noqa: E402
    EXPECTED_PILOT_MANIFEST_SHA256,
    load_pilot_manifest,
)

METRICS = (
    ("SR", "success", 100.0, "higher"),
    ("SPL", "spl", 100.0, "higher"),
    ("NE", "distance_to_goal", 1.0, "lower"),
    ("nDTW", "ndtw", 100.0, "higher"),
    ("OSR", "oracle_success", 100.0, "higher"),
    ("TL", "path_length", 1.0, "lower"),
)
RESOURCE_METRICS = (
    ("inference_latency_ms_mean", 1.0, "lower"),
    ("prompt_tokens_mean", 1.0, "lower"),
    ("l3_build_ms_mean", 1.0, "lower"),
    ("l3_fact_count_total", 1.0, "neutral"),
)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        raise FileNotFoundError(f"results.jsonl not found: {path}")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates = 0
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row {number} in {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row {number} in {path} is not an object")
            key = (str(row.get("scene_basename") or ""), str(row.get("episode_id") or ""))
            if not key[0] or not key[1]:
                raise ValueError(f"row {number} in {path} lacks scene/episode identity")
            if key in rows:
                duplicates += 1
            rows[key] = row
    return list(rows.values()), duplicates


def _row_value(row: dict[str, Any], key: str) -> float | None:
    if key in {"success", "oracle_success"}:
        return 1.0 if bool(row.get(key)) else 0.0 if key in row else None
    return finite_float(row.get(key))


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {"n_episodes": len(rows)}
    for name, key, scale, _direction in METRICS:
        values[name] = mean_or_none(
            value * scale for row in rows
            if (value := _row_value(row, key)) is not None
        )
    for key, _scale, _direction in RESOURCE_METRICS:
        values[key] = mean_or_none(
            finite_float(row.get(key))
            for row in rows
            if finite_float(row.get(key)) is not None
        )
    values["l3_context_steps"] = sum(
        int(row.get("l3_context_steps") or 0) for row in rows
    )
    values["l3_map_steps"] = sum(int(row.get("l3_map_steps") or 0) for row in rows)
    values["prompt_has_l3_steps"] = sum(
        int(row.get("prompt_has_l3_steps") or 0) for row in rows
    )
    values["official_prompt_sha256"] = sorted({
        str(row.get("official_prompt_sha256")) for row in rows
        if row.get("official_prompt_sha256")
    })
    # Prefer the explicit Pilot file digest; fall back to the old ambiguous
    # field only so pre-fix result files remain inspectable.
    values["pilot_manifest_sha256"] = sorted({
        str(row.get("pilot_manifest_sha256") or row.get("manifest_sha256"))
        for row in rows
        if row.get("pilot_manifest_sha256") or row.get("manifest_sha256")
    })
    values["episode_manifest_canonical_sha256"] = sorted({
        str(row.get("episode_manifest_canonical_sha256")) for row in rows
        if row.get("episode_manifest_canonical_sha256")
    })
    return values


def _delta_row(name: str, key: str, scale: float, direction: str,
               baseline_rows: list[dict[str, Any]],
               l3_rows: list[dict[str, Any]]) -> dict[str, Any]:
    a = [
        value * scale for row in baseline_rows
        if (value := _row_value(row, key)) is not None
    ]
    b = [
        value * scale for row in l3_rows
        if (value := _row_value(row, key)) is not None
    ]
    a_mean = mean_or_none(a)
    b_mean = mean_or_none(b)
    delta = (b_mean - a_mean) if a_mean is not None and b_mean is not None else None
    pct = (delta / a_mean * 100.0) if delta is not None and a_mean not in (None, 0.0) else None
    return {
        "metric": name,
        "baseline": a_mean,
        "l3": b_mean,
        "delta": delta,
        "delta_pct": pct,
        "preferred_direction": direction,
        "n_baseline": len(a),
        "n_l3": len(b),
    }


def _write_results_merged(path: Path, baseline_rows, l3_rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for group, rows in (("uninavid_baseline", baseline_rows),
                            ("uninavid_l3_only", l3_rows)):
            for row in rows:
                merged = dict(row)
                merged["group"] = group
                stream.write(json.dumps(merged, ensure_ascii=False) + "\n")


def _write_comparison(path: Path, comparison: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)


def _write_paired(path: Path, baseline_rows, l3_rows) -> None:
    fields = [
        "scene_basename", "episode_id", "instruction_sha256",
        "success_a", "success_b", "delta_success",
        "spl_a", "spl_b", "delta_spl",
        "ndtw_a", "ndtw_b", "delta_ndtw",
        "ne_a", "ne_b", "delta_ne",
        "osr_a", "osr_b", "delta_osr",
        "tl_a", "tl_b", "delta_tl",
        "inference_latency_ms_a", "inference_latency_ms_b",
        "prompt_tokens_a", "prompt_tokens_b",
        "l3_context_steps_b", "l3_fact_count_total_b",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for a_row, b_row in zip(baseline_rows, l3_rows):
            record = {
                "scene_basename": a_row["scene_basename"],
                "episode_id": a_row["episode_id"],
                "instruction_sha256": a_row.get("instruction_sha256"),
            }
            for prefix, key in (
                ("success", "success"), ("spl", "spl"), ("ndtw", "ndtw"),
                ("ne", "distance_to_goal"), ("osr", "oracle_success"),
                ("tl", "path_length"),
            ):
                a_value = finite_float(a_row.get(key))
                b_value = finite_float(b_row.get(key))
                record[f"{prefix}_a"] = a_value
                record[f"{prefix}_b"] = b_value
                record[f"delta_{prefix}"] = (
                    b_value - a_value if a_value is not None and b_value is not None else None
                )
            record["inference_latency_ms_a"] = finite_float(
                a_row.get("inference_latency_ms_mean")
            )
            record["inference_latency_ms_b"] = finite_float(
                b_row.get("inference_latency_ms_mean")
            )
            record["prompt_tokens_a"] = finite_float(a_row.get("prompt_tokens_mean"))
            record["prompt_tokens_b"] = finite_float(b_row.get("prompt_tokens_mean"))
            record["l3_context_steps_b"] = int(b_row.get("l3_context_steps") or 0)
            record["l3_fact_count_total_b"] = int(
                b_row.get("l3_fact_count_total") or 0
            )
            writer.writerow(record)


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_report(
    path: Path, *, baseline_metrics: dict[str, Any], l3_metrics: dict[str, Any],
    comparison: list[dict[str, Any]], paired: dict[str, Any],
    baseline_count: int, l3_count: int, manifest_sha: str | None,
    source_sha: dict[str, str | None], duplicates: tuple[int, int],
) -> None:
    lines = [
        "# Uni-NaVid L3-only paired experiment report",
        "",
        "Generated from the two groups' `results.jsonl` files. This is an",
        "**88-Episode paired pilot**, not a full val_unseen evaluation.",
        "",
        "## Experiment identity",
        "",
        f"- Baseline (A): `uninavid_baseline`, {baseline_count} Episodes",
        f"- L3-only (B): `uninavid_l3_only`, {l3_count} Episodes",
        f"- Pilot manifest file SHA256: `{manifest_sha or 'not recorded'}`",
        "- This is the raw `pilot_manifest.json` file digest, not the",
        "  completion-marker `episode_manifest.json` canonical digest.",
        f"- Results source SHA256 A/B: `{source_sha['baseline']}` / `{source_sha['l3']}`",
        "",
        "## Primary metrics",
        "",
        "| Metric | A baseline | B L3-only | Delta (B-A) | Preferred |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in comparison:
        lines.append(
            f"| {row['metric']} | {_format(row['baseline'])} | "
            f"{_format(row['l3'])} | {_format(row['delta'])} | "
            f"{row['preferred_direction']} |"
        )
    lines.extend([
        "",
        "SR/SPL/nDTW/OSR are percentages; NE and TL are metres.",
        "",
        "## Paired success flips",
        "",
        f"- A fail → B success: {paired['success_flips']['baseline_fail_l3_success']}",
        f"- A success → B fail: {paired['success_flips']['baseline_success_l3_fail']}",
        f"- Exact McNemar p-value: "
        f"{_format(paired['success_flips']['mcnemar']['p_value'], 6)}",
        "",
        "## Manipulation / integrity checks",
        "",
        f"- Baseline L3 map steps: {baseline_metrics['l3_map_steps']}",
        f"- Baseline prompt-with-L3 steps: {baseline_metrics['prompt_has_l3_steps']}",
        f"- L3 map steps: {l3_metrics['l3_map_steps']}",
        f"- L3 context-injected steps: {l3_metrics['l3_context_steps']}",
        f"- L3 prompt-with-L3 steps: {l3_metrics['prompt_has_l3_steps']}",
        f"- Duplicate rows ignored A/B: {duplicates[0]} / {duplicates[1]}",
        f"- Official prompt SHA A: {baseline_metrics['official_prompt_sha256']}",
        f"- Official prompt SHA B: {l3_metrics['official_prompt_sha256']}",
        "",
        "## Interpretation guardrails",
        "",
        "- A and B share the checkpoint, 88-Episode manifest, Habitat scene",
        "  configuration, early-stop protocol, action parser and inference",
        "  sampling temperature. Only the L3 sidecar differs.",
        "- A longer prompt or more injected facts is **not** evidence of a",
        "  better navigation policy. Read SR/SPL/NE/nDTW/OSR/TL and the paired",
        "  deltas together.",
        "- 88 Episodes give limited power. A non-significant McNemar result does",
        "  not prove equivalence, and a significant result on this pilot needs",
        "  replication on a larger split.",
        "- L3 uses RGB-D + pose + YOLO11n-seg. Depth/pose failures, closed-set",
        "  segmentation and uncalibrated geometry thresholds can all reduce",
        "  injection quality; inspect `steps.jsonl` for the per-step reasons.",
        "",
        "## Files",
        "",
        "- `results.jsonl` — merged per-Episode rows",
        "- `summary.json` — group metrics, paired tests, integrity checks",
        "- `comparison.csv` — headline metric table",
        "- `paired_episode_deltas.csv` — per-Episode paired deltas",
        "- `final_report.md` — this file",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--l3", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument(
        "--expected-sha256", default=EXPECTED_PILOT_MANIFEST_SHA256,
    )
    args = parser.parse_args(argv)

    if args.manifest and not args.expected_sha256:
        raise ValueError(
            "--manifest requires a nonempty --expected-sha256; never bypass "
            "the frozen manifest digest"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baseline_rows, baseline_duplicates = _read_jsonl(
        Path(args.baseline) / "results.jsonl"
    )
    l3_rows, l3_duplicates = _read_jsonl(Path(args.l3) / "results.jsonl")
    baseline_rows.sort(key=lambda row: (row["scene_basename"], row["episode_id"]))
    l3_rows.sort(key=lambda row: (row["scene_basename"], row["episode_id"]))
    baseline_keys = {(row["scene_basename"], row["episode_id"]) for row in baseline_rows}
    l3_keys = {(row["scene_basename"], row["episode_id"]) for row in l3_rows}
    if baseline_keys != l3_keys:
        raise ValueError(
            "A/B Episode keys differ: "
            f"missing_in_B={sorted(baseline_keys - l3_keys)[:5]}, "
            f"extra_in_B={sorted(l3_keys - baseline_keys)[:5]}"
        )
    if len(baseline_rows) != 88 or len(l3_rows) != 88:
        raise ValueError(
            f"paired report requires 88 Episodes per group; "
            f"found A={len(baseline_rows)}, B={len(l3_rows)}"
        )
    for a_row, b_row in zip(baseline_rows, l3_rows):
        a_key = (a_row["scene_basename"], a_row["episode_id"])
        b_key = (b_row["scene_basename"], b_row["episode_id"])
        if a_key != b_key:
            raise ValueError(
                f"A/B rows are not key-paired: {a_key} vs {b_key}"
            )
        if a_row.get("instruction_sha256") != b_row.get("instruction_sha256"):
            raise ValueError(
                "A/B instruction differs for "
                f"{a_key[0]}/{a_key[1]}"
            )

    manifest_sha = None
    manifest_report = None
    if args.manifest:
        manifest = load_pilot_manifest(
            args.manifest, expected_sha256=(args.expected_sha256 or None),
        )
        manifest_sha = manifest.file_sha256
        manifest_report = manifest.audit()
        # Both groups must cover exactly the frozen 88-task set; equal A/B
        # sets alone would not catch two identically wrong sets.
        manifest_keys = set(manifest.keys())
        for name, keys in (
            ("baseline", baseline_keys), ("l3", l3_keys),
        ):
            if keys != manifest_keys:
                raise ValueError(
                    f"{name} Episode set differs from the frozen manifest: "
                    f"missing={sorted(manifest_keys - keys)[:5]}, "
                    f"extra={sorted(keys - manifest_keys)[:5]}"
                )
        for row in baseline_rows + l3_rows:
            recorded = (
                row.get("pilot_manifest_sha256") or row.get("manifest_sha256")
            )
            if recorded not in (None, manifest_sha):
                raise ValueError(
                    "result rows record a different Pilot manifest file "
                    "SHA256 (not the episode-manifest canonical digest)"
                )

    comparison = [
        _delta_row(name, key, scale, direction, baseline_rows, l3_rows)
        for name, key, scale, direction in METRICS
    ]
    comparison.extend(
        _delta_row(name, name, scale, direction, baseline_rows, l3_rows)
        for name, scale, direction in RESOURCE_METRICS
    )
    paired = paired_summary(baseline_rows, l3_rows)
    baseline_metrics = _group_metrics(baseline_rows)
    l3_metrics = _group_metrics(l3_rows)
    baseline_prompt_l3 = baseline_metrics["prompt_has_l3_steps"]
    l3_prompt_l3 = l3_metrics["prompt_has_l3_steps"]
    if baseline_metrics["l3_map_steps"] or baseline_prompt_l3:
        raise ValueError(
            "baseline group contains L3 map/prompt steps; the L3-only "
            "comparison is invalid"
        )
    if not l3_prompt_l3 or not l3_metrics["l3_context_steps"]:
        raise ValueError("L3 group never injected L3 into the prompt")
    baseline_template_shas = baseline_metrics["official_prompt_sha256"]
    l3_template_shas = l3_metrics["official_prompt_sha256"]
    if (
        len(baseline_template_shas) != 1
        or len(l3_template_shas) != 1
        or baseline_template_shas != l3_template_shas
    ):
        raise ValueError("A/B report an official prompt template mismatch")


    source_sha = {
        "baseline": hashlib.sha256(
            (Path(args.baseline) / "results.jsonl").read_bytes()
        ).hexdigest(),
        "l3": hashlib.sha256(
            (Path(args.l3) / "results.jsonl").read_bytes()
        ).hexdigest(),
    }
    summary = {
        "protocol_version": "uninavid_l3_paired_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(baseline_rows),
        "manifest": manifest_report,
        "pilot_manifest_sha256": manifest_sha,
        "manifest_digest_kind": (
            "pilot_manifest.json raw byte SHA256; not the "
            "episode_manifest.json completion-marker canonical digest"
        ),
        "episode_manifest_canonical_sha256": sorted({
            row.get("episode_manifest_canonical_sha256")
            for row in baseline_rows + l3_rows
            if row.get("episode_manifest_canonical_sha256")
        }),
        "source_results_sha256": source_sha,
        "groups": {
            "uninavid_baseline": baseline_metrics,
            "uninavid_l3_only": l3_metrics,
        },
        "comparison": comparison,
        "paired": paired,
        "integrity": {
            "same_episode_ids": True,
            "same_instructions": True,
            "baseline_prompt_has_l3_steps": baseline_metrics["prompt_has_l3_steps"],
            "baseline_l3_map_steps": baseline_metrics["l3_map_steps"],
            "l3_prompt_has_l3_steps": l3_metrics["prompt_has_l3_steps"],
            "official_prompt_sha256_a": baseline_metrics["official_prompt_sha256"],
            "official_prompt_sha256_b": l3_metrics["official_prompt_sha256"],
            "duplicate_rows_ignored": {
                "baseline": baseline_duplicates, "l3": l3_duplicates,
            },
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_results_merged(output / "results.jsonl", baseline_rows, l3_rows)
    _write_comparison(output / "comparison.csv", comparison)
    _write_paired(output / "paired_episode_deltas.csv", baseline_rows, l3_rows)
    _write_report(
        output / "final_report.md",
        baseline_metrics=baseline_metrics,
        l3_metrics=l3_metrics,
        comparison=comparison,
        paired=paired,
        baseline_count=len(baseline_rows),
        l3_count=len(l3_rows),
        manifest_sha=manifest_sha,
        source_sha=source_sha,
        duplicates=(baseline_duplicates, l3_duplicates),
    )
    print(f"wrote {output}/summary.json, results.jsonl, comparison.csv, "
          f"paired_episode_deltas.csv, final_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

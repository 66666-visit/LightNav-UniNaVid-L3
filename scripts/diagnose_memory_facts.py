#!/usr/bin/env python3
"""Explain a zero-fact prompt-ablation run from its per-step diagnostics.

Usage on AutoDL (read-only; never reruns a model)::

    python scripts/diagnose_memory_facts.py \
        output/eight_group_smoke_20261121_100659/l1_l3_joint

The script reads ``steps.jsonl`` and prints an aggregate plus explicit
``possible_causes``.  It does not treat a non-empty 3D map as a retrieved fact:
``map_object_nodes`` and ``retrieved_nodes`` are reported separately.

Rows written before the pipeline diagnostics were added still load; the script
then says that the stage counters are unavailable instead of guessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:  # The script lives in this repository; keep a stdlib-only fallback.
    from l3_memory.lexical import _terms
except Exception:  # noqa: BLE001 - the checker still works without it
    _terms = None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _sum(rows: Iterable[dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key) or 0) for row in rows)


def _max(rows: Iterable[dict[str, Any]], key: str) -> int:
    return max((int(row.get(key) or 0) for row in rows), default=0)


def _any_positive(rows: Iterable[dict[str, Any]], key: str) -> bool:
    return any(int(row.get(key) or 0) > 0 for row in rows)


def _run_context(results_path: Path) -> dict[str, Any] | None:
    if not results_path.is_file():
        return None
    rows = _load_rows(results_path)
    if not rows:
        return None
    first = rows[0]
    summary = first.get("memory_summary") or {}
    labels = ((summary.get("map") or {}).get("labels") or {})
    instruction = first.get("instruction")
    instruction_terms = (
        sorted(_terms(instruction)) if _terms is not None and instruction else []
    )
    relevant_labels = []
    if _terms is not None and instruction:
        relevant_labels = sorted(
            label for label in labels
            if _terms(label) & _terms(instruction)
        )
    return {
        "episodes": len(rows),
        "lexical_available": _terms is not None,
        "instruction": instruction,
        "instruction_terms": instruction_terms,
        "map_labels": labels,
        "map_labels_with_instruction_overlap": relevant_labels,
        "l1_fact_count_total": sum(
            int(row.get("l1_fact_count_total") or 0) for row in rows
        ),
        "l3_fact_count_total": sum(
            int(row.get("l3_fact_count_total") or 0) for row in rows
        ),
        "context_steps_total": sum(
            int(row.get("l3_context_steps") or 0) for row in rows
        ),
    }


def summarize(
    rows: list[dict[str, Any]], name: str,
    run_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    legacy_schema = bool(rows) and all(
        "l1_detection_events" not in row for row in rows
    )
    summary: dict[str, Any] = {
        "name": name,
        "rows": len(rows),
        "legacy_row_schema": legacy_schema,
        "context_steps": sum(bool(row.get("context_injected")) for row in rows),
        "context_has_fact_steps": sum(
            bool(row.get("context_has_facts")) for row in rows
        ),
        "fact_steps": sum(int(row.get("fact_count") or 0) > 0 for row in rows),
        "l1_fact_total": _sum(rows, "l1_fact_count"),
        "l3_fact_total": _sum(rows, "l3_fact_count"),
        "context_tokens_total": _sum(rows, "context_tokens"),
        "memory_tokens_injected_total": _sum(rows, "memory_tokens_injected"),
        "segmentation_steps": sum(
            int(row.get("segmentation_count") or 0) > 0 for row in rows
        ),
        "map_object_nodes_max": _max(rows, "map_object_nodes"),
        "l3_map_object_count_max": _max(rows, "l3_map_object_count"),
        "l1_detection_events_total": _sum(rows, "l1_detection_events"),
        "l1_rejected_irrelevant_total": _sum(rows, "l1_rejected_irrelevant"),
        "l1_rejected_below_confidence_total": _sum(
            rows, "l1_rejected_below_confidence"
        ),
        "l1_rejected_missing_evidence_total": _sum(
            rows, "l1_rejected_missing_evidence"
        ),
        "l3_relevance_rejected_total": _sum(rows, "l3_relevance_rejected"),
        "l3_current_only_skipped_total": _sum(rows, "l3_current_only_skipped"),
        "l3_history_unavailable_total": _sum(rows, "l3_history_unavailable"),
        "retrieved_nodes_max": _max(rows, "retrieved_nodes"),
        "joint_l3_facts_considered_total": _sum(
            rows, "joint_l3_facts_considered"
        ),
        "joint_candidate_pairs_total": _sum(rows, "joint_candidate_pairs"),
        "joint_confirmed_total": _sum(rows, "joint_aligned_pairs"),
    }

    causes: list[str] = []
    if not rows:
        causes.append("steps_jsonl_is_empty")
        summary["possible_causes"] = causes
        return summary
    if legacy_schema:
        causes.append(
            "legacy_rows_lack_pipeline_counters: rerun the CPU/GPU smoke with "
            "this revision to attribute the zero fact count"
        )
    if summary["l1_fact_total"] == 0:
        if summary["l1_detection_events_total"] == 0:
            causes.append("no_l1_detection_event_reached_the_writer")
        else:
            if summary["l1_rejected_irrelevant_total"]:
                causes.append(
                    "l1_detections_were_written_but_no_class_overlapped_the_instruction"
                )
            if summary["l1_rejected_below_confidence_total"]:
                causes.append(
                    "l1_detections_were_below_the_prompt_floor_0.5"
                )
            if summary["l1_rejected_missing_evidence_total"]:
                causes.append(
                    "l1_events_were_written_but_the_prompt_lookup_did_not_find_them"
                )
    if summary["l3_fact_total"] == 0:
        if summary["segmentation_steps"] == 0:
            causes.append("no_l3_segmentation_step_ran")
        elif summary["map_object_nodes_max"] == 0:
            causes.append("segmentation_never_produced_a_valid_3d_object")
        else:
            if summary["l3_relevance_rejected_total"]:
                causes.append(
                    "3d_objects_exist_but_no_class_overlapped_the_instruction"
                )
            if summary["l3_current_only_skipped_total"]:
                causes.append(
                    "only_current_step_l3_evidence_was_available_to_the_prompt"
                )
            if summary["l3_history_unavailable_total"]:
                causes.append("l3_objects_had_no_usable_history_record")
    if (
        summary["l3_fact_total"] > 0
        and summary["joint_candidate_pairs_total"] == 0
    ):
        causes.append(
            "l3_facts_exist_but_no_l1_counterpart_was_available_for_alignment"
        )
    if (
        summary["l3_fact_total"] == 0
        and summary["joint_l3_facts_considered_total"] == 0
    ):
        causes.append("joint_alignment_had_no_l3_fact_to_consider")
    if summary["context_steps"] and summary["fact_steps"] == 0:
        causes.append("a_fixed_hint_block_was_injected_without_a_fact")
    if run_context:
        summary["run_context"] = run_context
        if (
            summary["l3_fact_total"] == 0
            and run_context.get("lexical_available")
            and run_context["map_labels"]
            and not run_context["map_labels_with_instruction_overlap"]
        ):
            causes.append(
                "map_labels_have_no_lexical_overlap_with_instruction"
            )
    if (
        summary["context_tokens_total"] == 0
        and summary["memory_tokens_injected_total"] > 0
    ):
        causes.append(
            "legacy_context_tokens_recording_bug: it was seeded to 0 before "
            "the token count was computed"
        )
    summary["possible_causes"] = causes
    return summary


def _inputs(argv: list[str]) -> list[tuple[Path, Path, Path | None]]:
    parser = argparse.ArgumentParser(
        description="Explain zero-fact steps from steps.jsonl (read-only).",
    )
    parser.add_argument(
        "paths", nargs="+",
        help="run directory containing steps.jsonl, or a steps.jsonl file",
    )
    args = parser.parse_args(argv)
    result: list[tuple[Path, Path, Path | None]] = []
    for value in args.paths:
        path = Path(value).expanduser()
        if path.is_dir():
            result.append((path, path / "steps.jsonl", path / "results.jsonl"))
        else:
            result.append((path, path, None))
    return result


def main(argv: list[str] | None = None) -> int:
    inputs = _inputs(sys.argv[1:] if argv is None else argv)
    reports = []
    for run_dir, steps_path, results_path in inputs:
        run_context = _run_context(
            results_path if results_path is not None
            else run_dir / "results.jsonl"
        )
        if not steps_path.is_file():
            reports.append({
                "name": str(steps_path), "error": "steps.jsonl not found",
                "run_context": run_context,
                "possible_causes": ["steps_jsonl_missing"],
            })
            continue
        reports.append(
            summarize(_load_rows(steps_path), str(steps_path), run_context)
        )
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return 0 if all("error" not in report for report in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())

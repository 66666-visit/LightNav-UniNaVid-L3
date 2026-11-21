"""Unit tests for the read-only zero-fact diagnostic classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "diagnose_memory_facts", ROOT / "scripts" / "diagnose_memory_facts.py",
)
assert _SPEC is not None and _SPEC.loader is not None
diagnose = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnose)


def _row(**values):
    row = {
        "step": 0,
        "context_injected": True,
        "context_has_facts": True,
        "fact_count": 1,
        "l1_fact_count": 1,
        "l3_fact_count": 1,
        "context_tokens": 20,
        "memory_tokens_injected": 25,
        "segmentation_count": 1,
        "map_object_nodes": 1,
        "l3_map_object_count": 1,
        "l1_detection_events": 2,
        "l1_rejected_irrelevant": 0,
        "l1_rejected_below_confidence": 0,
        "l1_rejected_missing_evidence": 0,
        "l3_relevance_rejected": 0,
        "l3_current_only_skipped": 0,
        "l3_history_unavailable": 0,
        "retrieved_nodes": 1,
        "joint_l3_facts_considered": 1,
        "joint_candidate_pairs": 1,
        "joint_aligned_pairs": 1,
    }
    row.update(values)
    return row


def test_irrelevant_detections_and_map_objects_are_classified_as_data_cause():
    summary = diagnose.summarize([_row(
        context_has_facts=False, fact_count=0,
        l1_fact_count=0, l3_fact_count=0,
        l1_rejected_irrelevant=2, l3_relevance_rejected=1,
        retrieved_nodes=0, joint_l3_facts_considered=0,
        joint_candidate_pairs=0, joint_aligned_pairs=0,
    )], "run")
    causes = summary["possible_causes"]
    assert "l1_detections_were_written_but_no_class_overlapped_the_instruction" in causes
    assert "3d_objects_exist_but_no_class_overlapped_the_instruction" in causes
    assert "a_fixed_hint_block_was_injected_without_a_fact" in causes
    assert "joint_alignment_had_no_l3_fact_to_consider" in causes


def test_valid_facts_are_not_reported_as_a_fixed_hint():
    summary = diagnose.summarize([_row()], "run")
    assert summary["l1_fact_total"] == 1
    assert summary["l3_fact_total"] == 1
    assert "a_fixed_hint_block_was_injected_without_a_fact" not in (
        summary["possible_causes"]
    )


def test_results_context_compares_map_labels_with_the_instruction(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text(
        '{"instruction": "go to the kitchen counter", '
        '"memory_summary": {"map": {"labels": {"chair": 2}}}}\n',
        encoding="utf-8",
    )
    context = diagnose._run_context(results)
    assert context is not None
    assert context["lexical_available"] is True
    assert "counter" in context["instruction_terms"]
    assert context["map_labels"] == {"chair": 2}
    assert context["map_labels_with_instruction_overlap"] == []

    summary = diagnose.summarize([_row(
        context_has_facts=False, fact_count=0,
        l1_fact_count=0, l3_fact_count=0,
    )], "run", context)
    assert "map_labels_have_no_lexical_overlap_with_instruction" in (
        summary["possible_causes"]
    )


def test_legacy_rows_are_not_over_interpreted():
    legacy = {
        "step": 0, "context_injected": True, "fact_count": 0,
        "l1_fact_count": 0, "l3_fact_count": 0,
    }
    summary = diagnose.summarize([legacy], "run")
    assert summary["legacy_row_schema"] is True
    assert any(
        cause.startswith("legacy_rows_lack_pipeline_counters")
        for cause in summary["possible_causes"]
    )

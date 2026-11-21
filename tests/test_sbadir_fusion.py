"""Source pairing, conservative conflict checks, and exact official task slot."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

from l1_integration.sbadir_backend import L1SelectedFact, L1SourceEvidence
from l3_integration.official_prompt import (
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE, build_prompt, split_official_template,
    verify_task_text_injection,
)
from l3_integration.sbadir_fusion import Claim, classify_claims, compose_sbadir


HEADER = "[Navigation Map Memory - L3; history, not current view]"
L3_LINE = "Map record (step 2, 1 view): the chair is in front of you, 2.0 m straight-line."


def _count(text):
    # Deterministic fake tokenizer; the real agent supplies checkpoint.encode.
    return len(text.split())


def _l3(label="chair", linked=("rgb-1",)):
    return SimpleNamespace(
        object_id="map-1", label=label, latest_step=2, last_step=2,
        source="yolo11n-seg:seg.pt", confidence=0.87,
        geometry_verified=True, evidence_ids=("seg-1",),
        linked_l1_evidence_ids=linked,
    )


def _l1(label="chair", evidence_id="rgb-1", sector="left"):
    source = L1SourceEvidence(
        evidence_id=evidence_id, observation_step=2, object_label=label,
        detection_confidence=0.91, detector_name="ultralytics_yolo:detect.pt",
        rgb_observation_source="observation.rgb",
        bbox_xyxy=(1.0, 2.0, 10.0, 12.0), source_evidence_ids=(),
    )
    return L1SelectedFact(
        text="Now: %s (%s), conf 0.91, 1 sighting since step 2, last step 2."
             % (label, sector),
        label=label, role="current", sector=sector, first_step=2,
        last_step=2, sightings=1, evidence_ids=(evidence_id,),
        evidence=(source,), selection_source="l1_prompt_organizer",
    )


def _compose(l3=None, l1=None, **kwargs):
    l3_facts = [_l3()] if l3 is None else l3
    l1_facts = [_l1()] if l1 is None else l1
    return compose_sbadir(
        template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
        instruction="find the chair",
        l3_text=HEADER + "\n" + L3_LINE if l3_facts else None,
        l3_facts=l3_facts, l1_facts=l1_facts,
        count_tokens=_count, max_prompt_tokens=10000,
        l1_max_tokens=160, **kwargs,
    )


def test_same_entity_same_label_merges_duplicate_assertion_and_keeps_both_sources():
    result = _compose()
    assert result.diagnostics["duplicated_facts"] == 1
    assert result.diagnostics["conflicting_facts"] == 0
    assert result.diagnostics["retained_facts"] == 2
    assert result.diagnostics["l3_core_unchanged"] is True
    assert result.l3_original_text == HEADER + "\n" + L3_LINE
    assert result.memory_text.splitlines()[1].startswith(L3_LINE)
    assert "L3 RGB-D map; source yolo11n-seg:seg.pt" in result.memory_text
    assert "evidence seg-1" in result.memory_text
    assert "L1 RGB observation also supports map node map-1" in result.l1_text
    assert "RGB input observation.rgb" in result.l1_text
    assert "detector conf 0.91" in result.l1_text
    assert "evidence rgb-1" in result.l1_text


def test_linked_conflicting_label_retains_both_values_and_evidence_without_winner():
    result = _compose(l1=[_l1(label="table")])
    assert result.diagnostics["conflicting_facts"] == 1
    assert result.diagnostics["conflicts"] == [{
        "type": "object_label", "entity_id": "map-1",
        "l1_value": "table", "l3_value": "chair",
        "l1_evidence_ids": ["rgb-1"], "l3_evidence_ids": ["seg-1"],
        "resolution": "both_retained",
    }]
    assert "the chair" in result.memory_text
    assert "L1 RGB fact: Now: table" in result.memory_text


def test_same_label_without_evidence_link_is_uncertain_not_forced_match():
    result = _compose(l3=[_l3(linked=())])
    assert result.diagnostics["uncertain_entity_matches"] == 1
    assert result.diagnostics["duplicated_facts"] == 0
    assert result.diagnostics["conflicting_facts"] == 0
    assert "L1 RGB fact: Now: chair" in result.l1_text


def test_aggregated_rgb_sightings_do_not_claim_one_map_entity_without_all_links():
    first = _l1()
    older = L1SourceEvidence(
        evidence_id="rgb-older", observation_step=1, object_label="chair",
        detection_confidence=0.78, detector_name="ultralytics_yolo:detect.pt",
        rgb_observation_source="observation.rgb", bbox_xyxy=None,
        source_evidence_ids=(),
    )
    aggregate = replace(
        first, sightings=2, evidence_ids=("rgb-1", "rgb-older"),
        evidence=(first.evidence[0], older),
    )
    result = _compose(l1=[aggregate])
    assert result.diagnostics["duplicated_facts"] == 0
    assert result.diagnostics["uncertain_entity_matches"] == 1
    assert "L1 RGB fact: Now: chair" in result.l1_text
    assert "evidence rgb-1" in result.l1_text
    assert "evidence rgb-older" not in result.l1_text
    assert result.diagnostics["l1_evidence_ids"] == [["rgb-1", "rgb-older"]]


def test_l3_provenance_pairs_latest_step_with_latest_evidence_id():
    fact = _l3()
    fact.evidence_ids = ("seg-old", "seg-new")
    fact.latest_step = 7
    result = _compose(l3=[fact], l1=[])
    assert "step 7; seg conf 0.87; evidence seg-new" in result.memory_text
    assert "evidence seg-old" not in result.memory_text


def test_matched_l1_visual_history_keeps_qualifier_and_place_source():
    fact = replace(
        _l1(), role="history", visual_place_id="place-1",
        visual_place_evidence_ids=("place-evidence-1",),
        visual_place_source="dinov2:local-small",
    )
    result = _compose(l1=[fact])
    assert "possible revisit; current direction and passage unknown" in result.l1_text
    assert "visual place source dinov2:local-small" in result.l1_text
    assert "visual place evidence place-evidence-1" in result.l1_text


def test_complementary_and_uncertain_claims_are_not_conflicts():
    first = Claim("map-1", "object_label", "chair", "L1", ("rgb-1",))
    assert classify_claims(first, Claim("map-1", "object_label", "chair", "L3", ("seg-1",))) == "duplicate"
    assert classify_claims(first, Claim("map-1", "range_m", "2.0", "L3", ("seg-1",))) == "complementary"
    assert classify_claims(first, Claim("map-2", "object_label", "table", "L3", ("seg-2",))) == "uncertain_entity"


def test_l1_only_uses_remaining_tokens_and_does_not_drop_l3():
    result = compose_sbadir(
        template=REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
        instruction="find the chair", l3_text=HEADER + "\n" + L3_LINE,
        l3_facts=[_l3()], l1_facts=[_l1()], count_tokens=_count,
        max_prompt_tokens=10000, l1_max_tokens=0,
    )
    assert result.diagnostics["l1_retained_facts"] == 0
    assert result.diagnostics["l1_omitted_facts"] == 1
    assert result.diagnostics["l3_fact_count"] == 1
    assert result.memory_text.splitlines()[1].startswith(L3_LINE)


def test_official_fixed_parts_and_visual_token_position_remain_exact():
    result = _compose()
    prompt = build_prompt(REFERENCE_OFFICIAL_PROMPT_TEMPLATE, result.task_text)
    prefix, suffix = split_official_template(REFERENCE_OFFICIAL_PROMPT_TEMPLATE)
    assert prompt == prefix + result.task_text + suffix
    assert prompt.count("<image>") == 1
    assert prompt.index("<image>") < prompt.index("find the chair")
    assert prompt.endswith(suffix)
    verify_task_text_injection(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, "find the chair",
        result.memory_text, prompt,
    )
    assert result.diagnostics["l3_original_sha256"] == hashlib.sha256(
        (HEADER + "\n" + L3_LINE).encode("utf-8")
    ).hexdigest()

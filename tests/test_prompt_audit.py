"""The CPU prompt audit must execute the real production prompt path."""

from __future__ import annotations

import hashlib
import json

import pytest

from l3_integration.official_prompt import (
    L3_SEPARATOR,
    NAVIGATION_IDENTIFIER,
    REFERENCE_OFFICIAL_PROMPT_SHA256,
    REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
    build_official_prompt,
    prompt_sha256,
    split_official_template,
    update_prompt_view,
)

from scripts import audit_prompt


@pytest.fixture(scope="module")
def audit():
    return audit_prompt.build_audit()


def test_baseline_prompt_is_the_byte_identical_official_prompt(audit) -> None:
    baseline = audit["baseline_prompt"]
    assert baseline == REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format(
        audit_prompt.FIXTURE_INSTRUCTION,
    )
    assert baseline == build_official_prompt(
        REFERENCE_OFFICIAL_PROMPT_TEMPLATE, audit_prompt.FIXTURE_INSTRUCTION,
    )
    assert prompt_sha256(REFERENCE_OFFICIAL_PROMPT_TEMPLATE) == (
        REFERENCE_OFFICIAL_PROMPT_SHA256
    )
    assert audit["l3_memory_block"] not in baseline
    assert "Navigation Map Memory" not in baseline


def test_l3_prompt_is_mode_b_inside_the_official_task_slot(audit) -> None:
    baseline = audit["baseline_prompt"]
    l3_prompt = audit["l3_prompt"]
    block = audit["l3_memory_block"]
    instruction = audit_prompt.FIXTURE_INSTRUCTION
    prefix, suffix = split_official_template(REFERENCE_OFFICIAL_PROMPT_TEMPLATE)

    assert l3_prompt == REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format(
        instruction + L3_SEPARATOR + block,
    )
    assert l3_prompt == prefix + instruction + L3_SEPARATOR + block + suffix
    assert l3_prompt != REFERENCE_OFFICIAL_PROMPT_TEMPLATE.format(instruction) + (
        L3_SEPARATOR + block
    )
    assert l3_prompt.startswith(prefix + instruction)
    assert l3_prompt.endswith(suffix)
    assert l3_prompt.index(block) == len(prefix) + len(instruction) + 1
    assert l3_prompt.index(block) < l3_prompt.index(suffix)
    assert len(l3_prompt) - len(baseline) == len(L3_SEPARATOR) + len(block)
    assert NAVIGATION_IDENTIFIER in l3_prompt


def test_l3_block_is_the_real_serializer_output(audit) -> None:
    block = audit["l3_memory_block"]
    assert block.startswith("[Navigation Map Memory - L3;")
    assert "Map record (step " in block
    assert "the chair is in front of you" in block
    assert block.count("\n") == 1
    assert audit["trace"]["facts"]["fact_count"] == 1
    assert audit["trace"]["facts"]["history_only"] is True
    assert audit["trace"]["facts"]["map_snapshot"]["objects"] >= 1


def test_l3_reaches_predict_inference_update_prompt_and_input_ids(audit) -> None:
    trace = audit["trace"]
    model_input = trace["model_input"]
    block = audit["l3_memory_block"]
    l3_prompt = audit["l3_prompt"]

    assert model_input["navigation_qs_is_l3_prompt"] is True
    assert model_input["l3_in_navigation_qs"] is True
    assert model_input["question"] == update_prompt_view(l3_prompt)
    assert model_input["cur_prompt"] == model_input["question"]
    assert model_input["update_prompt_argument"] == [[model_input["question"]]]
    assert model_input["update_prompt_equals_question"] is True
    assert block.replace("\n", "") in model_input["question"]
    assert model_input["l3_in_question"] is True
    assert model_input["l3_in_cur_prompt"] is True
    assert model_input["l3_in_input_ids"] is True
    assert model_input["official_navigation_identifier_in_question"] is True
    assert model_input["official_token_oracle_matches"] is True
    assert model_input["image_token_positions"] == [188]
    assert len(model_input["image_token_positions"]) == 1
    assert model_input["generate_called"] is True
    assert model_input["generate_kwargs"] == {
        "do_sample": True, "temperature": 0.2,
        "max_new_tokens": 1024, "use_cache": True,
    }
    assert model_input["vision_input_unchanged"] is True
    assert model_input["special_block_unchanged"] is True


def test_visual_tokens_and_generation_parameters_are_unchanged(audit) -> None:
    trace = audit["trace"]
    model_input = trace["model_input"]
    assert model_input["baseline_image_token_positions"] == (
        model_input["image_token_positions"]
    )
    assert model_input["structural_input_ids"]["baseline_length"] < (
        model_input["structural_input_ids"]["length"]
    )
    assert trace["invariants"]["visual_token_layout_unchanged"] is True
    assert trace["invariants"]["generation_parameters_unchanged"] is True
    assert trace["invariants"]["l1_not_injected"] is True


def test_the_audit_is_marked_as_a_synthetic_fixture(audit) -> None:
    trace = audit["trace"]
    assert trace["fixture"]["kind"] == "synthetic_fixture"
    assert trace["fixture"]["is_real_habitat_navigation_prompt"] is False
    assert "synthetic" in trace["fixture"]["description"].lower()
    assert trace["token_accounting"]["real_tokenizer_available"] is False
    assert trace["token_accounting"]["token_count"] is None
    assert trace["token_accounting"]["l3_block_token_count"] is None
    assert trace["l1"]["all_event_lists_empty"] is True
    assert trace["l1"]["linked_l1_evidence_count"] == 0


def test_written_artifacts_are_exact_and_self_consistent(audit, tmp_path) -> None:
    paths = audit_prompt.write_audit(tmp_path, audit)
    assert paths["baseline"].read_bytes() == audit["baseline_prompt"].encode()
    assert paths["l3"].read_bytes() == audit["l3_prompt"].encode()
    assert paths["block"].read_bytes() == audit["l3_memory_block"].encode()
    assert paths["diff"].read_text(encoding="utf-8") == audit["prompt_diff"]
    trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
    assert trace["prompts"]["baseline"]["sha256"] == hashlib.sha256(
        audit["baseline_prompt"].encode("utf-8")
    ).hexdigest()
    assert trace["prompts"]["l3"]["sha256"] == hashlib.sha256(
        audit["l3_prompt"].encode("utf-8")
    ).hexdigest()
    assert trace["prompts"]["l3_memory_block"]["sha256"] == hashlib.sha256(
        audit["l3_memory_block"].encode("utf-8")
    ).hexdigest()
    assert paths["diff"].read_text(encoding="utf-8").startswith(
        "--- baseline_prompt.txt\n+++ l3_prompt.txt\n@@"
    )


def test_call_chain_records_the_production_files_and_lines(audit) -> None:
    stages = {row["stage"]: row for row in audit["trace"]["call_chain"]}
    assert stages["l3_serialize_context"]["file"] == (
        "l3_memory/l3_official_prompt.py"
    )
    assert stages["l3_serialize_context"]["line"] is not None
    assert stages["official_format"]["file"] == "agent_uninavid.py"
    assert stages["official_format"]["line"] is not None
    assert stages["official_format_call"]["file"] == (
        "l3_integration/official_prompt.py"
    )
    assert stages["predict_inference"]["line"] is not None
    assert stages["question_view"]["line"] is not None
    assert stages["update_prompt"]["line"] is not None
    assert stages["generate"]["line"] is not None
    assert stages["model_prompt_use"]["file"] == (
        "uninavid/model/uninavid_arch.py"
    )
    assert stages["model_text_embeddings"]["line"] is not None

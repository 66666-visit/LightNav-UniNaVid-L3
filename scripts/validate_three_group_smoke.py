#!/usr/bin/env python3
"""Validate the three real one-Episode smokes before the formal 88-Episode runs.

A must come from the untouched 7602b25 runtime (no source-background/L1 keys).
B must have source background and no L1 work at all.  C must add L1 on top of
B's source pairing.  The three smokes must use the same Episode identity, seed
and instruction.  On success this writes each smoke's ``smoke_validation.json``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROMPT_SHA = "435d79868156aaa1352afcd98f4ceae2f03a54f8cdc5fe87e41d32774d5512c1"


def _load(directory: pathlib.Path) -> dict:
    path = directory / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 1:
        raise ValueError("{} requires exactly one Episode result".format(directory))
    row = rows[0]
    if not row.get("scene_basename") or not row.get("episode_id"):
        raise ValueError("smoke result lacks Episode identity in {}".format(directory))
    if row.get("official_prompt_sha256") != PROMPT_SHA:
        raise ValueError("smoke did not preserve the official Prompt in {}".format(directory))
    row["_directory"] = str(directory)
    return row


def _validate_a(row: dict) -> None:
    for key in ("source_background_facts_total", "l1_fact_count_total",
                "prompt_has_l1_steps", "l3_original_memory_unchanged",
                "l1_enabled_steps"):
        if key in row:
            raise ValueError("A smoke unexpectedly contains {}".format(key))
    if row.get("l3_enabled") is not True or row.get("l3_map_steps", 0) <= 0:
        raise ValueError("A smoke did not build the L3 map")
    if row.get("l3_fact_count_total", 0) <= 0:
        raise ValueError("A smoke produced no L3 fact; choose another SMOKE_SHARD")
    if row.get("l3_segmentation_error_count", 0) != 0:
        raise ValueError("A smoke encountered L3 segmentation errors")


def _validate_b(row: dict) -> None:
    if row.get("source_background_enabled") is not True:
        raise ValueError("B smoke did not enable source background")
    if row.get("l1_enabled") is not False:
        raise ValueError("B smoke enabled L1")
    if row.get("l1_enabled_steps", -1) != 0:
        raise ValueError("B smoke observed through L1")
    if row.get("l1_observation_attempt_steps", -1) != 0:
        raise ValueError("B smoke attempted L1 observation")
    if row.get("l1_fact_count_total", -1) != 0 or row.get("prompt_has_l1_steps", -1) != 0:
        raise ValueError("B smoke injected an L1 fact")
    if row.get("l3_fact_count_total", 0) <= 0:
        raise ValueError("B smoke produced no L3 fact; choose another SMOKE_SHARD")
    if row.get("source_background_facts_total") != row.get("l3_fact_count_total"):
        raise ValueError("B smoke did not pair every L3 fact with its source")
    if row.get("l3_original_memory_unchanged") is not True:
        raise ValueError("B smoke altered the original L3 memory")
    if row.get("l3_segmentation_error_count", 0) != 0:
        raise ValueError("B smoke encountered L3 segmentation errors")


def _validate_c(row: dict) -> None:
    if row.get("source_background_enabled") is not True:
        raise ValueError("C smoke did not enable source background")
    if row.get("l1_enabled") is not True:
        raise ValueError("C smoke did not enable L1")
    if row.get("l1_enabled_steps", 0) != row.get("env_steps"):
        raise ValueError("C smoke did not run L1 on every step")
    # LightNav's original LocalGraphFusion legitimately produces no card on
    # steps without a current instruction-relevant detection or history.  A
    # single Smoke Episode therefore may contain zero L1 facts.  That is not a
    # Smoke failure as long as the L1 chain ran and no error was recorded.
    l1_facts = int(row.get("l1_fact_count_total", 0) or 0)
    l1_retained = int(row.get("l1_retained_facts_total", 0) or 0)
    l1_built = int(row.get("l1_prompt_built_steps", 0) or 0)
    l1_prompt_steps = int(row.get("prompt_has_l1_steps", 0) or 0)
    if l1_facts == 0:
        if l1_retained or l1_built or l1_prompt_steps:
            raise ValueError(
                "C smoke built/injected an L1 card without an L1 fact"
            )
    else:
        if l1_built <= 0:
            raise ValueError(
                "C smoke produced L1 facts but never built the original L1 card"
            )
        if l1_retained <= 0:
            raise ValueError(
                "C smoke selected L1 facts but did not inject them verbatim"
            )
        if l1_prompt_steps <= 0:
            raise ValueError(
                "C smoke selected L1 facts but no prompt step contains the L1 card"
            )
    # C's SBAdir source background is attached to B's frozen L3 facts only.
    # The original L1 card is injected verbatim and has no source suffix.
    if row.get("source_background_facts_total") != row.get("l3_fact_count_total", 0):
        raise ValueError("C smoke did not pair every L3 fact with its source")
    if row.get("l3_original_memory_unchanged") is not True:
        raise ValueError("C smoke altered the original L3 memory")
    for key in ("l1_observe_error_count", "l1_selection_error_count",
                "l1_detector_error_count",
                "l1_place_error_count", "l3_segmentation_error_count",
                "cross_label_entity_link_error_count"):
        if row.get(key, 0) != 0:
            raise ValueError("C smoke encountered {}".format(key))
    budget_source = row.get("budget_source")
    if l1_prompt_steps > 0:
        if budget_source == "token_aware_context":
            base_tokens = row.get("base_prompt_tokens_mean")
            available = row.get("available_memory_tokens_min")
            injected = row.get("memory_tokens_injected_mean")
            if not isinstance(base_tokens, (int, float)) or base_tokens <= 0:
                raise ValueError("C smoke lacks a measured base prompt token count")
            if not isinstance(available, int) or available < 0:
                raise ValueError("C smoke lacks the measured L1 token budget")
            if not isinstance(injected, (int, float)) or injected < 0:
                raise ValueError("C smoke lacks the injected L1 token count")
            if row.get("memory_budget_violation_steps", 0) != 0:
                raise ValueError("C smoke injected more L1 tokens than the budget")
            visual = row.get("visual_tokens_mean")
            if not isinstance(visual, (int, float)) or visual < 0:
                raise ValueError("C smoke lacks a visual-token count")
            margin = row.get("context_margin_tokens_min")
            if not isinstance(margin, int) or margin < 0:
                raise ValueError("C smoke multimodal prompt exceeds context reserve")
            if row.get("multimodal_budget_violation_steps", 0) != 0:
                raise ValueError("C smoke exceeded the multimodal context reserve")
        elif budget_source == "char_fallback_320":
            if not row.get("budget_fallback_reason"):
                raise ValueError("C L1 char fallback lacks a reported reason")
        else:
            raise ValueError("C smoke lacks the L1 token-budget protocol")
    drops = row.get("token_overflow_drops", 0)
    if not isinstance(drops, int) or drops < 0:
        raise ValueError("C smoke has an invalid token overflow count")
    for key in ("l3_map_steps", "l3_fact_count_total", "prompt_has_l3_steps",
                "l1_place_attempt_steps", "l1_place_observed_steps",
                "l1_events_persisted_total"):
        if row.get(key, 0) <= 0:
            raise ValueError(
                "C smoke did not exercise {}; choose another SMOKE_SHARD".format(key)
            )


def _write(directory: pathlib.Path, row: dict, keys) -> None:
    record = {
        "group": row["group"],
        "group_name": row.get("group_name"),
        "episode": [row["scene_basename"], row["episode_id"]],
        "episode_seed": row.get("episode_seed"),
        "instruction_sha256": row.get("instruction_sha256"),
        "official_prompt_sha256": row.get("official_prompt_sha256"),
    }
    for key in keys:
        record[key] = row.get(key)
    path = directory / "smoke_validation.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


A_KEYS = ("l3_enabled", "env_steps", "l3_map_steps", "l3_fact_count_total",
          "prompt_has_l3_steps", "l3_segmentation_error_count")
B_KEYS = ("l3_enabled", "env_steps", "l3_map_steps", "l3_fact_count_total",
          "prompt_has_l3_steps", "source_background_facts_total",
          "l3_original_memory_unchanged", "source_background_enabled",
          "l1_enabled", "l1_enabled_steps", "l1_observation_attempt_steps",
          "l1_fact_count_total", "prompt_has_l1_steps",
          "l3_segmentation_error_count")
C_KEYS = ("l3_enabled", "env_steps", "l3_map_steps", "l3_fact_count_total",
          "prompt_has_l3_steps", "prompt_has_l1_steps",
          "source_background_facts_total", "l3_original_memory_unchanged",
          "source_background_enabled", "l1_enabled", "l1_enabled_steps",
          "l1_fact_count_total", "l1_retained_facts_total",
          "l1_omitted_facts_total", "l1_prompt_built_steps", "l1_empty_steps",
          "l1_observation_attempt_steps", "l1_events_persisted_total",
          "l1_detection_events_persisted_total",
          "l1_place_attempt_steps", "l1_place_observed_steps",
          "l1_observe_error_count", "l1_selection_error_count",
          "l1_detector_error_count",
          "l1_place_error_count", "l3_segmentation_error_count",
          "cross_label_entity_link_error_count", "budget_source",
          "context_limit_tokens", "base_prompt_tokens",
          "available_memory_tokens", "memory_tokens_injected",
          "token_overflow_drops", "budget_fallback_reason",
          "memory_budget_violation_steps",
          "text_prompt_tokens_mean", "visual_tokens_mean",
          "multimodal_prompt_tokens_mean", "context_margin_tokens_min",
          "multimodal_budget_violation_steps")


def validate(a_dir, b_dir, c_dir, a_group: str, b_group: str, c_group: str) -> dict:
    a = _load(pathlib.Path(a_dir))
    b = _load(pathlib.Path(b_dir))
    c = _load(pathlib.Path(c_dir))
    if [a["group"], b["group"], c["group"]] != [a_group, b_group, c_group]:
        raise ValueError("smoke groups are not A/B/C")
    identity = (a["scene_basename"], a["episode_id"])
    for name, row in (("B", b), ("C", c)):
        if (row["scene_basename"], row["episode_id"]) != identity:
            raise ValueError("{} smoke used a different Episode".format(name))
        if row.get("episode_seed") != a.get("episode_seed"):
            raise ValueError("{} smoke seed differs from A".format(name))
        if row.get("instruction_sha256") != a.get("instruction_sha256"):
            raise ValueError("{} smoke instruction differs from A".format(name))
    _validate_a(a)
    _validate_b(b)
    _validate_c(c)
    _write(pathlib.Path(a_dir), a, A_KEYS)
    _write(pathlib.Path(b_dir), b, B_KEYS)
    _write(pathlib.Path(c_dir), c, C_KEYS)
    return {
        "episode": list(identity),
        "a": a, "b": b, "c": c,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-dir", required=True)
    parser.add_argument("--b-dir", required=True)
    parser.add_argument("--c-dir", required=True)
    parser.add_argument("--a-group", required=True)
    parser.add_argument("--b-group", required=True)
    parser.add_argument("--c-group", required=True)
    args = parser.parse_args(argv)
    result = validate(args.a_dir, args.b_dir, args.c_dir,
                      args.a_group, args.b_group, args.c_group)
    a, b, c = result["a"], result["b"], result["c"]
    print("smokes completed", result["episode"][0], result["episode"][1])
    print("A L3 facts:", a.get("l3_fact_count_total"),
          "| B L3 facts:", b.get("l3_fact_count_total"),
          "B source backgrounds:", b.get("source_background_facts_total"),
          "| C L3/L1 facts:", c.get("l3_fact_count_total"),
          c.get("l1_fact_count_total"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    if row.get("l1_fact_count_total", 0) <= 0:
        raise ValueError("C smoke produced no L1 fact; choose another SMOKE_SHARD")
    if row.get("source_background_facts_total") != (
            row.get("l3_fact_count_total", 0) + row.get("l1_fact_count_total", 0)):
        raise ValueError("C smoke did not pair every retained fact with its source")
    if row.get("l3_original_memory_unchanged") is not True:
        raise ValueError("C smoke altered the original L3 memory")
    for key in ("l1_observe_error_count", "l1_detector_error_count",
                "l1_place_error_count", "l3_segmentation_error_count",
                "cross_label_entity_link_error_count"):
        if row.get(key, 0) != 0:
            raise ValueError("C smoke encountered {}".format(key))
    for key in ("l3_map_steps", "l3_fact_count_total", "prompt_has_l3_steps",
                "prompt_has_l1_steps", "l1_place_attempt_steps",
                "l1_place_observed_steps"):
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
          "l1_fact_count_total", "l1_observation_attempt_steps",
          "l1_place_attempt_steps", "l1_place_observed_steps",
          "l1_observe_error_count", "l1_detector_error_count",
          "l1_place_error_count", "l3_segmentation_error_count",
          "cross_label_entity_link_error_count")


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

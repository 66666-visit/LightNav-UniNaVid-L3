#!/usr/bin/env python3
"""Strict three-group paired report for the frozen L3 comparison.

A ``l3_organized_baseline_7602b25`` runs the untouched ``7602b25`` L3
Organized prompt.  B ``l3_sbadir_source_background_only`` keeps exactly the
same L3 facts and pairs each with its real source record.  C ``l3_l1_sbadir``
adds the L1 RGB memory on top of B with the same SBAdir fusion.

Every measurement comes from the three real result streams.  Episode identity,
manifest, official prompt, shared run conditions, the absence of any L1 work in
B, and the equality of the original L3 text/background on the common action
prefix are checked before any report file is written.  A is never paired by
execution order; every comparison uses ``(scene_basename, episode_id)``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from l3_integration.experiment_groups import (  # noqa: E402
    A_GROUP,
    A_NAME,
    B_GROUP,
    B_NAME,
    C_GROUP,
    C_NAME,
)
from l3_integration.metrics import finite_float, mean_or_none, paired_summary  # noqa: E402
from l3_integration.pilot_manifest import (  # noqa: E402
    EXPECTED_EPISODE_MANIFEST_SHA256,
    EXPECTED_PILOT_MANIFEST_SHA256,
    load_episode_manifest,
    load_pilot_manifest,
)

BASELINE_SHA = "7602b254ea514aa71b0057e41b5db3bdfedf9ffa"
PROMPT_SHA = "435d79868156aaa1352afcd98f4ceae2f03a54f8cdc5fe87e41d32774d5512c1"
HISTORICAL_SUCCESSES = 49
HISTORICAL_EPISODES = 88

REQUIRED_B_FIELDS = (
    "l3_fact_count_total", "l3_context_tokens_mean", "l3_original_memory_unchanged",
    "source_background_facts_total", "l3_provenance_count_total",
    "l1_fact_count_total", "l1_retained_facts_total", "l1_enabled_steps",
    "l1_observation_attempt_steps", "prompt_has_l1_steps",
    "duplicated_facts_total", "conflicting_facts_total",
    "l3_segmentation_error_count",
)
REQUIRED_C_FIELDS = (
    "l1_fact_count_total", "l3_fact_count_total", "l1_context_tokens_mean",
    "l3_context_tokens_mean", "duplicated_facts_total",
    "conflicting_facts_total", "retained_facts_total", "omitted_facts_total",
    "prompt_has_l1_steps", "l3_original_memory_unchanged",
    "source_background_facts_total", "l3_provenance_count_total",
    "l1_retained_facts_total", "l1_omitted_facts_total",
    "l1_prompt_built_steps", "l1_empty_steps",
    "l1_observation_attempt_steps", "l1_events_persisted_total",
    "l1_detection_events_persisted_total",
    "l1_enabled_steps", "l1_observe_error_count", "l1_selection_error_count",
    "l1_detector_error_count",
    "l1_place_attempt_steps", "l1_place_observed_steps", "l1_place_error_count",
    "l3_segmentation_error_count", "cross_label_entity_link_error_count",
    "l1_token_budget_steps", "l1_token_budget_fallback_steps",
    "token_overflow_drops_total", "memory_budget_violation_steps",
    "text_prompt_tokens_mean", "visual_tokens_mean",
    "multimodal_prompt_tokens_mean", "context_margin_tokens_min",
    "multimodal_budget_violation_steps",
)
METRICS = (
    ("SR", "success", 100.0, "%", "higher"),
    ("SPL", "spl", 100.0, "%", "higher"),
    ("nDTW", "ndtw", 100.0, "%", "higher"),
    ("NE", "distance_to_goal", 1.0, "m", "lower"),
    ("OSR", "oracle_success", 100.0, "%", "higher"),
    ("TL", "path_length", 1.0, "m", "lower"),
    ("Mean steps", "steps_taken", 1.0, "steps", "lower"),
    ("Mean inference latency", "inference_latency_ms_mean", 1.0, "ms", "lower"),
    ("Prompt tokens", "prompt_tokens_mean", 1.0, "tokens", "lower"),
    ("Memory facts", "memory_fact_count_total", 1.0, "fact uses", "descriptive"),
    ("Conflicts", "conflicting_facts_total", 1.0, "conflicts", "descriptive"),
    ("Source-paired facts", "source_background_facts_total", 1.0,
     "fact uses", "descriptive"),
)
COUNT_FIELDS = (
    "l3_fact_count_total", "l1_fact_count_total", "duplicated_facts_total",
    "conflicting_facts_total", "retained_facts_total", "omitted_facts_total",
    "prompt_has_l3_steps", "prompt_has_l1_steps", "source_background_facts_total",
    "l3_provenance_count_total", "l1_retained_facts_total",
    "l1_omitted_facts_total", "l1_prompt_built_steps", "l1_empty_steps",
    "l1_events_persisted_total", "l1_detection_events_persisted_total",
    "l1_enabled_steps", "l1_observation_attempt_steps", "l1_observe_error_count",
    "l1_selection_error_count", "l1_detector_error_count", "l1_place_attempt_steps",
    "l1_place_observed_steps", "l1_place_error_count",
    "l3_segmentation_error_count", "cross_label_entity_link_error_count",
    "cross_label_entity_links_total", "env_steps",
    "l1_token_budget_steps", "l1_token_budget_fallback_steps",
    "token_overflow_drops_total", "memory_budget_violation_steps",
    "context_margin_tokens_min", "multimodal_budget_violation_steps",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _b_source_hashes(root: Path) -> Dict[str, str]:
    names = [
        "run.py", "agent_uninavid.py", "l3_integration/map_sidecar.py",
        "l3_integration/sbadir_fusion.py", "l3_integration/experiment_groups.py",
    ]
    names += sorted(str(path.relative_to(root)) for path in (root / "l1_memory").glob("*.py"))
    names += sorted(str(path.relative_to(root)) for path in (root / "l1_integration").glob("*.py"))
    return {name: _sha256(root / name) for name in sorted(names)}


def _b_source_digest(hashes: Dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(
        hashes, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _official_agent_contract(baseline: Path, fused: Path) -> bool:
    def parse(path: Path):
        return ast.parse(path.read_text(encoding="utf-8"))

    def method(tree, name: str):
        nodes = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(nodes) != 1:
            raise ValueError("missing or duplicate official {} method".format(name))
        return nodes[0]

    def template(tree) -> str:
        values = [ast.literal_eval(node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Assign) and any(
                      isinstance(target, ast.Attribute) and target.attr == "promt_template"
                      for target in node.targets)]
        if len(values) != 1 or not isinstance(values[0], str):
            raise ValueError("official template is not one literal string")
        return values[0]

    a = parse(baseline / "agent_uninavid.py")
    b = parse(fused / "agent_uninavid.py")
    original = template(a)
    a_predict = method(a, "predict_inference")
    b_predict = method(b, "predict_inference")
    allowed_guard = ast.parse(
        'if (getattr(self, "source_background", False)\n'
        '        and self.last_prompt_tokens + 1024 > self.context_len):\n'
        '    raise RuntimeError("SBAdir exact input tokens exceed checkpoint context reserve")'
    ).body[0]
    matches = [index for index, node in enumerate(b_predict.body)
               if ast.dump(node) == ast.dump(allowed_guard)]
    if len(matches) != 1:
        return False
    index = matches[0]
    before = b_predict.body[index - 1] if index else None
    after = b_predict.body[index + 1] if index + 1 < len(b_predict.body) else None
    if not (isinstance(before, ast.Assign) and any(
        isinstance(target, ast.Attribute) and target.attr == "last_prompt_tokens"
        for target in before.targets
    ) and isinstance(after, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "stop_str"
        for target in after.targets
    )):
        return False
    del b_predict.body[index]
    return (
        original == template(b)
        and hashlib.sha256(original.encode("utf-8")).hexdigest() == PROMPT_SHA
        and ast.dump(method(a, "process_images")) == ast.dump(method(b, "process_images"))
        and ast.dump(a_predict) == ast.dump(b_predict)
    )


def _json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object in {}".format(path))
    return value


def _key(row: Dict[str, Any]) -> Tuple[str, str]:
    scene = row.get("scene_basename")
    episode = row.get("episode_id")
    if not isinstance(scene, str) or not scene or not isinstance(episode, str) or not episode:
        raise ValueError("result row lacks scene_basename/episode_id")
    return scene, episode


def _rows(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    rows = {}  # type: Dict[Tuple[str, str], Dict[str, Any]]
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("{}:{} is not a JSON object".format(path, number))
        key = _key(row)
        if key in rows:
            raise ValueError("duplicate Episode identity in {}: {}".format(path, key))
        rows[key] = row
    return rows


def _number(value: Any) -> Optional[float]:
    return finite_float(value)


def _int_nonnegative(row: Dict[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("{} must be a nonnegative integer".format(key))
    return value


def _metric_value(row: Dict[str, Any], name: str) -> Optional[float]:
    if name in ("success", "oracle_success"):
        value = row.get(name)
        return float(value) if isinstance(value, bool) else None
    if name == "memory_fact_count_total":
        l3 = row.get("l3_fact_count_total")
        l1 = row.get("l1_fact_count_total", 0)
        if isinstance(l3, int) and not isinstance(l3, bool) and isinstance(l1, int) and not isinstance(l1, bool):
            return float(l3 + l1)
        return None
    if name == "conflicting_facts_total" and name not in row:
        # A/B cannot create cross-layer conflicts because they have no L1.
        return 0.0
    return _number(row.get(name))


def _group_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {"episodes": len(rows), "successes": sum(row["success"] for row in rows)}
    for label, key, scale, _unit, _direction in METRICS:
        values = [_metric_value(row, key) for row in rows]
        metrics[label] = mean_or_none(
            value * scale for value in values if value is not None
        )
        metrics[label + "_available_episodes"] = sum(value is not None for value in values)
    for key in COUNT_FIELDS:
        if rows and all(key in row for row in rows):
            metrics[key] = sum(_int_nonnegative(row, key) for row in rows)
        else:
            metrics[key] = None
    for key in ("l3_context_tokens_mean", "l1_context_tokens_mean"):
        metrics[key] = mean_or_none(
            value for row in rows if (value := _number(row.get(key))) is not None
        )
    return metrics


def _validate_configs(
    a: Dict[str, Any], b: Dict[str, Any], c: Dict[str, Any],
    launch: Dict[str, Any],
) -> None:
    if a.get("group") != A_GROUP or b.get("group") != B_GROUP or c.get("group") != C_GROUP:
        raise ValueError("run_config groups do not match the frozen A/B/C code paths")
    for name, cfg in (("A", a), ("B", b), ("C", c)):
        if cfg.get("seed") != 42 or not cfg.get("formal_pilot"):
            raise ValueError("{} is not the frozen seed-42 formal Pilot".format(name))
        if not cfg.get("episode_manifest_verified") or not cfg.get("episode_manifest_aligned"):
            raise ValueError("{} lacks verified/aligned Episode Manifest".format(name))
        if cfg.get("pilot_manifest_file_sha256") != EXPECTED_PILOT_MANIFEST_SHA256:
            raise ValueError("{} Pilot SHA256 differs".format(name))
        if cfg.get("execution_manifest_canonical_sha256") != EXPECTED_EPISODE_MANIFEST_SHA256:
            raise ValueError("{} Episode Manifest canonical SHA256 differs".format(name))
    for key in (
        "model_name", "model_path", "exp_config", "split_num", "seed",
        "exp_save", "pilot_manifest_file_sha256", "execution_manifest_canonical_sha256",
        "pilot_order_digest", "execution_order_digest", "l3",
    ):
        if not (a.get(key) == b.get(key) == c.get(key)):
            raise ValueError("A/B/C run_config mismatch: {}".format(key))

    # A must be the untouched 7602b25 runtime.  The files written by that code
    # contain no L1, no source-background and no new group-name metadata, so a
    # dev-worktree run masquerading as A fails closed here.
    for key in ("source_background", "l1", "group_name"):
        if key in a:
            raise ValueError("A run_config contains {}: not the 7602b25 runtime".format(key))

    # B is L3 + source background only: never any L1 configuration.
    if b.get("l1") is not None:
        raise ValueError("B run_config contains an L1 block")
    b_background = b.get("source_background") or {}
    if b_background.get("enabled") is not True:
        raise ValueError("B did not enable the source background")
    if b_background.get("fusion") != "evidence_source_pairing":
        raise ValueError("B source background uses an unknown fusion mode")
    if b_background.get("l3_organized_facts_preserved") is not True:
        raise ValueError("B did not declare the original L3 facts preserved")
    if b.get("group_name") != B_NAME:
        raise ValueError("B run_config lacks its frozen group name")

    # C adds exactly one thing over B: the L1 backend.
    c_background = c.get("source_background") or {}
    if c.get("group_name") != C_NAME:
        raise ValueError("C run_config lacks its frozen group name")
    if c_background.get("enabled") is not True:
        raise ValueError("C did not enable the source background")
    if c_background.get("fusion") != b_background.get("fusion"):
        raise ValueError("B/C source-background fusion modes differ")
    l1_block = c.get("l1")
    if not isinstance(l1_block, dict):
        raise ValueError("C run_config lacks the L1 backend configuration")
    for key in ("det_model", "place_model"):
        if not l1_block.get(key):
            raise ValueError("C L1 configuration lacks {}".format(key))
    for key in ("det_confidence", "every_n_steps", "max_items",
                "prompt_max_chars", "builder"):
        if key not in l1_block:
            raise ValueError("C L1 configuration lacks {}".format(key))
    if l1_block.get("builder") != "local_graph_fusion":
        raise ValueError("C L1 is not the original LocalGraphFusion builder")
    if l1_block.get("prompt_max_chars") != 320:
        raise ValueError("C L1 does not keep the original 320-character fallback")
    if l1_block.get("max_items") != 2:
        raise ValueError("C L1 does not use the original two-landmark card")
    budget = l1_block.get("budget")
    if not isinstance(budget, dict):
        raise ValueError("C L1 configuration lacks the token-budget protocol")
    if budget.get("strategy") != "token_aware_context":
        raise ValueError("C L1 does not use the token-aware prompt budget")
    if budget.get("token_aware") is not True:
        raise ValueError("C L1 token budget is not enabled")
    if budget.get("original_char_budget_chars") != 320:
        raise ValueError("C L1 original-character fallback differs")
    if budget.get("generation_reserve") != "max_new_tokens":
        raise ValueError("C L1 reserves a value other than max_new_tokens")
    if budget.get("safety_margin_tokens") != 1:
        raise ValueError("C L1 safety margin is not the original one token")
    if budget.get("whole_lines_only") is not True:
        raise ValueError("C L1 token budget may cut a line")
    if budget.get("visual_token_accounting") != (
        "online_merge_upper_bound_from_actual_cache"
    ):
        raise ValueError("C L1 budget does not account for visual tokens")
    if budget.get("max_landmarks") != l1_block.get("max_items"):
        raise ValueError("C L1 budget landmark count differs")
    if budget.get("fallback") != "original_char_budget_with_reported_reason":
        raise ValueError("C L1 fallback protocol differs")
    if l1_block.get("fusion") != "original_l1_plus_l3_sbadir":
        raise ValueError("C L1 fusion mode differs")

    # No L1 weight path may appear anywhere in A's or B's run configuration.
    for name, cfg in (("A", a), ("B", b)):
        serialized = json.dumps(cfg, sort_keys=True)
        for key in ("l1_det_model", "l1_place_model"):
            value = launch.get(key)
            if value and value in serialized:
                raise ValueError(
                    "{} run_config records L1 weight path {}: {}".format(name, key, value)
                )

    if launch.get("baseline_commit") != BASELINE_SHA:
        raise ValueError("launch manifest does not identify exact 7602b25 baseline")
    if launch.get("official_prompt_sha256") != PROMPT_SHA:
        raise ValueError("launch manifest official Prompt SHA256 differs")
    if launch.get("pilot_manifest_file_sha256") != EXPECTED_PILOT_MANIFEST_SHA256:
        raise ValueError("launch manifest Pilot SHA256 differs")
    if launch.get("episode_manifest_canonical_sha256") != EXPECTED_EPISODE_MANIFEST_SHA256:
        raise ValueError("launch manifest Episode SHA256 differs")
    if launch.get("model_path") != a.get("model_path"):
        raise ValueError("launch and run checkpoint paths differ")
    if launch.get("chunks") != a.get("split_num"):
        raise ValueError("launch and run shard counts differ")
    if launch.get("temperature") != 0.2 or launch.get("max_new_tokens") != 1024:
        raise ValueError("official generation parameters differ")
    if list(launch.get("groups") or []) != [A_GROUP, B_GROUP, C_GROUP]:
        raise ValueError("launch manifest group order differs")
    source_hashes = _b_source_hashes(Path(launch["dev_worktree"]))
    if source_hashes != launch.get("b_source_files_sha256"):
        raise ValueError("B/C source files changed after launch")
    source_digest = _b_source_digest(source_hashes)
    if source_digest != launch.get("b_source_tree_sha256"):
        raise ValueError("B/C source tree digest differs from launch")
    if b_background.get("source_tree_sha256") != source_digest:
        raise ValueError("B run_config lacks the launched source tree digest")
    if c_background.get("source_tree_sha256") != source_digest:
        raise ValueError("C run_config lacks the launched source tree digest")
    if l1_block.get("source_tree_sha256") != source_digest:
        raise ValueError("C L1 block lacks the launched source tree digest")


def _validate_rows(
    a: Dict[Tuple[str, str], Dict[str, Any]],
    b: Dict[Tuple[str, str], Dict[str, Any]],
    c: Dict[Tuple[str, str], Dict[str, Any]],
    expected_keys: Iterable[Tuple[str, str]],
    *, expected_count: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    expected = set(expected_keys)
    if (len(expected) != expected_count
            or set(a) != expected or set(b) != expected or set(c) != expected):
        raise ValueError("A/B/C Episode identities differ from the frozen Manifest")
    triples = []
    for key in sorted(expected):
        ra, rb, rc = a[key], b[key], c[key]
        if (ra.get("group") != A_GROUP or rb.get("group") != B_GROUP
                or rc.get("group") != C_GROUP):
            raise ValueError("result group mismatch at {}".format(key))
        if not (ra.get("instruction_sha256") == rb.get("instruction_sha256")
                == rc.get("instruction_sha256")):
            raise ValueError("A/B/C instruction differs at {}".format(key))
        if not (ra.get("episode_seed") == rb.get("episode_seed")
                == rc.get("episode_seed")):
            raise ValueError("A/B/C Episode seed differs at {}".format(key))
        for name, row in (("A", ra), ("B", rb), ("C", rc)):
            if not isinstance(row.get("success"), bool):
                raise ValueError("{} lacks Boolean success at {}".format(name, key))
            if row.get("official_prompt_sha256") != PROMPT_SHA:
                raise ValueError("{} official Prompt hash differs at {}".format(name, key))
            if row.get("pilot_manifest_file_sha256") != EXPECTED_PILOT_MANIFEST_SHA256:
                raise ValueError("{} Pilot digest differs at {}".format(name, key))
            if row.get("episode_manifest_canonical_sha256") != EXPECTED_EPISODE_MANIFEST_SHA256:
                raise ValueError("{} Episode digest differs at {}".format(name, key))
            if row.get("episode_manifest_verified") is not True:
                raise ValueError("{} Episode Manifest not verified at {}".format(name, key))
        _int_nonnegative(ra, "l3_fact_count_total")
        _int_nonnegative(ra, "prompt_has_l3_steps")
        # A must come from the 7602b25 serializer: no source background/L1 keys.
        for field in ("source_background_facts_total", "l1_fact_count_total",
                      "l1_enabled_steps", "prompt_has_l1_steps",
                      "l3_original_memory_unchanged"):
            if field in ra:
                raise ValueError(
                    "A result contains {} at {}: not the 7602b25 runtime".format(field, key)
                )

        for field in REQUIRED_B_FIELDS:
            if field not in rb:
                raise ValueError("B result lacks required diagnostic {} at {}".format(field, key))
        for field in REQUIRED_B_FIELDS:
            if field in ("l3_context_tokens_mean", "l3_original_memory_unchanged"):
                continue
            _int_nonnegative(rb, field)
        if rb["l3_original_memory_unchanged"] is not True:
            raise ValueError("B altered original L3 memory at {}".format(key))
        if _int_nonnegative(rb, "l1_fact_count_total") != 0:
            raise ValueError("B injected an L1 fact at {}".format(key))
        if _int_nonnegative(rb, "l1_retained_facts_total") != 0:
            raise ValueError("B retained an L1 fact at {}".format(key))
        if _int_nonnegative(rb, "prompt_has_l1_steps") != 0:
            raise ValueError("B prompt contains L1 at {}".format(key))
        if _int_nonnegative(rb, "l1_enabled_steps") != 0:
            raise ValueError("B observed RGB through L1 at {}".format(key))
        if _int_nonnegative(rb, "l1_observation_attempt_steps") != 0:
            raise ValueError("B attempted L1 observation at {}".format(key))
        if _int_nonnegative(rb, "duplicated_facts_total") != 0:
            raise ValueError("B counted a cross-layer duplicate at {}".format(key))
        if _int_nonnegative(rb, "conflicting_facts_total") != 0:
            raise ValueError("B counted a cross-layer conflict at {}".format(key))
        l3_total = _int_nonnegative(rb, "l3_fact_count_total")
        if _int_nonnegative(rb, "source_background_facts_total") != l3_total:
            raise ValueError("B did not source-pair every L3 fact at {}".format(key))
        if _int_nonnegative(rb, "l3_provenance_count_total") != l3_total:
            raise ValueError("B L3 source-background count differs at {}".format(key))

        for field in REQUIRED_C_FIELDS:
            if field not in rc:
                raise ValueError("C result lacks required diagnostic {} at {}".format(field, key))
        token_steps_c = _int_nonnegative(rc, "l1_token_budget_steps")
        for field in REQUIRED_C_FIELDS:
            if field in ("l3_context_tokens_mean", "l1_context_tokens_mean",
                         "text_prompt_tokens_mean", "visual_tokens_mean",
                         "multimodal_prompt_tokens_mean",
                         "l3_original_memory_unchanged"):
                continue
            if field == "context_margin_tokens_min" and token_steps_c == 0:
                continue
            _int_nonnegative(rc, field)
        if rc["l3_original_memory_unchanged"] is not True:
            raise ValueError("C altered original L3 memory at {}".format(key))
        c_l3 = _int_nonnegative(rc, "l3_fact_count_total")
        c_l1 = _int_nonnegative(rc, "l1_fact_count_total")
        c_l1_retained = _int_nonnegative(rc, "l1_retained_facts_total")
        c_l1_built = _int_nonnegative(rc, "l1_prompt_built_steps")
        c_l1_prompt = _int_nonnegative(rc, "prompt_has_l1_steps")
        if c_l1_retained > c_l1:
            raise ValueError("C retained more L1 facts than selected at {}".format(key))
        if c_l1 == 0:
            if c_l1_retained or c_l1_built or c_l1_prompt:
                raise ValueError("C has an L1 card without an L1 fact at {}".format(key))
        else:
            if c_l1_built <= 0:
                raise ValueError("C built no original L1 card at {}".format(key))
            if c_l1_retained <= 0:
                raise ValueError("C did not inject a selected L1 fact at {}".format(key))
            if c_l1_prompt <= 0:
                raise ValueError("C prompt lacks the L1 card at {}".format(key))
        # C's SBAdir source background covers B's frozen L3 facts only.  The
        # original Local Graph L1 card is inserted verbatim without a suffix.
        if _int_nonnegative(rc, "source_background_facts_total") != c_l3:
            raise ValueError("C did not source-pair every L3 fact at {}".format(key))
        if _int_nonnegative(rc, "l3_provenance_count_total") != c_l3:
            raise ValueError("C L3 source-background count differs at {}".format(key))
        if _int_nonnegative(rc, "l1_enabled_steps") != _int_nonnegative(rc, "env_steps"):
            raise ValueError("C did not observe through L1 on every step at {}".format(key))
        token_steps = _int_nonnegative(rc, "l1_token_budget_steps")
        fallback_steps = _int_nonnegative(rc, "l1_token_budget_fallback_steps")
        if token_steps + fallback_steps > _int_nonnegative(rc, "env_steps"):
            raise ValueError("C token-budget step counts exceed env_steps at {}".format(key))
        if _int_nonnegative(rc, "token_overflow_drops_total") < 0:
            raise ValueError("C token overflow count is negative at {}".format(key))
        if _int_nonnegative(rc, "memory_budget_violation_steps") != 0:
            raise ValueError("C injected more L1 tokens than the budget at {}".format(key))
        if _int_nonnegative(rc, "multimodal_budget_violation_steps") != 0:
            raise ValueError("C multimodal prompt exceeded context reserve at {}".format(key))
        if _int_nonnegative(rc, "context_margin_tokens_min") < 0:
            raise ValueError("C multimodal context margin is negative at {}".format(key))
        if token_steps_c > 0:
            for field in ("text_prompt_tokens_mean", "visual_tokens_mean",
                          "multimodal_prompt_tokens_mean"):
                value = rc.get(field)
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError("C lacks a valid {} at {}".format(field, key))
        triples.append((ra, rb, rc))
    return triples


def _historical(path: Optional[Path], expected_keys: set) -> Dict[str, Any]:
    result = {
        "claimed_successes": HISTORICAL_SUCCESSES,
        "claimed_episodes": HISTORICAL_EPISODES,
        "claimed_sr_percent": HISTORICAL_SUCCESSES / HISTORICAL_EPISODES * 100.0,
        "historical_artifacts_verified": False,
        "observed_successes": None,
        "observed_sr_percent": None,
        "source": None if path is None else str(path),
        "missing_artifacts": [],
    }
    if path is None:
        result["missing_artifacts"] = ["historical run directory not supplied"]
        return result
    needed = ("run_config.json", "console.log", "results.jsonl", "execution_manifest.json")
    missing = [name for name in needed if not (path / name).is_file()]
    result["missing_artifacts"] = missing
    if not (path / "results.jsonl").is_file():
        return result
    rows = _rows(path / "results.jsonl")
    result["results_sha256"] = _sha256(path / "results.jsonl")
    result["episode_set_matches"] = set(rows) == expected_keys
    result["observed_successes"] = sum(row.get("success") is True for row in rows.values())
    result["observed_sr_percent"] = (
        100.0 * result["observed_successes"] / len(rows) if rows else None
    )
    if not missing and set(rows) == expected_keys and len(rows) == HISTORICAL_EPISODES:
        cfg = _json(path / "run_config.json")
        result["historical_artifacts_verified"] = (
            cfg.get("pilot_manifest_file_sha256") == EXPECTED_PILOT_MANIFEST_SHA256
            and cfg.get("execution_manifest_canonical_sha256") == EXPECTED_EPISODE_MANIFEST_SHA256
        )
    return result


def _step_audit(path: Path) -> Dict[str, Any]:
    """Check each original L3 line is an unchanged prefix of an enriched line."""
    if not path.is_file():
        raise FileNotFoundError(str(path))
    observed = 0
    observed_l1 = 0
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("{}:{} is not a step object".format(path, number))
            if row.get("prompt_has_l3"):
                l3 = row.get("l3_context_text")
                prompt = row.get("prompt_text")
                if not isinstance(l3, str) or not l3 or not isinstance(prompt, str):
                    raise ValueError("L3 prompt step lacks logged original text")
                expected_sha = hashlib.sha256(l3.encode("utf-8")).hexdigest()
                if row.get("l3_original_sha256") != expected_sha:
                    raise ValueError("original L3 digest differs at step {}".format(number))
                original_lines = [line for line in l3.splitlines() if line]
                prompt_lines = prompt.splitlines()
                search_from = 0
                for original in original_lines:
                    matches = [index for index in range(search_from, len(prompt_lines))
                               if prompt_lines[index].startswith(original)]
                    if len(matches) != 1:
                        raise ValueError("original L3 line missing or duplicated at step {}".format(number))
                    search_from = matches[0] + 1
                observed += 1
            if row.get("prompt_has_l1"):
                l1 = row.get("l1_context_text")
                prompt = row.get("prompt_text")
                if not isinstance(l1, str) or not l1.startswith("[Navigation Memory]\n"):
                    raise ValueError("L1 prompt step lacks the original header at step {}".format(number))
                budget_source = row.get("budget_source") or (
                    row.get("l1_selection") or {}
                ).get("budget_source")
                if budget_source == "token_aware_context":
                    available = row.get("available_memory_tokens")
                    injected = row.get("memory_tokens_injected")
                    if (
                        isinstance(available, int) and isinstance(injected, int)
                        and injected > available
                    ):
                        raise ValueError(
                            "L1 memory tokens exceed the measured budget at step {}".format(number)
                        )
                    text_tokens = row.get("text_prompt_tokens")
                    visual_tokens = row.get("visual_tokens")
                    base_tokens = row.get("base_prompt_tokens")
                    if all(isinstance(value, int) for value in (
                        text_tokens, visual_tokens, base_tokens,
                    )) and base_tokens != text_tokens + visual_tokens:
                        raise ValueError(
                            "multimodal base token accounting differs at step {}".format(number)
                        )
                    limit = row.get("multimodal_prompt_limit_tokens")
                    final = row.get("multimodal_prompt_tokens")
                    if isinstance(limit, int) and isinstance(final, int):
                        if final > limit:
                            raise ValueError(
                                "multimodal prompt exceeds context reserve at step {}".format(number)
                            )
                        margin = row.get("context_margin_tokens")
                        if isinstance(margin, int) and margin != limit - final:
                            raise ValueError(
                                "multimodal context margin differs at step {}".format(number)
                            )
                    if row.get("multimodal_budget_overflow") is True:
                        raise ValueError(
                            "multimodal budget overflow flag set at step {}".format(number)
                        )
                elif len(l1) > 320:
                    raise ValueError(
                        "L1 fallback card exceeds the original 320-character budget at step {}".format(number)
                    )
                if not isinstance(prompt, str) or l1 not in prompt:
                    raise ValueError("original L1 card is not verbatim in the prompt at step {}".format(number))
                if row.get("l1_prompt_chars") != len(l1):
                    raise ValueError("L1 card length differs at step {}".format(number))
                expected_sha = hashlib.sha256(l1.encode("utf-8")).hexdigest()
                if row.get("l1_prompt_sha256") != expected_sha:
                    raise ValueError("original L1 digest differs at step {}".format(number))
                if row.get("l1_budget_chars") != 320:
                    raise ValueError("L1 budget differs at step {}".format(number))
                observed_l1 += 1
    return {
        "l3_prompt_steps_checked": observed,
        "original_l3_line_prefixes_preserved": True,
        "l1_prompt_steps_checked": observed_l1,
        "original_l1_cards_verbatim": True,
    }


def _ordered_result_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _steps_by_identity(result_path: Path, steps_path: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Attach step segments to result identities without using reset order to pair."""
    results = _ordered_result_rows(result_path)
    segments = []  # type: List[List[Dict[str, Any]]]
    for number, line in enumerate(steps_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or isinstance(row.get("step"), bool) or not isinstance(row.get("step"), int):
            raise ValueError("invalid step row {} in {}".format(number, steps_path))
        if row["step"] == 0:
            segments.append([])
        if not segments or row["step"] != len(segments[-1]):
            raise ValueError("step sequence cannot be attached to Episodes in {}".format(steps_path))
        segments[-1].append(row)
    if len(segments) != len(results):
        raise ValueError("step segments and result rows differ in {}".format(steps_path))
    return {_key(result): segment for result, segment in zip(results, segments)}


#: L3 observation/retrieval fields that must not depend on L1 presence.
#: ``evidence_unmatched`` is excluded: with an L1 backend, a same-class L1 box
#: turns an "unmatched" L3 mask into a "linked" one, which is C's intended
#: entity association and does not change the mask, the map, or any fact.
#: ``context_injected`` is excluded because C can inject L1-only memory when
#: the L3 organizer selected no fact; ``context_available``/``fact_count`` are
#: the L3-only fields and are compared.
L3_STATE_KEYS = (
    "frame_valid", "segmentation_count", "segmentation_error",
    "evidence_minted", "observed_objects", "object_nodes", "viewpoints",
    "observed_from_edges", "traversed_to_edges", "context_available",
    "candidate_count", "verified_count",
    "unverified_count", "current_only_skipped", "omitted_items", "fact_count",
)


def _l3_state(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: row.get(key) for key in L3_STATE_KEYS}


def _paired_l3_step_audit(
    left_dir: Path, right_dir: Path, *, right_name: str,
) -> Dict[str, Any]:
    """Compare the L3 observation/fact state by Episode identity.

    Comparison stops at the first action divergence: from that step on the two
    runs observed different RGB-D histories, so their maps are not required to
    be identical.  Everything before the divergence must match: the L3 map and
    retrieval diagnostics of every common step, and the exact original L3 text
    of every common step that injected a fact.  B/C enriched source
    backgrounds are compared when both runs logged them.
    """
    left_steps = left_dir / "steps.jsonl"
    right_steps = right_dir / "steps.jsonl"
    if not left_steps.is_file() or not right_steps.is_file():
        return {
            "status": "unavailable",
            "reason": "{} or {} steps.jsonl is missing".format(
                left_dir.name, right_name),
            "comparable_steps": 0,
            "comparable_l3_state_steps": 0,
            "all_comparable_l3_state_equal": None,
            "comparable_nonempty_l3_steps": 0,
            "matching_nonempty_l3_steps": 0,
            "episodes_with_action_divergence": 0,
            "all_comparable_l3_text_equal": None,
            "comparable_l3_enriched_steps": 0,
            "all_comparable_l3_enriched_equal": None,
        }
    left = _steps_by_identity(left_dir / "results.jsonl", left_steps)
    right = _steps_by_identity(right_dir / "results.jsonl", right_steps)
    if set(left) != set(right):
        raise ValueError("step Episode identities differ: {} vs {}".format(
            left_dir.name, right_dir.name))
    comparable_steps = 0
    state_comparable = 0
    state_matching = 0
    comparable = 0
    matching = 0
    divergent_episodes = 0
    enriched_comparable = 0
    enriched_matching = 0
    for key in sorted(left):
        prior_actions_equal = True
        for left_row, right_row in zip(left[key], right[key]):
            if not prior_actions_equal:
                break
            comparable_steps += 1
            # The L3 map/fact-selection state is a trajectory-independent
            # fairness check: C's L1 links must not change it.
            if any(name in left_row or name in right_row
                   for name in L3_STATE_KEYS):
                state_comparable += 1
                if _l3_state(left_row) != _l3_state(right_row):
                    raise ValueError(
                        "L3 map/fact-selection state differs before action "
                        "divergence at {}".format(key))
                state_matching += 1
            left_text = left_row.get("l3_context_text")
            right_text = right_row.get("l3_context_text")
            if (left_row.get("prompt_text") is not None
                    and right_row.get("prompt_text") is not None):
                if left_text or right_text:
                    comparable += 1
                    if left_text != right_text:
                        raise ValueError(
                            "original L3 text differs before action divergence at {}".format(key))
                    matching += 1
                left_sha = left_row.get("l3_enriched_sha256")
                right_sha = right_row.get("l3_enriched_sha256")
                if isinstance(left_sha, str) and isinstance(right_sha, str):
                    enriched_comparable += 1
                    if left_sha != right_sha:
                        raise ValueError(
                            "L3 source background differs before action divergence at {}".format(key))
                    enriched_matching += 1
            if left_row.get("action") != right_row.get("action"):
                prior_actions_equal = False
                divergent_episodes += 1
    return {
        "status": (
            "verified_on_common_action_prefix"
            if state_comparable and comparable else "partial_on_common_action_prefix"
        ),
        "comparable_steps": comparable_steps,
        "comparable_l3_state_steps": state_comparable,
        "matching_l3_state_steps": state_matching,
        "all_comparable_l3_state_equal": (
            state_matching == state_comparable if state_comparable else None
        ),
        "comparable_nonempty_l3_steps": comparable,
        "matching_nonempty_l3_steps": matching,
        "episodes_with_action_divergence": divergent_episodes,
        "all_comparable_l3_text_equal": matching == comparable if comparable else None,
        "comparable_l3_enriched_steps": enriched_comparable,
        "matching_l3_enriched_steps": enriched_matching,
        "all_comparable_l3_enriched_equal": (
            enriched_matching == enriched_comparable if enriched_comparable else None
        ),
    }


def _smoke_audit(output_dir: Path, launch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not launch.get("smoke_required"):
        return None
    directories = launch.get("smoke_directories") or {
        A_GROUP: "smoke_" + A_NAME,
        B_GROUP: "smoke_" + B_NAME,
        C_GROUP: "smoke_" + C_NAME,
    }
    records = {}
    for group in (A_GROUP, B_GROUP, C_GROUP):
        directory = output_dir / directories[group]
        check = _json(directory / "smoke_validation.json")
        rows = _rows(directory / "results.jsonl")
        if len(rows) != 1:
            raise ValueError("{} smoke must contain exactly one Episode".format(group))
        key = next(iter(rows))
        row = rows[key]
        if list(check.get("episode", [])) != list(key):
            raise ValueError("{} smoke validation identity differs".format(group))
        if check.get("group") != group or row.get("group") != group:
            raise ValueError("{} smoke group differs".format(group))
        if check.get("official_prompt_sha256") != PROMPT_SHA:
            raise ValueError("{} smoke did not preserve the official Prompt".format(group))
        if row.get("official_prompt_sha256") != PROMPT_SHA:
            raise ValueError("{} smoke row Prompt hash differs".format(group))
        records[group] = {"episode": list(key), "row": row, "validation": check}
    identity = records[A_GROUP]["episode"]
    for group in (B_GROUP, C_GROUP):
        if records[group]["episode"] != identity:
            raise ValueError("A/B/C smokes used different Episodes")
        if records[group]["row"].get("episode_seed") != records[A_GROUP]["row"].get("episode_seed"):
            raise ValueError("A/B/C smoke seeds differ")
        if (records[group]["row"].get("instruction_sha256")
                != records[A_GROUP]["row"].get("instruction_sha256")):
            raise ValueError("A/B/C smoke instructions differ")

    a = records[A_GROUP]["row"]
    for field in ("source_background_facts_total", "l1_fact_count_total",
                  "l1_enabled_steps", "prompt_has_l1_steps",
                  "l3_original_memory_unchanged"):
        if field in a:
            raise ValueError("A smoke contains {}: not the 7602b25 runtime".format(field))
    if a.get("l3_fact_count_total", 0) <= 0 or a.get("l3_map_steps", 0) <= 0:
        raise ValueError("A smoke did not exercise the L3 Organized prompt")

    b = records[B_GROUP]["row"]
    if b.get("source_background_enabled") is not True:
        raise ValueError("B smoke did not enable the source background")
    if b.get("l1_enabled") is not False or b.get("l1_enabled_steps", -1) != 0:
        raise ValueError("B smoke enabled L1")
    if b.get("l1_observation_attempt_steps", -1) != 0:
        raise ValueError("B smoke attempted L1 observation")
    if b.get("l1_fact_count_total", -1) != 0 or b.get("prompt_has_l1_steps", -1) != 0:
        raise ValueError("B smoke injected an L1 fact")
    if b.get("l3_fact_count_total", 0) <= 0:
        raise ValueError("B smoke produced no L3 fact")
    if b.get("source_background_facts_total") != b.get("l3_fact_count_total"):
        raise ValueError("B smoke did not pair every L3 fact with its source")
    if b.get("l3_original_memory_unchanged") is not True:
        raise ValueError("B smoke altered original L3 memory")
    if b.get("l3_segmentation_error_count", 0) != 0:
        raise ValueError("B smoke encountered L3 segmentation errors")

    c = records[C_GROUP]["row"]
    if c.get("source_background_enabled") is not True:
        raise ValueError("C smoke did not enable the source background")
    if c.get("l1_enabled") is not True:
        raise ValueError("C smoke did not enable L1")
    if c.get("l1_enabled_steps") != c.get("env_steps"):
        raise ValueError("C smoke did not run L1 on every step")
    # The original LocalGraphFusion card may legitimately be empty for one
    # Episode.  Only require prompt facts when the L1 builder produced them.
    l1_facts = int(c.get("l1_fact_count_total", 0) or 0)
    l1_retained = int(c.get("l1_retained_facts_total", 0) or 0)
    l1_built = int(c.get("l1_prompt_built_steps", 0) or 0)
    l1_prompt_steps = int(c.get("prompt_has_l1_steps", 0) or 0)
    if l1_facts == 0:
        if l1_retained or l1_built or l1_prompt_steps:
            raise ValueError("C smoke built/injected an L1 card without an L1 fact")
    else:
        if l1_built <= 0:
            raise ValueError("C smoke produced L1 facts but never built the original L1 card")
        if l1_retained <= 0:
            raise ValueError("C smoke selected L1 facts but did not inject them verbatim")
        if l1_prompt_steps <= 0:
            raise ValueError("C smoke selected L1 facts but no prompt step contains the L1 card")
    # SBAdir pairs B's frozen L3 facts; the original L1 card is verbatim.
    if c.get("source_background_facts_total") != c.get("l3_fact_count_total", 0):
        raise ValueError("C smoke did not pair every L3 fact with its source")
    if c.get("l3_original_memory_unchanged") is not True:
        raise ValueError("C smoke altered original L3 memory")
    for key in ("l1_observe_error_count", "l1_selection_error_count",
                "l1_detector_error_count",
                "l1_place_error_count", "l3_segmentation_error_count",
                "cross_label_entity_link_error_count"):
        if c.get(key, 0) != 0:
            raise ValueError("C smoke encountered {}".format(key))
    for key in ("l3_map_steps", "l3_fact_count_total", "prompt_has_l3_steps",
                "l1_place_attempt_steps", "l1_place_observed_steps",
                "l1_events_persisted_total"):
        if c.get(key, 0) <= 0:
            raise ValueError("C smoke did not exercise {}".format(key))
    return {
        "episode": identity,
        "groups": {
            A_GROUP: {
                "l3_fact_count_total": a.get("l3_fact_count_total"),
                "source_background_facts_total": None,
                "l1_fact_count_total": None,
            },
            B_GROUP: {
                "l3_fact_count_total": b.get("l3_fact_count_total"),
                "source_background_facts_total": b.get("source_background_facts_total"),
                "l1_fact_count_total": b.get("l1_fact_count_total"),
            },
            C_GROUP: {
                "l3_fact_count_total": c.get("l3_fact_count_total"),
                "source_background_facts_total": c.get("source_background_facts_total"),
                "l1_fact_count_total": c.get("l1_fact_count_total"),
            },
        },
    }


def _comparison(
    left: Sequence[Dict[str, Any]], right: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    result = []
    for label, key, scale, unit, direction in METRICS:
        left_value = mean_or_none(
            value * scale for row in left if (value := _metric_value(row, key)) is not None
        )
        right_value = mean_or_none(
            value * scale for row in right if (value := _metric_value(row, key)) is not None
        )
        result.append({
            "metric": label, "unit": unit,
            "left": left_value, "right": right_value,
            "delta_right_minus_left": (
                None if left_value is None or right_value is None
                else right_value - left_value
            ),
            "preferred_direction": direction,
        })
    return result


def _paired_records(
    triples: Sequence[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    records = []
    keys = (
        ("success", "success"), ("spl", "spl"), ("ndtw", "ndtw"),
        ("ne", "distance_to_goal"), ("osr", "oracle_success"),
        ("tl", "path_length"), ("steps", "steps_taken"),
        ("inference_latency_ms", "inference_latency_ms_mean"),
        ("prompt_tokens", "prompt_tokens_mean"),
        ("memory_facts", "memory_fact_count_total"),
        ("conflicts", "conflicting_facts_total"),
        ("source_paired_facts", "source_background_facts_total"),
    )
    for a, b, c in triples:
        record = {"scene_basename": a["scene_basename"],
                  "episode_id": a["episode_id"],
                  "instruction_sha256": a["instruction_sha256"]}
        for label, field in keys:
            av = _metric_value(a, field)
            bv = _metric_value(b, field)
            cv = _metric_value(c, field)
            record[label + "_a"] = av
            record[label + "_b"] = bv
            record[label + "_c"] = cv
            record["delta_" + label + "_b_minus_a"] = (
                None if av is None or bv is None else bv - av)
            record["delta_" + label + "_c_minus_b"] = (
                None if bv is None or cv is None else cv - bv)
            record["delta_" + label + "_c_minus_a"] = (
                None if av is None or cv is None else cv - av)
        record["l3_fact_count_a"] = a.get("l3_fact_count_total")
        record["l3_fact_count_b"] = b.get("l3_fact_count_total")
        record["l3_fact_count_c"] = c.get("l3_fact_count_total")
        record["source_background_facts_b"] = b.get("source_background_facts_total")
        record["source_background_facts_c"] = c.get("source_background_facts_total")
        record["l1_fact_count_c"] = c.get("l1_fact_count_total")
        record["l3_memory_unchanged_b"] = b.get("l3_original_memory_unchanged")
        record["l3_memory_unchanged_c"] = c.get("l3_original_memory_unchanged")
        records.append(record)
    return records


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    return "{:.3f}".format(value) if isinstance(value, float) else str(value)


def _report_text(summary: Dict[str, Any]) -> str:
    groups = summary["groups"]
    a, b, c = groups[A_NAME], groups[B_NAME], groups[C_NAME]
    hist = summary["history"]
    integrity = summary["integrity"]
    comparisons = summary["comparisons"]
    flips = summary["paired"]
    lines = [
        "# Three-group L3 comparison: Organized L3, SBAdir source background, L1",
        "",
        "This report covers the frozen 88-Episode Pilot. A is the untouched",
        ("`{}` L3 Organized baseline. B keeps exactly the same L3 facts and "
         "pairs each with its real source record. C keeps B's L3 facts/source "
         "background and adds the L1 RGB memory through the same fusion."
         ).format(BASELINE_SHA),
        "",
        "| Metric | A L3 Organized | B L3+Source | C L3+Source+L1 | B-A | C-B | C-A |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_metric = {
        name: {row["metric"]: row for row in comparisons[name]}
        for name in ("a_vs_b", "b_vs_c", "a_vs_c")
    }
    for label, _key, _scale, unit, _direction in METRICS:
        row_ab = by_metric["a_vs_b"][label]
        lines.append("| {} ({}) | {} | {} | {} | {} | {} | {} |".format(
            label, unit, _format(row_ab["left"]), _format(row_ab["right"]),
            _format(by_metric["b_vs_c"][label]["right"]),
            _format(row_ab["delta_right_minus_left"]),
            _format(by_metric["b_vs_c"][label]["delta_right_minus_left"]),
            _format(by_metric["a_vs_c"][label]["delta_right_minus_left"]),
        ))
    lines.extend([
        "",
        "All A/B/C rows are paired by `(scene_basename, episode_id)`; execution",
        "order is never used. Memory facts and conflicts count fact uses across",
        "steps, averaged per Episode.",
        "",
        "## Fairness checks",
        "",
        "- B has no L1 configuration, no L1 fact, no L1 prompt step, and no L1",
        "  RGB observation: {}.".format(integrity["b_has_no_l1"]),
        "- Every B L3 fact is paired with a real source record: {}.".format(
            integrity["b_source_background_present"]),
        "- L3 map/fact-selection state is identical on the common action prefix:",
        "  A-B {} ({} steps), B-C {} ({} steps), A-C {} ({} steps).".format(
            _format(integrity["a_b_l3_state_equal_on_common_prefix"]),
            integrity["a_b_step_audit"]["comparable_l3_state_steps"],
            _format(integrity["b_c_l3_state_equal_on_common_prefix"]),
            integrity["b_c_step_audit"]["comparable_l3_state_steps"],
            _format(integrity["a_c_l3_state_equal_on_common_prefix"]),
            integrity["a_c_step_audit"]["comparable_l3_state_steps"]),
        "- A/B original L3 text is identical on the common action prefix: {} ({} nonempty steps).".format(
            _format(integrity["a_b_l3_text_equal_on_common_prefix"]),
            integrity["a_b_step_audit"]["comparable_nonempty_l3_steps"]),
        "- B/C original L3 text is identical on the common action prefix: {} ({} nonempty steps).".format(
            _format(integrity["b_c_l3_text_equal_on_common_prefix"]),
            integrity["b_c_step_audit"]["comparable_nonempty_l3_steps"]),
        "- B/C enriched L3 source background is identical on the common action",
        "  prefix: {} ({} steps).".format(
            _format(integrity["b_c_l3_background_equal_on_common_prefix"]),
            integrity["b_c_step_audit"]["comparable_l3_enriched_steps"]),
        "- A/C original L3 text is identical on the common action prefix: {}.".format(
            _format(integrity["a_c_l3_text_equal_on_common_prefix"])),
        "- C adds L1 on top of B: {}.".format(integrity["c_adds_l1_to_b"]),
        "- C keeps B's source-background fusion: {}.".format(
            integrity["c_keeps_b_source_background"]),
        "- Source fields are evidence IDs, steps, map nodes and recorded",
        "  confidences only; no sensor trust score is computed.",
        "",
        "## Historical 55.68% baseline",
        "",
        "- Claimed historical result: 49/88 = 55.6818%.",
        "- A result: {}/88 = {}%.".format(a["successes"], _format(a["SR"])),
        "- A matches the claimed 49/88 count: {}.".format(
            integrity["a_matches_claimed_49_of_88"]),
        "- Historical artifacts verified: {}.".format(hist["historical_artifacts_verified"]),
    ])
    if hist["observed_successes"] is not None:
        lines.append("- Historical results file: {}/{} successes ({}%).".format(
            hist["observed_successes"], HISTORICAL_EPISODES,
            _format(hist["observed_sr_percent"])))
    if hist["missing_artifacts"]:
        lines.append("- Missing historical artifacts: {}.".format(
            ", ".join(hist["missing_artifacts"])))
    if a["successes"] != HISTORICAL_SUCCESSES or (
        hist["historical_artifacts_verified"]
        and hist["observed_successes"] != HISTORICAL_SUCCESSES
    ):
        lines.append("- **Historical baseline not reproduced.** Do not claim the "
                     "55.68% result was established in this run.")
    elif not hist["historical_artifacts_verified"]:
        lines.append("- The 49/88 count matches the supplied claim, but the "
                     "historical source run remains unverified.")
    lines.extend([
        "",
        "## Real GPU smoke",
        "",
        "- A/B/C one-Episode smoke: {}.".format(
            "passed on {}".format(summary["smoke"]["episode"])
            if summary["smoke"] is not None else "not attached to this report"),
        "",
        "## Paired Episode changes",
        "",
        "- A fail → B success: {}; A success → B fail: {} (McNemar p={}).".format(
            flips["a_vs_b"]["success_flips"]["baseline_fail_l3_success"],
            flips["a_vs_b"]["success_flips"]["baseline_success_l3_fail"],
            _format(flips["a_vs_b"]["success_flips"]["mcnemar"]["p_value"])),
        "- B fail → C success: {}; B success → C fail: {} (McNemar p={}).".format(
            flips["b_vs_c"]["success_flips"]["baseline_fail_l3_success"],
            flips["b_vs_c"]["success_flips"]["baseline_success_l3_fail"],
            _format(flips["b_vs_c"]["success_flips"]["mcnemar"]["p_value"])),
        "- A fail → C success: {}; A success → C fail: {} (McNemar p={}).".format(
            flips["a_vs_c"]["success_flips"]["baseline_fail_l3_success"],
            flips["a_vs_c"]["success_flips"]["baseline_success_l3_fail"],
            _format(flips["a_vs_c"]["success_flips"]["mcnemar"]["p_value"])),
        "",
        "## Memory and source evidence",
        "",
        "- L3 facts across steps A/B/C: {} / {} / {}.".format(
            a["l3_fact_count_total"], b["l3_fact_count_total"],
            c["l3_fact_count_total"]),
        "- Source-paired facts B: {} (all of B's L3 facts); C: {}.".format(
            b["source_background_facts_total"], c["source_background_facts_total"]),
        "- L1 facts injected C: {} across {} prompt steps.".format(
            c["l1_fact_count_total"], c["prompt_has_l1_steps"]),
        "- Mean L3 memory tokens per Episode A/B/C: {} / {} / {}.".format(
            _format(a["l3_context_tokens_mean"]), _format(b["l3_context_tokens_mean"]),
            _format(c["l3_context_tokens_mean"])),
        "- Mean L1 memory tokens per Episode C: {}.".format(
            _format(c["l1_context_tokens_mean"])),
        "- Duplicate facts C: {}; conflicts C: {}; retained: {}; omitted: {}.".format(
            c["duplicated_facts_total"], c["conflicting_facts_total"],
            c["retained_facts_total"], c["omitted_facts_total"]),
        "- L1 RGB observation steps C: {}; DINOv2 place observations: {}.".format(
            c["l1_observation_attempt_steps"], c["l1_place_observed_steps"]),
        "- L1 observe/detector/place errors C: {} / {} / {}; L3 segmentation errors C: {}.".format(
            c["l1_observe_error_count"], c["l1_detector_error_count"],
            c["l1_place_error_count"], c["l3_segmentation_error_count"]),
        "- Conflict types: {}.".format(
            json.dumps(summary["conflict_type_counts"], sort_keys=True)),
        "- Conflict evidence IDs recorded: {}.".format(
            _format(summary["conflict_evidence_id_count"])),
        "- Unique cross-label entity links: {}.".format(
            c["cross_label_entity_links_total"]),
        "",
        "When a same-entity L1/L3 label disagreement is linked by verified",
        "same-frame evidence, both values and both evidence IDs are kept; the",
        "prompt states the shared map entity and never declares a winner.",
        "",
        "## Integrity",
        "",
        "- Same Pilot raw SHA256: `{}`.".format(EXPECTED_PILOT_MANIFEST_SHA256),
        "- Same Episode Manifest canonical SHA256: `{}`.".format(
            EXPECTED_EPISODE_MANIFEST_SHA256),
        "- Same official Prompt template SHA256: `{}`.".format(PROMPT_SHA),
        "- Same checkpoint, seed 42, Habitat configuration and L3 settings:",
        "  verified from the three `run_config.json` files and `launch_manifest.json`.",
        "- A ran the untouched 7602b25 runtime (no L1/source-background fields): {}.".format(
            integrity["a_is_7602b25_runtime"]),
        "- L3 organizer, map core and segmentation source SHA256 match baseline: {}.".format(
            integrity["l3_source_files_match"]),
        "- Visual encoding and generation methods match baseline AST: {}.".format(
            integrity["official_agent_methods_match"]),
        "- B source files equal C source files (one worktree): {}.".format(
            integrity["b_c_same_source_tree"]),
        "- Original L3 memory preserved in every B/C Episode: {} / {}.".format(
            integrity["b_original_l3_memory_preserved"],
            integrity["c_original_l3_memory_preserved"]),
        "- B step audit: {}; C step audit: {}.".format(
            integrity["b_step_audit"]["l3_prompt_steps_checked"],
            integrity["c_step_audit"]["l3_prompt_steps_checked"]),
        "",
        "## Files",
        "",
        "- `summary.json`: validated metrics, three comparisons and integrity.",
        "- `results.jsonl`: the three source streams, each tagged with its group.",
        "- `comparison.csv`: metric table for A vs B, B vs C and A vs C.",
        "- `paired_episode_deltas.csv`: identity-paired Episode deltas.",
        "- `final_report.md`: this report.",
    ])
    return "\n".join(lines) + "\n"


def build_report(
    baseline_dir: Path,
    source_background_dir: Path,
    fused_dir: Path,
    output_dir: Path,
    expected_keys: Iterable[Tuple[str, str]],
    manifest_audit: Dict[str, Any],
    episode_audit: Dict[str, Any],
    launch: Dict[str, Any],
    *,
    historical_dir: Optional[Path] = None,
    expected_count: int = 88,
) -> Dict[str, Any]:
    a_path = baseline_dir / "results.jsonl"
    b_path = source_background_dir / "results.jsonl"
    c_path = fused_dir / "results.jsonl"
    a, b, c = _rows(a_path), _rows(b_path), _rows(c_path)
    _validate_configs(
        _json(baseline_dir / "run_config.json"),
        _json(source_background_dir / "run_config.json"),
        _json(fused_dir / "run_config.json"),
        launch,
    )
    triples = _validate_rows(a, b, c, expected_keys, expected_count=expected_count)
    # Each run's derived execution artifact must have the same frozen identity.
    for name, directory in (("A", baseline_dir), ("B", source_background_dir),
                            ("C", fused_dir)):
        execution = load_episode_manifest(directory / "execution_manifest.json")
        if set(execution.keys()) != set(expected_keys):
            raise ValueError("{} execution Manifest Episode set differs".format(name))
    a_list = [item[0] for item in triples]
    b_list = [item[1] for item in triples]
    c_list = [item[2] for item in triples]
    a_metrics = _group_metrics(a_list)
    b_metrics = _group_metrics(b_list)
    c_metrics = _group_metrics(c_list)
    hist = _historical(historical_dir, set(expected_keys))
    if hist["historical_artifacts_verified"]:
        hist["a_matches_historical_success_count"] = (
            a_metrics["successes"] == hist["observed_successes"])
        historical_rows = _rows(historical_dir / "results.jsonl")
        hist["a_matches_historical_success_by_episode"] = all(
            a[key]["success"] == historical_rows[key]["success"] for key in a
        )
    type_counts = Counter()  # type: Counter[str]
    conflict_evidence_ids = set()  # type: set[str]
    conflict_evidence_records = []  # type: List[Dict[str, Any]]
    evidence_ids_recorded = False
    for row in c_list:
        for kind, count in (row.get("conflict_type_counts") or {}).items():
            if not isinstance(kind, str) or isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("invalid conflict_type_counts")
            type_counts[kind] += count
        if "conflict_evidence_ids" in row:
            evidence_ids_recorded = True
            ids = row["conflict_evidence_ids"]
            if not isinstance(ids, list):
                raise ValueError("invalid conflict_evidence_ids")
            for item in ids:
                if isinstance(item, str):
                    conflict_evidence_ids.add(item)
                elif isinstance(item, dict):
                    for source in ("l1", "l3"):
                        values = item.get(source, [])
                        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                            raise ValueError("invalid {} conflict evidence IDs".format(source))
                        conflict_evidence_ids.update(values)
                    conflict_evidence_records.append(item)
                else:
                    raise ValueError("invalid conflict evidence record")
    # The B/C serializer must be independently guarded: source code and per-step
    # runtime L3 text are checked before claiming that L3 stayed unchanged.
    base_root = Path(launch["baseline_worktree"])
    dev_root = Path(launch["dev_worktree"])
    source_files = [
        "l3_integration/official_prompt.py",
        "l3_integration/habitat_geometry.py",
    ] + sorted(str(path.relative_to(base_root))
               for path in (base_root / "l3_memory").glob("*.py"))
    relative_config = Path(_json(baseline_dir / "run_config.json")["exp_config"])
    if relative_config.is_absolute() or ".." in relative_config.parts:
        raise ValueError("run_config experiment config must be relative to both worktrees")
    task_config = Path("VLN_CE/habitat_extensions/config/vlnce_task_uninavid_l3_r2r.yaml")
    for root in (base_root, dev_root):
        if _sha256(root / relative_config) != launch.get("config_sha256"):
            raise ValueError("Habitat experiment config changed after launch")
        if _sha256(root / task_config) != launch.get("task_config_sha256"):
            raise ValueError("Habitat task config changed after launch")
    source_match = all(_sha256(base_root / name) == _sha256(dev_root / name)
                       for name in source_files)
    if not source_match:
        raise ValueError("B/C modified L3 source files from protected baseline")
    agent_contract = _official_agent_contract(base_root, dev_root)
    if not agent_contract:
        raise ValueError("B/C modified official Prompt, visual ordering, or generation methods")
    b_step_audit = _step_audit(source_background_dir / "steps.jsonl")
    c_step_audit = _step_audit(fused_dir / "steps.jsonl")
    a_b_audit = _paired_l3_step_audit(baseline_dir, source_background_dir,
                                      right_name=B_NAME)
    b_c_audit = _paired_l3_step_audit(source_background_dir, fused_dir,
                                      right_name=C_NAME)
    a_c_audit = _paired_l3_step_audit(baseline_dir, fused_dir, right_name=C_NAME)
    if not a_b_audit["comparable_l3_state_steps"]:
        raise ValueError("A/B share no comparable L3 observation step")
    if not a_b_audit["all_comparable_l3_state_equal"]:
        raise ValueError("A/B L3 map/fact-selection state differs")
    if not b_c_audit["comparable_l3_state_steps"]:
        raise ValueError("B/C share no comparable L3 observation step")
    if not b_c_audit["all_comparable_l3_state_equal"]:
        raise ValueError("B/C L3 map/fact-selection state differs")
    if not b_c_audit["comparable_l3_enriched_steps"]:
        raise ValueError("B/C share no comparable source-background step")
    smoke = _smoke_audit(output_dir, launch)
    if smoke is None:
        raise ValueError("the three real one-Episode smokes are required")
    b_has_no_l1 = all(
        row.get("l1_fact_count_total") == 0
        and row.get("prompt_has_l1_steps") == 0
        and row.get("l1_enabled_steps") == 0
        and row.get("l1_observation_attempt_steps") == 0
        for row in b_list
    )
    b_source_background = all(
        row.get("source_background_facts_total") == row.get("l3_fact_count_total")
        and row.get("l3_original_memory_unchanged") is True
        for row in b_list
    )
    c_adds_l1 = (
        c_metrics["l1_fact_count_total"] > 0
        and c_metrics["l1_prompt_built_steps"] > 0
        and c_metrics["prompt_has_l1_steps"] > 0
        and b_metrics["l1_fact_count_total"] == 0
    )
    c_keeps_b = all(
        row.get("source_background_facts_total") == row.get("l3_fact_count_total", 0)
        and row.get("l3_original_memory_unchanged") is True
        for row in c_list
    )
    comparison_pairs = {
        "a_vs_b": (a_list, b_list, A_NAME, B_NAME),
        "b_vs_c": (b_list, c_list, B_NAME, C_NAME),
        "a_vs_c": (a_list, c_list, A_NAME, C_NAME),
    }
    comparisons = {}
    for name, (left_rows, right_rows, left_name, right_name) in comparison_pairs.items():
        rows = _comparison(left_rows, right_rows)
        for row in rows:
            row["comparison"] = name
            row["left_group"] = left_name
            row["right_group"] = right_name
        comparisons[name] = rows
    summary = {
        "protocol_version": "l3_three_group_sbadir_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(triples),
        "pilot_manifest": manifest_audit,
        "episode_manifest": episode_audit,
        "launch": launch,
        "smoke": smoke,
        "source_results_sha256": {
            A_NAME: _sha256(a_path), B_NAME: _sha256(b_path), C_NAME: _sha256(c_path),
        },
        "groups": {A_NAME: a_metrics, B_NAME: b_metrics, C_NAME: c_metrics},
        "comparisons": comparisons,
        "paired": {
            "a_vs_b": paired_summary(a_list, b_list),
            "b_vs_c": paired_summary(b_list, c_list),
            "a_vs_c": paired_summary(a_list, c_list),
        },
        "history": hist,
        "conflict_type_counts": dict(sorted(type_counts.items())),
        "conflict_evidence_ids": sorted(conflict_evidence_ids) if evidence_ids_recorded else None,
        "conflict_evidence_records": conflict_evidence_records if evidence_ids_recorded else None,
        "conflict_evidence_id_count": len(conflict_evidence_ids) if evidence_ids_recorded else None,
        "integrity": {
            "same_episode_ids": True,
            "same_instructions": True,
            "same_episode_seeds": True,
            "same_checkpoint_config_l3": True,
            "official_prompt_sha256": PROMPT_SHA,
            "a_is_7602b25_runtime": True,
            "l3_source_files_match": source_match,
            "official_agent_methods_match": agent_contract,
            "b_c_same_source_tree": True,
            "b_step_audit": b_step_audit,
            "c_step_audit": c_step_audit,
            "a_b_step_audit": a_b_audit,
            "b_c_step_audit": b_c_audit,
            "a_c_step_audit": a_c_audit,
            "a_b_l3_state_equal_on_common_prefix": a_b_audit["all_comparable_l3_state_equal"],
            "b_c_l3_state_equal_on_common_prefix": b_c_audit["all_comparable_l3_state_equal"],
            "a_c_l3_state_equal_on_common_prefix": a_c_audit["all_comparable_l3_state_equal"],
            "a_b_l3_text_equal_on_common_prefix": a_b_audit["all_comparable_l3_text_equal"],
            "b_c_l3_text_equal_on_common_prefix": b_c_audit["all_comparable_l3_text_equal"],
            "b_c_l3_background_equal_on_common_prefix": b_c_audit["all_comparable_l3_enriched_equal"],
            "a_c_l3_text_equal_on_common_prefix": a_c_audit["all_comparable_l3_text_equal"],
            "b_has_no_l1": b_has_no_l1,
            "b_source_background_present": b_source_background,
            "c_adds_l1_to_b": c_adds_l1,
            "c_keeps_b_source_background": c_keeps_b,
            "b_original_l3_memory_preserved": all(
                row.get("l3_original_memory_unchanged") is True for row in b_list),
            "c_original_l3_memory_preserved": all(
                row.get("l3_original_memory_unchanged") is True for row in c_list),
            "a_matches_claimed_49_of_88": a_metrics["successes"] == HISTORICAL_SUCCESSES,
            "b_observation_paths_error_free": all(
                b_metrics.get(name) == 0 for name in (
                    "l3_segmentation_error_count",
                )
            ),
            "c_observation_paths_error_free": all(
                c_metrics.get(name) == 0 for name in (
                    "l1_observe_error_count", "l1_detector_error_count",
                    "l1_place_error_count", "l3_segmentation_error_count",
                    "cross_label_entity_link_error_count",
                )
            ),
        },
    }
    if not b_has_no_l1:
        raise ValueError("B contains L1 evidence")
    if not b_source_background:
        raise ValueError("B does not source-pair every L3 fact")
    if not c_adds_l1:
        raise ValueError("C did not add L1 on top of B")
    if not c_keeps_b:
        raise ValueError("C did not keep B's source pairing")
    paths = [output_dir / name for name in (
        "summary.json", "results.jsonl", "comparison.csv",
        "paired_episode_deltas.csv", "final_report.md",
    )]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite report files: {}".format(existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with paths[1].open("w", encoding="utf-8") as stream:
        for name, rows in ((A_NAME, a_list), (B_NAME, b_list), (C_NAME, c_list)):
            for row in rows:
                record = dict(row)
                record["comparison_group"] = name
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    comparison_rows = [
        dict(row) for name in ("a_vs_b", "b_vs_c", "a_vs_c")
        for row in summary["comparisons"][name]
    ]
    for path, records in ((paths[2], comparison_rows),
                          (paths[3], _paired_records(triples))):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    paths[4].write_text(_report_text(summary), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--source-background", required=True)
    parser.add_argument("--fused", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument("--launch-manifest", required=True)
    parser.add_argument("--historical", default="")
    args = parser.parse_args(argv)
    pilot = load_pilot_manifest(args.manifest, expected_sha256=EXPECTED_PILOT_MANIFEST_SHA256)
    episode = load_episode_manifest(
        args.episode_manifest,
        expected_canonical_sha256=EXPECTED_EPISODE_MANIFEST_SHA256,
    )
    if set(pilot.keys()) != set(episode.keys()):
        raise ValueError("Pilot and Episode Manifest sets differ")
    summary = build_report(
        Path(args.baseline), Path(args.source_background), Path(args.fused),
        Path(args.output), pilot.keys(),
        pilot.audit(), episode.audit(), _json(Path(args.launch_manifest)),
        historical_dir=Path(args.historical) if args.historical else None,
    )
    print("wrote {} paired Episodes to {}".format(summary["n_pairs"], args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Strict paired report for protected L3 baseline versus L3+L1 SBAdir.

Every measurement comes from the two real result streams. Missing historical
artifacts remain explicitly unverified. Episode identity, manifest, prompt,
and shared run conditions are checked before any report file is written.
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

from l3_integration.metrics import finite_float, mean_or_none, paired_summary  # noqa: E402
from l3_integration.pilot_manifest import (  # noqa: E402
    EXPECTED_EPISODE_MANIFEST_SHA256,
    EXPECTED_PILOT_MANIFEST_SHA256,
    load_episode_manifest,
    load_pilot_manifest,
)

A_NAME = "l3_organized_baseline_7602b25"
B_NAME = "l3_l1_sbadir"
A_GROUP = "uninavid_l3_only"
B_GROUP = "uninavid_l3_l1_sbadir"
BASELINE_SHA = "7602b254ea514aa71b0057e41b5db3bdfedf9ffa"
PROMPT_SHA = "435d79868156aaa1352afcd98f4ceae2f03a54f8cdc5fe87e41d32774d5512c1"
HISTORICAL_SUCCESSES = 49
HISTORICAL_EPISODES = 88
REQUIRED_B_FIELDS = (
    "l1_fact_count_total", "l3_fact_count_total", "l1_context_tokens_mean",
    "l3_context_tokens_mean", "duplicated_facts_total",
    "conflicting_facts_total", "retained_facts_total", "omitted_facts_total",
    "prompt_has_l1_steps", "l3_original_memory_unchanged",
    "source_background_facts_total", "l1_observation_attempt_steps",
    "l1_observe_error_count", "l1_detector_error_count",
    "l1_place_attempt_steps", "l1_place_observed_steps", "l1_place_error_count",
    "l3_segmentation_error_count",
    "cross_label_entity_link_error_count",
)
METRICS = (
    ("SR", "success", 100.0, "%", "higher"),
    ("SPL", "spl", 100.0, "%", "higher"),
    ("nDTW", "ndtw", 100.0, "%", "higher"),
    ("NE", "distance_to_goal", 1.0, "m", "lower"),
    ("OSR", "oracle_success", 100.0, "%", "higher"),
    ("TL", "path_length", 1.0, "m", "lower"),
    ("Mean inference latency", "inference_latency_ms_mean", 1.0, "ms", "lower"),
    ("Prompt tokens", "prompt_tokens_mean", 1.0, "tokens", "lower"),
    ("Memory facts", "memory_fact_count_total", 1.0, "fact uses", "descriptive"),
    ("Conflicts", "conflicting_facts_total", 1.0, "conflicts", "descriptive"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _b_source_hashes(root: Path) -> Dict[str, str]:
    names = [
        "run.py", "agent_uninavid.py", "l3_integration/map_sidecar.py",
        "l3_integration/sbadir_fusion.py",
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
        'if getattr(self, "l1", None) is not None and '
        'self.last_prompt_tokens + 1024 > self.context_len:\n'
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
        # A predates fusion; it cannot create cross-layer conflicts.
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
    count_fields = (
        "l3_fact_count_total", "l1_fact_count_total", "duplicated_facts_total",
        "conflicting_facts_total", "retained_facts_total", "omitted_facts_total",
        "prompt_has_l3_steps", "prompt_has_l1_steps", "source_background_facts_total",
        "l1_detector_error_count", "l1_place_error_count",
        "l3_segmentation_error_count", "cross_label_entity_link_error_count",
        "cross_label_entity_links_total",
        "l1_observation_attempt_steps", "l1_observe_error_count",
        "l1_place_attempt_steps", "l1_place_observed_steps",
    )
    for key in count_fields:
        if all(key in row for row in rows):
            metrics[key] = sum(_int_nonnegative(row, key) for row in rows)
        elif rows and rows[0].get("group") == A_GROUP and key in (
            "l1_fact_count_total", "duplicated_facts_total", "conflicting_facts_total",
            "retained_facts_total", "omitted_facts_total", "prompt_has_l1_steps",
            "source_background_facts_total",
            "l1_detector_error_count", "l1_place_error_count",
            "l3_segmentation_error_count", "cross_label_entity_link_error_count",
            "cross_label_entity_links_total",
        ):
            metrics[key] = 0
        else:
            metrics[key] = None
    for key in ("l3_context_tokens_mean", "l1_context_tokens_mean"):
        metrics[key] = mean_or_none(
            value for row in rows if (value := _number(row.get(key))) is not None
        )
    return metrics


def _validate_configs(a: Dict[str, Any], b: Dict[str, Any], launch: Dict[str, Any]) -> None:
    if a.get("group") != A_GROUP or b.get("group") != B_GROUP:
        raise ValueError("run_config groups do not match protected A/B code paths")
    for name, cfg in (("A", a), ("B", b)):
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
        if a.get(key) != b.get(key):
            raise ValueError("A/B run_config mismatch: {}".format(key))
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
    source_hashes = _b_source_hashes(Path(launch["dev_worktree"]))
    if source_hashes != launch.get("b_source_files_sha256"):
        raise ValueError("B source files changed after launch")
    source_digest = _b_source_digest(source_hashes)
    if source_digest != launch.get("b_source_tree_sha256"):
        raise ValueError("B source tree digest differs from launch")
    if (b.get("l1") or {}).get("source_tree_sha256") != source_digest:
        raise ValueError("B run_config lacks the launched source tree digest")


def _validate_rows(
    a: Dict[Tuple[str, str], Dict[str, Any]],
    b: Dict[Tuple[str, str], Dict[str, Any]],
    expected_keys: Iterable[Tuple[str, str]],
    *, expected_count: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    expected = set(expected_keys)
    if len(expected) != expected_count or set(a) != expected or set(b) != expected:
        raise ValueError("A/B Episode identities differ from the frozen Manifest")
    pairs = []
    for key in sorted(expected):
        ra, rb = a[key], b[key]
        if ra.get("group") != A_GROUP or rb.get("group") != B_GROUP:
            raise ValueError("result group mismatch at {}".format(key))
        if ra.get("instruction_sha256") != rb.get("instruction_sha256"):
            raise ValueError("A/B instruction differs at {}".format(key))
        if ra.get("episode_seed") != rb.get("episode_seed"):
            raise ValueError("A/B Episode seed differs at {}".format(key))
        for name, row in (("A", ra), ("B", rb)):
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
        for field in REQUIRED_B_FIELDS:
            if field not in rb:
                raise ValueError("B result lacks required diagnostic {} at {}".format(field, key))
        for field in REQUIRED_B_FIELDS:
            if field not in ("l1_context_tokens_mean", "l3_context_tokens_mean", "l3_original_memory_unchanged"):
                _int_nonnegative(rb, field)
        if rb["l3_original_memory_unchanged"] is not True:
            raise ValueError("B altered original L3 memory at {}".format(key))
        pairs.append((ra, rb))
    return pairs


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
    return {"l3_prompt_steps_checked": observed, "original_l3_line_prefixes_preserved": True}


def _ordered_result_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _steps_by_identity(result_path: Path, steps_path: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Attach step segments to result identities without using reset order to pair A/B."""
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


def _paired_l3_step_audit(
    baseline_dir: Path, fused_dir: Path,
) -> Dict[str, Any]:
    a_steps = baseline_dir / "steps.jsonl"
    b_steps = fused_dir / "steps.jsonl"
    if not a_steps.is_file() or not b_steps.is_file():
        return {
            "status": "unavailable", "reason": "A or B steps.jsonl is missing",
            "comparable_nonempty_l3_steps": 0,
            "all_comparable_l3_text_equal": None,
        }
    a = _steps_by_identity(baseline_dir / "results.jsonl", a_steps)
    b = _steps_by_identity(fused_dir / "results.jsonl", b_steps)
    if set(a) != set(b):
        raise ValueError("A/B step Episode identities differ")
    comparable = 0
    matching = 0
    divergent_episodes = 0
    for key in sorted(a):
        prior_actions_equal = True
        for a_row, b_row in zip(a[key], b[key]):
            if not prior_actions_equal:
                break
            a_text = a_row.get("l3_context_text")
            b_text = b_row.get("l3_context_text")
            if a_row.get("prompt_text") is not None and b_row.get("prompt_text") is not None:
                if a_text or b_text:
                    comparable += 1
                    if a_text != b_text:
                        raise ValueError("A/B original L3 text differs on common action prefix: {}".format(key))
                    matching += 1
            if a_row.get("action") != b_row.get("action"):
                prior_actions_equal = False
                divergent_episodes += 1
    return {
        "status": "verified_on_common_action_prefix" if comparable else "no_comparable_nonempty_l3_steps",
        "comparable_nonempty_l3_steps": comparable,
        "matching_nonempty_l3_steps": matching,
        "episodes_with_action_divergence": divergent_episodes,
        "all_comparable_l3_text_equal": matching == comparable if comparable else None,
    }


def _smoke_audit(output_dir: Path, launch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not launch.get("smoke_required"):
        return None
    directory = output_dir / "smoke_l3_l1_sbadir"
    check = _json(directory / "smoke_validation.json")
    rows = _rows(directory / "results.jsonl")
    if len(rows) != 1 or list(rows)[0] != tuple(check.get("episode", [])):
        raise ValueError("one-Episode smoke validation does not match its result")
    row = next(iter(rows.values()))
    for key in (
        "l1_observation_attempt_steps", "l3_map_steps", "l3_fact_count_total",
        "l1_fact_count_total", "source_background_facts_total",
        "prompt_has_l1_steps", "prompt_has_l3_steps",
        "l1_place_attempt_steps", "l1_place_observed_steps",
    ):
        if _number(check.get(key)) is None or check[key] <= 0:
            raise ValueError("real smoke did not exercise {}".format(key))
        if check[key] != row.get(key):
            raise ValueError("smoke validation differs from result: {}".format(key))
    for key in ("l1_observe_error_count", "l1_detector_error_count",
                "l3_segmentation_error_count", "l1_place_error_count",
                "cross_label_entity_link_error_count"):
        if check.get(key) != 0:
            raise ValueError("real smoke encountered {}".format(key))
        if check[key] != row.get(key):
            raise ValueError("smoke validation differs from result: {}".format(key))
    if check.get("official_prompt_sha256") != PROMPT_SHA or check.get("l3_original_memory_unchanged") is not True:
        raise ValueError("real smoke did not preserve official/L3 Prompt")
    if check["official_prompt_sha256"] != row.get("official_prompt_sha256") or row.get("l3_original_memory_unchanged") is not True:
        raise ValueError("smoke validation Prompt differs from result")
    return check


def _comparison(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for label, key, scale, unit, direction in METRICS:
        av = mean_or_none(
            value * scale for row in a if (value := _metric_value(row, key)) is not None
        )
        bv = mean_or_none(
            value * scale for row in b if (value := _metric_value(row, key)) is not None
        )
        result.append({
            "metric": label, "unit": unit, "l3_only": av, "l3_l1_sbadir": bv,
            "delta_b_minus_a": None if av is None or bv is None else bv - av,
            "preferred_direction": direction,
        })
    return result


def _paired_records(pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]) -> List[Dict[str, Any]]:
    records = []
    keys = (
        ("success", "success"), ("spl", "spl"), ("ndtw", "ndtw"),
        ("ne", "distance_to_goal"), ("osr", "oracle_success"),
        ("tl", "path_length"), ("prompt_tokens", "prompt_tokens_mean"),
        ("inference_latency_ms", "inference_latency_ms_mean"),
        ("memory_facts", "memory_fact_count_total"),
        ("conflicts", "conflicting_facts_total"),
    )
    for a, b in pairs:
        record = {"scene_basename": a["scene_basename"], "episode_id": a["episode_id"],
                  "instruction_sha256": a["instruction_sha256"]}
        for label, field in keys:
            av, bv = _metric_value(a, field), _metric_value(b, field)
            record[label + "_a"] = av
            record[label + "_b"] = bv
            record["delta_" + label] = None if av is None or bv is None else bv - av
        record["l1_fact_count_b"] = b["l1_fact_count_total"]
        record["l3_fact_count_a"] = a.get("l3_fact_count_total")
        record["l3_fact_count_b"] = b["l3_fact_count_total"]
        record["l3_memory_unchanged_b"] = b["l3_original_memory_unchanged"]
        records.append(record)
    return records


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    return "{:.3f}".format(value) if isinstance(value, float) else str(value)


def _report_text(summary: Dict[str, Any]) -> str:
    a = summary["groups"][A_NAME]
    b = summary["groups"][B_NAME]
    hist = summary["history"]
    flips = summary["paired"]["success_flips"]
    lines = [
        "# L3 Organized baseline vs L3+L1 SBAdir",
        "",
        "This report covers the frozen 88-Episode Pilot. A uses exact commit",
        "`{}`; B uses the SBAdir development worktree.".format(BASELINE_SHA),
        "",
        "| Metric | L3 Only | L3+L1 | Delta (B-A) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["comparison"]:
        lines.append("| {} ({}) | {} | {} | {} |".format(
            row["metric"], row["unit"], _format(row["l3_only"]),
            _format(row["l3_l1_sbadir"]), _format(row["delta_b_minus_a"]),
        ))
    lines.extend([
        "",
        "Memory facts and conflicts count fact uses across steps, averaged per Episode.",
        "An L3 fact can appear at more than one step.",
        "",
        "## Historical 55.68% baseline",
        "",
        "- Claimed historical result: 49/88 = 55.6818%.",
        "- A result: {}/88 = {}%.".format(a["successes"], _format(a["SR"])),
        "- A matches the claimed 49/88 count: {}.".format(summary["integrity"]["a_matches_claimed_49_of_88"]),
        "- Historical artifacts verified: {}.".format(hist["historical_artifacts_verified"]),
    ])
    if hist["observed_successes"] is not None:
        lines.append("- Historical results file: {}/{} successes ({}%).".format(
            hist["observed_successes"], HISTORICAL_EPISODES,
            _format(hist["observed_sr_percent"])))
    if hist["missing_artifacts"]:
        lines.append("- Missing historical artifacts: {}.".format(", ".join(hist["missing_artifacts"])))
    if a["successes"] != HISTORICAL_SUCCESSES or (
        hist["historical_artifacts_verified"]
        and hist["observed_successes"] != HISTORICAL_SUCCESSES
    ):
        lines.append("- **Historical baseline not reproduced.** Do not claim the 55.68% result was established in this run.")
    elif not hist["historical_artifacts_verified"]:
        lines.append("- The 49/88 count matches the supplied claim, but the historical source run remains unverified.")
    lines.extend([
        "",
        "## Real GPU smoke",
        "",
        "- One-Episode smoke: {}.".format(
            "passed; {}".format(summary["smoke"]["episode"])
            if summary["smoke"] is not None else "not attached to this report"),
        "",
        "## Paired Episode changes",
        "",
        "- A failure → B success: {}.".format(flips["baseline_fail_l3_success"]),
        "- A success → B failure: {}.".format(flips["baseline_success_l3_fail"]),
        "- Exact McNemar p-value: {}.".format(_format(flips["mcnemar"]["p_value"])),
        "",
        "## Memory and source evidence",
        "",
        "- B L1 facts injected across steps: {}.".format(b["l1_fact_count_total"]),
        "- B prompts with L1 facts: {}.".format(b["prompt_has_l1_steps"]),
        "- B L3 facts across steps: {}; A: {}.".format(b["l3_fact_count_total"], a["l3_fact_count_total"]),
        "- Mean L3 memory tokens per Episode A/B: {} / {}.".format(
            _format(a["l3_context_tokens_mean"]), _format(b["l3_context_tokens_mean"])),
        "- Mean L1 memory tokens per Episode B: {}.".format(_format(b["l1_context_tokens_mean"])),
        "- Duplicate facts: {}; conflicts: {}; retained: {}; omitted: {}.".format(
            b["duplicated_facts_total"], b["conflicting_facts_total"],
            b["retained_facts_total"], b["omitted_facts_total"]),
        "- Source background facts logged: {}.".format(_format(b["source_background_facts_total"])),
        "- L1 RGB observation attempts: {}; DINOv2 place observations: {}.".format(
            b["l1_observation_attempt_steps"], b["l1_place_observed_steps"]),
        "- L1 observe/detector/place errors: {} / {} / {}; L3 segmentation errors: {}.".format(
            b["l1_observe_error_count"], b["l1_detector_error_count"],
            b["l1_place_error_count"], b["l3_segmentation_error_count"]),
        "- Original L3 memory preserved in every B Episode: {}.".format(summary["integrity"]["b_original_l3_memory_preserved"]),
        "- B prompt steps checked for exact original L3 text: {}.".format(
            summary["integrity"]["b_step_audit"]["l3_prompt_steps_checked"]),
        "- A/B original L3 text on shared action prefixes: {} ({} comparable nonempty steps).".format(
            summary["integrity"]["paired_l3_step_audit"]["status"],
            summary["integrity"]["paired_l3_step_audit"]["comparable_nonempty_l3_steps"],
        ),
        "- After A/B actions diverge, L3 memories can reflect different RGB-D histories and are not paired as identical observations.",
        "- Conflict types: {}.".format(json.dumps(summary["conflict_type_counts"], sort_keys=True)),
        "- Conflict evidence IDs recorded: {}.".format(_format(summary["conflict_evidence_id_count"])),
        "- Unique cross-label entity links: {}; L1 detector/DINO/L3 segmenter/matcher errors: {}/{}/{}/{}.".format(
            b["cross_label_entity_links_total"], b["l1_detector_error_count"],
            b["l1_place_error_count"], b["l3_segmentation_error_count"],
            b["cross_label_entity_link_error_count"],
        ),
        "",
        "Source backgrounds and conflicting evidence were exposed to the model",
        "when logged above. The paired outcomes alone do not establish that source",
        "backgrounds caused a success change; inspect `steps.jsonl` for each",
        "conflict's fact text and evidence IDs.",
        "",
        "## Integrity",
        "",
        "- Same Pilot raw SHA256: `{}`.".format(EXPECTED_PILOT_MANIFEST_SHA256),
        "- Same Episode Manifest canonical SHA256: `{}`.".format(EXPECTED_EPISODE_MANIFEST_SHA256),
        "- Same official Prompt template SHA256: `{}`.".format(PROMPT_SHA),
        "- Same checkpoint, seed 42, Habitat configuration and L3 settings:",
        "  verified from `run_config.json` and `launch_manifest.json`.",
        "- L3 organizer, map core and segmentation source SHA256 match baseline: {}.".format(
            summary["integrity"]["l3_source_files_match"]),
        "- Visual encoding and generation methods match baseline AST: {}.".format(
            summary["integrity"]["official_agent_methods_match"]),
        "",
        "## Files",
        "",
        "- `summary.json`: validated metrics, history status and integrity.",
        "- `results.jsonl`: paired source rows, each tagged with A or B.",
        "- `comparison.csv`: metric comparison.",
        "- `paired_episode_deltas.csv`: identity-paired Episode deltas.",
        "- `final_report.md`: this report.",
    ])
    return "\n".join(lines) + "\n"


def build_report(
    baseline_dir: Path,
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
    a_path, b_path = baseline_dir / "results.jsonl", fused_dir / "results.jsonl"
    a, b = _rows(a_path), _rows(b_path)
    _validate_configs(_json(baseline_dir / "run_config.json"),
                      _json(fused_dir / "run_config.json"), launch)
    pairs = _validate_rows(a, b, expected_keys, expected_count=expected_count)
    # Each run's derived execution artifact must have the same frozen identity.
    for name, directory in (("A", baseline_dir), ("B", fused_dir)):
        execution = load_episode_manifest(directory / "execution_manifest.json")
        if set(execution.keys()) != set(expected_keys):
            raise ValueError("{} execution Manifest Episode set differs".format(name))
    a_list = [item[0] for item in pairs]
    b_list = [item[1] for item in pairs]
    a_metrics = _group_metrics(a_list)
    b_metrics = _group_metrics(b_list)
    paired = paired_summary(a_list, b_list)
    hist = _historical(historical_dir, set(expected_keys))
    if hist["historical_artifacts_verified"]:
        hist["a_matches_historical_success_count"] = a_metrics["successes"] == hist["observed_successes"]
        historical_rows = _rows(historical_dir / "results.jsonl")
        hist["a_matches_historical_success_by_episode"] = all(
            a[key]["success"] == historical_rows[key]["success"] for key in a
        )
    type_counts = Counter()  # type: Counter[str]
    conflict_evidence_ids = set()  # type: set[str]
    conflict_evidence_records = []  # type: List[Dict[str, Any]]
    evidence_ids_recorded = False
    for row in b_list:
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
    # The B serializer must be independently guarded: source code and per-step
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
        raise ValueError("B modified L3 source files from protected baseline")
    agent_contract = _official_agent_contract(base_root, dev_root)
    if not agent_contract:
        raise ValueError("B modified official Prompt, visual ordering, or generation methods")
    step_audit = _step_audit(fused_dir / "steps.jsonl")
    paired_l3_audit = _paired_l3_step_audit(baseline_dir, fused_dir)
    smoke = _smoke_audit(output_dir, launch)
    summary = {
        "protocol_version": "l3_vs_l3l1_sbadir_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(pairs),
        "pilot_manifest": manifest_audit,
        "episode_manifest": episode_audit,
        "launch": launch,
        "smoke": smoke,
        "source_results_sha256": {A_NAME: _sha256(a_path), B_NAME: _sha256(b_path)},
        "groups": {A_NAME: a_metrics, B_NAME: b_metrics},
        "comparison": _comparison(a_list, b_list),
        "paired": paired,
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
            "l3_source_files_match": source_match,
            "official_agent_methods_match": agent_contract,
            "b_original_l3_memory_preserved": True,
            "b_step_audit": step_audit,
            "paired_l3_step_audit": paired_l3_audit,
            "a_b_original_l3_text_equal_on_comparable_steps": paired_l3_audit["all_comparable_l3_text_equal"],
            "a_matches_claimed_49_of_88": a_metrics["successes"] == HISTORICAL_SUCCESSES,
            "l1_injected": b_metrics["l1_fact_count_total"] > 0 and b_metrics["prompt_has_l1_steps"] > 0,
            "b_observation_paths_error_free": all(
                b_metrics.get(name) == 0 for name in (
                    "l1_observe_error_count", "l1_detector_error_count", "l1_place_error_count",
                    "l3_segmentation_error_count", "cross_label_entity_link_error_count",
                )
            ),
        },
    }
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
        for name, rows in ((A_NAME, a_list), (B_NAME, b_list)):
            for row in rows:
                record = dict(row)
                record["comparison_group"] = name
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    for path, records in ((paths[2], summary["comparison"]),
                          (paths[3], _paired_records(pairs))):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    paths[4].write_text(_report_text(summary), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
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
        Path(args.baseline), Path(args.fused), Path(args.output), pilot.keys(),
        pilot.audit(), episode.audit(), _json(Path(args.launch_manifest)),
        historical_dir=Path(args.historical) if args.historical else None,
    )
    print("wrote {} paired Episodes to {}".format(summary["n_pairs"], args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

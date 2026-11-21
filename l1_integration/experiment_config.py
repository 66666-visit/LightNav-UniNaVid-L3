"""Independent per-group configuration for the eight prompt-ablation groups.

Each group is started through its own file, so changing one group cannot
silently change another.  The file may declare the expected layer sources and
combination; those declarations are checked against LightNav's ``mode_plan`` so
a typo cannot turn, for example, ``l1_original_only`` into a joint run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from l1_integration.memory_backend import EIGHT_ABLATION_GROUPS, group_config

_REQUIRED = ("group", "prompt_mode")
_LAYER_FIELDS = (
    "l1_prompt", "l3_prompt", "l1_evidence_writer", "l3_map",
    "combination", "l1_builder", "l3_builder", "preserves_l1_verbatim",
)
_MEMORY_DEFAULTS: dict[str, Any] = {
    "max_landmarks": 2,
    "prompt_max_chars": 320,
    "max_context_chars": 600,
    "scope": "episode_local",
    "token_budget": True,
    "place_every_n_steps": 5,
    "evidence_min_iou": 0.5,
    "evidence_ambiguity_margin": 0.1,
    "min_injection_confidence": 0.5,
}


def _load_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError:
            # The CPU-only Python 3.8 test venv has no PyYAML.  Every YAML
            # config ships a byte-equivalent JSON sibling for this case; the
            # formal Habitat environment reads the YAML file itself.
            sibling = path.with_suffix(".json")
            if sibling.is_file():
                data = json.loads(sibling.read_text(encoding="utf-8"))
            else:
                raise RuntimeError(
                    "PyYAML is required to read this group YAML file"
                )
        else:
            data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError(f"group config must be a mapping: {path}")
    return data


def load_group_config(path: str) -> dict[str, Any]:
    """Load, normalise and validate one group configuration."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"group config not found: {source}")
    data = _load_file(source)
    for name in _REQUIRED:
        if name not in data:
            raise ValueError(f"group config lacks {name!r}: {source}")
    group = data["group"]
    if group not in EIGHT_ABLATION_GROUPS:
        raise ValueError(
            f"group config must name one of the eight ablation groups: {group!r}"
        )
    expected = group_config(group)
    if data["prompt_mode"] != expected["prompt_mode"]:
        raise ValueError(
            f"{group}: prompt_mode {data['prompt_mode']!r} differs from the "
            f"LightNav plan {expected['prompt_mode']!r}"
        )
    layers = data.get("layers") if isinstance(data.get("layers"), dict) else data
    for field in _LAYER_FIELDS:
        if field in layers and layers[field] != expected.get(field):
            raise ValueError(
                f"{group}: declared {field}={layers[field]!r} but the LightNav "
                f"plan requires {expected.get(field)!r}"
            )
    memory = dict(_MEMORY_DEFAULTS)
    configured = data.get("memory")
    if configured is not None:
        if not isinstance(configured, dict):
            raise ValueError("group config 'memory' must be a mapping")
        unknown = set(configured) - set(_MEMORY_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown memory keys in group config: {sorted(unknown)}")
        memory.update(configured)
    for name in (
        "max_landmarks", "prompt_max_chars", "max_context_chars",
        "place_every_n_steps",
    ):
        if type(memory[name]) is not int or memory[name] < 1:
            raise ValueError(f"memory.{name} must be a positive integer")
    if memory["scope"] != "episode_local":
        raise ValueError(
            "memory.scope must be episode_local for the formal eight groups"
        )
    if type(memory["token_budget"]) is not bool:
        raise ValueError("memory.token_budget must be boolean")
    return {
        "source": str(source),
        "group": group,
        "prompt_mode": expected["prompt_mode"],
        "expected": expected,
        "memory": memory,
    }


def group_config_artifact(config: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe record written into ``run_config.json``."""
    return {
        "source": config["source"],
        "group": config["group"],
        "prompt_mode": config["prompt_mode"],
        "memory": dict(config["memory"]),
        "expected": dict(config["expected"]),
    }

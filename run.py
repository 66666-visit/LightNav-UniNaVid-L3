#!/usr/bin/env python3

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

from habitat.datasets import make_dataset
from VLN_CE.vlnce_baselines.config.default import get_config

from habitat import Env
from tqdm import trange
import numpy as np

from l3_integration.episode_execution import (
    EpisodeExecutionError,
    EpisodeExecutionTracker,
    metadata_from_execution_manifest,
    metadata_from_pilot_manifest,
)
from l1_integration.experiment_config import (
    group_config_artifact,
    load_group_config,
)
from l1_integration.memory_backend import (
    ALL_GROUPS,
    EIGHT_ABLATION_GROUPS,
    PromptMemoryBackend,
    group_config as group_semantics,
)
from l3_integration.pilot_manifest import (
    EXPECTED_EPISODES,
    EXPECTED_EPISODE_MANIFEST_SHA256,
    EXPECTED_PILOT_MANIFEST_SHA256,
    audit_manifest,
    build_execution_manifest,
    episode_instruction,
    load_episode_manifest,
    load_pilot_manifest,
    scene_basename,
    select_pilot_episodes,
    write_execution_manifest,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--exp-save",
        type=str,
        required=True,
        help="results types requried to be saved",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="names of evaluation model",
    )

    parser.add_argument(
        "--split-num",
        type=int,
        required=True,
        help="chunks of evluation"
    )

    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
        help="chunks ID of evluation"

    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="location of model weights"

    )

    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="location to save results"

    )

    parser.add_argument(
        "--pilot-manifest",
        type=str,
        default="",
        help="existing 88-Episode pilot_manifest.json; never generated here",
    )
    parser.add_argument(
        "--pilot-sha256",
        type=str,
        default=EXPECTED_PILOT_MANIFEST_SHA256,
        help=(
            "required raw-file SHA256 of pilot_manifest.json; pass '' only for "
            "non-formal smoke tests"
        ),
    )
    parser.add_argument(
        "--episode-manifest",
        type=str,
        default="",
        help=(
            "optional evaluation episode_manifest.json; when supplied, its "
            "canonical completion-marker digest is verified, its 88-Episode "
            "set is matched to the Pilot, and its order is the execution order"
        ),
    )
    parser.add_argument(
        "--episode-manifest-sha256",
        type=str,
        default=EXPECTED_EPISODE_MANIFEST_SHA256,
        help=(
            "expected canonical (not raw-file) SHA256 of episode_manifest.json"
        ),
    )
    parser.add_argument(
        "--parent-manifest",
        type=str,
        default="",
        help="optional parent sampling manifest for raw-file SHA256 verification",
    )
    parser.add_argument(
        "--require-episode-manifest", action="store_true",
        help=(
            "fail closed unless --episode-manifest is supplied; the formal "
            "launcher uses this so the historical marker digest is verified"
        ),
    )
    parser.add_argument(
        "--experiment-group",
        choices=ALL_GROUPS,
        default="uninavid_baseline",
    )
    parser.add_argument(
        "--group-config", type=str, default="",
        help=(
            "independent YAML/JSON configuration for one of the eight "
            "prompt-ablation groups; its declared layers are checked against "
            "the LightNav prompt-mode plan"
        ),
    )
    parser.add_argument(
        "--memory-l1-model", type=str, default="",
        help="local YOLO11n detection .pt for the L1 memory writer",
    )
    parser.add_argument("--memory-l1-device", type=str, default="cuda:0")
    parser.add_argument("--memory-l1-confidence", type=float, default=0.25)
    parser.add_argument("--memory-l1-every-n-steps", type=int, default=1)
    parser.add_argument(
        "--memory-place-model", type=str, default="",
        help=(
            "local DINOv2 model directory used by the original L1 Local Graph "
            "history; without it the original history channel is empty and "
            "this is recorded in the diagnostics"
        ),
    )
    parser.add_argument("--memory-place-device", type=str, default="cuda:0")
    parser.add_argument("--memory-place-every-n-steps", type=int, default=5)
    parser.add_argument(
        "--memory-token-budget", type=int, choices=(0, 1), default=1,
        help=(
            "1 = measure the official prompt headroom per step and fall back "
            "to the character budget with a recorded reason (default); "
            "0 = use the character budget only"
        ),
    )
    parser.add_argument(
        "--no-force-stop", action="store_true",
        help=(
            "do not inject an early stop before Habitat's MAX_EPISODE_STEPS; "
            "the formal prompt-ablation launcher sets this with NO_FORCE_STOP=1"
        ),
    )
    parser.add_argument(
        "--smoke-episodes", type=int, default=0,
        help=(
            "non-formal only: run at most N Episodes from the selected split. "
            "Requires --allow-nonpilot-eval and is rejected with a frozen pilot"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-episodes", type=int, default=0,
        help="optional exact episode count check (88 for the formal pilot)",
    )
    parser.add_argument("--l3-seg-model", type=str, default="")
    parser.add_argument("--l3-seg-device", type=str, default="cuda:0")
    parser.add_argument("--l3-seg-confidence", type=float, default=0.25)
    parser.add_argument("--l3-every-n-steps", type=int, default=5)
    parser.add_argument("--l3-evidence-min-iou", type=float, default=0.5)
    parser.add_argument("--l3-evidence-ambiguity-margin", type=float, default=0.1)
    parser.add_argument("--l3-min-confidence", type=float, default=0.5)
    parser.add_argument("--l3-max-items", type=int, default=2)
    parser.add_argument("--l3-max-chars", type=int, default=600)
    parser.add_argument("--l3-scope", type=str, default="episode_local")
    parser.add_argument(
        "--allow-nonpilot-eval", action="store_true",
        help=(
            "legacy/non-pilot evaluation only; disables the mandatory frozen "
            "88-Episode manifest checks. Never use for the formal L3 A/B run."
        ),
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def _stable_episode_seed(seed: int, scene_id: str, episode_id: str) -> int:
    payload = f"{seed}\0{scene_id}\0{episode_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2 ** 31 - 1)


def _seed_episode_rngs(episode_seed: int) -> None:
    """Apply one actual Episode's deterministic seed to the model RNGs.

    The actual Episode identity is only known after ``env.reset()``; this is
    called immediately after the identity check and before ``agent.reset()``
    and the first model call.  A/B therefore share the same seed for the same
    Episode even when Habitat visits Episodes in a different order.
    """
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(episode_seed)


def _json_value(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _json_value(value)


def _write_run_metadata(path: str, metadata: dict) -> None:
    target = Path(path)
    metadata = _json_safe(metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError(
                f"result path already contains a different {target.name}: {target}"
            )
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_exp(exp_config: str, split_num: int, split_id: int, model_path: str,
            result_path: str, model_name: str, exp_save: str, opts=None,
            *, pilot_manifest: str = "",
            pilot_sha256: str = EXPECTED_PILOT_MANIFEST_SHA256,
            experiment_group: str = "uninavid_baseline", seed: int = 42,
            expected_episodes: int = 0, episode_manifest: str = "",
            episode_manifest_sha256: str = EXPECTED_EPISODE_MANIFEST_SHA256,
            parent_manifest: str = "", require_episode_manifest: bool = False,
            l3_seg_model: str = "",
            l3_seg_device: str = "cuda:0", l3_seg_confidence: float = 0.25,
            l3_every_n_steps: int = 5, l3_evidence_min_iou: float = 0.5,
            l3_evidence_ambiguity_margin: float = 0.1,
            l3_min_confidence: float = 0.5, l3_max_items: int = 2,
            l3_max_chars: int = 600, l3_scope: str = "episode_local",
            group_config: str = "",
            memory_l1_model: str = "", memory_l1_device: str = "cuda:0",
            memory_l1_confidence: float = 0.25,
            memory_l1_every_n_steps: int = 1,
            memory_place_model: str = "", memory_place_device: str = "cuda:0",
            memory_place_every_n_steps: int = 5,
            memory_token_budget: int = 1,
            no_force_stop: bool = False,
            smoke_episodes: int = 0,
            allow_nonpilot_eval: bool = False) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """
    if split_num < 1 or not 0 <= split_id < split_num:
        raise ValueError("split-id must lie in [0, split-num)")
    if l3_scope != "episode_local":
        raise ValueError("L3 may not persist across Episodes; use episode_local")
    if memory_token_budget not in (0, 1):
        raise ValueError("--memory-token-budget must be 0 or 1")
    if type(smoke_episodes) is not int or smoke_episodes < 0:
        raise ValueError("--smoke-episodes must be a nonnegative integer")
    if smoke_episodes:
        if not allow_nonpilot_eval:
            raise ValueError(
                "--smoke-episodes is non-formal and requires --allow-nonpilot-eval"
            )
        if pilot_manifest or require_episode_manifest:
            raise ValueError(
                "--smoke-episodes must not be combined with the frozen pilot or "
                "the formal execution manifest"
            )
    group_spec = group_semantics(experiment_group)
    loaded_group_config = None
    memory_prompt_max_chars = 320
    if group_config:
        loaded_group_config = load_group_config(group_config)
        if loaded_group_config["group"] != experiment_group:
            raise ValueError(
                f"--group-config names {loaded_group_config['group']!r} but "
                f"--experiment-group is {experiment_group!r}"
            )
        memory = loaded_group_config["memory"]
        l3_max_items = memory["max_landmarks"]
        l3_max_chars = memory["max_context_chars"]
        memory_prompt_max_chars = memory["prompt_max_chars"]
        l3_min_confidence = memory["min_injection_confidence"]
        l3_evidence_min_iou = memory["evidence_min_iou"]
        l3_evidence_ambiguity_margin = memory["evidence_ambiguity_margin"]
        memory_place_every_n_steps = memory["place_every_n_steps"]
        memory_token_budget = 1 if memory["token_budget"] else 0
    if experiment_group in EIGHT_ABLATION_GROUPS:
        if not group_config:
            raise ValueError(
                "each of the eight prompt-ablation groups requires its own "
                "--group-config file (configs/prompt_ablation/<group>.yaml)"
            )
        if group_spec["requires_l1_model"] and not memory_l1_model:
            raise ValueError(
                f"{experiment_group} requires --memory-l1-model"
            )
        if group_spec["requires_seg_model"] and not l3_seg_model:
            raise ValueError(
                f"{experiment_group} requires --l3-seg-model"
            )
        if (
            group_spec["requires_l1_model"]
            and not memory_place_model
            and not allow_nonpilot_eval
        ):
            raise ValueError(
                f"{experiment_group} formal runs require --memory-place-model: "
                "the original L1 Local Graph history is DINO-place-gated, and a "
                "run without it would not be the declared L1 condition"
            )
        if not group_spec["requires_l1_model"] and memory_l1_model:
            raise ValueError(
                f"{experiment_group} has no L1 layer; do not pass --memory-l1-model"
            )
        if not group_spec["requires_seg_model"] and l3_seg_model:
            raise ValueError(
                f"{experiment_group} has no L3 layer; do not pass --l3-seg-model"
            )
    elif experiment_group == "uninavid_l3_only":
        if not l3_seg_model:
            raise ValueError("uninavid_l3_only requires --l3-seg-model")
        if memory_l1_model or memory_place_model:
            raise ValueError(
                "uninavid_l3_only is the frozen L3-only path; L1 models are not used"
            )
    else:
        if l3_seg_model or memory_l1_model or memory_place_model:
            raise ValueError(
                "baseline group must not receive any memory model"
            )
    if experiment_group != "uninavid_baseline" and model_name != "uni-navid":
        raise ValueError("the memory experiment groups are only defined for uni-navid")
    if model_name == "uni-navid" and not allow_nonpilot_eval:
        # The formal A/B comparison must fail closed on the frozen pilot file.
        if not pilot_manifest:
            raise ValueError(
                "uni-navid formal evaluation requires --pilot-manifest; "
                "use --allow-nonpilot-eval only for a non-pilot legacy run"
            )
        if (
            not isinstance(pilot_sha256, str)
            or pilot_sha256 != EXPECTED_PILOT_MANIFEST_SHA256
        ):
            raise ValueError(
                "formal evaluation requires --pilot-sha256 "
                f"{EXPECTED_PILOT_MANIFEST_SHA256} (raw pilot file SHA256); "
                "empty/different digests are not allowed"
            )
        if expected_episodes != EXPECTED_EPISODES:
            raise ValueError(
                f"formal evaluation requires --expected-episodes {EXPECTED_EPISODES}"
            )
        if (
            episode_manifest
            and episode_manifest_sha256 != EXPECTED_EPISODE_MANIFEST_SHA256
        ):
            raise ValueError(
                "formal evaluation requires --episode-manifest-sha256 "
                f"{EXPECTED_EPISODE_MANIFEST_SHA256} (historical canonical "
                "digest); empty/different digests are not allowed"
            )
    if not isinstance(allow_nonpilot_eval, bool):
        raise ValueError("allow_nonpilot_eval must be explicit boolean")
    if episode_manifest and not pilot_manifest:
        raise ValueError(
            "--episode-manifest requires --pilot-manifest for the strict "
            "88-Episode set comparison and execution-order reconciliation"
        )
    if parent_manifest and not pilot_manifest:
        raise ValueError(
            "--parent-manifest requires --pilot-manifest to read the declared "
            "parent_manifest_sha256"
        )
    if not episode_manifest and episode_manifest_sha256 not in (
        "", None, EXPECTED_EPISODE_MANIFEST_SHA256,
    ):
        raise ValueError(
            "--episode-manifest-sha256 was supplied without --episode-manifest"
        )
    if episode_manifest and not episode_manifest_sha256:
        raise ValueError(
            "--episode-manifest requires --episode-manifest-sha256; never "
            "bypass the historical completion-marker digest"
        )
    if require_episode_manifest and not episode_manifest:
        raise ValueError(
            "--require-episode-manifest requires --episode-manifest; supply "
            "the historical evaluation manifest so its canonical digest, "
            "88-Episode set and execution order can be verified"
        )

    config = get_config(exp_config, opts)

    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(seed)

    manifest = None
    manifest_audit = None
    execution_manifest = None
    execution_report = None
    episode_order = {}
    episode_metadata = None
    if pilot_manifest:
        effective_pilot_sha = (
            None if allow_nonpilot_eval and not pilot_sha256
            else (pilot_sha256 or EXPECTED_PILOT_MANIFEST_SHA256)
        )
        manifest = load_pilot_manifest(
            pilot_manifest, expected_sha256=effective_pilot_sha,
        )
        if episode_manifest:
            historical_episode = load_episode_manifest(
                episode_manifest,
                expected_canonical_sha256=episode_manifest_sha256 or None,
            )
            execution_manifest = build_execution_manifest(
                historical_episode, manifest,
            )
        selected = select_pilot_episodes(
            dataset.episodes, manifest,
            execution_manifest=execution_manifest,
        )
        if expected_episodes and len(selected) != expected_episodes:
            raise ValueError(
                f"pilot manifest selected {len(selected)} Episodes; "
                f"expected {expected_episodes}"
            )
        if len(selected) != EXPECTED_EPISODES:
            raise ValueError(
                f"pilot manifest selected {len(selected)} Episodes; "
                f"expected {EXPECTED_EPISODES}"
            )
        # Metadata is indexed once by Episode identity.  Neither the dataset
        # list position nor Habitat's reset order is used for association.
        if execution_manifest is not None:
            episode_metadata = metadata_from_execution_manifest(
                execution_manifest,
            )
        else:
            episode_metadata = metadata_from_pilot_manifest(manifest)
        dataset_split = copy.copy(dataset)
        dataset_split.episodes = selected[split_id::split_num]
        instructions = [episode_instruction(episode) for episode in selected]
        instruction_by_key = {
            (scene_basename(episode.scene_id), str(episode.episode_id)):
                episode_instruction(episode)
            for episode in selected
        }
        pilot_instructions = [
            instruction_by_key[(row.scene_basename, row.episode_id)]
            for row in manifest.rows
        ]
        manifest_audit = audit_manifest(
            pilot_manifest,
            expected_sha256=effective_pilot_sha,
            dataset_episodes=dataset.episodes,
            episode_manifest=episode_manifest or None,
            expected_episode_manifest_sha256=(
                episode_manifest_sha256 or None
            ),
            execution_manifest=execution_manifest,
            parent_manifest=parent_manifest or None,
        )
        manifest_audit["pilot_order_digest_with_instructions"] = (
            manifest.order_digest(instructions=pilot_instructions)
        )
        manifest_audit["dataset_order_digest_with_instructions"] = (
            execution_manifest.order_digest(instructions=instructions)
            if execution_manifest is not None
            else manifest.order_digest(instructions=instructions)
        )
        manifest_audit["execution_order"] = (
            execution_manifest.order_source
            if execution_manifest is not None else "pilot_manifest.json"
        )
        manifest_audit["shard"] = {
            "split_num": split_num, "split_id": split_id,
            "selected_count": len(dataset_split.episodes),
        }
        # Diagnostic fallback only: the formal path always associates Pilot
        # metadata through episode_metadata, keyed by Episode identity.
        for index, episode in enumerate(selected):
            episode_order[
                (scene_basename(episode.scene_id), str(episode.episode_id))
            ] = index
        if execution_manifest is not None:
            execution_report = write_execution_manifest(
                os.path.join(result_path, "execution_manifest.json"),
                execution_manifest,
                expected_canonical_sha256=episode_manifest_sha256,
            )
            manifest_audit["execution_manifest_verification"] = execution_report
            _write_run_metadata(
                os.path.join(result_path, "manifest_alignment.json"),
                {
                    "aligned": True,
                    "execution_order": execution_manifest.order_source,
                    "pilot_manifest": {
                        "path": str(manifest.path),
                        "file_sha256": manifest.file_sha256,
                        "expected_file_sha256": effective_pilot_sha,
                    },
                    "episode_manifest": {
                        "path": str(historical_episode.path),
                        "file_sha256": historical_episode.file_sha256,
                        "canonical_sha256": historical_episode.canonical_sha256,
                        "expected_canonical_sha256": episode_manifest_sha256,
                    },
                    "pilot_comparison": manifest_audit["episode_manifest"][
                        "pilot_comparison"
                    ],
                    "execution_manifest": {
                        key: value for key, value in execution_report.items()
                        if key != "created"
                    },
                },
            )
    else:
        dataset_split = dataset.get_splits(split_num)[split_id]
        if smoke_episodes:
            dataset_split = copy.copy(dataset_split)
            dataset_split.episodes = list(dataset_split.episodes)[:smoke_episodes]
            if not dataset_split.episodes:
                raise ValueError("--smoke-episodes selected no Episode")
        if expected_episodes and len(dataset_split.episodes) != expected_episodes:
            raise ValueError(
                f"split selected {len(dataset_split.episodes)} Episodes; "
                f"expected {expected_episodes}"
            )

    evaluate_agent(
        config, split_id, dataset_split, model_path, result_path, model_name,
        exp_save, split_num=split_num, experiment_group=experiment_group, seed=seed,
        l3_options={
            "seg_model": l3_seg_model,
            "seg_device": l3_seg_device,
            "seg_confidence": l3_seg_confidence,
            "every_n_steps": l3_every_n_steps,
            "evidence_min_iou": l3_evidence_min_iou,
            "evidence_ambiguity_margin": l3_evidence_ambiguity_margin,
            "min_confidence": l3_min_confidence,
            "max_items": l3_max_items,
            "max_chars": l3_max_chars,
            "scope": l3_scope,
        },
        memory_options={
            "l1_model": memory_l1_model,
            "l1_device": memory_l1_device,
            "l1_confidence": memory_l1_confidence,
            "l1_every_n_steps": memory_l1_every_n_steps,
            "place_model": memory_place_model,
            "place_device": memory_place_device,
            "place_every_n_steps": memory_place_every_n_steps,
            "token_budget": bool(memory_token_budget),
            "max_landmarks": l3_max_items,
            "max_context_chars": l3_max_chars,
            "prompt_max_chars": memory_prompt_max_chars,
            "seg_model": l3_seg_model,
            "seg_device": l3_seg_device,
            "seg_confidence": l3_seg_confidence,
            "seg_every_n_steps": l3_every_n_steps,
            "evidence_min_iou": l3_evidence_min_iou,
            "evidence_ambiguity_margin": l3_evidence_ambiguity_margin,
            "min_confidence": l3_min_confidence,
        },
        group_config_spec=loaded_group_config,
        no_force_stop=bool(no_force_stop),
        manifest_audit=manifest_audit,
        episode_order=episode_order,
        episode_metadata=episode_metadata,
        execution_manifest=execution_manifest,
        execution_report=execution_report,
        exp_config=exp_config,
        allow_nonpilot_eval=allow_nonpilot_eval,
    )


def _build_l3_sidecar(config, l3_options):
    from l3_integration.habitat_geometry import camera_spec_from_config
    from l3_integration.map_sidecar import L3MapSidecar
    from l3_memory.spatial_l3_config import SpatialL3Config

    l3_config = SpatialL3Config(
        mode="inject",
        seg_model_path=l3_options["seg_model"],
        seg_device=l3_options["seg_device"],
        seg_confidence=l3_options["seg_confidence"],
        seg_every_n_steps=l3_options["every_n_steps"],
        evidence_min_iou=l3_options["evidence_min_iou"],
        evidence_ambiguity_margin=l3_options["evidence_ambiguity_margin"],
        min_injection_confidence=l3_options["min_confidence"],
    )
    camera_spec = camera_spec_from_config(config)
    sidecar = L3MapSidecar(
        l3_config,
        camera_spec=camera_spec,
        max_items=l3_options["max_items"],
        max_chars=l3_options["max_chars"],
        scope=l3_options["scope"],
    )
    return sidecar, camera_spec, l3_config


def _build_memory_backend(config, experiment_group, memory_options):
    """Build the real L1/L3 backend for one of the eight ablation groups."""
    from pathlib import Path as _Path

    from l1_memory.place_recognition import DINOv2PlaceDescriptorExtractor
    from l1_memory.yolo_scene_extractor import YOLOSceneExtractor
    from l3_integration.habitat_geometry import camera_spec_from_config
    from l3_memory.yolo_seg_extractor import YOLOSegExtractor

    spec = group_semantics(experiment_group)
    camera_spec = camera_spec_from_config(config)
    l1_extractor = None
    if spec["requires_l1_model"]:
        l1_extractor = YOLOSceneExtractor(
            memory_options["l1_model"], device=memory_options["l1_device"],
            confidence=memory_options["l1_confidence"],
            every_n_steps=memory_options["l1_every_n_steps"],
        )
    seg_extractor = None
    if spec["requires_seg_model"]:
        seg_extractor = YOLOSegExtractor(
            memory_options["seg_model"], device=memory_options["seg_device"],
            confidence=memory_options["seg_confidence"],
            every_n_steps=memory_options["seg_every_n_steps"],
        )
    place_extractor = None
    if spec["requires_l1_model"] and memory_options.get("place_model"):
        place_extractor = DINOv2PlaceDescriptorExtractor(
            memory_options["place_model"], device=memory_options["place_device"],
        )
    backend = PromptMemoryBackend(
        spec["prompt_mode"],
        l1_extractor=l1_extractor,
        seg_extractor=seg_extractor,
        place_extractor=place_extractor,
        max_landmarks=memory_options["max_landmarks"],
        max_context_chars=memory_options["max_context_chars"],
        prompt_max_chars=memory_options["prompt_max_chars"],
        scope="episode_local",
        evidence_min_iou=memory_options["evidence_min_iou"],
        evidence_ambiguity_margin=memory_options["evidence_ambiguity_margin"],
        min_injection_confidence=memory_options["min_confidence"],
        place_every_n_steps=memory_options["place_every_n_steps"],
        seg_model_path=memory_options.get("seg_model", ""),
        seg_source=(
            "yolo11n-seg:"
            + _Path(memory_options.get("seg_model", "")).name
            if memory_options.get("seg_model") else "yolo11n-seg"
        ),
    )
    return backend, camera_spec


def evaluate_agent(config, split_id, dataset, model_path, result_path, model_name,
                   exp_save, *, split_num: int = 1,
                   experiment_group: str = "uninavid_baseline",
                   seed: int = 42,
                   l3_options: Optional[Dict[str, Any]] = None,
                   memory_options: Optional[Dict[str, Any]] = None,
                   group_config_spec: Optional[Dict[str, Any]] = None,
                   no_force_stop: bool = False,
                   manifest_audit: Optional[Dict[str, Any]] = None,
                   episode_order: Optional[Dict[Any, Any]] = None,
                   episode_metadata: Optional[Dict[Any, Any]] = None,
                   execution_manifest=None,
                   execution_report: Optional[Dict[str, Any]] = None,
                   exp_config: str = "",
                   allow_nonpilot_eval: bool = False) -> None:

    if experiment_group != "uninavid_baseline" and model_name != "uni-navid":
        raise ValueError("the memory experiment groups are only defined for uni-navid")
    if experiment_group in EIGHT_ABLATION_GROUPS and group_config_spec is None:
        raise ValueError(
            "the eight prompt-ablation groups require their --group-config"
        )
    if (
        experiment_group == "uninavid_baseline"
        and isinstance(l3_options, dict)
        and l3_options.get("seg_model")
    ):
        raise ValueError("baseline group must not receive an L3 segmentation model")
    if no_force_stop is False and os.environ.get("NO_FORCE_STOP") == "1":
        no_force_stop = True

    if episode_metadata is None and execution_manifest is not None:
        episode_metadata = metadata_from_execution_manifest(execution_manifest)

    env = Env(config.TASK_CONFIG, dataset)
    l3_sidecar = None
    memory_backend = None
    camera_spec = None
    l3_config = None
    if experiment_group == "uninavid_l3_only":
        l3_sidecar, camera_spec, l3_config = _build_l3_sidecar(
            config, l3_options or {},
        )
    elif experiment_group in EIGHT_ABLATION_GROUPS:
        memory_backend, camera_spec = _build_memory_backend(
            config, experiment_group, memory_options or {},
        )

    if model_name == "navid":
        from agent_navid import NaVid_Agent
        agent = NaVid_Agent(model_path, result_path, exp_save)

    elif model_name == "uni-navid":
        from agent_uninavid import UniNaVid_Agent
        agent = UniNaVid_Agent(
            model_path, result_path, exp_save,
            l3_sidecar=l3_sidecar, memory_backend=memory_backend,
            env=env, config=config, experiment_group=experiment_group,
            camera_spec=camera_spec,
            memory_token_budget=(
                True if not memory_options
                else bool(memory_options.get("token_budget", True))
            ),
        )
    else:
        raise ValueError(f"unknown model name: {model_name}")

    os.makedirs(result_path, exist_ok=True)
    execution_audit = (
        execution_manifest.audit() if execution_manifest is not None else None
    )
    run_metadata = {
        "group": experiment_group,
        "model_name": model_name,
        "model_path": str(Path(model_path).expanduser()),
        "exp_config": str(Path(exp_config).expanduser()) if exp_config else "",
        "split_num": int(split_num),
        "seed": int(seed),
        "seed_protocol": (
            "sha256(seed,scene,episode) per actual Episode, applied after "
            "env.reset() has identified the Episode and before agent.reset() "
            "and the model call"
        ),
        "exp_save": exp_save,
        "group_config": (
            group_config_artifact(group_config_spec)
            if group_config_spec is not None else None
        ),
        "group_semantics": group_semantics(experiment_group),
        "memory_options": (
            {key: value for key, value in (memory_options or {}).items()
             if key not in {"l1_model", "seg_model", "place_model"}}
        ),
        "memory_weights": {
            "l1_model": (memory_options or {}).get("l1_model") or None,
            "seg_model": (memory_options or {}).get("seg_model") or None,
            "place_model": (memory_options or {}).get("place_model") or None,
        },
        "no_force_stop": bool(no_force_stop),
        "max_episode_steps": (
            config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
            if getattr(getattr(config, "TASK_CONFIG", None), "ENVIRONMENT", None)
            is not None else None
        ),
        "pilot_manifest": (
            manifest_audit.get("path") if manifest_audit else None
        ),
        "pilot_manifest_file_sha256": (
            manifest_audit.get("file_sha256") if manifest_audit else None
        ),
        "pilot_manifest_expected_file_sha256": (
            EXPECTED_PILOT_MANIFEST_SHA256
        ),
        "pilot_expected_count": (
            manifest_audit.get("expected_count") if manifest_audit else None
        ),
        "formal_pilot": bool(
            not allow_nonpilot_eval
            and model_name == "uni-navid"
            and manifest_audit is not None
            and manifest_audit.get("file_sha256")
            == EXPECTED_PILOT_MANIFEST_SHA256
            and manifest_audit.get("expected_count") == EXPECTED_EPISODES
        ),
        "episode_manifest_verified": bool(
            ((manifest_audit or {}).get("episode_manifest") or {}).get("verified")
        ),
        "episode_manifest_aligned": bool(
            execution_manifest is not None
            and execution_manifest.canonical_sha256
            == execution_manifest.episode.canonical_sha256
        ),
        "execution_order": (
            manifest_audit.get("execution_order") if manifest_audit else None
        ),
        "execution_manifest_canonical_sha256": (
            execution_manifest.canonical_sha256
            if execution_manifest is not None else None
        ),
        "execution_manifest": execution_audit,
        "execution_manifest_verification": (
            {
                key: value for key, value in execution_report.items()
                if key != "created"
            }
            if execution_report is not None else None
        ),
        "manifests": {
            "pilot": {
                "path": (
                    manifest_audit.get("path") if manifest_audit else None
                ),
                "file_sha256": (
                    manifest_audit.get("file_sha256")
                    if manifest_audit else None
                ),
                "expected_file_sha256": EXPECTED_PILOT_MANIFEST_SHA256,
                "expected_count": (
                    manifest_audit.get("expected_count")
                    if manifest_audit else None
                ),
            },
            "parent": (
                manifest_audit.get("parent_manifest")
                if manifest_audit else None
            ),
            "episode": (
                manifest_audit.get("episode_manifest")
                if manifest_audit else None
            ),
            "execution": execution_audit,
        },
        "pilot_order_digest": (
            manifest_audit.get("pilot_order_digest_with_instructions")
            if manifest_audit else None
        ),
        "execution_order_digest": (
            manifest_audit.get("dataset_order_digest_with_instructions")
            if manifest_audit else None
        ),
        "l3": None if l3_config is None else {
            "mode": "inject",
            "seg_model": l3_options["seg_model"],
            "seg_device": l3_options["seg_device"],
            "seg_confidence": l3_options["seg_confidence"],
            "every_n_steps": l3_options["every_n_steps"],
            "evidence_min_iou": l3_options["evidence_min_iou"],
            "evidence_ambiguity_margin": l3_options["evidence_ambiguity_margin"],
            "min_confidence": l3_options["min_confidence"],
            "max_items": l3_options["max_items"],
            "max_chars": l3_options["max_chars"],
            "scope": l3_options["scope"],
            "camera": None if camera_spec is None else {
                "width": camera_spec.width,
                "height": camera_spec.height,
                "hfov_deg": camera_spec.hfov_deg,
                "min_depth_m": camera_spec.min_depth_m,
                "max_depth_m": camera_spec.max_depth_m,
                "normalize_depth": camera_spec.normalize_depth,
                "rgb_key": camera_spec.rgb_key,
                "depth_key": camera_spec.depth_key,
            },
        },
    }
    _write_run_metadata(os.path.join(result_path, "run_config.json"), run_metadata)
    if memory_backend is not None:
        from l1_memory.prompt_modes import effective_layer_activation
        from l3_integration.official_prompt import (
            REFERENCE_OFFICIAL_PROMPT_TEMPLATE,
            prompt_sha256,
        )

        _write_run_metadata(
            os.path.join(result_path, "prompt_protocol.json"),
            {
                "protocol_version": "uninavid_l1_l3_prompt_ablation_v1",
                "group": experiment_group,
                "prompt_mode": memory_backend.plan.mode,
                "activation": effective_layer_activation(memory_backend.plan),
                "group_config": (
                    group_config_artifact(group_config_spec)
                    if group_config_spec is not None else None
                ),
                "official_prompt_sha256": prompt_sha256(
                    REFERENCE_OFFICIAL_PROMPT_TEMPLATE
                ),
                "injection_point": "official template single {} task slot after the original instruction",
                "budget_policy": (
                    "token_aware_context_with_char_fallback"
                    if (memory_options or {}).get("token_budget", True)
                    else "char_budget"
                ),
                "generation": {
                    "do_sample": True, "temperature": 0.2,
                    "max_new_tokens": 1024, "use_cache": True,
                },
                "no_force_stop": bool(no_force_stop),
            },
        )

    num_episodes = len(env.episodes)
    assigned_episodes = list(env.episodes)
    if len(assigned_episodes) != num_episodes:
        raise RuntimeError("env.episodes changed length before evaluation")
    tracker = EpisodeExecutionTracker(
        assigned_episodes,
        episode_metadata,
        require_full_metadata=(
            episode_metadata is not None
            and split_num == 1
            and num_episodes > 0
        ),
        context=(
            "the frozen 88-Episode Pilot"
            if episode_metadata is not None else "this evaluation split"
        ),
    )
    if tracker.expected_count != num_episodes:
        raise RuntimeError(
            "the Episode identity tracker did not receive the complete "
            "assigned execution set"
        )
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS

    target_key = {
        "distance_to_goal", "success", "spl", "ndtw", "path_length",
        "oracle_success", "steps_taken",
    }

    results_path = os.path.join(result_path, "results.jsonl")
    try:
        for habitat_execution_index in trange(
            num_episodes, desc=config.EVAL.IDENTIFICATION+"-{}".format(split_id),
        ):
            episode_started = time.perf_counter()
            try:
                obs = env.reset()
            except StopIteration as exc:
                missing = tracker.missing_keys()
                raise EpisodeExecutionError(
                    "Habitat's Episode iterator ended before the expected "
                    f"execution set was covered; missing {len(missing)} "
                    "Episode(s): "
                    + ", ".join(
                        f"{scene}/{episode}"
                        for scene, episode in missing[:5]
                    )
                    + (" ..." if len(missing) > 5 else "")
                ) from exc

            # The actual Episode, not the dataset/list position, is the only
            # source of truth for identity and manifest metadata.
            episode = env.current_episode
            episode_key, episode_meta = tracker.accept(episode)
            scene_id = str(episode.scene_id)
            episode_id = str(episode.episode_id)
            episode_seed = _stable_episode_seed(seed, scene_id, episode_id)
            _seed_episode_rngs(episode_seed)
            instruction = episode_instruction(episode)
            if (
                episode_meta is not None
                and episode_meta.instruction is not None
                and instruction != episode_meta.instruction
            ):
                raise EpisodeExecutionError(
                    "Episode instruction differs from the frozen manifest for "
                    f"{episode_key[0]}/{episode_key[1]}"
                )
            if model_name == "uni-navid":
                agent.reset(episode=episode, instruction=instruction)
            else:
                agent.reset()
            iter_step = 0

            continuse_rotation_count = 0
            last_dtg = 999
            while not env.episode_over:

                info = env.get_metrics()

                if info["distance_to_goal"] != last_dtg:
                    last_dtg = info["distance_to_goal"]
                    continuse_rotation_count=0
                else :
                    continuse_rotation_count +=1


                action = agent.act(obs, info, episode_id)

                if (
                    not no_force_stop
                    and (
                        continuse_rotation_count > EARLY_STOP_ROTATION
                        or iter_step > EARLY_STOP_STEPS
                    )
                ):
                    action = {"action": 0}

                iter_step+=1
                obs = env.step(action)

            info = env.get_metrics()
            episode_time_sec = time.perf_counter() - episode_started
            result_dict = {
                k: _json_value(info[k]) for k in target_key if k in info
            }
            result_dict["episode_time_sec"] = episode_time_sec
            result_dict["iter_steps"] = iter_step
            result_dict["success"] = bool(info.get("success", False))
            result_dict["spl"] = _finite_float(info.get("spl"))
            result_dict["ndtw"] = _finite_float(info.get("ndtw"))
            result_dict["oracle_success"] = bool(info.get("oracle_success", False))
            result_dict["osr"] = result_dict["oracle_success"]
            result_dict["distance_to_goal"] = _finite_float(
                info.get("distance_to_goal")
            )
            result_dict["ne"] = result_dict["distance_to_goal"]
            result_dict["path_length"] = _finite_float(info.get("path_length"))
            result_dict["tl"] = result_dict["path_length"]
            result_dict["steps_taken"] = _json_value(info.get("steps_taken"))
            if hasattr(agent, "episode_summary"):
                result_dict.update(agent.episode_summary())

            parent_manifest = (manifest_audit or {}).get("parent_manifest") or {}
            episode_manifest = (
                (manifest_audit or {}).get("episode_manifest") or {}
            )
            # Authoritative identity and manifest metadata, written after any
            # optional agent summary so a test double cannot override them.
            result_dict.update({
                "id": episode_id,
                "episode_id": episode_id,
                "scene_id": scene_id,
                "scene_basename": episode_key[0],
                "instruction": instruction,
                "instruction_sha256": hashlib.sha256(
                    instruction.encode("utf-8")
                ).hexdigest(),
                "episode_seed": episode_seed,
                # Diagnostic only: the actual Habitat reset position.  It is
                # never used to look up manifest metadata.
                "habitat_execution_index": habitat_execution_index,
                "manifest_order_index": (
                    episode_meta.manifest_order_index
                    if episode_meta is not None
                    else (episode_order or {}).get(episode_key)
                ),
                "pilot_manifest_index": (
                    episode_meta.pilot_manifest_index
                    if episode_meta is not None else None
                ),
                "distance_group": (
                    episode_meta.distance_group
                    if episode_meta is not None else None
                ),
                "execution_order": (
                    (manifest_audit or {}).get("execution_order")
                ),
                "group": experiment_group,
                "experiment_group": experiment_group,
                "pilot_manifest_file_sha256": (
                    manifest_audit.get("file_sha256")
                    if manifest_audit else None
                ),
                "parent_manifest_sha256": parent_manifest.get(
                    "declared_sha256_in_pilot"
                ),
                "episode_manifest_canonical_sha256": episode_manifest.get(
                    "canonical_sha256"
                ),
                "episode_manifest_verified": bool(
                    episode_manifest.get("verified")
                ),
            })

            if "data" in exp_save:
                with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(episode_id)), "w") as f:
                    json.dump(_json_safe(result_dict), f, indent=4)

            with open(results_path, "a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(_json_safe(result_dict), ensure_ascii=False) + "\n"
                )
        tracker.assert_complete()
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()

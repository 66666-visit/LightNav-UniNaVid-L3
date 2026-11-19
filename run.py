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

from habitat.datasets import make_dataset
from VLN_CE.vlnce_baselines.config.default import get_config

from habitat import Env
from tqdm import trange
import numpy as np

from l3_integration.pilot_manifest import (
    EXPECTED_EPISODES,
    EXPECTED_EPISODE_MANIFEST_SHA256,
    EXPECTED_PILOT_MANIFEST_SHA256,
    audit_manifest,
    episode_instruction,
    load_pilot_manifest,
    scene_basename,
    select_pilot_episodes,
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
            "canonical completion-marker digest and Episode order are verified"
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
        choices=("uninavid_baseline", "uninavid_l3_only"),
        default="uninavid_baseline",
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
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError(
                f"result path already contains a different run_config.json: {target}"
            )
        return
    target.write_text(
        json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
    if experiment_group == "uninavid_l3_only" and not l3_seg_model:
        raise ValueError("uninavid_l3_only requires --l3-seg-model")
    if experiment_group == "uninavid_baseline" and l3_seg_model:
        raise ValueError("baseline group must not receive an L3 segmentation model")
    if experiment_group != "uninavid_baseline" and model_name != "uni-navid":
        raise ValueError("the L3 experiment groups are only defined for uni-navid")
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
    if not isinstance(allow_nonpilot_eval, bool):
        raise ValueError("allow_nonpilot_eval must be explicit boolean")
    if episode_manifest and not pilot_manifest:
        raise ValueError(
            "--episode-manifest requires --pilot-manifest for the ordered "
            "88-Episode identity comparison"
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
            "the historical evaluation manifest so its canonical digest and "
            "ordered 88 Episodes can be verified"
        )

    config = get_config(exp_config, opts)

    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(seed)

    manifest = None
    manifest_audit = None
    episode_order = {}
    if pilot_manifest:
        effective_pilot_sha = (
            None if allow_nonpilot_eval and not pilot_sha256
            else (pilot_sha256 or EXPECTED_PILOT_MANIFEST_SHA256)
        )
        manifest = load_pilot_manifest(
            pilot_manifest, expected_sha256=effective_pilot_sha,
        )
        selected = select_pilot_episodes(dataset.episodes, manifest)
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
        dataset_split = copy.copy(dataset)
        dataset_split.episodes = selected[split_id::split_num]
        instructions = [episode_instruction(episode) for episode in selected]
        manifest_audit = audit_manifest(
            pilot_manifest,
            expected_sha256=effective_pilot_sha,
            dataset_episodes=dataset.episodes,
            episode_manifest=episode_manifest or None,
            expected_episode_manifest_sha256=(
                episode_manifest_sha256 or None
            ),
            parent_manifest=parent_manifest or None,
        )
        manifest_audit["dataset_order_digest_with_instructions"] = (
            manifest.order_digest(instructions=instructions)
        )
        manifest_audit["shard"] = {
            "split_num": split_num, "split_id": split_id,
            "selected_count": len(dataset_split.episodes),
        }
        for index, episode in enumerate(selected):
            episode_order[
                (scene_basename(episode.scene_id), str(episode.episode_id))
            ] = index
    else:
        dataset_split = dataset.get_splits(split_num)[split_id]
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
        manifest_audit=manifest_audit,
        episode_order=episode_order,
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


def evaluate_agent(config, split_id, dataset, model_path, result_path, model_name,
                   exp_save, *, split_num: int = 1,
                   experiment_group: str = "uninavid_baseline",
                   seed: int = 42, l3_options: dict | None = None,
                   manifest_audit: dict | None = None,
                   episode_order: dict | None = None,
                   exp_config: str = "",
                   allow_nonpilot_eval: bool = False) -> None:

    if experiment_group == "uninavid_l3_only" and model_name != "uni-navid":
        raise ValueError("uninavid_l3_only is only defined for uni-navid")
    if (
        experiment_group == "uninavid_baseline"
        and isinstance(l3_options, dict)
        and l3_options.get("seg_model")
    ):
        raise ValueError("baseline group must not receive an L3 segmentation model")

    env = Env(config.TASK_CONFIG, dataset)
    l3_sidecar = None
    camera_spec = None
    l3_config = None
    if experiment_group == "uninavid_l3_only":
        l3_sidecar, camera_spec, l3_config = _build_l3_sidecar(
            config, l3_options or {},
        )

    if model_name == "navid":
        from agent_navid import NaVid_Agent
        agent = NaVid_Agent(model_path, result_path, exp_save)

    elif model_name == "uni-navid":
        from agent_uninavid import UniNaVid_Agent
        if experiment_group == "uninavid_l3_only":
            agent = UniNaVid_Agent(
                model_path, result_path, exp_save, l3_sidecar=l3_sidecar,
                env=env, config=config, experiment_group=experiment_group,
            )
        else:
            agent = UniNaVid_Agent(
                model_path, result_path, exp_save, l3_sidecar=None,
                env=env, config=config, experiment_group=experiment_group,
            )
    else:
        raise ValueError(f"unknown model name: {model_name}")

    os.makedirs(result_path, exist_ok=True)
    run_metadata = {
        "group": experiment_group,
        "model_name": model_name,
        "model_path": str(Path(model_path).expanduser()),
        "exp_config": str(Path(exp_config).expanduser()) if exp_config else "",
        "split_num": int(split_num),
        "seed": int(seed),
        "seed_protocol": (
            "sha256(seed,scene,episode) per Episode, applied before env.reset() "
            "and before the model call"
        ),
        "exp_save": exp_save,
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
            manifest_audit is not None
            and manifest_audit.get("episode_manifest", {}).get("verified")
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
        },
        "pilot_order_digest": (
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

    num_episodes = len(env.episodes)
    planned_episodes = list(env.episodes)
    if len(planned_episodes) != num_episodes:
        raise RuntimeError("env.episodes changed length before evaluation")
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS

    target_key = {
        "distance_to_goal", "success", "spl", "ndtw", "path_length",
        "oracle_success", "steps_taken",
    }

    results_path = os.path.join(result_path, "results.jsonl")
    try:
        for episode_index in trange(
            num_episodes, desc=config.EVAL.IDENTIFICATION+"-{}".format(split_id),
        ):
            episode_started = time.perf_counter()
            planned_episode = planned_episodes[episode_index]
            episode_seed = _stable_episode_seed(
                seed, str(planned_episode.scene_id),
                str(planned_episode.episode_id),
            )
            # Seed before env.reset() as well as before the model call so both
            # groups start the same Episode from the same RNG state.
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            try:
                import torch
                torch.manual_seed(episode_seed)
            except ImportError:
                pass
            obs = env.reset()
            episode = env.current_episode
            if (
                str(episode.scene_id), str(episode.episode_id)
            ) != (
                str(planned_episode.scene_id), str(planned_episode.episode_id)
            ):
                raise RuntimeError(
                    "env.reset() did not return the Episode at the expected "
                    f"index {episode_index}; manifest order would be violated"
                )
            instruction = episode_instruction(episode)
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


                action = agent.act(obs, info, env.current_episode.episode_id)

                if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step>EARLY_STOP_STEPS:
                    action = {"action": 0}

                iter_step+=1
                obs = env.step(action)

            info = env.get_metrics()
            episode_time_sec = time.perf_counter() - episode_started
            result_dict = dict()
            result_dict = {k: _json_value(info[k]) for k in target_key if k in info}
            result_dict["id"] = env.current_episode.episode_id
            result_dict["scene_id"] = str(env.current_episode.scene_id)
            result_dict["instruction"] = instruction
            result_dict["instruction_sha256"] = hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest()
            result_dict["episode_time_sec"] = episode_time_sec
            result_dict["episode_seed"] = episode_seed
            result_dict["iter_steps"] = iter_step
            result_dict["group"] = experiment_group
            result_dict["pilot_manifest_file_sha256"] = (
                manifest_audit.get("file_sha256") if manifest_audit else None
            )
            result_dict["parent_manifest_sha256"] = (
                manifest_audit.get("parent_manifest", {}).get(
                    "declared_sha256_in_pilot"
                )
                if manifest_audit else None
            )
            result_dict["episode_manifest_canonical_sha256"] = (
                manifest_audit.get("episode_manifest", {}).get(
                    "canonical_sha256"
                )
                if manifest_audit else None
            )
            result_dict["episode_manifest_verified"] = bool(
                manifest_audit is not None
                and manifest_audit.get("episode_manifest", {}).get("verified")
            )
            result_dict["manifest_order_index"] = (
                (episode_order or {}).get(
                    (scene_basename(env.current_episode.scene_id),
                     str(env.current_episode.episode_id))
                )
            )
            result_dict["scene_basename"] = scene_basename(
                env.current_episode.scene_id
            )
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
            # Authoritative Episode identity from the environment, after any
            # optional agent summary (the summary may omit it in a test double).
            result_dict["id"] = str(env.current_episode.episode_id)
            result_dict["episode_id"] = str(env.current_episode.episode_id)
            result_dict["scene_id"] = str(env.current_episode.scene_id)
            result_dict["scene_basename"] = scene_basename(
                env.current_episode.scene_id
            )
            result_dict["instruction"] = instruction
            result_dict["instruction_sha256"] = hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest()
            result_dict["group"] = experiment_group
            result_dict["experiment_group"] = experiment_group

            if "data" in exp_save:
                with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
                    json.dump(_json_safe(result_dict), f, indent=4)

            with open(results_path, "a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(_json_safe(result_dict), ensure_ascii=False) + "\n"
                )
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()

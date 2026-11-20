"""L3-only sidecar: real map update + official-style retrieval for Uni-NaVid.

The sidecar owns one ``ObjectMap`` and one L3 evidence store per Episode.  It
is deliberately not a prompt designer: it calls the vendored LightNav L3
pipeline (``make_spatial_evidence`` -> ``ObjectMap.observe``) and the vendored
official-style serializer (``L3OfficialPromptOrganizer``).  It passes an empty
L1 event list, so no L1 text or L1 fusion can enter the map or the prompt.

No Habitat import is required here.  The default frame builder is the pure
``frame_from_habitat`` adapter; tests inject a synthetic frame builder.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from l3_memory.conceptgraphs_lite import ObjectMap
from l3_memory.l3_official_prompt import L3OfficialPromptOrganizer
from l3_memory.l3_prompt_organizer import L3PromptOrganizer
from l3_memory.spatial_graph_adapter import make_spatial_evidence
from l3_memory.spatial_l3_config import SpatialL3Config
from l3_memory.v3_store import V3MemoryStore
from l3_memory.yolo_seg_extractor import YOLOSegExtractor

from l3_integration.habitat_geometry import (
    CameraSpec,
    frame_from_habitat,
)

logger = logging.getLogger(__name__)

FrameBuilder = Callable[
    [Mapping[str, Any], Any, CameraSpec], Tuple[Any, Dict[str, Any]]
]



class L3MapSidecar:
    """Episode-isolated L3 map and official-style context builder."""

    def __init__(
        self,
        config: SpatialL3Config,
        *,
        camera_spec: CameraSpec,
        max_items: int = 2,
        max_chars: int = 600,
        scope: str = "episode_local",
        extractor: Optional[Any] = None,
        frame_builder: Optional[FrameBuilder] = None,
        organizer: Optional[L3OfficialPromptOrganizer] = None,
    ) -> None:
        if not isinstance(config, SpatialL3Config):
            raise ValueError("config must be a SpatialL3Config")
        if config.mode not in {"shadow", "inject"}:
            raise ValueError("L3 sidecar requires mode 'shadow' or 'inject'")
        if not isinstance(camera_spec, CameraSpec):
            raise ValueError("camera_spec must be a CameraSpec")
        if scope != "episode_local":
            raise ValueError(
                "L3 may not persist across Episodes; scope must be episode_local"
            )
        if type(max_items) is not int or max_items < 1:
            raise ValueError("max_items must be a positive integer")
        if type(max_chars) is not int or max_chars < 1:
            raise ValueError("max_chars must be a positive integer")
        self.config = config
        self.camera_spec = camera_spec
        self.max_items = max_items
        self.max_chars = max_chars
        self.scope = scope
        self.extractor = extractor if extractor is not None else YOLOSegExtractor(
            config.seg_model_path,
            device=config.seg_device,
            confidence=config.seg_confidence,
            every_n_steps=config.seg_every_n_steps,
        )
        self.frame_builder = frame_builder or frame_from_habitat
        self.organizer = organizer or L3OfficialPromptOrganizer(
            max_items=max_items,
            collector=L3PromptOrganizer(max_items=max_items),
        )
        self.source = (
            "yolo11n-seg:"
            + Path(config.seg_model_path).name
            if config.seg_model_path else "yolo11n-seg"
        )
        self.active = False
        self.store: Optional[V3MemoryStore] = None
        self.object_map: Optional[ObjectMap] = None
        self.scene_id: Optional[str] = None
        self.episode_id: Optional[str] = None
        self.instruction = ""
        self.last_frame = None
        self.last_frame_step: Optional[int] = None
        self.last_observation_diagnostic: Dict[str, Any] = {}
        self.last_context_diagnostic: Dict[str, Any] = {}
        self.last_context_text: Optional[str] = None
        self._step_observations = 0
        self._segmentation_attempts = 0

    # -- lifecycle ------------------------------------------------------------

    @staticmethod
    def _instruction_from_episode(episode: Any) -> str:
        instruction = getattr(episode, "instruction", None)
        if instruction is None:
            return ""
        if isinstance(instruction, str):
            return instruction
        if isinstance(instruction, Mapping):
            return str(
                instruction.get("instruction_text")
                or instruction.get("text")
                or ""
            )
        return str(
            getattr(instruction, "instruction_text", None)
            or getattr(instruction, "text", None)
            or ""
        )

    def reset_episode(
        self, episode: Any, instruction: str = "",
    ) -> Dict[str, Any]:
        """Start a new Episode with brand-new map and store objects."""
        if episode is None:
            raise ValueError("episode is required to reset the L3 map")
        scene_id = getattr(episode, "scene_id", None)
        episode_id = getattr(episode, "episode_id", None)
        if not scene_id or not episode_id:
            raise ValueError("episode requires scene_id and episode_id")
        if not instruction:
            instruction = self._instruction_from_episode(episode)
        self.scene_id = str(scene_id)
        self.episode_id = str(episode_id)
        self.instruction = str(instruction)

        # New objects, not clear(): no previous Episode reference can survive.
        self.object_map = ObjectMap(self.config.geometry)
        self.object_map.reset_episode(self.scene_id, self.episode_id, rollout_idx=0)
        self.store = V3MemoryStore(scope=self.scope)
        self.store.begin_episode(
            self.scene_id, self.episode_id, self.instruction, rollout_idx=0,
        )
        reset = getattr(self.extractor, "reset_episode", None)
        if callable(reset):
            reset(self.episode_id, self.instruction)
        self.active = True
        self.last_frame = None
        self.last_frame_step = None
        self.last_observation_diagnostic = {}
        self.last_context_diagnostic = {}
        self.last_context_text = None
        self._step_observations = 0
        self._segmentation_attempts = 0
        return {
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "instruction_sha256": _sha256(self.instruction),
            "scope": self.scope,
        }

    def close_episode(self) -> None:
        close = getattr(self.extractor, "close_episode", None)
        if callable(close):
            close()
        self.active = False
        self.last_frame = None
        self.last_frame_step = None
        self.last_context_text = None

    # -- map update -----------------------------------------------------------

    def observe(
        self,
        observations: Mapping[str, Any],
        agent_state: Any,
        step: int,
        *,
        transition_record: bool = False,
    ) -> Dict[str, Any]:
        """Update the L3 map from the current RGB-D observation.

        Failures are recorded in the returned diagnostic and never raised into
        the navigation loop.  A missing depth channel, pose, calibration or
        segmentation dependency therefore leaves the official RGB path intact
        and marks L3 unavailable for that step.
        """
        if not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        diagnostic: Dict[str, Any] = {
            "step": step,
            "l3_active": self.active,
            "frame_valid": False,
            "frame_error": None,
            "segmentation_count": 0,
            "segmentation_ms": None,
            "segmentation_error": None,
            "evidence_minted": 0,
            "evidence_unmatched": 0,
            "observed_objects": 0,
            "object_nodes": 0,
            "viewpoints": 0,
            "observed_from_edges": 0,
            "traversed_to_edges": 0,
            "observe_ms": 0.0,
        }
        if not self.active:
            diagnostic["frame_error"] = "no active Episode"
            self.last_observation_diagnostic = diagnostic
            return diagnostic
        if self.last_frame_step is not None and step <= self.last_frame_step:
            diagnostic["frame_error"] = (
                f"non-increasing map step {step} after {self.last_frame_step}"
            )
            self.last_observation_diagnostic = diagnostic
            return diagnostic

        started = time.perf_counter()
        try:
            frame, _metadata = self.frame_builder(
                observations, agent_state, self.camera_spec,
            )
            diagnostic["frame_valid"] = True
            self.last_frame = frame
            self.last_frame_step = step
            instances = self.extractor.extract(frame.rgb, step)
            self._segmentation_attempts += 1
            diagnostic["segmentation_count"] = len(instances)
            diagnostic["segmentation_ms"] = getattr(
                self.extractor, "last_detection_ms", None,
            )
            diagnostic["segmentation_error"] = getattr(
                self.extractor, "last_error", None,
            )
            masks, counts = make_spatial_evidence(
                instances, [], self.store,
                scene_id=self.scene_id,
                episode_id=self.episode_id,
                rollout_idx=0,
                step=step,
                source=self.source,
                min_iou=self.config.evidence_min_iou,
                ambiguity_margin=self.config.evidence_ambiguity_margin,
            )
            diagnostic["evidence_minted"] = counts["minted"]
            # No L1 events are supplied, so every minted mask is "unmatched".
            # This is the explicit L1-off state, not a detection failure.
            diagnostic["evidence_unmatched"] = counts["unmatched"]
            observed = self.object_map.observe(
                frame, masks, step_id=step,
                transition_record=bool(transition_record),
            )
            diagnostic["observed_objects"] = len(observed)
            diagnostic["object_nodes"] = len(self.object_map.objects)
            diagnostic["viewpoints"] = len(self.object_map.viewpoints)
            diagnostic["observed_from_edges"] = sum(
                edge.relation_type == "observed_from" for edge in self.object_map.edges
            )
            diagnostic["traversed_to_edges"] = sum(
                edge.relation_type == "traversed_to" for edge in self.object_map.edges
            )
            self._step_observations += 1
        except Exception as exc:  # noqa: BLE001 - optional sidecar must fail closed
            diagnostic["frame_valid"] = False
            diagnostic["frame_error"] = f"{type(exc).__name__}: {exc}"
            self.last_frame = None
            self.last_frame_step = None
            logger.warning(
                "L3 map update unavailable at step %s: %s", step, diagnostic["frame_error"],
            )
        diagnostic["observe_ms"] = (time.perf_counter() - started) * 1000.0
        self.last_observation_diagnostic = diagnostic
        return diagnostic

    # -- prompt context -------------------------------------------------------

    def build_context(
        self, instruction: str, step: int,
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Retrieve and serialize L3 facts; return ``(organization, diagnostic)``."""
        diagnostic: Dict[str, Any] = {
            "step": step,
            "context_available": False,
            "context_injected": False,
            "context_reason": None,
            "context_chars": 0,
            "fact_count": 0,
            "candidate_count": 0,
            "verified_count": 0,
            "unverified_count": 0,
            "current_only_skipped": 0,
            "omitted_items": 0,
            "empty_reason": None,
            "retrieval_ms": 0.0,
            "expression": None,
        }
        if not self.active or self.object_map is None or self.store is None:
            diagnostic["context_reason"] = "no active Episode"
            self.last_context_diagnostic = diagnostic
            return None, diagnostic
        if self.last_frame is None or self.last_frame_step != step:
            diagnostic["context_reason"] = "current RGB-D or pose unavailable"
            self.last_context_diagnostic = diagnostic
            return None, diagnostic
        if not isinstance(instruction, str):
            diagnostic["context_reason"] = "instruction unavailable"
            self.last_context_diagnostic = diagnostic
            return None, diagnostic

        started = time.perf_counter()
        try:
            organization = self.organizer.build(
                self.store, self.object_map,
                scene_id=self.scene_id,
                episode_id=self.episode_id,
                rollout_idx=0,
                step=step,
                instruction=instruction,
                frame=self.last_frame,
                scope=self.scope,
                max_chars=self.max_chars,
                min_confidence=self.config.min_injection_confidence,
            )
            diagnostic["context_available"] = organization.context is not None
            diagnostic["context_injected"] = organization.context is not None
            diagnostic["candidate_count"] = organization.candidate_count
            diagnostic["verified_count"] = organization.verified_count
            diagnostic["unverified_count"] = organization.unverified_count
            diagnostic["current_only_skipped"] = organization.current_only_skipped
            diagnostic["omitted_items"] = organization.omitted_items
            diagnostic["fact_count"] = len(organization.selected_facts)
            diagnostic["empty_reason"] = organization.empty_reason
            diagnostic["expression"] = (
                dict(organization.expression) if organization.expression else None
            )
            if organization.context is not None:
                text = organization.context.text
                diagnostic["context_chars"] = len(text)
                self.last_context_text = text
            else:
                self.last_context_text = None
                diagnostic["context_reason"] = organization.empty_reason or "no L3 fact"
        except Exception as exc:  # noqa: BLE001
            diagnostic["context_reason"] = f"{type(exc).__name__}: {exc}"
            self.last_context_text = None
            organization = None
            logger.warning("L3 context unavailable at step %s: %s", step, exc)
        diagnostic["retrieval_ms"] = (time.perf_counter() - started) * 1000.0
        self.last_context_diagnostic = diagnostic
        return organization, diagnostic

    # -- audit ----------------------------------------------------------------

    def map_snapshot(self) -> Dict[str, Any]:
        """Small, JSON-safe map summary for per-Episode diagnostics."""
        if self.object_map is None:
            return {"objects": 0, "viewpoints": 0, "evidence": 0, "edges": 0}
        labels: Dict[str, int] = {}
        for obj in self.object_map.objects.values():
            labels[obj.class_name] = labels.get(obj.class_name, 0) + 1
        return {
            "objects": len(self.object_map.objects),
            "viewpoints": len(self.object_map.viewpoints),
            "evidence": len(self.object_map.evidence),
            "edges": len(self.object_map.edges),
            "labels": dict(sorted(labels.items())),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "active": self.active,
            "scope": self.scope,
            "step_observations": self._step_observations,
            "segmentation_attempts": self._segmentation_attempts,
            "map": self.map_snapshot(),
        }


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()

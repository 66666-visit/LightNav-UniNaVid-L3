"""Episode-local L3 map update + official-style retrieval for Uni-NaVid.

The sidecar owns one ``ObjectMap`` and one L3 evidence store per Episode.  It
is deliberately not a prompt designer: it calls the vendored LightNav L3
pipeline (``make_spatial_evidence`` -> ``ObjectMap.observe``) and the vendored
official-style serializer (``L3OfficialPromptOrganizer``).  The baseline
passes no L1 events.  SBAdir may add verified same-frame L1 evidence links;
these are provenance only and do not change the mask, map, or fact serializer.

No Habitat import is required here.  The default frame builder is the pure
``frame_from_habitat`` adapter; tests inject a synthetic frame builder.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

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


class _Py38Description(str):
    """Supply ``str.removeprefix`` to the frozen LightNav matcher on Python 3.8."""

    def removeprefix(self, prefix):
        return self[len(prefix):] if self.startswith(prefix) else str(self)



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
        self.cross_sensor_links: Dict[str, str] = {}

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
        self.cross_sensor_links = {}
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
        l1_events: Optional[list[Any]] = None,
        l1_store: Optional[V3MemoryStore] = None,
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
            "evidence_linked": 0,
            "evidence_ambiguous": 0,
            "cross_label_entity_links": 0,
            "cross_label_entity_link_error": None,
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
            linked_events = []
            if l1_events and l1_store is not None:
                from l3_memory.v3_schemas import canonical_evidence_key
                for event in l1_events:
                    if (getattr(event, "kind", None) not in
                            {"object_observation", "landmark"}
                            or getattr(event, "step_id", None) != step):
                        continue
                    key = canonical_evidence_key(
                        self.scene_id, self.episode_id, 0, event.event_id,
                    )
                    record = l1_store.get_evidence(key)
                    if (record is None or record.scene_id != self.scene_id
                            or record.episode_id != self.episode_id
                            or record.step != step
                            or record.observation_status != "observed"
                            or not record.provenance_validity):
                        continue
                    if self.store.get_evidence(key) is None:
                        self.store.add(record)
                    linked_events.append(SimpleNamespace(
                        kind=event.kind,
                        description=_Py38Description(event.description),
                        metadata=event.metadata,
                        event_id=event.event_id,
                        step_id=event.step_id,
                    ))
            masks, counts = make_spatial_evidence(
                instances, linked_events, self.store,
                scene_id=self.scene_id,
                episode_id=self.episode_id,
                rollout_idx=0,
                step=step,
                source=self.source,
                min_iou=self.config.evidence_min_iou,
                ambiguity_margin=self.config.evidence_ambiguity_margin,
            )
            diagnostic["evidence_minted"] = counts["minted"]
            diagnostic["evidence_unmatched"] = counts["unmatched"]
            diagnostic["evidence_linked"] = counts["linked"]
            diagnostic["evidence_ambiguous"] = counts["ambiguous"]
            observed = self.object_map.observe(
                frame, masks, step_id=step,
                transition_record=bool(transition_record),
            )
            if linked_events:
                try:
                    diagnostic["cross_label_entity_links"] = self._link_cross_label_boxes(
                        instances, linked_events,
                    )
                except Exception as exc:
                    diagnostic["cross_label_entity_link_error"] = (
                        "%s: %s" % (type(exc).__name__, exc)
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

    def _link_cross_label_boxes(self, instances, l1_events):
        """Link only unique same-frame boxes with strong mask overlap.

        This auxiliary identity proof is used for conflict recording.  It does
        not alter the L3 map, its evidence records, or its official serializer.
        A class disagreement is never resolved here.
        """
        from l3_memory.v3_schemas import canonical_evidence_key

        node_by_evidence = {
            edge.evidence_id: edge.object_id
            for edge in self.object_map.edges
            if edge.relation_type == "observed_from"
        }
        proposals = []
        for index, instance in enumerate(instances):
            l3_id = canonical_evidence_key(
                self.scene_id, self.episode_id, 0,
                "l3_seg_%d_%d" % (self.last_frame_step, index),
            )
            record = self.object_map.evidence.get(l3_id)
            if (record is None or record.geometry_status != "observed"
                    or l3_id not in node_by_evidence):
                continue
            mask = np.asarray(instance.mask)
            if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.any():
                continue
            sx1, sy1, sx2, sy2 = (float(value) for value in instance.bbox_xyxy)
            if not all(math.isfinite(value) for value in (sx1, sy1, sx2, sy2)):
                continue
            for event in l1_events:
                l1_label = str(event.description)
                if l1_label.startswith("Observed "):
                    l1_label = l1_label[len("Observed "):]
                cross_label = l1_label.strip().casefold() != instance.class_name.casefold()
                box = event.metadata.get("bbox_xyxy")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = (float(value) for value in box)
                except (TypeError, ValueError):
                    continue
                if (not all(math.isfinite(value) for value in (x1, y1, x2, y2))
                        or x1 < 0 or y1 < 0 or x2 > mask.shape[1]
                        or y2 > mask.shape[0] or x2 <= x1 or y2 <= y1):
                    continue
                overlap = max(0.0, min(sx2, x2) - max(sx1, x1)) * max(
                    0.0, min(sy2, y2) - max(sy1, y1)
                )
                union = (sx2 - sx1) * (sy2 - sy1) + (x2 - x1) * (y2 - y1) - overlap
                if union <= 0 or overlap / union < 0.7:
                    continue
                xlo, xhi = max(0, int(math.floor(x1))), min(mask.shape[1], int(math.ceil(x2)))
                ylo, yhi = max(0, int(math.floor(y1))), min(mask.shape[0], int(math.ceil(y2)))
                if float(mask[ylo:yhi, xlo:xhi].sum()) / float(mask.sum()) < 0.8:
                    continue
                l1_id = canonical_evidence_key(
                    self.scene_id, self.episode_id, 0, event.event_id,
                )
                proposals.append((l1_id, l3_id, node_by_evidence[l3_id], cross_label))
        by_l1 = {}
        by_l3 = {}
        for l1_id, l3_id, _node, _cross_label in proposals:
            by_l1[l1_id] = by_l1.get(l1_id, 0) + 1
            by_l3[l3_id] = by_l3.get(l3_id, 0) + 1
        added = 0
        for l1_id, l3_id, node_id, cross_label in proposals:
            if cross_label and by_l1[l1_id] == 1 and by_l3[l3_id] == 1:
                self.cross_sensor_links[l1_id] = node_id
                added += 1
        return added

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

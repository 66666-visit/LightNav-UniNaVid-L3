"""Uni-NaVid adapter around the real LightNav L1 + L3 memory backends.

This is deliberately a *thin* adapter, not a second algorithm:

* the L1 event writer, V3 evidence store, Local Graph fusion, organized L1/L3
  serializers, joint fusion and prompt assembly are the vendored LightNav
  modules, called with the same arguments;
* the two blocks copied from ``MemoryManager`` are ``_observe_place_v3`` and
  ``_persist_v3_events`` (the real L1 write path) and the small
  ``observe_spatial`` body (the real L3 write path).  The only changes are the
  RGB-D frame source (Uni-NaVid's Habitat observation instead of LightNav's
  server payload) and the omission of legacy RGM/V4 branches that the eight
  groups never enable.

No method here chooses an action or edits the official Uni-NaVid prompt.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import replace
from typing import Any

from l1_memory.episodic_memory import EpisodicMemory
from l1_memory.joint_prompt import JointPromptBuilder
from l1_memory.l1_prompt_organizer import L1PromptOrganizer
from l1_memory.local_graph_config import LocalGraphConfig
from l1_memory.local_graph_fusion import LocalGraphFusion
from l1_memory.memory_writer import MemoryWriter
from l1_memory.prompt_assembly import PromptContextResult, build_prompt_context
from l1_memory.prompt_modes import (
    PromptModePlan,
    effective_layer_activation,
    mode_plan,
    resolve_prompt_budget,
    validate_mode_backing,
)
from l1_memory.scene_graph import SceneGraph
from l1_memory.token_budget import (
    UNBOUNDED_CHAR_LIMIT,
    MemoryTokenBudget,
    TokenBudgetFailure,
)
from l1_memory.working_memory import WorkingMemoryStore
from l3_memory.conceptgraphs_lite import ObjectMap, RGBDFrame
from l3_memory.l3_official_prompt import L3OfficialPromptOrganizer
from l3_memory.l3_prompt_organizer import L3PromptOrganizer
from l3_memory.schemas import MemoryEvent, ObservationPacket
from l3_memory.spatial_graph_adapter import SpatialGraphAdapter, make_spatial_evidence
from l3_memory.spatial_l3_config import SpatialL3Config
from l3_memory.v3_schemas import EvidenceRecord
from l3_memory.v3_store import V3MemoryStore

logger = logging.getLogger(__name__)

#: The eight ablation groups accepted by :func:`group_config`/``run.py``.
EIGHT_ABLATION_GROUPS: tuple[str, ...] = (
    "l1_original_only",
    "l1_organized_only",
    "l3_original_only",
    "l3_organized_only",
    "l1_l3_original_concat",
    "l1_l3_joint",
    "l1_original_l3_organized_concat",
    "l1_original_l3_organized_joint",
)

#: The existing two-group L3 integration stays available and unchanged.
LEGACY_GROUPS: tuple[str, ...] = ("uninavid_baseline", "uninavid_l3_only")

ALL_GROUPS: tuple[str, ...] = (*LEGACY_GROUPS, *EIGHT_ABLATION_GROUPS)


def group_config(group: str) -> dict[str, Any]:
    """Return the explicit, machine-checkable semantics of one group.

    The values are derived from LightNav's ``prompt_modes`` plans; the mapping
    below is the single place where a Uni-NaVid group name is bound to a plan.
    """
    if group in LEGACY_GROUPS:
        return {
            "group": group,
            "prompt_mode": "legacy" if group == "uninavid_baseline" else "l3_official_style_only",
            "legacy_l3_sidecar": group == "uninavid_l3_only",
            "l1_prompt": False,
            "l3_prompt": group == "uninavid_l3_only",
            "l1_evidence_writer": False,
            "l3_map": group == "uninavid_l3_only",
            "combination": "legacy" if group == "uninavid_baseline" else "single",
            "requires_l1_model": False,
            "requires_seg_model": group == "uninavid_l3_only",
        }
    plan = mode_plan(group)
    l1_builder = {
        "original": "local_graph_fusion",
        "organized": "l1_prompt_organizer",
        None: None,
    }[plan.l1_source]
    l3_builder = {
        "original": "spatial_graph_adapter",
        "organized": "l3_prompt_organizer",
        "official": "l3_official_prompt",
        None: None,
    }[plan.l3_source]
    return {
        "group": group,
        "prompt_mode": plan.mode,
        "legacy_l3_sidecar": False,
        "l1_prompt": plan.uses_l1,
        "l3_prompt": plan.uses_l3,
        # Every eight-group condition runs the real L1 writer: it is the
        # observation layer that L3 links its evidence to.  Only the L1 *prompt
        # builder* is disabled in the L3-only conditions.
        "l1_evidence_writer": True,
        "l3_map": plan.uses_l3,
        "combination": plan.combination,
        "l1_builder": l1_builder,
        "l3_builder": l3_builder,
        "requires_l1_model": True,
        "requires_seg_model": bool(plan.uses_l3),
        "preserves_l1_verbatim": plan.preserves_l1_verbatim,
        "activation": effective_layer_activation(plan),
    }


class PromptMemoryBackend:
    """Episode-isolated L1/L3 memory and one curated prompt context per step."""

    def __init__(
        self,
        prompt_mode: str,
        *,
        l1_extractor: Any | None = None,
        seg_extractor: Any | None = None,
        place_extractor: Any | None = None,
        max_landmarks: int = 2,
        max_context_chars: int = 600,
        prompt_max_chars: int = 320,
        scope: str = "episode_local",
        evidence_min_iou: float = 0.5,
        evidence_ambiguity_margin: float = 0.1,
        min_injection_confidence: float = 0.5,
        place_every_n_steps: int = 5,
        seg_model_path: str = "",
        seg_source: str = "yolo11n-seg",
        frame_builder: Callable[[Any, Any], RGBDFrame] | None = None,
    ) -> None:
        self.plan: PromptModePlan = mode_plan(prompt_mode)
        if self.plan.mode == "legacy":
            raise ValueError("legacy/baseline uses the official path, not this backend")
        if scope != "episode_local":
            raise ValueError(
                "this Uni-NaVid adapter is Episode-isolated; scope must be "
                "episode_local (cross-Episode L1 retention is not part of the "
                "eight formal groups)"
            )
        if type(max_landmarks) is not int or max_landmarks < 1:
            raise ValueError("max_landmarks must be a positive integer")
        if type(max_context_chars) is not int or max_context_chars < 1:
            raise ValueError("max_context_chars must be a positive integer")
        if type(prompt_max_chars) is not int or prompt_max_chars < 1:
            raise ValueError("prompt_max_chars must be a positive integer")
        if type(place_every_n_steps) is not int or place_every_n_steps < 1:
            raise ValueError("place_every_n_steps must be a positive integer")

        self.l1_extractor = l1_extractor
        self.seg_extractor = seg_extractor
        self.place_extractor = place_extractor
        self.place_every_n_steps = place_every_n_steps
        self.frame_builder = frame_builder
        self.max_landmarks = max_landmarks
        self.max_context_chars = max_context_chars
        self.scope = scope
        self.evidence_min_iou = evidence_min_iou
        self.evidence_ambiguity_margin = evidence_ambiguity_margin
        self.min_injection_confidence = min_injection_confidence
        self.seg_model_path = seg_model_path or seg_source
        self.seg_source = seg_source

        self.local_graph_config = LocalGraphConfig(
            enabled=True, scope=scope, prompt_max_chars=prompt_max_chars,
            max_landmarks=max_landmarks,
        )
        self.spatial_l3_config = SpatialL3Config(
            mode="inject" if self.plan.uses_l3 else "off",
            seg_model_path=self.seg_model_path,
            evidence_min_iou=evidence_min_iou,
            evidence_ambiguity_margin=evidence_ambiguity_margin,
            min_injection_confidence=min_injection_confidence,
        )
        validate_mode_backing(
            self.plan, local_graph_enabled=True,
            spatial_l3_mode=self.spatial_l3_config.mode,
        )
        if self.plan.uses_l1 and l1_extractor is None:
            raise ValueError(f"{self.plan.mode} requires a local L1 detector")
        if self.plan.uses_l3 and seg_extractor is None:
            raise ValueError(f"{self.plan.mode} requires an independent L3 segmenter")

        self.prompt_budget_chars = resolve_prompt_budget(
            max_context_chars=max_context_chars,
            local_graph_prompt_max_chars=prompt_max_chars,
        )
        self.l1_prompt_organizer = L1PromptOrganizer(max_items=max_landmarks)
        self.l3_prompt_organizer = L3PromptOrganizer(max_items=max_landmarks)
        self.l3_official_prompt_organizer = L3OfficialPromptOrganizer(
            max_items=max_landmarks, collector=self.l3_prompt_organizer,
        )
        self.joint_prompt_builder = JointPromptBuilder(max_items=max_landmarks + 1)
        self.local_graph_fusion = LocalGraphFusion(self.local_graph_config)
        # Experimental modes lift only the builder's internal character cap;
        # the token budget still limits the block after assembly.
        self.prompt_local_graph_fusion = LocalGraphFusion(replace(
            self.local_graph_config, prompt_max_chars=UNBOUNDED_CHAR_LIMIT,
        ))
        self.spatial_adapter = SpatialGraphAdapter() if self.plan.uses_l3 else None
        self.object_map: ObjectMap | None = None
        self.writer: MemoryWriter | None = None
        self.v3_store: V3MemoryStore | None = None
        self.working = WorkingMemoryStore()
        self.episodic = EpisodicMemory()
        self.scene_graph = SceneGraph()

        self.active = False
        self.scene_id: str | None = None
        self.episode_id: str | None = None
        self.instruction = ""
        self.v3_rollout_idx = 0
        self.v3_place_id: str | None = None
        self.v3_descriptor: tuple[float, ...] | None = None
        self._v3_prior_place_id: str | None = None
        self._v3_pending_descriptors: dict[str, tuple[float, ...]] = {}
        self.last_frame: RGBDFrame | None = None
        self.last_frame_step: int | None = None
        self.last_spatial_diagnostic: dict[str, Any] = {}
        self.last_prompt_context: PromptContextResult | None = None
        self._prompt_context_pending = False
        self._prompt_context_pending_ids = frozenset()
        self._active_prompt_fusion = self.prompt_local_graph_fusion
        self._step_events: list[MemoryEvent] = []
        self.last_observe_diagnostic: dict[str, Any] = {}

    # -- lifecycle -----------------------------------------------------------

    def reset_episode(self, episode: Any, instruction: str = "") -> dict[str, Any]:
        if episode is None:
            raise ValueError("episode is required")
        scene_id = str(getattr(episode, "scene_id", ""))
        episode_id = str(getattr(episode, "episode_id", ""))
        if not scene_id or not episode_id:
            raise ValueError("episode requires scene_id and episode_id")
        if not instruction:
            instruction = _episode_instruction(episode)
        self.close_episode()
        self.scene_id = scene_id
        self.episode_id = episode_id
        self.instruction = str(instruction)
        self.v3_rollout_idx = 0
        self.v3_place_id = None
        self.v3_descriptor = None
        self._v3_prior_place_id = None
        self._v3_pending_descriptors = {}
        self.v3_store = V3MemoryStore(scope=self.scope)
        self.v3_store.begin_episode(
            scene_id, episode_id, self.instruction, rollout_idx=0,
        )
        self.object_map = (
            ObjectMap(self.spatial_l3_config.geometry)
            if self.plan.uses_l3 else None
        )
        if self.object_map is not None:
            self.object_map.reset_episode(scene_id, episode_id, 0)
        self.last_frame = None
        self.last_frame_step = None
        self.last_spatial_diagnostic = {}
        self.local_graph_fusion.reset_episode(scene_id, episode_id, 0)
        self.prompt_local_graph_fusion.reset_episode(scene_id, episode_id, 0)
        self._active_prompt_fusion = self.prompt_local_graph_fusion
        self.working.reset(episode_id, self.instruction)
        self.episodic = EpisodicMemory()
        self.scene_graph = SceneGraph()
        if self.l1_extractor is not None:
            reset = getattr(self.l1_extractor, "reset_episode", None)
            if callable(reset):
                reset(episode_id, self.instruction)
        if self.seg_extractor is not None and self.plan.uses_l3:
            reset = getattr(self.seg_extractor, "reset_episode", None)
            if callable(reset):
                reset(episode_id, self.instruction)
        if self.place_extractor is not None:
            reset = getattr(self.place_extractor, "reset_episode", None)
            if callable(reset):
                reset(episode_id, self.instruction)
        self.writer = (
            MemoryWriter(
                self.working, self.episodic, self.scene_graph, self.l1_extractor,
            )
            if self.l1_extractor is not None else None
        )
        self._prompt_context_pending = False
        self._prompt_context_pending_ids = frozenset()
        self.last_prompt_context = None
        self._step_events = []
        self.active = True
        return {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "instruction_sha256": _sha256(self.instruction),
            "prompt_mode": self.plan.mode,
            "scope": self.scope,
            "activation": effective_layer_activation(self.plan),
        }

    def close_episode(self) -> None:
        for extractor in (self.l1_extractor, self.seg_extractor, self.place_extractor):
            close = getattr(extractor, "close_episode", None)
            if callable(close):
                close()
        self.active = False
        self.last_frame = None
        self.last_frame_step = None
        self.writer = None

    # -- L1 write path (copied from LightNav MemoryManager) ------------------

    def observe(
        self, observations: Any, step: int, *,
        last_action: dict[str, Any] | None = None,
        prediction: str | None = None,
    ) -> tuple[list[MemoryEvent], dict[str, Any]]:
        """Write one L1 observation and, when enabled, one DINO place record."""
        if not self.active or self.v3_store is None:
            raise RuntimeError("memory episode is not active")
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if self.writer is None:
            # L3-only groups still need the observation step to advance; no L1
            # detector and no L1 text are involved.
            diagnostic = {
                "step": step, "events_written": 0, "event_kinds": {},
                "place_extraction_enabled": False, "place_id": None,
                "evidence_records": len(
                    self.v3_store.evidence_for_scene(self.scene_id)
                ),
                "l1_observe_ms": 0.0, "l1_writer_enabled": False,
            }
            self.last_observe_diagnostic = diagnostic
            return [], diagnostic
        rgb = observations.get("rgb") if isinstance(observations, dict) else None
        if rgb is None:
            raise ValueError("observation has no RGB frame")
        packet = ObservationPacket(
            step_id=step, rgb=rgb, instruction=self.instruction,
            last_action=deepcopy(last_action) if last_action is not None else None,
            prediction=prediction,
        )
        started = time.perf_counter()
        events = self.writer.observe(packet)
        if self.place_extractor is not None and step % self.place_every_n_steps == 0:
            self._observe_place_v3(packet, events)
        if self.plan.uses_l1 or self.plan.uses_l3:
            self._persist_v3_events(events)
        self._step_events = list(events)
        diagnostic = {
            "step": step,
            "events_written": len(events),
            "event_kinds": _counts(event.kind for event in events),
            "place_extraction_enabled": self.place_extractor is not None,
            "place_id": self.v3_place_id,
            "evidence_records": len(self.v3_store.evidence_for_scene(self.scene_id)),
            "l1_observe_ms": (time.perf_counter() - started) * 1000.0,
        }
        self.last_observe_diagnostic = diagnostic
        return events, diagnostic

    def _observe_place_v3(self, packet: ObservationPacket, events: list[MemoryEvent]) -> None:
        """Copied from LightNav ``MemoryManager._observe_place_v3``."""
        assert self.place_extractor is not None and self.v3_store is not None
        try:
            feature_method = getattr(self.place_extractor, "extract_features", None)
            features = (
                feature_method(packet) if feature_method is not None
                else self.place_extractor.extract(packet)
            )
            if features is None:
                return
            values = getattr(features, "global_descriptor", features)
            descriptor = tuple(float(value) for value in values)
            if not descriptor:
                return
            event_id = f"place_observation_{packet.step_id}"
            self._v3_pending_descriptors[event_id] = descriptor
            event = MemoryEvent(
                event_id=event_id, step_id=packet.step_id,
                kind="place_observation",
                description="Observed RGB visual place descriptor",
                confidence=1.0, source=self.place_extractor.source,
                importance=0.6, episode_id=self.working.state.episode_id,
                metadata={"descriptor_dim": len(descriptor)},
            )
            self.episodic.add(event)
            self.working.record_event(event)
            events.append(event)
            self.v3_descriptor = descriptor
        except Exception as exc:  # noqa: BLE001 - optional place channel
            logger.warning("V3 place extraction failed at step=%s: %s", packet.step_id, exc)

    def _persist_v3_events(self, events: list[MemoryEvent]) -> None:
        """Copied from LightNav ``MemoryManager._persist_v3_events``.

        The legacy ``rgm_adopted``/gate branches are omitted because the eight
        groups never enable the RGM gate; every event written here is a real
        observation and is persisted exactly as the upstream function does.
        """
        assert self.v3_store is not None and self.scene_id is not None
        instruction_words = set(_words(self.instruction))
        direct_by_label: dict[str, str] = {}
        visible_landmarks: list[tuple[str, str]] = []
        for event in events:
            is_detection = event.kind in {"object_observation", "landmark"}
            label = (
                _removeprefix(event.description, "Observed ").strip()
                if is_detection else None
            )
            payload = {"description": event.description}
            if event.kind in {"action", "turn"}:
                action = event.metadata.get("action")
                if isinstance(action, dict):
                    payload["action"] = deepcopy(action)
            source_refs: tuple[str, ...] = ()
            if event.kind == "instruction_match":
                matched_label = _removeprefix(
                    event.description, "Observed instruction-related ",
                ).strip().casefold()
                if matched_label in direct_by_label:
                    source_refs = (direct_by_label[matched_label],)
            terms = tuple(sorted(instruction_words & set(
                _words(label or event.description)
            )))
            record = EvidenceRecord(
                evidence_id=event.event_id, scene_id=self.scene_id,
                episode_id=event.episode_id, rollout_idx=self.v3_rollout_idx,
                step=event.step_id, event_type=event.kind, semantic_label=label,
                payload=payload, visual_descriptor=self._v3_pending_descriptors.pop(
                    event.event_id, None,
                ), detector_name=event.source if is_detection else None,
                raw_detector_confidence=event.confidence if is_detection else None,
                instruction_terms=terms, source_evidence_ids=source_refs,
                observation_status=("observed" if is_detection or event.kind in {
                    "place_observation", "visual_change", "place_visit",
                } else "unknown"),
                spatial_status="observed" if event.kind == "place_observation" else "unknown",
                traversability_status="unknown",
                provenance_validity=(
                    event.kind != "instruction_match" or bool(source_refs)
                ),
            )
            stored = self.v3_store.add(record)
            if is_detection:
                direct_by_label[label.casefold()] = stored.key
                landmark = self.v3_store.add_landmark(stored)
                visible_landmarks.append((landmark.node_id, stored.key))
            elif event.kind == "place_observation":
                node = self.v3_store.add_place(stored)
                for landmark_id, landmark_key in visible_landmarks:
                    self.v3_store.add_relation(
                        node.node_id, landmark_id, "inferred_relation",
                        evidence_keys=(stored.key, landmark_key),
                    )
                prior = self._v3_prior_place_id
                if prior is not None and prior != node.node_id:
                    prior_node = next((item for item in self.v3_store.nodes_for_scene(
                        self.scene_id,
                    ) if item.node_id == prior), None)
                    if prior_node is not None and prior_node.evidence_ids:
                        self.v3_store.add_relation(
                            prior, node.node_id, "inferred_relation",
                            evidence_keys=(prior_node.evidence_ids[-1], stored.key),
                        )
                self.v3_place_id = node.node_id
                self._v3_prior_place_id = node.node_id

    # -- L3 write path (copied from LightNav MemoryManager.observe_spatial) --

    def observe_spatial(
        self, frame: RGBDFrame, events: list[MemoryEvent], step: int, *,
        transition_record: bool = False,
    ) -> dict[str, Any]:
        """Update the L3 ObjectMap from one aligned RGB-D frame.

        This body is the upstream ``observe_spatial`` without the LightNav
        server's ``RGBDFrame.from_habitat`` call; the caller supplies the frame
        built by ``l3_integration.habitat_geometry.frame_from_habitat``.
        """
        if not self.plan.uses_l3:
            return {"l3_enabled": False, "step": step}
        if not self.active or self.seg_extractor is None or self.object_map is None:
            raise RuntimeError("L3 memory episode is not active")
        diagnostic: dict[str, Any] = {
            "mode": "inject", "step": step,
            "frame_valid": False, "segmentation_count": 0,
            "evidence_minted": 0, "evidence_linked": 0, "evidence_unmatched": 0,
            "evidence_ambiguous": 0, "observed_objects": 0,
            "object_nodes": len(self.object_map.objects),
            "viewpoints": len(self.object_map.viewpoints),
            "observed_from_edges": sum(
                edge.relation_type == "observed_from"
                for edge in self.object_map.edges
            ),
            "traversed_to_edges": sum(
                edge.relation_type == "traversed_to"
                for edge in self.object_map.edges
            ),
            "rejection_reason": None,
            "l3_observe_ms": 0.0,
        }
        started = time.perf_counter()
        self.last_frame = None
        try:
            self.last_frame = frame
            self.last_frame_step = step
            diagnostic["frame_valid"] = True
            instances = self.seg_extractor.extract(frame.rgb, step)
            diagnostic["segmentation_count"] = len(instances)
            masks, counts = make_spatial_evidence(
                instances, events, self.v3_store,
                scene_id=self.scene_id,
                episode_id=self.working.state.episode_id,
                rollout_idx=self.v3_rollout_idx,
                step=step,
                source=self.seg_source,
                min_iou=self.evidence_min_iou,
                ambiguity_margin=self.evidence_ambiguity_margin,
            )
            diagnostic["evidence_minted"] = counts["minted"]
            diagnostic["evidence_linked"] = counts["linked"]
            diagnostic["evidence_unmatched"] = counts["unmatched"]
            diagnostic["evidence_ambiguous"] = counts["ambiguous"]
            observed = self.object_map.observe(
                frame, masks, step_id=step,
                transition_record=bool(transition_record),
            )
            diagnostic["observed_objects"] = len(observed)
            diagnostic["object_nodes"] = len(self.object_map.objects)
            diagnostic["viewpoints"] = len(self.object_map.viewpoints)
            diagnostic["observed_from_edges"] = sum(
                edge.relation_type == "observed_from"
                for edge in self.object_map.edges
            )
            diagnostic["traversed_to_edges"] = sum(
                edge.relation_type == "traversed_to"
                for edge in self.object_map.edges
            )
        except Exception as exc:  # noqa: BLE001 - optional sidecar fails closed
            diagnostic["rejection_reason"] = f"{type(exc).__name__}: {exc}"
            diagnostic["frame_valid"] = False
            self.last_frame = None
            self.last_frame_step = None
            logger.warning("spatial L3 rejected step=%s: %s", step, exc)
        diagnostic["l3_observe_ms"] = (time.perf_counter() - started) * 1000.0
        self.last_spatial_diagnostic = diagnostic
        return dict(diagnostic)

    # -- prompt context (copied from LightNav MemoryManager) -----------------

    def build_context(
        self, *, instruction: str, step: int, rgb_width: int,
        frame: RGBDFrame | None = None,
        token_budget: MemoryTokenBudget | None = None,
        budget_failure: TokenBudgetFailure | None = None,
    ) -> PromptContextResult:
        if not self.active or self.v3_store is None:
            raise RuntimeError("memory episode is not active")
        token_limited = token_budget is not None and token_budget.available
        fusion = (
            self.prompt_local_graph_fusion
            if self.plan.l1_source == "original" and token_limited
            else self.local_graph_fusion
        )
        self._active_prompt_fusion = fusion
        char_limit = (
            UNBOUNDED_CHAR_LIMIT if token_limited else self.prompt_budget_chars
        )
        result = build_prompt_context(
            self.plan,
            store=self.v3_store,
            object_map=self.object_map,
            frame=frame,
            local_graph_fusion=fusion,
            spatial_adapter=self.spatial_adapter,
            l1_organizer=self.l1_prompt_organizer,
            l3_organizer=self.l3_prompt_organizer,
            official_l3_organizer=self.l3_official_prompt_organizer,
            joint_builder=self.joint_prompt_builder,
            scene_id=self.scene_id,
            episode_id=self.working.state.episode_id,
            rollout_idx=self.v3_rollout_idx,
            step=step,
            instruction=instruction,
            rgb_width=rgb_width,
            events=self._step_events,
            current_place_id=self.v3_place_id,
            scope=self.local_graph_config.scope,
            max_landmarks=self.local_graph_config.max_landmarks,
            min_confidence=self.min_injection_confidence,
            budget_chars=self.prompt_budget_chars,
            char_limit=char_limit,
            token_budget=token_budget,
            budget_failure=budget_failure,
        )
        try:
            from l1_integration.memory_diagnostics import diagnose_prompt_pipeline

            result.diagnostics["pipeline"] = diagnose_prompt_pipeline(
                result=result,
                store=self.v3_store,
                object_map=self.object_map,
                scene_id=self.scene_id,
                episode_id=self.working.state.episode_id,
                rollout_idx=self.v3_rollout_idx,
                step=step,
                instruction=instruction,
                events=self._step_events,
                min_confidence=self.min_injection_confidence,
                scope=self.local_graph_config.scope,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must fail open
            result.diagnostics["pipeline_error"] = f"{type(exc).__name__}: {exc}"
        if self.plan.l1_source == "original" and not result.local_graph_pending:
            self._active_prompt_fusion.discard_selection()
        self.last_prompt_context = result
        self._prompt_context_pending = result.local_graph_pending
        self._prompt_context_pending_ids = frozenset(result.l1_evidence_ids)
        return result

    def confirm_context(self, evidence_ids: Iterable[str], *, step: int) -> None:
        if not self._prompt_context_pending:
            return
        self._prompt_context_pending = False
        confirmed = [
            evidence_id for evidence_id in evidence_ids
            if evidence_id in self._prompt_context_pending_ids
        ]
        self._prompt_context_pending_ids = frozenset()
        if confirmed:
            self._active_prompt_fusion.confirm_injected(confirmed, step)
        else:
            self._active_prompt_fusion.discard_selection()

    def discard_context(self) -> None:
        if not self._prompt_context_pending:
            return
        self._prompt_context_pending = False
        self._prompt_context_pending_ids = frozenset()
        self._active_prompt_fusion.discard_selection()

    # -- audit ---------------------------------------------------------------

    def map_snapshot(self) -> dict[str, Any]:
        if self.object_map is None:
            return {"objects": 0, "viewpoints": 0, "evidence": 0, "edges": 0}
        labels: dict[str, int] = {}
        for obj in self.object_map.objects.values():
            labels[obj.class_name] = labels.get(obj.class_name, 0) + 1
        return {
            "objects": len(self.object_map.objects),
            "viewpoints": len(self.object_map.viewpoints),
            "evidence": len(self.object_map.evidence),
            "edges": len(self.object_map.edges),
            "labels": dict(sorted(labels.items())),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "active": self.active,
            "prompt_mode": self.plan.mode,
            "activation": effective_layer_activation(self.plan),
            "scope": self.scope,
            "evidence_records": (
                len(self.v3_store.evidence_for_scene(self.scene_id))
                if self.v3_store is not None and self.scene_id else 0
            ),
            "place_nodes": (
                sum(node.node_type == "place" for node
                    in self.v3_store.nodes_for_scene(self.scene_id))
                if self.v3_store is not None and self.scene_id else 0
            ),
            "map": self.map_snapshot(),
        }


def _episode_instruction(episode: Any) -> str:
    instruction = getattr(episode, "instruction", None)
    if instruction is None:
        return ""
    if isinstance(instruction, str):
        return instruction
    return str(
        getattr(instruction, "instruction_text", None)
        or getattr(instruction, "text", None)
        or ""
    )


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _words(value: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", (value or "").casefold())


def _removeprefix(value: str, prefix: str) -> str:
    # Python 3.8 has no str.removeprefix; the explicit slice is intentional.
    return value[len(prefix):] if value.startswith(prefix) else value  # noqa: FURB188


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))

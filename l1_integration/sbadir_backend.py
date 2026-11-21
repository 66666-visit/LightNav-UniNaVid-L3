"""Run the LightNav L1 writer on Uni-NaVid RGB observations.

The writer, V3 store, DINOv2 place history, LocalGraphFusion and
L1PromptOrganizer remain the LightNav algorithms. This adapter only supplies
the observation and exposes selected facts with their actual source records.
It never creates a second navigation prompt or a sensor trust score.
"""

from __future__ import annotations

import logging
import re
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Iterable

from l1_memory.episodic_memory import EpisodicMemory
from l1_memory.l1_prompt_organizer import L1Fact, L1PromptOrganizer
from l1_memory.local_graph_config import LocalGraphConfig
from l1_memory.local_graph_fusion import LocalGraphFusion
from l1_memory.memory_writer import MemoryWriter
from l1_memory.scene_graph import SceneGraph
from l1_memory.working_memory import WorkingMemoryStore
from l3_memory.schemas import LayerMemoryItem, MemoryEvent, ObservationPacket
from l3_memory.v3_schemas import EvidenceRecord
from l3_memory.v3_store import V3MemoryStore

logger = logging.getLogger(__name__)
_DETECTION_KINDS = frozenset({"object_observation", "landmark"})


@dataclass(frozen=True)
class L1SourceEvidence:
    """One real RGB detection and its record-level provenance."""

    evidence_id: str
    observation_step: int
    object_label: str | None
    detection_confidence: float | None
    detector_name: str | None
    rgb_observation_source: str | None
    bbox_xyxy: tuple[float, float, float, float] | None
    source_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "observation_step": self.observation_step,
            "object_label": self.object_label,
            "detection_confidence": self.detection_confidence,
            "detector_name": self.detector_name,
            "rgb_observation_source": self.rgb_observation_source,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,
            "source_evidence_ids": list(self.source_evidence_ids),
        }


@dataclass(frozen=True)
class L1SelectedFact:
    """A LightNav-selected fact; detector labels are not physical identities."""

    text: str
    label: str
    role: str
    sector: str | None
    first_step: int
    last_step: int
    sightings: int
    evidence_ids: tuple[str, ...]
    evidence: tuple[L1SourceEvidence, ...]
    selection_source: str
    entity_id: str | None = None
    attribute: str = "object_label"
    visual_place_id: str | None = None
    visual_place_evidence_ids: tuple[str, ...] = ()
    visual_place_source: str | None = None

    @property
    def value(self) -> str:
        return self.label

    @property
    def source(self) -> str | None:
        """Detector reported for the selected source record, if present."""
        return self.evidence[0].detector_name if self.evidence else None

    @property
    def rgb_observation_source(self) -> str | None:
        return self.evidence[0].rgb_observation_source if self.evidence else None

    @property
    def confidence(self) -> float | None:
        """An actual recorded detector score, never a source trust score."""
        scores = [
            entry.detection_confidence for entry in self.evidence
            if entry.detection_confidence is not None
        ]
        if not scores:
            return None
        return scores[0] if self.role == "current" else max(scores)

    def line(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "role": self.role,
            "sector": self.sector,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "sightings": self.sightings,
            "evidence_ids": list(self.evidence_ids),
            "evidence": [entry.to_dict() for entry in self.evidence],
            "selection_source": self.selection_source,
            "entity_id": self.entity_id,
            "attribute": self.attribute,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "rgb_observation_source": self.rgb_observation_source,
            "visual_place_id": self.visual_place_id,
            "visual_place_evidence_ids": list(self.visual_place_evidence_ids),
            "visual_place_source": self.visual_place_source,
        }


@dataclass(frozen=True)
class L1Selection:
    organized_facts: tuple[L1SelectedFact, ...]
    local_graph_facts: tuple[L1SelectedFact, ...]
    diagnostics: dict[str, Any]

    @property
    def selected_facts(self) -> tuple[L1SelectedFact, ...]:
        """Keep organizer order and merge distinct graph-place context."""
        merged = list(self.organized_facts)
        used = {key for fact in merged for key in fact.evidence_ids}
        for fact in self.local_graph_facts:
            if set(fact.evidence_ids).issubset(used):
                if fact.role == "history" and fact.visual_place_id is not None:
                    for index, existing in enumerate(merged):
                        if (existing.label.casefold() == fact.label.casefold()
                                and set(fact.evidence_ids).issubset(set(existing.evidence_ids))):
                            qualifier = (
                                " Visual place history: near a visually similar area "
                                "(possible revisit; current direction and passage unknown)."
                            )
                            merged[index] = replace(
                                existing,
                                text=existing.text + qualifier,
                                selection_source="l1_prompt_organizer+local_graph_fusion",
                                visual_place_id=fact.visual_place_id,
                                visual_place_evidence_ids=fact.visual_place_evidence_ids,
                                visual_place_source=fact.visual_place_source,
                            )
                            break
                continue
            merged.append(fact)
            used.update(fact.evidence_ids)
        return tuple(merged)


class L1EvidenceBackend:
    """Episode-local RGB L1 memory with provenance-preserving fact selection."""

    def __init__(
        self,
        detector: Any,
        *,
        place_extractor: Any | None = None,
        max_landmarks: int = 2,
        place_every_n_steps: int = 5,
    ) -> None:
        if detector is None:
            raise ValueError("L1 requires a real RGB detector")
        if type(max_landmarks) is not int or max_landmarks < 1:
            raise ValueError("max_landmarks must be a positive integer")
        if type(place_every_n_steps) is not int or place_every_n_steps < 1:
            raise ValueError("place_every_n_steps must be a positive integer")
        self.detector = detector
        self.place_extractor = place_extractor
        self.place_every_n_steps = place_every_n_steps
        self.organizer = L1PromptOrganizer(max_items=max_landmarks)
        self.local_graph = LocalGraphFusion(LocalGraphConfig(
            enabled=True, scope="episode_local", prompt_max_chars=65536,
            max_landmarks=max_landmarks,
        ))
        self.working = WorkingMemoryStore()
        self.episodic = EpisodicMemory()
        self.scene_graph = SceneGraph()
        self.writer: MemoryWriter | None = None
        self.v3_store: V3MemoryStore | None = None
        self.scene_id: str | None = None
        self.episode_id: str | None = None
        self.instruction = ""
        self.rollout_idx = 0
        self.place_id: str | None = None
        self._prior_place_id: str | None = None
        self._pending_descriptors: dict[str, tuple[float, ...]] = {}
        self.last_events: list[MemoryEvent] = []
        self.last_step: int | None = None
        self.last_diagnostic: dict[str, Any] = {}
        self.last_place_error: str | None = None
        self.last_place_observed = False
        self._local_graph_selected_ids: frozenset[str] = frozenset()

    def reset_episode(
        self, scene_id: str, episode_id: str, instruction: str,
        *, rollout_idx: int = 0,
    ) -> None:
        if not scene_id or not episode_id:
            raise ValueError("scene_id and episode_id are required")
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")
        if type(rollout_idx) is not int or rollout_idx < 0:
            raise ValueError("rollout_idx must be a nonnegative integer")
        self.close_episode()
        self.scene_id = scene_id
        self.episode_id = episode_id
        self.instruction = instruction
        self.rollout_idx = rollout_idx
        self.v3_store = V3MemoryStore(scope="episode_local")
        self.v3_store.begin_episode(scene_id, episode_id, instruction, rollout_idx)
        self.working.reset(episode_id, instruction)
        self.episodic = EpisodicMemory()
        self.scene_graph = SceneGraph()
        self.detector.reset_episode(episode_id, instruction)
        if self.place_extractor is not None:
            self.place_extractor.reset_episode(episode_id, instruction)
        self.writer = MemoryWriter(
            self.working, self.episodic, self.scene_graph, self.detector,
        )
        self.local_graph.reset_episode(scene_id, episode_id, rollout_idx)
        self.place_id = None
        self._prior_place_id = None
        self._pending_descriptors.clear()
        self.last_events = []
        self.last_step = None
        self.last_diagnostic = {}
        self.last_place_error = None
        self.last_place_observed = False
        self._local_graph_selected_ids = frozenset()

    def close_episode(self) -> None:
        self.local_graph.close_episode()
        if self.v3_store is not None and self.v3_store.active_episode is not None:
            self.v3_store.close_episode()
        for extractor in (self.detector, self.place_extractor):
            if extractor is not None:
                close = getattr(extractor, "close_episode", None)
                if callable(close):
                    close()
        self.writer = None
        self.v3_store = None
        self.scene_id = None
        self.episode_id = None
        self.last_events = []
        self.last_step = None
        self.last_place_error = None
        self.last_place_observed = False
        self._local_graph_selected_ids = frozenset()

    def observe(
        self,
        rgb: Any,
        step: int,
        *,
        rgb_observation_source: str = "observation.rgb",
        last_action: dict[str, Any] | None = None,
        prediction: str | None = None,
    ) -> tuple[list[MemoryEvent], dict[str, Any]]:
        if self.writer is None or self.v3_store is None or self.scene_id is None:
            raise RuntimeError("L1 episode is not active")
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if not isinstance(rgb_observation_source, str) or not rgb_observation_source:
            raise ValueError("rgb_observation_source must be a nonempty string")
        packet = ObservationPacket(
            step_id=step, rgb=rgb, instruction=self.instruction,
            last_action=deepcopy(last_action) if last_action is not None else None,
            prediction=prediction,
        )
        started = time.perf_counter()
        events = self.writer.observe(packet)
        place_attempted = (
            self.place_extractor is not None and step % self.place_every_n_steps == 0
        )
        self.last_place_error = None
        self.last_place_observed = False
        if place_attempted:
            self._observe_place(packet, events)
        self._persist_events(events, rgb_observation_source)
        self.last_events = list(events)
        self.last_step = step
        detections = [event for event in events if event.kind in _DETECTION_KINDS]
        diagnostic = {
            "step": step,
            "rgb_observation_source": rgb_observation_source,
            "event_count": len(events),
            "detection_count": len(detections),
            "detection_evidence_ids": [
                self._evidence_key(event.event_id) for event in detections
            ],
            "place_attempted": place_attempted,
            "place_observed": self.last_place_observed,
            "place_error": self.last_place_error,
            "place_id": self.place_id,
            "evidence_record_count": len(self.v3_store.evidence_for_scene(self.scene_id)),
            "detector_error": getattr(self.detector, "last_error", None),
            "observe_ms": (time.perf_counter() - started) * 1000.0,
        }
        self.last_diagnostic = diagnostic
        return list(events), dict(diagnostic)

    def select(self, *, step: int, rgb_width: int) -> L1Selection:
        if (self.v3_store is None or self.scene_id is None or self.episode_id is None
                or self.last_step != step):
            raise RuntimeError("select requires an observed step in the active episode")
        organization = self.organizer.build(
            self.v3_store, scene_id=self.scene_id, episode_id=self.episode_id,
            rollout_idx=self.rollout_idx, step=step,
            instruction=self.instruction, rgb_width=rgb_width,
            events=self.last_events, max_chars=65536,
        )
        organized = tuple(
            self._from_organized(fact) for fact in organization.selected_facts
        )
        graph_context = self.local_graph.build(
            self.v3_store, scene_id=self.scene_id, episode_id=self.episode_id,
            rollout_idx=self.rollout_idx, step=step,
            instruction=self.instruction,
            current_place_id=self.place_id if self.last_place_observed else None,
            rgb_width=rgb_width, events=self.last_events,
        )
        graph = tuple(
            self._from_graph(item) for item in graph_context.layer_items
        ) if graph_context is not None else ()
        self._local_graph_selected_ids = frozenset(
            evidence_id for fact in graph for evidence_id in fact.evidence_ids
        )
        diagnostics = {
            "step": step,
            "organized_fact_count": len(organized),
            "local_graph_fact_count": len(graph),
            "organized_evidence_ids": sorted({
                key for fact in organized for key in fact.evidence_ids
            }),
            "local_graph_evidence_ids": sorted(self._local_graph_selected_ids),
            "organizer": organization.diagnostics(),
            "local_graph_candidate_count": len(graph),
        }
        selection = L1Selection(organized, graph, diagnostics)
        diagnostics["selected_fact_count"] = len(selection.selected_facts)
        return selection

    def confirm_local_graph(self, evidence_ids: Iterable[str], *, step: int) -> None:
        if self._local_graph_selected_ids:
            selected = set(evidence_ids) & self._local_graph_selected_ids
            if selected:
                self.local_graph.confirm_injected(selected, step)
            else:
                self.local_graph.discard_selection()
        self._local_graph_selected_ids = frozenset()

    def discard_local_graph(self) -> None:
        self.local_graph.discard_selection()
        self._local_graph_selected_ids = frozenset()

    def _observe_place(self, packet: ObservationPacket, events: list[MemoryEvent]) -> None:
        assert self.place_extractor is not None
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
            self.last_place_observed = True
            event_id = "place_observation_%d" % packet.step_id
            self._pending_descriptors[event_id] = descriptor
            event = MemoryEvent(
                event_id=event_id, step_id=packet.step_id,
                kind="place_observation",
                description="Observed RGB visual place descriptor",
                confidence=1.0, source=self.place_extractor.source,
                importance=0.6, episode_id=self.episode_id or "",
                metadata={"descriptor_dim": len(descriptor)},
            )
            self.episodic.add(event)
            self.working.record_event(event)
            events.append(event)
        except Exception as exc:  # optional descriptor must not fabricate a place
            self.last_place_error = "%s: %s" % (type(exc).__name__, exc)
            logger.warning("L1 place extraction failed at step=%s: %s", packet.step_id, exc)

    def _persist_events(self, events: list[MemoryEvent], rgb_source: str) -> None:
        assert self.v3_store is not None and self.scene_id is not None
        instruction_words = set(_words(self.instruction))
        direct_by_label: dict[str, str] = {}
        visible_landmarks: list[tuple[str, str]] = []
        for event in events:
            is_detection = event.kind in _DETECTION_KINDS
            label = _remove_prefix(event.description, "Observed ").strip() if is_detection else None
            payload: dict[str, Any] = {"description": event.description}
            if is_detection or event.kind == "place_observation":
                payload["rgb_observation_source"] = rgb_source
            if event.kind == "place_observation":
                payload["visual_place_source"] = event.source
            if is_detection and "bbox_xyxy" in event.metadata:
                payload["bbox_xyxy"] = list(event.metadata["bbox_xyxy"])
            if event.kind in {"action", "turn"}:
                action = event.metadata.get("action")
                if isinstance(action, dict):
                    payload["action"] = deepcopy(action)
            source_refs: tuple[str, ...] = ()
            if event.kind == "instruction_match":
                matched_label = _remove_prefix(
                    event.description, "Observed instruction-related ",
                ).strip().casefold()
                if matched_label in direct_by_label:
                    source_refs = (direct_by_label[matched_label],)
            terms = tuple(sorted(instruction_words & set(
                _words(label or event.description)
            )))
            record = EvidenceRecord(
                evidence_id=event.event_id, scene_id=self.scene_id,
                episode_id=event.episode_id, rollout_idx=self.rollout_idx,
                step=event.step_id, event_type=event.kind,
                semantic_label=label, payload=payload,
                visual_descriptor=self._pending_descriptors.pop(event.event_id, None),
                detector_name=event.source if is_detection else None,
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
                assert label is not None
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
                prior = self._prior_place_id
                if prior is not None and prior != node.node_id:
                    prior_node = next((item for item in self.v3_store.nodes_for_scene(
                        self.scene_id,
                    ) if item.node_id == prior), None)
                    if prior_node is not None and prior_node.evidence_ids:
                        self.v3_store.add_relation(
                            prior, node.node_id, "inferred_relation",
                            evidence_keys=(prior_node.evidence_ids[-1], stored.key),
                        )
                self.place_id = node.node_id
                self._prior_place_id = node.node_id

    def _evidence_key(self, event_id: str) -> str:
        from l3_memory.v3_schemas import canonical_evidence_key

        assert self.scene_id is not None and self.episode_id is not None
        return canonical_evidence_key(
            self.scene_id, self.episode_id, self.rollout_idx, event_id,
        )

    def _from_organized(self, fact: L1Fact) -> L1SelectedFact:
        evidence = self._source_evidence(fact.evidence_ids)
        return L1SelectedFact(
            text=fact.line(), label=fact.label, role=fact.role,
            sector=fact.sector, first_step=fact.first_step,
            last_step=fact.last_step, sightings=fact.sightings,
            evidence_ids=fact.evidence_ids, evidence=evidence,
            selection_source="l1_prompt_organizer",
        )

    def _from_graph(self, item: LayerMemoryItem) -> L1SelectedFact:
        ids = tuple(dict.fromkeys(ref.event_id for ref in item.evidence_refs))
        evidence = self._source_evidence(ids)
        if not evidence:
            raise ValueError("local graph selected a fact without source evidence")
        steps = [entry.observation_step for entry in evidence]
        place_id = item.metadata.get("place_id")
        place_evidence_ids: tuple[str, ...] = ()
        place_source = None
        if isinstance(place_id, str) and self.v3_store is not None:
            node = next((node for node in self.v3_store.nodes_for_scene(self.scene_id)
                         if node.node_id == place_id and node.node_type == "place"), None)
            if node is not None:
                place_evidence_ids = tuple(node.evidence_ids)
                if place_evidence_ids:
                    record = self.v3_store.get_evidence(place_evidence_ids[-1])
                    if record is not None:
                        place_source = record.payload.get("visual_place_source")
        return L1SelectedFact(
            text=item.description, label=item.label or evidence[0].object_label or "",
            role=str(item.metadata.get("role", "history")),
            sector=item.metadata.get("current_view_sector"),
            first_step=min(steps), last_step=max(steps),
            sightings=len(evidence), evidence_ids=ids, evidence=evidence,
            selection_source="local_graph_fusion",
            visual_place_id=place_id if isinstance(place_id, str) else None,
            visual_place_evidence_ids=place_evidence_ids,
            visual_place_source=place_source,
        )

    def _source_evidence(self, evidence_ids: Iterable[str]) -> tuple[L1SourceEvidence, ...]:
        assert self.v3_store is not None
        result = []
        for key in evidence_ids:
            record = self.v3_store.get_evidence(key)
            if record is None or record.event_type not in _DETECTION_KINDS:
                raise ValueError("selected L1 fact lacks a persisted RGB detection")
            raw_bbox = record.payload.get("bbox_xyxy")
            bbox = None
            if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                bbox = tuple(float(value) for value in raw_bbox)
            raw_rgb_source = record.payload.get("rgb_observation_source")
            result.append(L1SourceEvidence(
                evidence_id=record.key,
                observation_step=record.step,
                object_label=record.semantic_label,
                detection_confidence=record.raw_detector_confidence,
                detector_name=record.detector_name,
                rgb_observation_source=(
                    raw_rgb_source if isinstance(raw_rgb_source, str) else None
                ),
                bbox_xyxy=bbox,
                source_evidence_ids=record.source_evidence_ids,
            ))
        return tuple(result)


def _words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _remove_prefix(value: str, prefix: str) -> str:
    return value[len(prefix):] if value.startswith(prefix) else value

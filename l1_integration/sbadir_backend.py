"""Run LightNav's original ``LocalGraphFusion`` L1 path on Uni-NaVid RGB.

This module is an adapter, not an L1 algorithm.  It supplies Uni-NaVid's RGB
observation to the vendored LightNav ``MemoryWriter``, the original place
observation method (``MemoryManager._observe_place_v3``), the original V3
evidence persistence method (``MemoryManager._persist_v3_events``), and the
original ``LocalGraphFusion`` card.  There is no L1 prompt organizer, no
second fact-selection rule, and no SBAdir rewrite of the L1 lines here.  The
only additional step is LightNav's token-budget whole-line selector over the
frozen card; it can omit trailing original lines but cannot create or rewrite
a fact.

The only code that is not a literal copy from LightNav is the interface:
``observe`` accepts a Uni-NaVid RGB array and a step number, ``select`` returns
the original ``MemoryContext`` plus the exact text that the original context
builder produced, and ``reset_episode`` accepts the Episode identities already
known by the Uni-NaVid runner.
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional

from l1_integration.token_budget import (
    ORIGINAL_CHAR_FALLBACK,
    UNBOUNDED_CHAR_LIMIT,
    MemoryTokenBudget,
    TokenBudgetFailure,
    select_context_lines,
)
from l1_memory.compat import removeprefix
from l1_memory.episodic_memory import EpisodicMemory
from l1_memory.local_graph_config import LocalGraphConfig
from l1_memory.local_graph_fusion import LocalGraphFusion
from l1_memory.memory_context_builder import build_context_text
from l1_memory.memory_writer import MemoryWriter
from l1_memory.scene_graph import SceneGraph
from l1_memory.working_memory import WorkingMemoryStore
from l3_memory.schemas import LayerMemoryItem, MemoryContext, MemoryEvent, ObservationPacket
from l3_memory.v3_schemas import EvidenceRecord
from l3_memory.v3_store import V3MemoryStore

logger = logging.getLogger(__name__)

#: Exact values of the 75% Local Graph baseline (LightNav tag
#: ``baseline-local-graph-fusion-88-sr75``, commit ``2fb184e``).
ORIGINAL_L1_PROMPT_MAX_CHARS = 320
ORIGINAL_L1_MAX_LANDMARKS = 2
ORIGINAL_L1_REACTIVATION_STEPS = 8
ORIGINAL_L1_PLACE_EVERY_N_STEPS = 5
ORIGINAL_L1_SCOPE = "episode_local"

_DETECTION_KINDS = frozenset({"object_observation", "landmark"})
_DETECTION_WORDS = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class L1Selection:
    """The original Local Graph card and the exact text for the prompt.

    ``context`` is the ``LocalGraphFusion.build`` result after the optional
    whole-line token selection; every line is an unchanged original line.
    ``prompt_text`` is what LightNav's context builder injects: HTML-escaped
    and bounded by the measured token budget, or by the original 320-character
    module budget on the fallback path.
    """

    context: MemoryContext | None
    prompt_text: str | None
    evidence_ids: tuple[str, ...]
    diagnostics: dict[str, Any]

    @property
    def fact_count(self) -> int:
        return len(self.context.layer_items) if self.context is not None else 0

    @property
    def text(self) -> str:
        return self.prompt_text or ""


class L1EvidenceBackend:
    """Episode-local RGB L1 memory using only LightNav's original algorithms."""

    def __init__(
        self,
        detector: Any,
        *,
        place_extractor: Any | None = None,
        max_landmarks: int = ORIGINAL_L1_MAX_LANDMARKS,
        place_every_n_steps: int = ORIGINAL_L1_PLACE_EVERY_N_STEPS,
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
        self.max_landmarks = max_landmarks
        # This is the exact LocalGraphConfig of the 75% baseline.  The
        # character ceiling is not lifted and no token-derived replacement is
        # used for L1.
        self.local_graph = LocalGraphFusion(LocalGraphConfig(
            enabled=True,
            scope=ORIGINAL_L1_SCOPE,
            prompt_max_chars=ORIGINAL_L1_PROMPT_MAX_CHARS,
            max_landmarks=max_landmarks,
            reactivation_steps=ORIGINAL_L1_REACTIVATION_STEPS,
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
        self.v3_descriptor: tuple[float, ...] | None = None
        self._prior_place_id: str | None = None
        self._pending_descriptors: dict[str, tuple[float, ...]] = {}
        self.last_events: list[MemoryEvent] = []
        self.last_step: int | None = None
        self.last_diagnostic: dict[str, Any] = {}
        self.last_v3_diagnostic: dict[str, Any] = {}
        self.last_place_error: str | None = None
        self.last_place_observed = False
        self.last_place_attempted = False
        self._selected_ids: frozenset[str] = frozenset()

    # -- lifecycle -----------------------------------------------------------

    def reset_episode(
        self,
        scene_id: str,
        episode_id: str,
        instruction: str,
        *,
        rollout_idx: int = 0,
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
        self.v3_store = V3MemoryStore(scope=ORIGINAL_L1_SCOPE)
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
        self.v3_descriptor = None
        self._prior_place_id = None
        self._pending_descriptors.clear()
        self.last_events = []
        self.last_step = None
        self.last_diagnostic = {}
        self.last_v3_diagnostic = {}
        self.last_place_error = None
        self.last_place_observed = False
        self.last_place_attempted = False
        self._selected_ids = frozenset()

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
        self.last_place_attempted = False
        self._selected_ids = frozenset()

    # -- observation ---------------------------------------------------------

    def observe(
        self,
        rgb: Any,
        step: int,
        *,
        rgb_observation_source: str = "observation.rgb",
        last_action: dict[str, Any] | None = None,
        prediction: str | None = None,
    ) -> tuple[list[MemoryEvent], dict[str, Any]]:
        """Consume one RGB observation through the original L1 write path.

        ``rgb_observation_source`` is accepted for runner diagnostics only.  It
        is deliberately not injected into the L1 record payload, matching the
        75% baseline's ``_persist_v3_events``.
        """
        if self.writer is None or self.v3_store is None or self.scene_id is None:
            raise RuntimeError("L1 episode is not active")
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if not isinstance(rgb_observation_source, str) or not rgb_observation_source:
            raise ValueError("rgb_observation_source must be a nonempty string")
        packet = ObservationPacket(
            step_id=step,
            rgb=rgb,
            instruction=self.instruction,
            last_action=deepcopy(last_action) if last_action is not None else None,
            prediction=prediction,
        )
        started = time.perf_counter()
        # Original MemoryManager.observe: a positive velocity command drops the
        # carried visual place until a new descriptor is observed.  Uni-NaVid
        # emits discrete actions, so this branch normally does not fire.
        if MemoryWriter._has_positive_forward_command(last_action):
            self.place_id = None
        events = self.writer.observe(packet)
        self.last_place_attempted = (
            self.place_extractor is not None
            and step % self.place_every_n_steps == 0
        )
        self.last_place_error = None
        self.last_place_observed = False
        if self.last_place_attempted:
            self._observe_place_v3(packet, events)
        self._persist_v3_events(events)
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
            "place_attempted": self.last_place_attempted,
            "place_observed": self.last_place_observed,
            "place_error": self.last_place_error,
            "place_id": self.place_id,
            "evidence_record_count": len(
                self.v3_store.evidence_for_scene(self.scene_id)
            ),
            "events_persisted": int(self.last_v3_diagnostic.get("events_persisted", 0)),
            "detection_events_persisted": int(
                self.last_v3_diagnostic.get("detection_events_persisted", 0)
            ),
            "landmark_nodes": int(self.last_v3_diagnostic.get("landmark_nodes", 0)),
            "place_nodes": int(self.last_v3_diagnostic.get("place_nodes", 0)),
            "detector_error": getattr(self.detector, "last_error", None),
            "observe_ms": (time.perf_counter() - started) * 1000.0,
        }
        self.last_diagnostic = diagnostic
        return list(events), dict(diagnostic)

    # -- original Local Graph selection -------------------------------------

    @contextmanager
    def _internal_character_limit(self, value: Optional[int]):
        """Temporarily set the Local Graph serialization cap.

        ``LocalGraphFusion`` itself is unchanged.  LightNav's token-aware mode
        disables only its internal character ceiling while a measured token
        budget owns line selection; the original 320-character value remains
        this adapter's fallback and is restored after every call.
        """
        original = self.local_graph.config
        if value is None:
            yield
            return
        self.local_graph.config = replace(original, prompt_max_chars=value)
        try:
            yield
        finally:
            self.local_graph.config = original

    def _build_context(self, *, step: int, rgb_width: int,
                       internal_char_limit: Optional[int]):
        assert self.v3_store is not None
        assert self.scene_id is not None and self.episode_id is not None
        with self._internal_character_limit(internal_char_limit):
            return self.local_graph.build(
                self.v3_store,
                scene_id=self.scene_id,
                episode_id=self.episode_id,
                rollout_idx=self.rollout_idx,
                step=step,
                instruction=self.instruction,
                # Original MemoryManager.get_local_graph_context passes the last
                # observed place every step, not only on a place-extraction step.
                current_place_id=self.place_id,
                rgb_width=rgb_width,
                events=self.last_events,
            )

    def select(
        self,
        *,
        step: int,
        rgb_width: int,
        token_budget: Optional[MemoryTokenBudget] = None,
        budget_failure: Optional[TokenBudgetFailure] = None,
    ) -> L1Selection:
        """Return the original Local Graph card bounded by the measured budget.

        With a measured ``token_budget`` the Local Graph's internal 320-char
        cap is disabled exactly as in LightNav's token-aware prompt mode, then
        whole lines are kept while the exact Uni-NaVid prompt-token delta fits
        ``available_memory_tokens``.  Without a measurement, the original
        320-character card is used and the failure reason is reported.
        """
        if (
            self.v3_store is None
            or self.scene_id is None
            or self.episode_id is None
            or self.last_step != step
        ):
            raise RuntimeError("select requires an observed step in the active episode")

        token_aware = token_budget is not None and token_budget.measured
        fallback_reason: Optional[str] = (
            budget_failure.reason
            if budget_failure is not None and not token_aware else None
        )
        selection = None
        builder_context = None
        if token_aware:
            try:
                builder_context = self._build_context(
                    step=step, rgb_width=rgb_width,
                    internal_char_limit=UNBOUNDED_CHAR_LIMIT,
                )
                if builder_context is not None:
                    selection = select_context_lines(
                        builder_context,
                        capacity_tokens=token_budget.available_tokens,
                        count_tokens=token_budget.counter,
                    )
            except Exception as exc:  # noqa: BLE001 - fall back, never guess
                fallback_reason = "token_selection_failed: %s: %s" % (
                    type(exc).__name__, exc,
                )
                logger.warning("L1 token selection failed at step=%s: %s", step, exc)
                token_aware = False
                selection = None

        if token_aware:
            context = selection.context if selection is not None else None
            if context is not None and not context.layer_items:
                # A header without a complete fact line is not a memory card.
                context = None
            prompt_text = (
                build_context_text(context, max_chars=UNBOUNDED_CHAR_LIMIT)
                if context is not None else ""
            )
            budget_diagnostics = dict(token_budget.diagnostics())
            budget_diagnostics["memory_block_tokens"] = (
                selection.block_tokens if selection is not None else 0
            )
            budget_diagnostics["omitted_lines_by_tokens"] = (
                selection.omitted_lines if selection is not None else 0
            )
            token_overflow_drops = (
                selection.omitted_lines if selection is not None else 0
            )
            internal_char_limit = "token_budget_owned"
            if budget_diagnostics.get("available_memory_tokens", 0) <= 0:
                empty_reason = "no_token_headroom"
            else:
                empty_reason = "token_selection_empty"
        else:
            builder_context = self._build_context(
                step=step, rgb_width=rgb_width, internal_char_limit=None,
            )
            context = builder_context
            prompt_text = (
                build_context_text(
                    context, max_chars=ORIGINAL_CHAR_FALLBACK,
                )
                if context is not None else ""
            )
            failure = budget_failure
            if failure is None and fallback_reason is not None:
                failure = TokenBudgetFailure(
                    reason=fallback_reason,
                    context_limit_tokens=(
                        token_budget.context_limit_tokens
                        if token_budget is not None else None
                    ),
                    base_prompt_tokens=(
                        token_budget.base_prompt_tokens
                        if token_budget is not None else None
                    ),
                    generation_reserve_tokens=(
                        token_budget.generation_reserve_tokens
                        if token_budget is not None else None
                    ),
                )
            budget_diagnostics = (
                failure.diagnostics() if failure is not None else {
                    "budget_source": "char_fallback_320",
                    "context_limit_tokens": None,
                    "base_prompt_tokens": None,
                    "generation_reserve_tokens": None,
                    "safety_margin_tokens": 1,
                    "available_memory_tokens": None,
                    "token_budget_unavailable_reason": None,
                    "budget_fallback_reason": None,
                }
            )
            token_overflow_drops = 0
            internal_char_limit = "native_320"
            empty_reason = "original_context_builder_empty"

        fact_count = (
            len(builder_context.layer_items) if builder_context is not None else 0
        )
        evidence_ids = tuple(dict.fromkeys(
            ref.event_id
            for item in (context.layer_items if context is not None else ())
            for ref in item.evidence_refs
        ))
        retained_fact_count = (
            len(context.layer_items) if context is not None else 0
        )
        selected = bool(context is not None and prompt_text and evidence_ids)
        self._selected_ids = frozenset(evidence_ids) if selected else frozenset()
        if selected:
            empty_reason = None
        elif context is None and builder_context is None:
            empty_reason = "original_builder_empty"
        budget_diagnostics["budget_fallback_reason"] = fallback_reason
        diagnostics = {
            "step": step,
            "source": "local_graph_fusion",
            "budget_chars": ORIGINAL_CHAR_FALLBACK,
            "budget_source": budget_diagnostics.get("budget_source"),
            "context_limit_tokens": budget_diagnostics.get("context_limit_tokens"),
            "base_prompt_tokens": budget_diagnostics.get("base_prompt_tokens"),
            "text_prompt_tokens": budget_diagnostics.get(
                "text_prompt_tokens"
            ),
            "visual_tokens": budget_diagnostics.get("visual_tokens"),
            "visual_token_accounting": budget_diagnostics.get(
                "visual_token_accounting"
            ),
            "generation_reserve_tokens": budget_diagnostics.get(
                "generation_reserve_tokens"
            ),
            "safety_margin_tokens": budget_diagnostics.get("safety_margin_tokens"),
            "available_memory_tokens": budget_diagnostics.get(
                "available_memory_tokens"
            ),
            "token_budget_unavailable_reason": budget_diagnostics.get(
                "token_budget_unavailable_reason"
            ),
            "budget_fallback_reason": fallback_reason,
            "internal_char_limit": internal_char_limit,
            "max_landmarks": self.max_landmarks,
            "fact_count": fact_count,
            "retained_fact_count": retained_fact_count,
            "omitted_facts": max(0, fact_count - retained_fact_count),
            "token_overflow_drops": token_overflow_drops,
            "prompt_chars": len(prompt_text),
            "prompt_sha256": _sha256(prompt_text) if prompt_text else None,
            "evidence_ids": list(evidence_ids),
            "empty_reason": None if selected else empty_reason,
            "budget": budget_diagnostics,
        }
        for item in (builder_context.layer_items if builder_context is not None else ()):
            if not isinstance(item, LayerMemoryItem):
                raise TypeError("Local Graph returned a non-LayerMemoryItem")
        return L1Selection(
            context=context,
            prompt_text=prompt_text or None,
            evidence_ids=evidence_ids,
            diagnostics=diagnostics,
        )

    def confirm_injected(self, evidence_ids: Iterable[str], *, step: int) -> None:
        """Confirm only when every selected evidence ID reached the prompt."""
        selected = frozenset(self._selected_ids)
        self._selected_ids = frozenset()
        if not selected:
            return
        confirmed = frozenset(evidence_ids)
        if confirmed == selected:
            self.local_graph.confirm_injected(confirmed, step)
        else:
            # Token selection can drop whole trailing lines.  The pending
            # selection is released, and no reactivation signature is claimed.
            self.local_graph.discard_selection()

    def discard_selection(self) -> None:
        self._selected_ids = frozenset()
        self.local_graph.discard_selection()

    # -- literal copies of the original LightNav methods --------------------

    def _observe_place_v3(
        self, packet: ObservationPacket, events: list[MemoryEvent],
    ) -> None:
        """Capture one DINO observation without claiming physical traversal."""
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
                event_id=event_id,
                step_id=packet.step_id,
                kind="place_observation",
                description="Observed RGB visual place descriptor",
                confidence=1.0,
                source=self.place_extractor.source,
                importance=0.6,
                episode_id=self.episode_id or "",
                metadata={"descriptor_dim": len(descriptor)},
            )
            self.episodic.add(event)
            self.working.record_event(event)
            events.append(event)
            self.v3_descriptor = descriptor
        except Exception as exc:  # noqa: BLE001 - optional descriptor fails closed
            self.last_place_error = "%s: %s" % (type(exc).__name__, exc)
            logger.warning(
                "V3 place extraction failed at step=%s: %s",
                packet.step_id, exc,
            )

    def _persist_v3_events(self, events: list[MemoryEvent]) -> None:
        """Write every valid observed event with a Scene/Episode-qualified key.

        This is the original 75% ``MemoryManager._persist_v3_events`` body.
        The only difference is that the C adapter always uses the long layer
        (the formal L1 group runs ``short,mid,long``), so the original
        ``"long" in self.layers`` condition is always true.
        """
        assert self.v3_store is not None and self.scene_id is not None
        instruction_words = set(_DETECTION_WORDS.findall(self.instruction.casefold()))
        direct_by_label: dict[str, str] = {}
        persisted = 0
        detection_persisted = 0
        visible_landmarks: list[tuple[str, str]] = []
        landmark_nodes = 0
        place_nodes = 0
        for event in events:
            is_detection = event.kind in _DETECTION_KINDS
            label = (
                removeprefix(event.description, "Observed ").strip()
                if is_detection else None
            )
            payload = {"description": event.description}
            if event.kind in {"action", "turn"}:
                action = event.metadata.get("action")
                if isinstance(action, dict):
                    payload["action"] = deepcopy(action)
            source_refs: tuple[str, ...] = ()
            if event.kind == "instruction_match":
                matched_label = removeprefix(
                    event.description, "Observed instruction-related ",
                ).strip().casefold()
                if matched_label in direct_by_label:
                    source_refs = (direct_by_label[matched_label],)
            terms = tuple(sorted(instruction_words & set(
                _DETECTION_WORDS.findall((label or event.description).casefold())
            )))
            record = EvidenceRecord(
                evidence_id=event.event_id,
                scene_id=self.scene_id,
                episode_id=event.episode_id,
                rollout_idx=self.rollout_idx,
                step=event.step_id,
                event_type=event.kind,
                semantic_label=label,
                payload=payload,
                visual_descriptor=self._pending_descriptors.pop(
                    event.event_id, None,
                ),
                detector_name=event.source if is_detection else None,
                raw_detector_confidence=event.confidence if is_detection else None,
                instruction_terms=terms,
                source_evidence_ids=source_refs,
                observation_status=(
                    "observed" if is_detection or event.kind in {
                        "place_observation", "visual_change", "place_visit",
                    } else "unknown"
                ),
                spatial_status=(
                    "observed" if event.kind == "place_observation" else "unknown"
                ),
                traversability_status="unknown",
                provenance_validity=(
                    event.kind != "instruction_match" or bool(source_refs)
                ),
            )
            stored = self.v3_store.add(record)
            persisted += 1
            if is_detection:
                detection_persisted += 1
                assert label is not None
                direct_by_label[label.casefold()] = stored.key
                landmark = self.v3_store.add_landmark(stored)
                visible_landmarks.append((landmark.node_id, stored.key))
                landmark_nodes += 1
            elif event.kind == "place_observation":
                node = self.v3_store.add_place(stored)
                place_nodes += 1
                for landmark_id, landmark_key in visible_landmarks:
                    self.v3_store.add_relation(
                        node.node_id,
                        landmark_id,
                        "inferred_relation",
                        evidence_keys=(stored.key, landmark_key),
                    )
                prior = self._prior_place_id
                if prior is not None and prior != node.node_id:
                    prior_node = next((
                        item for item in self.v3_store.nodes_for_scene(self.scene_id)
                        if item.node_id == prior
                    ), None)
                    if prior_node is not None and prior_node.evidence_ids:
                        self.v3_store.add_relation(
                            prior,
                            node.node_id,
                            "inferred_relation",
                            evidence_keys=(prior_node.evidence_ids[-1], stored.key),
                        )
                self.place_id = node.node_id
                self._prior_place_id = node.node_id
        self.last_v3_diagnostic = {
            "events_observed": len(events),
            "events_persisted": persisted,
            "detection_events_persisted": detection_persisted,
            "landmark_nodes": landmark_nodes,
            "place_nodes": place_nodes,
            "evidence_records": len(
                self.v3_store.evidence_for_scene(self.scene_id)
            ),
        }

    def _evidence_key(self, event_id: str) -> str:
        from l3_memory.v3_schemas import canonical_evidence_key

        assert self.scene_id is not None and self.episode_id is not None
        return canonical_evidence_key(
            self.scene_id, self.episode_id, self.rollout_idx, event_id,
        )


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()

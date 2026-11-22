"""The C L1 path is LightNav's original Local Graph implementation.

These tests intentionally assert the original card text and the original
write-path shape (no organizer fields, no SBAdir suffix, no extra instructions
or confidence filters beyond ``LocalGraphFusion._MIN_CONFIDENCE``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from l1_integration.sbadir_backend import (
    ORIGINAL_L1_PROMPT_MAX_CHARS,
    L1EvidenceBackend,
)
from l1_memory.memory_writer import MockSceneExtractor
from l1_memory.place_recognition import MockPlaceDescriptorExtractor
from l1_memory.yolo_scene_extractor import YOLOSceneExtractor
from l3_memory.schemas import DetectedObject, SceneExtraction

HEADER = "[Navigation Memory]"


class _FakeYolo:
    task = "detect"
    names = {0: "chair"}

    def __init__(self) -> None:
        self.input_frame = None

    def predict(self, *, source, conf, device, verbose, save):
        self.input_frame = source.copy()
        return [SimpleNamespace(
            names=self.names,
            boxes=SimpleNamespace(cls=[0], conf=[0.87], xyxy=[[0, 0, 3, 3]]),
        )]


def _rgb() -> np.ndarray:
    rgb = np.zeros((4, 9, 3), dtype=np.uint8)
    rgb[0, 0] = [10, 20, 30]
    return rgb


def _detector(confidence: float = 0.91, step: int = 0):
    return MockSceneExtractor({
        step: SceneExtraction(
            detections=[DetectedObject("chair", confidence, (0, 0, 3, 3))],
            source="mock_scene_extractor", confidence=confidence,
        ),
    })


def test_rgb_detector_builds_the_exact_original_local_graph_card(tmp_path):
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"test fixture")
    model = _FakeYolo()
    detector = YOLOSceneExtractor(
        str(weights), model_factory=lambda _: model, every_n_steps=1,
    )
    backend = L1EvidenceBackend(detector)
    backend.reset_episode("scene-A", "episode-1", "Find the chair")

    events, diagnostic = backend.observe(_rgb(), 0)
    selection = backend.select(step=0, rgb_width=9)

    assert model.input_frame[0, 0].tolist() == [30, 20, 10]
    assert diagnostic["detection_count"] == 1
    assert diagnostic["events_persisted"] >= 1
    assert backend.last_events == events
    assert selection.fact_count == 1
    expected = (
        HEADER + "\n"
        "Current RGB: instruction landmark chair in the left of the current "
        "view (detected; passage unknown)."
    )
    assert selection.context.text == expected
    assert selection.prompt_text == expected
    assert len(selection.prompt_text) <= ORIGINAL_L1_PROMPT_MAX_CHARS
    assert selection.diagnostics["budget_chars"] == 320
    assert selection.diagnostics["max_landmarks"] == 2
    assert selection.diagnostics["source"] == "local_graph_fusion"
    assert len(selection.evidence_ids) == 1
    record = backend.v3_store.get_evidence(selection.evidence_ids[0])
    assert record is not None
    assert record.semantic_label == "chair"
    assert record.raw_detector_confidence == 0.87
    # The original write path stores only the description (and action for an
    # action event); it does not add an L1 source-background payload.
    assert set(record.payload) == {"description"}
    assert not hasattr(backend, "organizer")
    backend.close_episode()


def test_original_history_is_available_on_a_non_place_step():
    detector = MockSceneExtractor({
        0: SceneExtraction(
            detections=[DetectedObject("chair", 0.91, (0, 0, 3, 3))],
            source="mock_scene_extractor", confidence=0.91,
        ),
        1: SceneExtraction(
            detections=[DetectedObject("chair", 0.91, (0, 0, 3, 3))],
            source="mock_scene_extractor", confidence=0.91,
        ),
    })
    places = MockPlaceDescriptorExtractor({0: (1.0, 0.0)})
    backend = L1EvidenceBackend(
        detector, place_extractor=places, place_every_n_steps=2,
    )
    backend.reset_episode("scene-A", "episode-1", "Find the chair")
    backend.observe(_rgb(), 0)
    first = backend.select(step=0, rgb_width=9)
    assert first.context is not None
    backend.discard_selection()
    events, diagnostic = backend.observe(_rgb(), 1)
    second = backend.select(step=1, rgb_width=9)

    assert diagnostic["place_attempted"] is False
    assert backend.last_place_observed is False
    # Original MemoryManager.get_local_graph_context keeps passing the last
    # observed place on intermediate steps.
    lines = second.context.text.splitlines()
    assert lines[0] == HEADER
    assert any(line.startswith("Local history: chair was seen near") for line in lines)
    assert backend.place_id is not None
    assert any(event.kind == "place_observation" for event in events) is False
    backend.close_episode()


def test_no_detection_yields_an_empty_original_card_without_an_error():
    backend = L1EvidenceBackend(MockSceneExtractor())
    backend.reset_episode("scene-A", "episode-1", "Find the chair")
    events, diagnostic = backend.observe(_rgb(), 0)
    selection = backend.select(step=0, rgb_width=9)

    assert diagnostic["detection_count"] == 0
    assert selection.context is None
    assert selection.prompt_text is None
    assert selection.fact_count == 0
    assert selection.evidence_ids == ()
    assert selection.diagnostics["empty_reason"] == "original_builder_empty"
    assert events == []  # no detection and no action event at step 0
    assert diagnostic["events_persisted"] == 0
    backend.close_episode()


def test_low_confidence_detection_is_rejected_by_the_original_filter_only():
    backend = L1EvidenceBackend(_detector(confidence=0.30))
    backend.reset_episode("scene-A", "episode-1", "Find the chair")
    backend.observe(_rgb(), 0)
    selection = backend.select(step=0, rgb_width=9)

    # The event is persisted (observation_layer), but LocalGraphFusion's
    # original 0.5 current-candidate threshold excludes it from the prompt.
    assert len(backend.v3_store.evidence_for_scene("scene-A")) >= 1
    assert selection.context is None
    assert selection.prompt_text is None
    backend.close_episode()


def test_confirming_the_card_suppresses_the_same_signature_within_8_steps():
    backend = L1EvidenceBackend(_detector())
    backend.reset_episode("scene-A", "episode-1", "Find the chair")
    backend.observe(_rgb(), 0)
    selection = backend.select(step=0, rgb_width=9)
    backend.confirm_injected(selection.evidence_ids, step=0)

    backend.observe(_rgb(), 1)
    repeated = backend.select(step=1, rgb_width=9)
    assert repeated.context is None
    assert repeated.prompt_text is None
    backend.close_episode()


def test_episode_reset_does_not_reuse_prior_episode_observations():
    detector = _detector()
    backend = L1EvidenceBackend(detector)
    backend.reset_episode("scene-A", "episode-1", "Find the chair")
    backend.observe(_rgb(), 0)
    assert backend.select(step=0, rgb_width=9).context is not None

    detector.by_step.clear()
    backend.reset_episode("scene-A", "episode-2", "Find the chair")
    backend.observe(_rgb(), 0)
    selection = backend.select(step=0, rgb_width=9)
    assert selection.context is None
    assert selection.prompt_text is None
    assert backend.v3_store.evidence_for_scene("scene-A") == ()
    backend.close_episode()

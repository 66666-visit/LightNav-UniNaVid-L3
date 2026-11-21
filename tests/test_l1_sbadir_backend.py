"""Real RGB-to-L1 write path and source-record preservation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from l1_integration.sbadir_backend import L1EvidenceBackend
from l1_memory.memory_writer import MockSceneExtractor
from l1_memory.place_recognition import MockPlaceDescriptorExtractor
from l1_memory.yolo_scene_extractor import YOLOSceneExtractor
from l3_memory.schemas import DetectedObject, SceneExtraction


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


def test_rgb_detector_writes_fact_with_actual_source_confidence_and_bbox(tmp_path):
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
    assert backend.last_events == events
    assert len(selection.organized_facts) == 1
    assert len(selection.selected_facts) == 1
    fact = selection.organized_facts[0]
    assert fact.label == "chair"
    assert fact.role == "current"
    assert fact.sector == "left"
    assert fact.entity_id is None
    assert fact.attribute == "object_label"
    assert fact.value == "chair"
    assert fact.evidence_ids == (diagnostic["detection_evidence_ids"][0],)
    assert len(fact.evidence) == 1
    source = fact.evidence[0]
    assert source.evidence_id == fact.evidence_ids[0]
    assert source.observation_step == 0
    assert source.object_label == "chair"
    assert source.detection_confidence == 0.87
    assert source.detector_name == "ultralytics_yolo:yolo11n.pt"
    assert source.rgb_observation_source == "observation.rgb"
    assert source.bbox_xyxy == (0.0, 0.0, 3.0, 3.0)
    assert backend.v3_store.get_evidence(source.evidence_id) is not None
    backend.close_episode()


def test_dinov2_place_history_retrieves_observed_fact_without_identity_claim():
    detector = MockSceneExtractor({
        0: SceneExtraction(
            detections=[DetectedObject("chair", 0.91, (0, 0, 3, 3))],
            source="mock_scene_extractor", confidence=0.91,
        ),
    })
    places = MockPlaceDescriptorExtractor({0: (1.0, 0.0), 1: (1.0, 0.0)})
    backend = L1EvidenceBackend(
        detector, place_extractor=places, place_every_n_steps=1,
    )
    backend.reset_episode("scene-A", "episode-1", "Find the chair")
    backend.observe(_rgb(), 0)
    first = backend.select(step=0, rgb_width=9)
    backend.discard_local_graph()
    backend.observe(_rgb(), 1)
    second = backend.select(step=1, rgb_width=9)

    assert first.organized_facts[0].role == "current"
    assert backend.place_id is not None
    assert any(fact.role == "history" for fact in second.organized_facts)
    assert any(fact.role == "history" for fact in second.local_graph_facts)
    assert len(second.selected_facts) == 1
    merged = second.selected_facts[0]
    assert "possible revisit; current direction and passage unknown" in merged.text
    assert merged.selection_source == "l1_prompt_organizer+local_graph_fusion"
    assert merged.visual_place_id == backend.place_id
    assert merged.visual_place_evidence_ids
    assert merged.visual_place_source == "mock_place_descriptor"
    assert backend.v3_store.get_evidence(merged.visual_place_evidence_ids[-1]) is not None
    for fact in (*second.organized_facts, *second.local_graph_facts):
        assert fact.entity_id is None
        assert fact.evidence[0].observation_step == 0
        assert fact.evidence[0].detection_confidence == 0.91
        assert fact.evidence[0].rgb_observation_source == "observation.rgb"
    backend.close_episode()


def test_no_detector_observation_yields_no_l1_fact():
    backend = L1EvidenceBackend(MockSceneExtractor())
    backend.reset_episode("scene-A", "episode-1", "Find the chair")
    _, diagnostic = backend.observe(_rgb(), 0)
    selection = backend.select(step=0, rgb_width=9)
    assert diagnostic["detection_count"] == 0
    assert selection.organized_facts == ()
    assert selection.local_graph_facts == ()
    backend.close_episode()


def test_unsampled_place_step_does_not_claim_current_visual_similarity():
    detector = MockSceneExtractor({
        0: SceneExtraction(
            detections=[DetectedObject("chair", 0.91, (0, 0, 3, 3))],
            source="mock_scene_extractor", confidence=0.91,
        ),
    })
    places = MockPlaceDescriptorExtractor({0: (1.0, 0.0)})
    backend = L1EvidenceBackend(detector, place_extractor=places, place_every_n_steps=2)
    backend.reset_episode("scene-A", "episode-1", "Find the chair")
    backend.observe(_rgb(), 0)
    backend.select(step=0, rgb_width=9)
    backend.discard_local_graph()
    backend.observe(_rgb(), 1, last_action={"action": 1})
    selection = backend.select(step=1, rgb_width=9)
    assert backend.last_place_observed is False
    assert selection.organized_facts[0].role == "history"
    assert not any(fact.visual_place_id for fact in selection.selected_facts)
    assert not any("possible revisit" in fact.text for fact in selection.selected_facts)
    backend.close_episode()


def test_episode_reset_does_not_reuse_prior_episode_observations():
    detector = MockSceneExtractor({
        0: SceneExtraction(
            detections=[DetectedObject("chair", 0.91, (0, 0, 3, 3))],
            source="mock_scene_extractor", confidence=0.91,
        ),
    })
    backend = L1EvidenceBackend(detector)
    backend.reset_episode("scene-A", "episode-1", "Find the chair")
    backend.observe(_rgb(), 0)
    assert backend.select(step=0, rgb_width=9).organized_facts

    detector.by_step.clear()
    backend.reset_episode("scene-A", "episode-2", "Find the chair")
    backend.observe(_rgb(), 0)
    selection = backend.select(step=0, rgb_width=9)
    assert selection.organized_facts == ()
    assert selection.local_graph_facts == ()
    assert backend.v3_store.evidence_for_scene("scene-A") == ()
    backend.close_episode()

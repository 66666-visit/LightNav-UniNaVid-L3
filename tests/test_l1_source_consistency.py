"""Pin the C L1 implementation to the 75% LightNav baseline.

The 75% score is the LightNav tag ``baseline-local-graph-fusion-88-sr75`` at
commit ``2fb184eeaaf3bbc4c85d86f630e237e572e78e6d``.  That commit predates
``L1PromptOrganizer``; the group uses ``LocalGraphFusion``.  The vendored files
are checked against that commit, not against a later prompt-ablation commit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path("/home/lilixing/projects/LightNav-0-codex")
BASELINE_COMMIT = "2fb184eeaaf3bbc4c85d86f630e237e572e78e6d"
COPIED_FROM_COMMIT = "6aefffada23de3d4edd422c3c9289182067dbad2"
TAG = "baseline-local-graph-fusion-88-sr75"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_show(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "show", f"{commit}:{path}"],
    )


def test_provenance_pins_the_75_percent_baseline_and_hashes() -> None:
    provenance = json.loads(
        (ROOT / "l1_memory/provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["source_commit"] == BASELINE_COMMIT
    assert provenance["source_tag"] == TAG
    assert "SR 75%" in provenance["source_tag_message"]
    assert provenance["evidence"]["l1_group_builder"] == "LocalGraphFusion"
    assert provenance["evidence"]["l1_original_only_uses_l1_prompt_organizer"] is False
    assert provenance["original_constants"] == {
        "prompt_max_chars": 320,
        "max_landmarks": 2,
        "reactivation_steps": 8,
        "place_every_n_steps": 5,
        "detector_every_n_steps": 5,
        "detector_confidence": 0.25,
        "scope": "episode_local",
    }
    assert len(provenance["files"]) == 11
    for entry in provenance["files"]:
        vendored = ROOT / entry["vendored"]
        assert _sha(vendored) == entry["vendored_sha256"], entry["name"]
        if entry["source"] is None:
            continue
        source = Path(entry["source"])
        if not source.is_file():
            continue
        assert _sha(source) == entry["source_sha256"], entry["name"]


def test_source_files_still_match_the_pinned_commit() -> None:
    if not (SOURCE_REPO / ".git").exists():
        pytest.skip("LightNav source checkout is not available")
    provenance = json.loads(
        (ROOT / "l1_memory/provenance.json").read_text(encoding="utf-8")
    )
    tag_commit = subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "rev-parse", TAG + "^{commit}"],
        text=True,
    ).strip()
    assert tag_commit == BASELINE_COMMIT
    for entry in provenance["files"]:
        if entry["source_commit"] is None:
            continue
        relative = "src/lightnav/memory/" + entry["name"]
        assert hashlib.sha256(_git_show(BASELINE_COMMIT, relative)).hexdigest() == (
            entry["source_sha256"]
        ), entry["name"]


def test_local_graph_fusion_is_identical_at_the_baseline_and_later_copy() -> None:
    if not (SOURCE_REPO / ".git").exists():
        pytest.skip("LightNav source checkout is not available")
    baseline = _git_show(BASELINE_COMMIT, "src/lightnav/memory/local_graph_fusion.py")
    later = _git_show(COPIED_FROM_COMMIT, "src/lightnav/memory/local_graph_fusion.py")
    assert hashlib.sha256(baseline).hexdigest() == hashlib.sha256(later).hexdigest()
    # The original builder is the one imported by the C adapter.
    vendored = (ROOT / "l1_memory/local_graph_fusion.py").read_text(encoding="utf-8")
    assert "class LocalGraphFusion" in vendored
    assert "_MIN_CONFIDENCE = 0.5" in vendored
    assert "prompt_max_chars" in vendored


def test_c_adapter_does_not_import_or_call_the_organizer() -> None:
    backend = (ROOT / "l1_integration/sbadir_backend.py").read_text(encoding="utf-8")
    agent = (ROOT / "agent_uninavid.py").read_text(encoding="utf-8")
    for text in (backend, agent):
        assert "L1PromptOrganizer" not in text
        assert "l1_prompt_organizer" not in text
        assert "65536" not in text
        assert "l1_max_tokens" not in text
    # L1SelectedFact/selected_facts were the old organizer-merge surface.
    assert "L1SelectedFact" not in backend
    assert "selected_facts" not in backend


def test_backend_uses_the_original_320_character_and_two_landmark_values() -> None:
    from l1_integration import sbadir_backend

    assert sbadir_backend.ORIGINAL_L1_PROMPT_MAX_CHARS == 320
    assert sbadir_backend.ORIGINAL_L1_MAX_LANDMARKS == 2
    assert sbadir_backend.ORIGINAL_L1_REACTIVATION_STEPS == 8
    assert sbadir_backend.ORIGINAL_L1_PLACE_EVERY_N_STEPS == 5

"""Strict loading and matching of the fixed 88-Episode R2R pilot manifest.

Three different digests are involved and **must not be conflated**:

1. ``pilot_manifest.json`` file SHA256 (raw bytes) -- the frozen Pilot input,
   pinned by :data:`EXPECTED_PILOT_MANIFEST_SHA256`.
2. ``parent_manifest_sha256`` -- the raw file SHA256 of the older manifest the
   Pilot was sampled from; it is a field *inside* the Pilot file, not a hash of
   the Pilot file.
3. The evaluation ``episode_manifest.json`` canonical SHA256 -- the digest
   written by LightNav's completion marker as
   ``sha256(json.dumps(validate_manifest(episode_manifest.json), sort_keys=True))``.
   It is *not* the file's raw SHA256.  :func:`canonical_episode_manifest_sha256`
   reproduces that exact convention.

The Pilot is an experiment input, never a value this project may invent.  If a
file is missing or its digest differs from the frozen value, loading fails
closed and the caller must obtain the original file.  No sampling, replacement,
reordering or instruction rewriting happens here.

Matching uses the Episode ID and the scene file's basename.  The manifest's
scene path and Habitat's SCENES_DIR-joined path may use different prefixes on
AutoDL, but the HM3D scene basename is unique in the R2R val_unseen split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

#: Frozen raw-file SHA256 of the 88-row ``pilot_manifest.json`` (the Pilot
#: selection input).  This is a file digest, not the completion-marker digest.
EXPECTED_PILOT_MANIFEST_SHA256 = (
    "cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a"
)

#: Historical completion-marker digest of the 88-row evaluation
#: ``episode_manifest.json``.  It is the canonical JSON digest produced by
#: ``sha256(json.dumps(validate_manifest(...), sort_keys=True))`` and is *not*
#: a raw file SHA256.  Keep it separate from the Pilot file digest above.
EXPECTED_EPISODE_MANIFEST_SHA256 = (
    "30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98"
)

EXPECTED_EPISODES = 88
EXPECTED_SCENES = 11
EXPECTED_PER_SCENE = 8
EXPECTED_STRATA = {"short": 3, "medium": 3, "long": 2}


class PilotManifestError(ValueError):
    """The supplied 88-Episode manifest is missing or inconsistent."""


@dataclass(frozen=True)
class PilotRow:
    scene_id: str
    episode_id: str
    instruction: str | None
    distance_group: str | None
    raw: dict[str, Any]

    @property
    def scene_basename(self) -> str:
        return scene_basename(self.scene_id)


@dataclass(frozen=True)
class PilotManifest:
    path: Path
    #: Raw SHA256 of the Pilot file bytes.  Never compare this with
    #: ``EXPECTED_EPISODE_MANIFEST_SHA256``.
    file_sha256: str
    rows: tuple[PilotRow, ...]
    protocol_version: str | None
    dataset_split: str | None
    sample_seed: int | None
    instructions_available: bool
    #: Raw SHA256 declared for the parent (sampling) manifest, if present.
    parent_manifest_sha256: str | None = None

    @property
    def expected_count(self) -> int:
        return len(self.rows)

    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple((row.scene_basename, row.episode_id) for row in self.rows)

    def scene_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.scene_basename] = counts.get(row.scene_basename, 0) + 1
        return dict(sorted(counts.items()))

    def strata_counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for row in self.rows:
            if row.distance_group is None:
                continue
            scene = counts.setdefault(
                row.scene_basename, {"short": 0, "medium": 0, "long": 0},
            )
            scene[row.distance_group] += 1
        return dict(sorted(counts.items()))

    def order_digest(self, instructions: Iterable[str] | None = None) -> str:
        return order_digest(self.rows, instructions=instructions)

    def audit(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "digest_kind": "pilot_manifest.json raw byte SHA256",
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "protocol_version": self.protocol_version,
            "dataset_split": self.dataset_split,
            "sample_seed": self.sample_seed,
            "expected_count": self.expected_count,
            "scene_counts": self.scene_counts(),
            "strata_counts": self.strata_counts(),
            "instructions_available": self.instructions_available,
            "order_digest": self.order_digest(),
            "episode_keys": [
                {"scene_basename": row.scene_basename, "episode_id": row.episode_id}
                for row in self.rows
            ],
        }


def scene_basename(scene_id: object) -> str:
    """Return a stable scene file name, rejecting empty/ambiguous values."""
    if not isinstance(scene_id, str) or not scene_id.strip() or "\x00" in scene_id:
        raise PilotManifestError("scene_id must be a nonempty path string")
    name = Path(scene_id.strip().rstrip("/")).name
    if not name:
        raise PilotManifestError(f"scene_id has no file name: {scene_id!r}")
    return name


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    try:
        if not target.is_file():
            raise FileNotFoundError(
                f"manifest file not found: {target}. "
                "Provide the existing file; do not regenerate or substitute it."
            )
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except PermissionError as exc:
        raise PilotManifestError(
            f"manifest file is not readable: {target}. "
            "Provide the original file with read permission; do not regenerate "
            "or substitute it."
        ) from exc
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise PilotManifestError(
            f"manifest file could not be read: {target} ({exc})"
        ) from exc


def _normalized_scene_key(scene_id: object) -> str:
    """Mirror LightNav's ``_normalize_scene_id`` for duplicate detection."""
    if not isinstance(scene_id, str) or not scene_id.strip() or "\x00" in scene_id:
        raise PilotManifestError(
            "episode manifest scene_id must be a nonempty path string"
        )
    try:
        return str(Path(scene_id).resolve(strict=False))
    except (OSError, RuntimeError, ValueError) as exc:
        raise PilotManifestError(
            f"episode manifest scene_id is invalid: {scene_id!r}"
        ) from exc


def _episode_manifest_key(
    row: Mapping[str, Any], *, result: bool = False,
) -> tuple[str, str]:
    """Mirror LightNav's ``_key`` for an episode-manifest row."""
    if not isinstance(row, Mapping):
        raise PilotManifestError("episode manifest episode must be an object")
    scene = row.get("scene_id")
    episode = row.get("habitat_episode_id" if result else "episode_id")
    if episode is None or not str(episode).strip() or "\x00" in str(episode):
        raise PilotManifestError(
            "episode manifest row requires nonempty scene_id and episode_id"
        )
    return _normalized_scene_key(scene), str(episode)


def canonicalize_episode_manifest(data: Any) -> dict[str, Any]:
    """Return the exact dict LightNav's completion marker hashes.

    This is a line-for-line mirror of
    ``lightnav.habitat.experiment_integrity.validate_manifest``: unknown fields
    are dropped, ``episode_id`` is stringified, duplicate identities are
    rejected, and stored scene path spellings are preserved.
    """
    if not isinstance(data, dict) or not isinstance(data.get("split"), str):
        raise PilotManifestError("episode manifest requires a split string")
    rows = data.get("episodes")
    if not isinstance(rows, list):
        raise PilotManifestError("episode manifest requires an episodes list")
    expected = data.get("expected_count")
    if type(expected) is not int or expected != len(rows):
        raise PilotManifestError(
            "episode manifest expected_count differs from episode list length"
        )
    seen: set[tuple[str, str]] = set()
    episodes: list[dict[str, str]] = []
    for row in rows:
        key = _episode_manifest_key(row)
        if key in seen:
            raise PilotManifestError(f"duplicate episode in manifest: {key}")
        seen.add(key)
        # Keep the original scene path spelling; only the identity is
        # normalized.  This matches the marker's canonical payload.
        episodes.append({"scene_id": row["scene_id"], "episode_id": key[1]})
    return {
        "split": data["split"],
        "expected_count": expected,
        "episodes": episodes,
    }


def canonical_episode_manifest_sha256(data: Any) -> str:
    """Completion-marker digest: canonical JSON SHA256, **not** file bytes.

    LightNav writes ``pilot_complete.json["manifest_sha256"]`` as
    ``sha256(json.dumps(validate_manifest(episode_manifest.json), sort_keys=True))``.
    Consequently ``sha256sum episode_manifest.json`` will not (in general)
    equal the marker digest.  This function is the verifier for that marker.
    """
    canonical = canonicalize_episode_manifest(data)
    payload = json.dumps(canonical, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class EpisodeRow:
    scene_id: str
    episode_id: str

    @property
    def scene_basename(self) -> str:
        return scene_basename(self.scene_id)


@dataclass(frozen=True)
class EpisodeManifest:
    path: Path
    file_sha256: str
    canonical_sha256: str
    split: str
    expected_count: int
    rows: tuple[EpisodeRow, ...]

    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple((row.scene_basename, row.episode_id) for row in self.rows)

    def scene_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.scene_basename] = counts.get(row.scene_basename, 0) + 1
        return dict(sorted(counts.items()))

    def audit(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "canonical_sha256": self.canonical_sha256,
            "digest_kind": (
                "canonical JSON SHA256 used by pilot_complete.json"
                "['manifest_sha256'], not the file SHA256"
            ),
            "split": self.split,
            "expected_count": self.expected_count,
            "scene_counts": self.scene_counts(),
            "episode_keys": [
                {"scene_basename": row.scene_basename,
                 "episode_id": row.episode_id}
                for row in self.rows
            ],
        }


def load_episode_manifest(
    path: str | Path,
    *,
    expected_canonical_sha256: str | None = EXPECTED_EPISODE_MANIFEST_SHA256,
) -> EpisodeManifest:
    """Load the evaluation manifest and verify the *canonical* digest."""
    target = Path(path)
    file_digest = sha256_file(target)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PilotManifestError(
            f"episode manifest is not valid JSON: {target}"
        ) from exc
    canonical = canonicalize_episode_manifest(data)
    digest = canonical_episode_manifest_sha256(data)
    if expected_canonical_sha256 is not None:
        if (
            not isinstance(expected_canonical_sha256, str)
            or len(expected_canonical_sha256) != 64
        ):
            raise PilotManifestError(
                "expected episode manifest digest must be a 64-character digest"
            )
        if digest != expected_canonical_sha256:
            raise PilotManifestError(
                "episode manifest canonical SHA256 mismatch "
                "(completion-marker digest, not file SHA256): "
                f"expected {expected_canonical_sha256}, got {digest}; "
                f"file SHA256 is {file_digest} ({target})"
            )
    return EpisodeManifest(
        path=target,
        file_sha256=file_digest,
        canonical_sha256=digest,
        split=canonical["split"],
        expected_count=canonical["expected_count"],
        rows=tuple(
            EpisodeRow(row["scene_id"], row["episode_id"])
            for row in canonical["episodes"]
        ),
    )


def compare_episode_manifest_to_pilot(
    episode: EpisodeManifest, pilot: PilotManifest,
) -> dict[str, Any]:
    """Require the exact 88 ordered (scene basename, Episode ID) identities."""
    if episode.split != "val_unseen":
        raise PilotManifestError(
            f"episode manifest split must be val_unseen; got {episode.split!r}"
        )
    expected = pilot.keys()
    actual = episode.keys()
    if len(expected) != EXPECTED_EPISODES:
        raise PilotManifestError(
            f"pilot manifest has {len(expected)} Episodes; expected "
            f"{EXPECTED_EPISODES}"
        )
    if len(actual) != EXPECTED_EPISODES:
        raise PilotManifestError(
            f"episode manifest has {len(actual)} Episodes; expected "
            f"{EXPECTED_EPISODES}"
        )
    for index, (pilot_key, episode_key) in enumerate(zip(expected, actual)):
        if pilot_key != episode_key:
            raise PilotManifestError(
                "episode manifest order/identity differs from the Pilot at "
                f"index {index}: pilot={pilot_key}, episode={episode_key}"
            )
    return {
        "match": True,
        "count": len(actual),
        "pilot_count": len(expected),
        "order": "pilot_manifest.episodes order preserved",
        "first_key": actual[0],
        "last_key": actual[-1],
        "pilot_scene_counts": pilot.scene_counts(),
        "episode_scene_counts": episode.scene_counts(),
    }


def load_completion_marker(
    path: str | Path,
    *,
    expected_sha256: str = EXPECTED_EPISODE_MANIFEST_SHA256,
    expected_count: int = EXPECTED_EPISODES,
) -> dict[str, Any]:
    """Verify LightNav's ``pilot_complete.json`` marker without rewriting it."""
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PilotManifestError(
            f"completion marker is not valid JSON: {target}"
        ) from exc
    if not isinstance(data, dict):
        raise PilotManifestError("completion marker must be a JSON object")
    count = data.get("expected_count")
    digest = data.get("manifest_sha256")
    if type(count) is not int or count != expected_count:
        raise PilotManifestError(
            "completion marker expected_count mismatch: "
            f"expected {expected_count}, got {count!r} ({target})"
        )
    if not isinstance(digest, str) or len(digest) != 64:
        raise PilotManifestError(
            "completion marker manifest_sha256 must be a 64-character digest"
        )
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise PilotManifestError(
            "expected completion marker digest must be a 64-character digest"
        )
    if digest != expected_sha256:
        raise PilotManifestError(
            "completion marker canonical digest mismatch: "
            f"expected {expected_sha256}, got {digest} ({target})"
        )
    return {
        "path": str(target),
        "expected_count": count,
        "manifest_sha256": digest,
        "expected_manifest_sha256": expected_sha256,
        "verified": True,
        "digest_kind": (
            "canonical episode_manifest.json digest from pilot_complete.json"
        ),
    }


def scan_episode_manifests(
    root: str | Path,
    *,
    expected_canonical_sha256: str = EXPECTED_EPISODE_MANIFEST_SHA256,
) -> list[dict[str, Any]]:
    """Recompute the marker digest for every ``episode_manifest.json``.

    This is intentionally a canonical-JSON recomputation, not ``sha256sum``:
    the historical completion marker never stored the raw file digest.
    """
    target = Path(root)
    if not target.is_dir():
        raise PilotManifestError(f"scan root is not a directory: {target}")
    results: list[dict[str, Any]] = []
    for path in sorted(target.rglob("episode_manifest.json")):
        entry: dict[str, Any] = {"path": str(path)}
        try:
            entry["file_sha256"] = sha256_file(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            entry["canonical_sha256"] = canonical_episode_manifest_sha256(data)
            entry["matches_expected"] = (
                entry["canonical_sha256"] == expected_canonical_sha256
            )
            entry["error"] = None
        except Exception as exc:  # noqa: BLE001 - scan reports, never hides
            entry.update({
                "file_sha256": entry.get("file_sha256"),
                "canonical_sha256": None,
                "matches_expected": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
        results.append(entry)
    return results


def _row_instruction(row: Mapping[str, Any]) -> str | None:
    for key in ("instruction", "instruction_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, Mapping):
            nested = value.get("instruction_text") or value.get("text")
            if isinstance(nested, str) and nested.strip():
                return nested
    return None


def _row_episode_id(row: Mapping[str, Any]) -> str:
    for key in ("habitat_episode_id", "episode_id"):
        value = row.get(key)
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    raise PilotManifestError("pilot row lacks habitat_episode_id")


def validate_pilot_manifest(data: Any) -> tuple[PilotRow, ...]:
    """Validate exactly 88 unique val_unseen Episodes and 11x8 scene balance."""
    if not isinstance(data, dict):
        raise PilotManifestError("pilot manifest must be a JSON object")
    rows = data.get("episodes")
    if not isinstance(rows, list):
        raise PilotManifestError("pilot manifest requires an episodes list")
    expected = data.get("expected_count")
    if expected != EXPECTED_EPISODES or len(rows) != EXPECTED_EPISODES:
        raise PilotManifestError(
            f"pilot manifest must contain exactly {EXPECTED_EPISODES} Episodes; "
            f"expected_count={expected!r}, rows={len(rows)}"
        )
    split = data.get("dataset_split")
    if split is not None and split != "val_unseen":
        raise PilotManifestError("pilot manifest split must be val_unseen")
    seed = data.get("sample_seed")
    if seed is not None and seed != 42:
        raise PilotManifestError("pilot manifest sample_seed must be 42")
    protocol = data.get("protocol_version")
    if protocol is not None and not (
        isinstance(protocol, str) and protocol.startswith("rgm_pilot_88")
    ):
        raise PilotManifestError(
            f"pilot manifest protocol_version is not an rgm_pilot_88 protocol: {protocol!r}"
        )
    parent_sha = data.get("parent_manifest_sha256")
    if parent_sha is not None and not (
        isinstance(parent_sha, str) and len(parent_sha) == 64
    ):
        raise PilotManifestError("parent_manifest_sha256 must be a 64-character digest")

    parsed: list[PilotRow] = []
    seen: set[tuple[str, str]] = set()
    instruction_presence: list[bool] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PilotManifestError(f"pilot row {index} is not an object")
        scene = row.get("scene_id")
        if not isinstance(scene, str) or not scene.strip():
            raise PilotManifestError(f"pilot row {index} lacks scene_id")
        episode = _row_episode_id(row)
        key = (scene_basename(scene), episode)
        if key in seen:
            raise PilotManifestError(f"duplicate pilot Episode: {key}")
        seen.add(key)
        instruction = _row_instruction(row)
        instruction_presence.append(instruction is not None)
        group = row.get("distance_group")
        if group is not None and group not in ("short", "medium", "long"):
            raise PilotManifestError(
                f"pilot row {index} has invalid distance_group: {group!r}"
            )
        parsed.append(PilotRow(
            scene_id=scene, episode_id=episode, instruction=instruction,
            distance_group=group, raw=dict(row),
        ))
    if any(instruction_presence) and not all(instruction_presence):
        raise PilotManifestError(
            "pilot manifest instruction fields are present for only some rows"
        )

    scene_counts: dict[str, int] = {}
    strata: dict[str, dict[str, int]] = {}
    for row in parsed:
        scene = row.scene_basename
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
        if row.distance_group is not None:
            current = strata.setdefault(scene, {"short": 0, "medium": 0, "long": 0})
            current[row.distance_group] += 1
    if len(scene_counts) != EXPECTED_SCENES:
        raise PilotManifestError(
            f"pilot manifest must cover {EXPECTED_SCENES} scenes; "
            f"found {len(scene_counts)}"
        )
    if any(count != EXPECTED_PER_SCENE for count in scene_counts.values()):
        raise PilotManifestError(
            f"pilot manifest must contain {EXPECTED_PER_SCENE} Episodes per scene; "
            f"found {scene_counts}"
        )
    if strata:
        if set(strata) != set(scene_counts):
            raise PilotManifestError("distance_group is missing for some scenes")
        for scene, counts in strata.items():
            if counts != EXPECTED_STRATA:
                raise PilotManifestError(
                    f"scene {scene} distance strata must be 3/3/2; found {counts}"
                )
    quotas = data.get("scene_quotas")
    if quotas is not None:
        if not isinstance(quotas, dict) or len(quotas) != EXPECTED_SCENES:
            raise PilotManifestError("scene_quotas must cover all 11 scenes")
        if any(type(value) is not int or value != EXPECTED_PER_SCENE
               for value in quotas.values()):
            raise PilotManifestError("scene_quotas must contain 8 for every scene")
    return tuple(parsed)


def load_pilot_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = EXPECTED_PILOT_MANIFEST_SHA256,
) -> PilotManifest:
    """Load and validate the frozen Pilot; never create a replacement.

    ``expected_sha256`` is always compared with the **raw file bytes** of
    ``pilot_manifest.json``.  The historical completion-marker digest belongs
    to the evaluation ``episode_manifest.json`` and is checked by
    :func:`load_episode_manifest` instead.
    """
    target = Path(path)
    digest = sha256_file(target)
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise PilotManifestError("expected_sha256 must be a 64-character digest")
        if digest != expected_sha256:
            raise PilotManifestError(
                "pilot manifest file SHA256 mismatch: "
                f"expected {expected_sha256}, got {digest} ({target})"
            )
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PilotManifestError(f"pilot manifest is not valid JSON: {target}") from exc
    rows = validate_pilot_manifest(data)
    return PilotManifest(
        path=target,
        file_sha256=digest,
        rows=rows,
        protocol_version=data.get("protocol_version"),
        dataset_split=data.get("dataset_split"),
        sample_seed=data.get("sample_seed"),
        instructions_available=all(row.instruction is not None for row in rows),
        parent_manifest_sha256=data.get("parent_manifest_sha256"),
    )


def episode_instruction(episode: Any) -> str:
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


def _dataset_index(dataset_episodes: Iterable[Any]) -> dict[tuple[str, str], Any]:
    index: dict[tuple[str, str], Any] = {}
    for episode in dataset_episodes:
        scene = getattr(episode, "scene_id", None)
        episode_id = getattr(episode, "episode_id", None)
        if not scene or episode_id is None:
            raise PilotManifestError(
                "dataset episode lacks scene_id or episode_id"
            )
        key = (scene_basename(scene), str(episode_id))
        if key in index:
            raise PilotManifestError(f"dataset has duplicate Episode: {key}")
        index[key] = episode
    return index


def select_pilot_episodes(
    dataset_episodes: Iterable[Any],
    manifest: PilotManifest,
    *,
    check_instruction: bool = True,
) -> list[Any]:
    """Return the actual dataset Episodes in manifest order.

    This is the only selection path used by both experiment groups.  It never
    samples, sorts or substitutes.  A missing key is a hard failure.
    """
    index = _dataset_index(dataset_episodes)
    selected: list[Any] = []
    missing: list[tuple[str, str]] = []
    for row in manifest.rows:
        key = (row.scene_basename, row.episode_id)
        episode = index.get(key)
        if episode is None:
            missing.append(key)
            continue
        if (
            check_instruction
            and manifest.instructions_available
            and row.instruction is not None
        ):
            actual = episode_instruction(episode)
            if actual != row.instruction:
                raise PilotManifestError(
                    f"instruction differs from the pilot manifest for {key}"
                )
        selected.append(episode)
    if missing:
        raise PilotManifestError(
            "dataset is missing pilot Episodes: "
            + ", ".join(f"{scene}/{episode}" for scene, episode in missing[:5])
            + ("" if len(missing) <= 5 else f" (+{len(missing) - 5} more)")
        )
    if len(selected) != EXPECTED_EPISODES:
        raise PilotManifestError(
            f"selected {len(selected)} Episodes; expected {EXPECTED_EPISODES}"
        )
    return selected


def order_digest(
    rows: Iterable[PilotRow], *, instructions: Iterable[str] | None = None,
) -> str:
    """Digest ordered (scene basename, Episode ID, instruction) identities."""
    values = list(instructions) if instructions is not None else None
    records: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        record = {
            "scene": row.scene_basename,
            "episode_id": row.episode_id,
        }
        if values is not None:
            record["instruction_sha256"] = hashlib.sha256(
                str(values[index]).encode("utf-8")
            ).hexdigest()
        records.append(record)
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = EXPECTED_PILOT_MANIFEST_SHA256,
    dataset_episodes: Iterable[Any] | None = None,
    episode_manifest: str | Path | None = None,
    expected_episode_manifest_sha256: str | None = (
        EXPECTED_EPISODE_MANIFEST_SHA256
    ),
    parent_manifest: str | Path | None = None,
    expected_parent_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe audit with the three manifest digests separated.

    * Pilot: raw file SHA256, verified against
      :data:`EXPECTED_PILOT_MANIFEST_SHA256`.
    * Parent: the raw SHA256 declared *inside* the Pilot, optionally verified
      against the actual parent file.
    * Episode: the canonical JSON SHA256 used by the historical
      ``pilot_complete.json`` marker, verified by
      :func:`load_episode_manifest`, not by ``sha256sum``.
    """
    manifest = load_pilot_manifest(path, expected_sha256=expected_sha256)
    report = manifest.audit()

    declared_parent = manifest.parent_manifest_sha256
    if parent_manifest is not None:
        expected_parent = (
            expected_parent_manifest_sha256 or declared_parent
        )
        if expected_parent is None:
            raise PilotManifestError(
                "parent manifest supplied but the Pilot declares no "
                "parent_manifest_sha256; pass --expected-parent-sha256"
            )
        parent_digest = sha256_file(parent_manifest)
        if parent_digest != expected_parent:
            raise PilotManifestError(
                "parent manifest file SHA256 mismatch: "
                f"expected {expected_parent}, got {parent_digest} "
                f"({parent_manifest})"
            )
        report["parent_manifest"] = {
            "path": str(Path(parent_manifest)),
            "file_sha256": parent_digest,
            "declared_sha256_in_pilot": declared_parent,
            "expected_sha256": expected_parent,
            "verified": True,
            "digest_kind": "raw file SHA256 of the sampling parent manifest",
        }
    else:
        report["parent_manifest"] = {
            "path": None,
            "file_sha256": None,
            "declared_sha256_in_pilot": declared_parent,
            "expected_sha256": declared_parent,
            "verified": False,
            "digest_kind": "raw file SHA256 of the sampling parent manifest",
            "reason": (
                "parent manifest file not supplied; only the digest declared "
                "inside pilot_manifest.json is reported"
            ),
        }

    if episode_manifest is not None:
        episode = load_episode_manifest(
            episode_manifest,
            expected_canonical_sha256=expected_episode_manifest_sha256,
        )
        comparison = compare_episode_manifest_to_pilot(episode, manifest)
        episode_report = episode.audit()
        episode_report.update({
            "expected_canonical_sha256": expected_episode_manifest_sha256,
            "verified": True,
            "pilot_comparison": comparison,
        })
        report["episode_manifest"] = episode_report
    else:
        report["episode_manifest"] = {
            "path": None,
            "file_sha256": None,
            "canonical_sha256": None,
            "digest_kind": (
                "canonical JSON SHA256 used by pilot_complete.json"
                "['manifest_sha256'], not the file SHA256"
            ),
            "expected_canonical_sha256": expected_episode_manifest_sha256,
            "verified": False,
            "reason": (
                "episode manifest not supplied; the historical completion "
                "marker digest cannot be verified from the Pilot file alone"
            ),
        }

    if dataset_episodes is not None:
        selected = select_pilot_episodes(dataset_episodes, manifest)
        instructions = [episode_instruction(episode) for episode in selected]
        report["dataset_match"] = True
        report["dataset_count"] = len(selected)
        report["dataset_episode_keys"] = [
            {
                "scene_basename": scene_basename(episode.scene_id),
                "episode_id": str(episode.episode_id),
            }
            for episode in selected
        ]
        report["instruction_sha256"] = [
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in instructions
        ]
        report["dataset_order_digest"] = order_digest(
            manifest.rows, instructions=instructions,
        )
        report["instructions_available"] = manifest.instructions_available
    return report

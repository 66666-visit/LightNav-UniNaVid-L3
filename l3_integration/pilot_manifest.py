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

    def index(self) -> dict[tuple[str, str], PilotRow]:
        """Index Pilot metadata by Episode identity, never by list position."""
        return {(row.scene_basename, row.episode_id): row for row in self.rows}

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


@dataclass(frozen=True)
class ExecutionRow:
    """One pilot Episode in the historical evaluation manifest's order.

    ``scene_id`` is kept byte-for-byte as stored in the historical
    ``episode_manifest.json`` so that the generated execution manifest
    reproduces its canonical digest.  Pilot metadata (``distance_group`` and
    ``instruction``) is attached through ``(scene_basename, episode_id)``;
    ``pilot_index`` is diagnostic only and is never used for association.
    """

    scene_id: str
    episode_id: str
    distance_group: str | None
    instruction: str | None
    pilot_scene_id: str
    pilot_index: int
    execution_index: int

    @property
    def scene_basename(self) -> str:
        return scene_basename(self.scene_id)

    @property
    def key(self) -> tuple[str, str]:
        return (self.scene_basename, self.episode_id)

    def canonical_entry(self) -> dict[str, str]:
        """The only fields LightNav's completion marker hashes."""
        return {"scene_id": self.scene_id, "episode_id": self.episode_id}

    def manifest_entry(self) -> dict[str, Any]:
        """Execution row with Pilot metadata, for the generated artifact."""
        entry: dict[str, Any] = self.canonical_entry()
        if self.distance_group is not None:
            entry["distance_group"] = self.distance_group
        if self.instruction is not None:
            entry["instruction"] = self.instruction
        return entry

    def audit(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "scene_basename": self.scene_basename,
            "pilot_scene_id": self.pilot_scene_id,
            "distance_group": self.distance_group,
            "instruction": self.instruction,
            "execution_index": self.execution_index,
            "pilot_index": self.pilot_index,
        }


@dataclass(frozen=True)
class ExecutionManifest:
    """Execution plan: historical Episode order plus Pilot metadata by key."""

    episode: EpisodeManifest
    pilot: PilotManifest
    rows: tuple[ExecutionRow, ...]
    canonical_sha256: str
    order_source: str = "episode_manifest.json"

    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(row.key for row in self.rows)

    def index(self) -> dict[tuple[str, str], ExecutionRow]:
        return {row.key: row for row in self.rows}

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

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "split": self.episode.split,
            "expected_count": len(self.rows),
            "episodes": [row.canonical_entry() for row in self.rows],
        }

    def manifest_payload(self) -> dict[str, Any]:
        """JSON-safe artifact; unknown top-level keys do not change the digest."""
        return {
            "split": self.episode.split,
            "expected_count": len(self.rows),
            "episodes": [row.manifest_entry() for row in self.rows],
            "execution_order": self.order_source,
            "pilot_manifest_file_sha256": self.pilot.file_sha256,
            "episode_manifest_file_sha256": self.episode.file_sha256,
            "episode_manifest_canonical_sha256": self.episode.canonical_sha256,
            "execution_manifest_canonical_sha256": self.canonical_sha256,
            "execution_identity_digest": self.order_digest(),
        }

    def order_digest(self, instructions: Iterable[str] | None = None) -> str:
        return order_digest(self.rows, instructions=instructions)

    def comparison(self) -> dict[str, Any]:
        """Report a strict set match while retaining both orders."""
        episode_keys = _unique_keys(
            self.keys(), manifest_name="execution manifest",
        )
        pilot_keys = _unique_keys(
            self.pilot.keys(), manifest_name="pilot manifest",
        )
        if len(episode_keys) != EXPECTED_EPISODES:
            raise PilotManifestError(
                f"execution manifest has {len(episode_keys)} Episodes; "
                f"expected {EXPECTED_EPISODES}"
            )
        if len(pilot_keys) != EXPECTED_EPISODES:
            raise PilotManifestError(
                f"pilot manifest has {len(pilot_keys)} Episodes; "
                f"expected {EXPECTED_EPISODES}"
            )
        episode_set = set(episode_keys)
        pilot_set = set(pilot_keys)
        missing = [key for key in pilot_keys if key not in episode_set]
        extra = [key for key in episode_keys if key not in pilot_set]
        if missing or extra:
            raise PilotManifestError(
                "execution manifest Episode set differs from the Pilot: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        order_matches = episode_keys == pilot_keys
        return {
            "match": True,
            "sets_equal": True,
            "count": len(episode_keys),
            "pilot_count": len(pilot_keys),
            "missing_count": 0,
            "extra_count": 0,
            "order_matches": order_matches,
            "order_relation": (
                "same_order" if order_matches else "same_set_different_order"
            ),
            "order_source": self.order_source,
            "execution_order": "episode_manifest.json order preserved",
            "pilot_order": "pilot_manifest.episodes order; metadata only",
            "first_key": episode_keys[0],
            "last_key": episode_keys[-1],
            "pilot_scene_counts": self.pilot.scene_counts(),
            "episode_scene_counts": self.episode.scene_counts(),
            "execution_scene_counts": self.scene_counts(),
            "pilot_identity_digest": order_digest(self.pilot.rows),
            "execution_identity_digest": self.order_digest(),
        }

    def audit(self) -> dict[str, Any]:
        aligned = self.canonical_sha256 == self.episode.canonical_sha256
        return {
            "order_source": self.order_source,
            "digest_kind": (
                "canonical JSON SHA256 used by pilot_complete.json"
                "['manifest_sha256'], computed after reordering into the "
                "historical episode_manifest.json order; not a file SHA256"
            ),
            "count": len(self.rows),
            "pilot_count": len(self.pilot.rows),
            "canonical_sha256": self.canonical_sha256,
            "historical_episode_manifest_canonical_sha256": (
                self.episode.canonical_sha256
            ),
            "aligned": aligned,
            "pilot_file_sha256": self.pilot.file_sha256,
            "episode_file_sha256": self.episode.file_sha256,
            "order_matches_pilot": self.keys() == self.pilot.keys(),
            "pilot_identity_digest": order_digest(self.pilot.rows),
            "execution_identity_digest": self.order_digest(),
            "scene_counts": self.scene_counts(),
            "strata_counts": self.strata_counts(),
            "episodes": [row.audit() for row in self.rows],
        }


def _unique_keys(
    keys: Iterable[tuple[str, str]], *, manifest_name: str,
) -> tuple[tuple[str, str], ...]:
    values = tuple(keys)
    seen: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []
    for key in values:
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise PilotManifestError(
            f"{manifest_name} has duplicate Episode identities: {duplicates[:5]}"
        )
    return values


def build_execution_manifest(
    episode: EpisodeManifest, pilot: PilotManifest,
) -> ExecutionManifest:
    """Reconcile the historical manifest with the Pilot by Episode identity.

    The Episode **set** must match exactly; order is deliberately ignored.  The
    historical ``episode_manifest.json`` order is the execution order, and
    every Pilot metadata field is looked up by
    ``(scene_basename, episode_id)`` rather than by list position.
    """
    if episode.split != "val_unseen":
        raise PilotManifestError(
            f"episode manifest split must be val_unseen; got {episode.split!r}"
        )
    if pilot.dataset_split not in (None, "val_unseen"):
        raise PilotManifestError(
            f"pilot manifest split must be val_unseen; got {pilot.dataset_split!r}"
        )
    if (
        len(episode.rows) != EXPECTED_EPISODES
        or episode.expected_count != EXPECTED_EPISODES
    ):
        raise PilotManifestError(
            f"episode manifest has {len(episode.rows)} Episodes; expected "
            f"{EXPECTED_EPISODES}"
        )
    if (
        len(pilot.rows) != EXPECTED_EPISODES
        or pilot.expected_count != EXPECTED_EPISODES
    ):
        raise PilotManifestError(
            f"pilot manifest has {len(pilot.rows)} Episodes; expected "
            f"{EXPECTED_EPISODES}"
        )
    episode_keys = _unique_keys(episode.keys(), manifest_name="episode manifest")
    pilot_keys = _unique_keys(pilot.keys(), manifest_name="pilot manifest")
    episode_set = set(episode_keys)
    pilot_set = set(pilot_keys)
    missing = [key for key in pilot_keys if key not in episode_set]
    extra = [key for key in episode_keys if key not in pilot_set]
    if missing or extra:
        raise PilotManifestError(
            "episode manifest Episode set differs from the Pilot: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    episode_scenes = episode.scene_counts()
    pilot_scenes = pilot.scene_counts()
    if len(episode_scenes) != EXPECTED_SCENES:
        raise PilotManifestError(
            f"episode manifest must cover {EXPECTED_SCENES} scenes; "
            f"found {len(episode_scenes)}"
        )
    if set(episode_scenes) != set(pilot_scenes):
        raise PilotManifestError(
            "episode manifest scene coverage differs from the Pilot: "
            f"episode={sorted(episode_scenes)}, pilot={sorted(pilot_scenes)}"
        )
    if any(count != EXPECTED_PER_SCENE for count in episode_scenes.values()):
        raise PilotManifestError(
            f"episode manifest must contain {EXPECTED_PER_SCENE} Episodes per "
            f"scene; found {episode_scenes}"
        )
    if pilot_scenes != episode_scenes:
        raise PilotManifestError(
            "episode manifest per-scene counts differ from the Pilot: "
            f"episode={episode_scenes}, pilot={pilot_scenes}"
        )

    pilot_rows_by_key = pilot.index()
    pilot_positions = {
        (row.scene_basename, row.episode_id): position
        for position, row in enumerate(pilot.rows)
    }
    rows: list[ExecutionRow] = []
    for execution_index, episode_row in enumerate(episode.rows):
        key = (episode_row.scene_basename, episode_row.episode_id)
        pilot_row = pilot_rows_by_key[key]
        rows.append(ExecutionRow(
            scene_id=episode_row.scene_id,
            episode_id=episode_row.episode_id,
            distance_group=pilot_row.distance_group,
            instruction=pilot_row.instruction,
            pilot_scene_id=pilot_row.scene_id,
            pilot_index=pilot_positions[key],
            execution_index=execution_index,
        ))

    canonical_payload = {
        "split": episode.split,
        "expected_count": len(rows),
        "episodes": [row.canonical_entry() for row in rows],
    }
    digest = canonical_episode_manifest_sha256(canonical_payload)
    if digest != episode.canonical_sha256:
        raise PilotManifestError(
            "generated execution manifest canonical digest does not align "
            f"with the historical episode_manifest.json: {digest} != "
            f"{episode.canonical_sha256}"
        )
    return ExecutionManifest(
        episode=episode,
        pilot=pilot,
        rows=tuple(rows),
        canonical_sha256=digest,
    )


def compare_episode_manifest_to_pilot(
    episode: EpisodeManifest, pilot: PilotManifest,
) -> dict[str, Any]:
    """Require the same 88-Episode set, ignoring order, and report both orders.

    A different order is not an error: the historical ``episode_manifest.json``
    order becomes the execution order.  Missing, extra, duplicate, wrong-count
    and wrong-scene Episodes still fail closed.
    """
    return build_execution_manifest(episode, pilot).comparison()


def write_execution_manifest(
    path: str | Path,
    execution: ExecutionManifest,
    *,
    expected_canonical_sha256: str | None = None,
) -> dict[str, Any]:
    """Write the execution manifest once and verify its canonical digest.

    The artifact stores the historical scene path spelling and the historical
    order, so ``canonical_episode_manifest_sha256`` equals the completion-marker
    digest.  Pilot metadata is extra JSON and is stripped by the same
    canonicalisation, so it cannot perturb the digest.
    """
    expected = (
        execution.episode.canonical_sha256
        if expected_canonical_sha256 is None else expected_canonical_sha256
    )
    if not isinstance(expected, str) or len(expected) != 64:
        raise PilotManifestError(
            "expected execution manifest digest must be a 64-character digest"
        )
    if execution.canonical_sha256 != execution.episode.canonical_sha256:
        raise PilotManifestError(
            "execution manifest is not aligned with the historical "
            "episode_manifest.json canonical digest"
        )
    payload = execution.manifest_payload()
    generated_digest = canonical_episode_manifest_sha256(payload)
    if generated_digest != expected:
        raise PilotManifestError(
            "execution manifest canonical SHA256 mismatch: "
            f"expected {expected}, got {generated_digest}"
        )
    target = Path(path)
    for source_name, source_path in (
        ("historical episode_manifest.json", execution.episode.path),
        ("pilot_manifest.json", execution.pilot.path),
    ):
        if target.resolve(strict=False) == source_path.resolve(strict=False):
            raise PilotManifestError(
                f"refusing to overwrite the {source_name}: {target}"
            )
    created = False
    if target.exists():
        existing = load_episode_manifest(
            target, expected_canonical_sha256=expected,
        )
        if existing.keys() != execution.keys():
            raise PilotManifestError(
                f"existing execution manifest has different Episode order/IDs: {target}"
            )
        try:
            existing_payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PilotManifestError(
                f"existing execution manifest is not valid JSON: {target}"
            ) from exc
        if existing_payload != payload:
            raise PilotManifestError(
                "existing execution manifest metadata differs from this run: "
                f"{target}; use a fresh output directory or remove the stale file"
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        try:
            with target.open("x", encoding="utf-8") as stream:
                stream.write(text)
            created = True
        except FileExistsError:
            # A concurrent shard created the same deterministic artifact.
            pass
        existing = load_episode_manifest(
            target, expected_canonical_sha256=expected,
        )
        if existing.keys() != execution.keys():
            raise PilotManifestError(
                "concurrent execution manifest has different Episode "
                f"order/IDs: {target}"
            )
    return {
        "path": str(target),
        "created": created,
        "verified": True,
        "aligned": True,
        "digest_kind": (
            "canonical JSON SHA256 used by pilot_complete.json"
            "['manifest_sha256']; not a file SHA256"
        ),
        "canonical_sha256": existing.canonical_sha256,
        "historical_episode_manifest_canonical_sha256": (
            execution.episode.canonical_sha256
        ),
        "expected_canonical_sha256": expected,
        "file_sha256": existing.file_sha256,
        "count": len(existing.rows),
        "order_source": execution.order_source,
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
    execution_manifest: ExecutionManifest | None = None,
) -> list[Any]:
    """Return the actual dataset Episodes in execution order.

    With an ``ExecutionManifest`` (the formal path), the historical
    ``episode_manifest.json`` order is used.  Without one, Pilot order is kept
    for the Pilot-only preflight.  This is the only selection path used by both
    experiment groups.  It never samples, sorts or substitutes.  A missing key
    is a hard failure.
    """
    if execution_manifest is not None:
        if execution_manifest.pilot.keys() != manifest.keys():
            raise PilotManifestError(
                "execution manifest was built from a different Pilot manifest"
            )
        if (
            execution_manifest.canonical_sha256
            != execution_manifest.episode.canonical_sha256
        ):
            raise PilotManifestError(
                "execution manifest does not reproduce the historical "
                "episode_manifest.json canonical digest"
            )
        order_rows: Iterable[PilotRow | ExecutionRow] = execution_manifest.rows
    else:
        order_rows = manifest.rows
    index = _dataset_index(dataset_episodes)
    selected: list[Any] = []
    missing: list[tuple[str, str]] = []
    for row in order_rows:
        key = (row.scene_basename, row.episode_id)
        episode = index.get(key)
        if episode is None:
            missing.append(key)
            continue
        if check_instruction and row.instruction is not None:
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
    rows: Iterable[PilotRow | ExecutionRow],
    *, instructions: Iterable[str] | None = None,
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
    execution_manifest: ExecutionManifest | None = None,
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

    When the episode manifest is supplied, its Episode set is compared to the
    Pilot as a set and its order becomes the execution order.  The generated
    execution manifest is reported with the same canonical digest as the
    historical file.
    """
    if execution_manifest is not None and episode_manifest is None:
        raise PilotManifestError(
            "an execution manifest requires the historical episode_manifest.json"
        )
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

    execution: ExecutionManifest | None = None
    if episode_manifest is not None:
        episode = load_episode_manifest(
            episode_manifest,
            expected_canonical_sha256=expected_episode_manifest_sha256,
        )
        if execution_manifest is None:
            execution = build_execution_manifest(episode, manifest)
        else:
            execution = execution_manifest
            if (
                execution.episode.file_sha256 != episode.file_sha256
                or execution.episode.canonical_sha256 != episode.canonical_sha256
                or execution.pilot.file_sha256 != manifest.file_sha256
            ):
                raise PilotManifestError(
                    "supplied execution manifest does not belong to the "
                    "verified pilot/episode manifests"
                )
        comparison = execution.comparison()
        episode_report = episode.audit()
        episode_report.update({
            "expected_canonical_sha256": expected_episode_manifest_sha256,
            "verified": True,
            "pilot_comparison": comparison,
        })
        report["episode_manifest"] = episode_report
        report["execution_manifest"] = execution.audit()
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
        report["execution_manifest"] = None

    if dataset_episodes is not None:
        selected = select_pilot_episodes(
            dataset_episodes, manifest, execution_manifest=execution,
        )
        instructions = [episode_instruction(episode) for episode in selected]
        order_rows: Iterable[PilotRow | ExecutionRow] = (
            execution.rows if execution is not None else manifest.rows
        )
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
            order_rows, instructions=instructions,
        )
        report["dataset_order_source"] = (
            execution.order_source if execution is not None
            else "pilot_manifest.json"
        )
        report["instructions_available"] = manifest.instructions_available
    return report

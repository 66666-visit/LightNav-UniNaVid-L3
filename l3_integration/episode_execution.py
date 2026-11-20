"""Identity-based execution tracking for the frozen 88-Episode Pilot.

Habitat's ``EpisodeIterator`` is allowed to shuffle, group by scene and cycle
Episodes.  Its reset order is therefore *not* the manifest order.  Execution
must be validated against the Episode set, never against a list position:

* every actual Episode must belong to the assigned execution set;
* an Episode identity may be executed at most once;
* every assigned Episode must eventually be executed.

Pilot metadata is looked up by ``(scene_basename, episode_id)``, the same
identity used to match the frozen manifest to the dataset.  ``scene_id`` is
normalized to its file basename because the Pilot, the historical
``episode_manifest.json`` and Habitat's ``SCENES_DIR`` may spell the same HM3D
scene with different prefixes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .pilot_manifest import PilotManifestError, scene_basename

#: Stable Episode identity: ``(scene file basename, episode_id)``.
EpisodeKey = tuple[str, str]


class EpisodeExecutionError(RuntimeError):
    """The environment did not execute the expected Episode set exactly once."""


def episode_identity(episode: Any) -> EpisodeKey:
    """Return the identity of an actual Habitat Episode object.

    ``scene_id`` is normalized to the HM3D scene basename, matching
    :func:`l3_integration.pilot_manifest.select_pilot_episodes`.  This is the
    only place where an Episode object is converted to a manifest key.
    """
    scene = getattr(episode, "scene_id", None)
    episode_id = getattr(episode, "episode_id", None)
    if scene is None or episode_id is None:
        raise EpisodeExecutionError(
            "Habitat Episode lacks scene_id or episode_id; its identity cannot "
            "be matched to the frozen manifest"
        )
    try:
        scene_name = scene_basename(scene)
    except PilotManifestError as exc:
        raise EpisodeExecutionError(
            f"Habitat Episode has an invalid scene_id: {scene!r}"
        ) from exc
    if not str(episode_id):
        raise EpisodeExecutionError("Habitat Episode has an empty episode_id")
    return scene_name, str(episode_id)


@dataclass(frozen=True)
class EpisodeMetadata:
    """Frozen-manifest metadata attached to one Episode identity."""

    manifest_order_index: int | None
    pilot_manifest_index: int | None
    distance_group: str | None
    instruction: str | None = None
    manifest_scene_id: str | None = None
    manifest_episode_id: str | None = None


def _row_key(row: Any) -> EpisodeKey:
    key = getattr(row, "key", None)
    if isinstance(key, tuple) and len(key) == 2:
        return str(key[0]), str(key[1])
    scene_name = getattr(row, "scene_basename", None)
    if not scene_name:
        scene_name = scene_basename(row.scene_id)
    return str(scene_name), str(row.episode_id)


def metadata_from_execution_manifest(
    execution_manifest: Any,
) -> dict[EpisodeKey, EpisodeMetadata]:
    """Index the execution manifest by Episode identity, never by position.

    ``manifest_order_index`` is the row's position in the historical
    ``episode_manifest.json`` (the marker order).  ``pilot_manifest_index`` is
    the row's position in ``pilot_manifest.json``.  Both are diagnostics; the
    key is what associates them with a Habitat Episode.
    """
    metadata: dict[EpisodeKey, EpisodeMetadata] = {}
    for row in execution_manifest.rows:
        key = _row_key(row)
        if key in metadata:
            raise EpisodeExecutionError(
                f"execution manifest has duplicate Episode identity: {key}"
            )
        metadata[key] = EpisodeMetadata(
            manifest_order_index=int(row.execution_index),
            pilot_manifest_index=int(row.pilot_index),
            distance_group=row.distance_group,
            instruction=row.instruction,
            manifest_scene_id=str(row.scene_id),
            manifest_episode_id=str(row.episode_id),
        )
    return metadata


def metadata_from_pilot_manifest(
    pilot_manifest: Any,
) -> dict[EpisodeKey, EpisodeMetadata]:
    """Index Pilot rows by identity for a Pilot-order (non-historical) run."""
    metadata: dict[EpisodeKey, EpisodeMetadata] = {}
    for index, row in enumerate(pilot_manifest.rows):
        key = _row_key(row)
        if key in metadata:
            raise EpisodeExecutionError(
                f"pilot manifest has duplicate Episode identity: {key}"
            )
        metadata[key] = EpisodeMetadata(
            manifest_order_index=index,
            pilot_manifest_index=index,
            distance_group=getattr(row, "distance_group", None),
            instruction=getattr(row, "instruction", None),
            manifest_scene_id=getattr(row, "scene_id", None),
            manifest_episode_id=str(row.episode_id),
        )
    return metadata


def _format_key(key: EpisodeKey) -> str:
    return f"{key[0]}/{key[1]}"


def _format_keys(keys: Iterable[EpisodeKey], limit: int = 5) -> str:
    values = list(keys)
    text = ", ".join(_format_key(key) for key in values[:limit])
    if len(values) > limit:
        text += f" (+{len(values) - limit} more)"
    return text or "<none>"


class EpisodeExecutionTracker:
    """Track the assigned Episode set exactly once, independent of order."""

    def __init__(
        self,
        expected_episodes: Iterable[Any],
        metadata_by_key: Mapping[EpisodeKey, EpisodeMetadata] | None = None,
        *,
        require_full_metadata: bool = False,
        context: str = "the expected execution set",
    ) -> None:
        expected_keys: list[EpisodeKey] = []
        for episode in expected_episodes:
            expected_keys.append(episode_identity(episode))

        seen_expected: set[EpisodeKey] = set()
        duplicates: list[EpisodeKey] = []
        for key in expected_keys:
            if key in seen_expected and key not in duplicates:
                duplicates.append(key)
            seen_expected.add(key)
        if duplicates:
            raise EpisodeExecutionError(
                f"{context} contains duplicate Episode identities: "
                f"{_format_keys(duplicates)}"
            )

        self.context = context
        self._expected_keys = tuple(expected_keys)
        self._expected_set = frozenset(expected_keys)
        self._metadata = (
            dict(metadata_by_key) if metadata_by_key is not None else None
        )
        if self._metadata is not None:
            missing_metadata = [
                key for key in self._expected_keys if key not in self._metadata
            ]
            if missing_metadata:
                raise EpisodeExecutionError(
                    f"{context} contains Episodes with no frozen-manifest "
                    f"metadata: {_format_keys(missing_metadata)}"
                )
            if require_full_metadata:
                extra = [
                    key for key in self._metadata
                    if key not in self._expected_set
                ]
                if extra:
                    raise EpisodeExecutionError(
                        f"{context} does not cover the complete frozen "
                        f"manifest; unassigned Episodes include "
                        f"{_format_keys(extra)}"
                    )

        self._seen: list[EpisodeKey] = []
        self._seen_set: set[EpisodeKey] = set()
        self._first_index: dict[EpisodeKey, int] = {}
        self._records: list[
            tuple[int, EpisodeKey, EpisodeMetadata | None]
        ] = []

    @property
    def expected_count(self) -> int:
        return len(self._expected_keys)

    @property
    def completed_count(self) -> int:
        return len(self._seen)

    @property
    def expected_keys(self) -> tuple[EpisodeKey, ...]:
        return self._expected_keys

    def missing_keys(self) -> tuple[EpisodeKey, ...]:
        return tuple(
            key for key in self._expected_keys if key not in self._seen_set
        )

    def _missing_suffix(self) -> str:
        missing = self.missing_keys()
        if not missing:
            return ""
        return f"; still missing {_format_keys(missing)}"

    def accept(self, episode: Any) -> tuple[EpisodeKey, EpisodeMetadata | None]:
        """Validate one actual Episode and return its key and metadata."""
        key = episode_identity(episode)
        if key not in self._expected_set:
            if self._metadata is not None and key in self._metadata:
                detail = (
                    "it belongs to the frozen manifest but not to this "
                    "shard's assigned execution set"
                )
            elif self._metadata is not None:
                detail = "it is not one of the frozen manifest Episodes"
            else:
                detail = "it is not one of the Episodes assigned to this run"
            raise EpisodeExecutionError(
                f"Habitat returned Episode {_format_key(key)} outside "
                f"{self.context}: {detail}{self._missing_suffix()}"
            )
        if key in self._seen_set:
            first = self._first_index[key]
            raise EpisodeExecutionError(
                f"Habitat returned duplicate Episode {_format_key(key)} "
                f"(first executed at habitat index {first}); each Episode "
                "must be executed exactly once"
                f"{self._missing_suffix()}"
            )

        position = len(self._seen)
        metadata = None if self._metadata is None else self._metadata[key]
        self._seen.append(key)
        self._seen_set.add(key)
        self._first_index[key] = position
        self._records.append((position, key, metadata))
        return key, metadata

    def assert_complete(self) -> None:
        """Fail unless every assigned Episode identity was executed exactly once."""
        missing = self.missing_keys()
        if missing:
            raise EpisodeExecutionError(
                f"Habitat completed {len(self._seen)}/{len(self._expected_keys)} "
                f"Episodes in {self.context}; missing "
                f"{_format_keys(missing)}"
            )

    def coverage(self) -> dict[str, Any]:
        """JSON-safe execution summary; diagnostics only, never an association."""
        return {
            "context": self.context,
            "expected_count": len(self._expected_keys),
            "completed_count": len(self._seen),
            "missing_count": len(self.missing_keys()),
            "duplicate_count": 0,
            "out_of_set_count": 0,
            "complete": not self.missing_keys(),
            "habitat_order": [
                {
                    "habitat_execution_index": position,
                    "scene_basename": key[0],
                    "episode_id": key[1],
                    "manifest_order_index": (
                        metadata.manifest_order_index
                        if metadata is not None else None
                    ),
                    "pilot_manifest_index": (
                        metadata.pilot_manifest_index
                        if metadata is not None else None
                    ),
                    "distance_group": (
                        metadata.distance_group
                        if metadata is not None else None
                    ),
                }
                for position, key, metadata in self._records
            ],
        }


__all__ = [
    "EpisodeExecutionError",
    "EpisodeExecutionTracker",
    "EpisodeKey",
    "EpisodeMetadata",
    "episode_identity",
    "metadata_from_execution_manifest",
    "metadata_from_pilot_manifest",
]

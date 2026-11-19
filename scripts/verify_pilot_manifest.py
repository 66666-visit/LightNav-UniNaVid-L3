#!/usr/bin/env python3
"""Verify the fixed 88-Episode manifests without regenerating them.

Three distinct digests are checked separately:

* ``--expected-sha256`` is the **raw file SHA256** of
  ``pilot_manifest.json`` (``cf1755...``).
* ``--parent-manifest`` / ``--expected-parent-sha256`` check the raw file
  SHA256 of the older sampling manifest declared *inside* the Pilot.
* ``--episode-manifest`` / ``--expected-episode-manifest-sha256`` check the
  canonical JSON digest (``30260e...``) used by the historical
  ``pilot_complete.json`` marker.  This is **not** a file SHA256.

Usage:
  python scripts/verify_pilot_manifest.py --manifest /path/pilot_manifest.json
  python scripts/verify_pilot_manifest.py --manifest ... \
      --episode-manifest /path/episode_manifest.json \
      --data-path /data/R2R_VLNCE_v1-3_preprocessed
  python scripts/verify_pilot_manifest.py \
      --scan-root /root/autodl-tmp/LightNav-0-codex/output \
      --require-scan-match

This command only reads files.  It never samples, rewrites or substitutes an
Episode.  A missing manifest is reported as a blocker.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from l3_integration.pilot_manifest import (  # noqa: E402
    EXPECTED_EPISODE_MANIFEST_SHA256,
    EXPECTED_PILOT_MANIFEST_SHA256,
    PilotManifestError,
    audit_manifest,
    load_completion_marker,
    scan_episode_manifests,
)


def _resolve_dataset(data_path: str) -> Path:
    target = Path(data_path).expanduser()
    if target.is_file() and target.name == "val_unseen.json.gz":
        return target
    candidates = [
        target / "val_unseen" / "val_unseen.json.gz",
        target / "R2R_VLNCE_v1-3_preprocessed" / "val_unseen" / "val_unseen.json.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"val_unseen.json.gz not found under {target}; provide --data-path"
    )


def _dataset_episodes(path: Path, scenes_dir: str) -> list[SimpleNamespace]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    episodes = []
    for row in data.get("episodes", []):
        raw_scene = str(row.get("scene_id", ""))
        if scenes_dir:
            prefix = "data/scene_datasets/"
            if raw_scene.startswith(prefix):
                raw_scene = raw_scene[len(prefix):]
            scene = os.path.join(scenes_dir, raw_scene)
        else:
            scene = raw_scene
        instruction = row.get("instruction") or {}
        text = instruction.get("instruction_text") or instruction.get("text") or ""
        episodes.append(SimpleNamespace(
            scene_id=scene,
            episode_id=str(row.get("episode_id")),
            instruction=SimpleNamespace(instruction_text=str(text)),
        ))
    if not episodes:
        raise ValueError(f"dataset has no episodes: {path}")
    return episodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="")
    parser.add_argument(
        "--expected-sha256", default=EXPECTED_PILOT_MANIFEST_SHA256,
        help="raw file SHA256 of pilot_manifest.json",
    )
    parser.add_argument("--episode-manifest", default="")
    parser.add_argument(
        "--expected-episode-manifest-sha256",
        default=EXPECTED_EPISODE_MANIFEST_SHA256,
        help="canonical completion-marker digest of episode_manifest.json",
    )
    parser.add_argument("--parent-manifest", default="")
    parser.add_argument("--expected-parent-sha256", default="")
    parser.add_argument(
        "--scan-root", default="",
        help="recursively recompute the canonical digest of every episode_manifest.json",
    )
    parser.add_argument(
        "--completion-marker", default="",
        help="optional pilot_complete.json to verify expected_count and marker digest",
    )
    parser.add_argument(
        "--require-scan-match", action="store_true",
        help="exit nonzero when --scan-root finds no matching canonical digest",
    )
    parser.add_argument(
        "--require-episode-manifest", action="store_true",
        help=(
            "fail closed unless --episode-manifest is supplied; the formal "
            "launcher sets this so the historical marker digest is verified"
        ),
    )
    parser.add_argument("--data-path", default="")
    parser.add_argument(
        "--scenes-dir", default=os.environ.get("SCENES_DIR", ""),
        help="same SCENES_DIR used by the Habitat dataset loader",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    if not args.manifest and not args.scan_root and not args.completion_marker:
        parser.error("provide --manifest, --scan-root and/or --completion-marker")
    if args.manifest and not args.expected_sha256:
        parser.error(
            "--expected-sha256 must not be empty; never bypass the frozen "
            "Pilot file digest"
        )
    if args.episode_manifest and not args.expected_episode_manifest_sha256:
        parser.error(
            "--expected-episode-manifest-sha256 must not be empty; never "
            "bypass the historical completion-marker digest"
        )
    if args.episode_manifest and not args.manifest:
        parser.error(
            "--episode-manifest requires --manifest for the strict "
            "88-Episode set comparison"
        )
    if args.require_episode_manifest and not args.episode_manifest:
        print(
            "BLOCKED: --require-episode-manifest was set but "
            "--episode-manifest was not supplied; the historical completion-"
            "marker digest cannot be verified from the Pilot file alone",
            file=sys.stderr,
        )
        return 1
    if args.expected_parent_sha256 and not args.parent_manifest:
        parser.error("--expected-parent-sha256 requires --parent-manifest")

    report = None
    scan_report = None
    try:
        if args.scan_root:
            scanned = scan_episode_manifests(
                args.scan_root,
                expected_canonical_sha256=(
                    args.expected_episode_manifest_sha256
                    or EXPECTED_EPISODE_MANIFEST_SHA256
                ),
            )
            matches = [entry for entry in scanned if entry["matches_expected"]]
            scan_report = {
                "root": str(Path(args.scan_root)),
                "expected_canonical_sha256": (
                    args.expected_episode_manifest_sha256
                    or EXPECTED_EPISODE_MANIFEST_SHA256
                ),
                "file_count": len(scanned),
                "match_count": len(matches),
                "matches": matches,
                "note": (
                    "canonical_sha256 is sha256(json.dumps(validate_manifest(...), "
                    "sort_keys=True)); comparing sha256sum of the file will not "
                    "match the historical marker digest"
                ),
                "entries": scanned,
            }

        if args.manifest:
            episodes = None
            dataset_path = None
            if args.data_path:
                dataset_path = _resolve_dataset(args.data_path)
                episodes = _dataset_episodes(dataset_path, args.scenes_dir)
            report = audit_manifest(
                args.manifest,
                expected_sha256=args.expected_sha256,
                dataset_episodes=episodes,
                episode_manifest=args.episode_manifest or None,
                expected_episode_manifest_sha256=(
                    args.expected_episode_manifest_sha256 or None
                ),
                parent_manifest=args.parent_manifest or None,
                expected_parent_manifest_sha256=(
                    args.expected_parent_sha256 or None
                ),
            )
            if dataset_path is not None:
                report["dataset_path"] = str(dataset_path)
                report["scenes_dir"] = args.scenes_dir
            report["expected_pilot_file_sha256"] = args.expected_sha256
            report["expected_episode_manifest_canonical_sha256"] = (
                args.expected_episode_manifest_sha256
            )
        else:
            report = {}
        if args.completion_marker:
            report["completion_marker"] = load_completion_marker(
                args.completion_marker,
                expected_sha256=(
                    args.expected_episode_manifest_sha256
                    or EXPECTED_EPISODE_MANIFEST_SHA256
                ),
            )
    except (FileNotFoundError, PilotManifestError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1

    if scan_report is not None:
        report["episode_manifest_scan"] = scan_report
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    print(payload, end="")

    if report.get("file_sha256"):
        counts = report.get("scene_counts", {})
        print(
            f"verified {report.get('expected_count')} Episodes across "
            f"{len(counts)} scenes; pilot_file_sha256="
            f"{report.get('file_sha256')}",
            file=sys.stderr,
        )
        episode = report.get("episode_manifest", {})
        if episode.get("verified"):
            comparison = episode.get("pilot_comparison", {})
            print(
                "episode_manifest canonical_sha256="
                f"{episode.get('canonical_sha256')} (file SHA256="
                f"{episode.get('file_sha256')}); set_match="
                f"{comparison.get('sets_equal')} order_matches_pilot="
                f"{comparison.get('order_matches')} execution_order="
                f"{comparison.get('order_source')}",
                file=sys.stderr,
            )
        else:
            print(
                "episode_manifest canonical digest NOT verified: "
                f"{episode.get('reason')}",
                file=sys.stderr,
            )
        parent = report.get("parent_manifest", {})
        print(
            "parent_manifest declared_sha256="
            f"{parent.get('declared_sha256_in_pilot')} "
            f"(verified={parent.get('verified')})",
            file=sys.stderr,
        )
    if report.get("completion_marker"):
        marker = report["completion_marker"]
        print(
            "completion marker verified: expected_count="
            f"{marker['expected_count']} manifest_sha256="
            f"{marker['manifest_sha256']}",
            file=sys.stderr,
        )
    if scan_report is not None:
        print(
            f"scanned {scan_report['file_count']} episode_manifest.json files; "
            f"{scan_report['match_count']} match the canonical digest "
            f"{scan_report['expected_canonical_sha256']}",
            file=sys.stderr,
        )
    if args.require_scan_match and scan_report is not None:
        if scan_report["match_count"] == 0:
            print(
                "BLOCKED: no episode_manifest.json matches the historical "
                "completion-marker digest",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

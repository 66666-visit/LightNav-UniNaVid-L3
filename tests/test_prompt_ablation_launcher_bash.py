"""Real Bash regressions for the eight-group formal launcher.

These tests execute ``scripts/run_prompt_ablation.sh`` in a subprocess.  They do
not import the launcher into Python and they do not re-implement its group
resolution.  The only stand-in is the heavyweight Python preflight (Habitat /
torch / ultralytics and the frozen manifest files), which cannot exist on this
CPU-only host; every group-selection branch still runs in the real Bash script.

The root-user bug was caused by Bash's special ``GROUPS`` array: assignments to
it are ignored, so ``GROUP=all`` expanded to the literal id ``0``.  The tests
therefore set ``GROUPS=0`` in the environment and require the launcher to print
all eight experiment names anyway.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from l1_integration.memory_backend import EIGHT_ABLATION_GROUPS

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_prompt_ablation.sh"

_STUB = """\
#!{python}
import os
import subprocess
import sys

args = sys.argv[1:]
if args and args[0] == "-":
    data = sys.stdin.buffer.read()
    if b"import habitat, torch, ultralytics" in data:
        print("habitat 0.0.0")
        print("torch 0.0.0 cuda True")
        print("ultralytics 0.0.0")
        sys.exit(0)
    completed = subprocess.run(
        [sys.executable, "-", *args[1:]], input=data,
    )
    sys.exit(completed.returncode)
if args and args[0].endswith("verify_pilot_manifest.py"):
    print("stub manifest verification")
    sys.exit(0)
if args and args[0] == "run.py":
    if "--result-path" in args:
        result_path = args[args.index("--result-path") + 1]
        os.makedirs(result_path, exist_ok=True)
        with open(os.path.join(result_path, "results.jsonl"), "w") as stream:
            stream.write('{{"partial": true}}\\n')
    sys.exit(7)
print("unexpected stub python invocation: %r" % (args,), file=sys.stderr)
sys.exit(91)
"""


def _fake_environment(tmp_path: Path) -> dict[str, str]:
    model = tmp_path / "model"
    place = tmp_path / "place"
    model.mkdir()
    place.mkdir()
    l1_model = tmp_path / "yolo11n.pt"
    seg_model = tmp_path / "yolo11n-seg.pt"
    pilot = tmp_path / "pilot_manifest.json"
    episode = tmp_path / "episode_manifest.json"
    for path in (l1_model, seg_model, pilot, episode):
        path.write_text("{}\n", encoding="utf-8")
    stub = tmp_path / "python_stub"
    stub.write_text(_STUB.format(python=sys.executable), encoding="utf-8")
    stub.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": str(stub),
        "MODEL_PATH": str(model),
        "PLACE_MODEL": str(place),
        "L1_MODEL": str(l1_model),
        "SEG_MODEL": str(seg_model),
        "PILOT_MANIFEST": str(pilot),
        "EPISODE_MANIFEST": str(episode),
        "OUTPUT_ROOT": str(tmp_path / "out"),
        "CONFIRM": "NO",
        "NO_FORCE_STOP": "1",
    })
    return env


def _run(env: dict[str, str], **updates: str) -> subprocess.CompletedProcess:
    merged = dict(env)
    merged.update(updates)
    return subprocess.run(
        ["bash", str(SCRIPT)], cwd=str(ROOT), env=merged,
        text=True, capture_output=True, check=False,
    )


def test_launcher_has_valid_bash_syntax() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)], cwd=str(ROOT),
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_root_like_special_groups_array_still_expands_all_eight(tmp_path) -> None:
    env = _fake_environment(tmp_path)
    completed = _run(
        env, GROUP="all", RESOLVE_GROUPS_ONLY="1", GROUPS="0",
    )
    assert completed.returncode == 0, completed.stderr
    selected = [line for line in completed.stdout.splitlines() if line.strip()]
    assert selected == list(EIGHT_ABLATION_GROUPS)
    assert "0" not in selected


def test_single_group_selection_is_exact(tmp_path) -> None:
    env = _fake_environment(tmp_path)
    completed = _run(
        env, GROUP="l1_l3_joint", RESOLVE_GROUPS_ONLY="1", GROUPS="0",
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["l1_l3_joint"]


def test_invalid_group_fails_before_any_model_or_result(tmp_path) -> None:
    env = _fake_environment(tmp_path)
    completed = _run(
        env, GROUP="0", RESOLVE_GROUPS_ONLY="1", GROUPS="0",
        CONFIRM="YES",
    )
    assert completed.returncode != 0
    assert "GROUP must be 'all' or one of:" in completed.stderr
    assert not (tmp_path / "out").exists()


def test_dry_run_prints_all_eight_commands_and_creates_no_output(tmp_path) -> None:
    env = _fake_environment(tmp_path)
    completed = _run(env, GROUP="all", CONFIRM="NO", GROUPS="0")
    assert completed.returncode == 0, completed.stderr
    commands = [
        line for line in completed.stdout.splitlines() if " run.py " in line
    ]
    assert len(commands) == len(EIGHT_ABLATION_GROUPS)
    for group in EIGHT_ABLATION_GROUPS:
        assert f"--experiment-group {group} " in "\n".join(commands)
    assert "DRY RUN ONLY" in completed.stdout
    assert not (tmp_path / "out").exists()
    assert "failed" not in {path.name for path in tmp_path.iterdir()}


def test_failed_formal_group_is_moved_out_of_the_result_path(tmp_path) -> None:
    env = _fake_environment(tmp_path)
    env["CONFIRM"] = "YES"
    completed = _run(env, GROUP="l1_original_only", CONFIRM="YES", GROUPS="0")
    assert completed.returncode != 0
    assert "partial output moved to" in completed.stderr
    out = tmp_path / "out"
    assert not (out / "l1_original_only").exists()
    failed = list((out / "failed").glob("l1_original_only.*"))
    assert len(failed) == 1
    assert (failed[0] / "results.jsonl").is_file()
    assert not (out / "l1_original_only" / "results.jsonl").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

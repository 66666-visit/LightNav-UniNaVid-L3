"""Real-import regressions for shipped modules and the Python 3.8 shim.

These tests import the repository's own packages.  They never install stub
modules, so a reference to a file that is missing from the checkout fails here
exactly as it fails in the AutoDL smoke run.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("l1_memory", "l3_memory", "l1_integration", "l3_integration")
L3_PACKAGES = ("l3_memory", "l3_integration")
L1_PACKAGES = ("l1_memory", "l1_integration")


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _shipped_modules() -> list[str]:
    return [
        _module_name(path)
        for package in PACKAGES
        for path in sorted((ROOT / package).glob("*.py"))
    ]


SHIPPED_MODULES = _shipped_modules()


def _resolved_imports(path: Path):
    """Yield ``(lineno, module)`` for every import, including relative ones."""
    package_parts = path.parent.relative_to(ROOT).parts
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - node.level + 1
                if keep <= 0:
                    continue
                module = ".".join(package_parts[:keep])
                if node.module:
                    module = f"{module}.{node.module}"
            else:
                module = node.module or ""
            if module:
                yield node.lineno, module


def _local_imports(path: Path):
    for lineno, module in _resolved_imports(path):
        if module.split(".")[0] in PACKAGES:
            yield lineno, module


def _module_exists(module: str) -> bool:
    relative = Path(*module.split("."))
    return (ROOT / relative).with_suffix(".py").is_file() or (
        ROOT / relative / "__init__.py"
    ).is_file()


@pytest.mark.parametrize("module_name", SHIPPED_MODULES)
def test_every_shipped_module_imports_for_real(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__name__ == module_name


@pytest.mark.parametrize("module_name", SHIPPED_MODULES)
def test_every_local_import_target_exists(module_name: str) -> None:
    path = ROOT.joinpath(*module_name.split(".")).with_suffix(".py")
    if not path.is_file():
        path = ROOT.joinpath(*module_name.split(".")) / "__init__.py"
    missing = [
        f"{path.relative_to(ROOT)}:{lineno}: {target}"
        for lineno, target in _local_imports(path)
        if not _module_exists(target)
    ]
    assert not missing, "missing local module imports:\n" + "\n".join(missing)


def test_l3_packages_do_not_import_l1_packages() -> None:
    offenders = []
    for package in L3_PACKAGES:
        for path in sorted((ROOT / package).glob("*.py")):
            for lineno, module in _resolved_imports(path):
                if module.split(".")[0] in L1_PACKAGES:
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{lineno}: {module}"
                    )
    assert not offenders, "L3-only code imports L1 code:\n" + "\n".join(offenders)


def test_l3_compat_removeprefix_matches_l1_and_python39() -> None:
    from l1_memory import compat as l1_compat
    from l3_memory import compat as l3_compat

    cases = (
        ("Observed chair", "Observed ", "chair"),
        ("chair", "Observed ", "chair"),
        ("Observed ", "Observed ", ""),
        ("", "Observed ", ""),
        ("xx", "x", "x"),
        ("éclair", "é", "clair"),
    )
    for value, prefix, expected in cases:
        assert l3_compat.removeprefix(value, prefix) == expected
        assert l3_compat.removeprefix(value, prefix) == l1_compat.removeprefix(
            value, prefix
        )

    with pytest.raises(TypeError):
        l3_compat.removeprefix(None, "Observed ")
    with pytest.raises(TypeError):
        l3_compat.removeprefix(b"Observed chair", "Observed ")


def test_spatial_graph_adapter_uses_the_real_compat_module() -> None:
    from l3_memory import compat as l3_compat
    from l3_memory import spatial_graph_adapter

    assert spatial_graph_adapter.removeprefix is l3_compat.removeprefix


def test_provenance_marks_compat_as_a_local_helper_not_vendored_code() -> None:
    provenance = json.loads(
        (ROOT / "l3_memory" / "provenance.json").read_text(encoding="utf-8")
    )
    vendored = {entry["name"] for entry in provenance["files"]}
    helpers = {
        entry["name"]: entry
        for entry in provenance["compatibility_helpers"]
    }

    assert "compat.py" not in vendored
    assert "compat.py" in helpers
    helper = helpers["compat.py"]
    assert helper["kind"] == "project_local_python38_helper"
    assert helper["source"] is None
    assert helper["path"] == "l3_memory/compat.py"
    digest = hashlib.sha256(
        (ROOT / helper["path"]).read_bytes()
    ).hexdigest()
    assert helper["sha256"] == digest

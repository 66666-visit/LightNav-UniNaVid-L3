"""Frozen A/B/C experiment group definitions for the L3 comparison.

A is the untouched ``7602b25`` L3 Organized baseline and is executed from the
protected worktree.  B adds only the SBAdir evidence/source pairing to the same
L3 facts.  C adds the L1 RGB memory on top of B.  The three flags below are the
only behavioural switches; every group keeps the official Uni-NaVid prompt,
visual token order, action decoding and generation parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

#: Experiment group values passed to ``run.py --experiment-group``.
A_GROUP = "uninavid_l3_only"
B_GROUP = "uninavid_l3_source_background"
C_GROUP = "uninavid_l3_l1_sbadir"

#: Result directory names produced by the launcher and report.
A_NAME = "l3_organized_baseline_7602b25"
B_NAME = "l3_sbadir_source_background_only"
C_NAME = "l3_l1_sbadir"

ALL_GROUPS: Tuple[str, ...] = (A_GROUP, B_GROUP, C_GROUP)


@dataclass(frozen=True)
class GroupSpec:
    """One frozen experimental condition."""

    group: str
    name: str
    uses_l3: bool
    source_background: bool
    uses_l1: bool
    worktree: str = "dev"

    @property
    def is_baseline_code(self) -> bool:
        return self.worktree == "baseline"


_SPECS: Dict[str, GroupSpec] = {
    A_GROUP: GroupSpec(
        group=A_GROUP, name=A_NAME, uses_l3=True,
        source_background=False, uses_l1=False, worktree="baseline",
    ),
    B_GROUP: GroupSpec(
        group=B_GROUP, name=B_NAME, uses_l3=True,
        source_background=True, uses_l1=False,
    ),
    C_GROUP: GroupSpec(
        group=C_GROUP, name=C_NAME, uses_l3=True,
        source_background=True, uses_l1=True,
    ),
}

_BY_NAME: Dict[str, GroupSpec] = {spec.name: spec for spec in _SPECS.values()}


def spec(group: str) -> GroupSpec:
    """Return the frozen spec; unknown groups fail closed."""
    try:
        return _SPECS[group]
    except KeyError:
        raise ValueError(
            "unknown experiment group {!r}; expected one of {}".format(
                group, ", ".join(ALL_GROUPS),
            )
        ) from None


def spec_by_name(name: str) -> GroupSpec:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(
            "unknown experiment group name {!r}; expected one of {}".format(
                name, ", ".join(spec.name for spec in _SPECS.values()),
            )
        ) from None


def source_background_enabled(group: str) -> bool:
    return spec(group).source_background


def l1_enabled(group: str) -> bool:
    return spec(group).uses_l1


def validate_model_paths(
    group: str, *, l3_seg_model: str, l1_det_model: str, l1_place_model: str,
) -> None:
    """Fail closed unless the supplied model paths match the frozen group.

    B, and every group without L1, must never receive an L1 weight path; C
    must receive both.  This is checked before any backend is constructed.
    """
    if group == "uninavid_baseline":
        if l3_seg_model:
            raise ValueError("baseline group must not receive an L3 segmentation model")
        if l1_det_model or l1_place_model:
            raise ValueError("baseline group must not receive L1 model paths")
        return
    frozen = spec(group)
    if frozen.uses_l3 and not l3_seg_model:
        raise ValueError("L3 groups require --l3-seg-model")
    if frozen.uses_l1 and (not l1_det_model or not l1_place_model):
        raise ValueError(
            "the L1 SBAdir group requires --l1-det-model and --l1-place-model"
        )
    if not frozen.uses_l1 and (l1_det_model or l1_place_model):
        raise ValueError(
            "group {} must not receive L1 model paths".format(group)
        )

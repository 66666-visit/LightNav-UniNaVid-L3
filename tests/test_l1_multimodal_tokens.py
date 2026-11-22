"""Visual-token accounting for Uni-NaVid's navigation multimodal prompt."""

from __future__ import annotations

import pytest

from l1_integration.multimodal_tokens import (
    NAVIGATION_FRAME_TOKENS,
    navigation_nav_size,
    navigation_visual_token_accounting,
)
from l1_integration.token_budget import build_token_budget


def _accounting(**kwargs):
    defaults = dict(
        cached_frames=0,
        cached_long_entries=0,
        new_frames=1,
        nav_size=4,
    )
    defaults.update(kwargs)
    return navigation_visual_token_accounting(**defaults)


def test_nav_size_comes_from_the_real_compress_type():
    assert navigation_nav_size("grid:2") == 4
    assert navigation_nav_size("grid:4") == 16
    assert navigation_nav_size("mean") == 1
    with pytest.raises(ValueError, match="compress_type"):
        navigation_nav_size(None)
    with pytest.raises(ValueError, match="Unsupported"):
        navigation_nav_size("grid:8")


def test_first_frames_are_exact_because_online_merge_does_not_apply():
    one = _accounting(cached_frames=0, new_frames=1, nav_size=4)
    assert one["exact"] is True
    assert one["visual_groups"] == 1
    assert one["result_visual_tokens"] == 4
    assert one["separator_tokens"] == 0
    assert one["navigation_tokens"] == NAVIGATION_FRAME_TOKENS
    # 4 video + 64 nav tokens replace the one <image> token.
    assert one["visual_tokens_total"] == 68
    assert one["visual_tokens_net"] == 67

    full = _accounting(cached_frames=63, new_frames=1, nav_size=4)
    assert full["frames_after"] == 64
    assert full["exact"] is True
    assert full["visual_groups"] == 64
    assert full["result_visual_tokens"] == 64 * 4
    assert full["separator_tokens"] == 63
    assert full["visual_tokens_net"] == 64 * 4 + 63 + 64 - 1


def test_online_merge_upper_bound_after_64_frames():
    first_crossing = _accounting(cached_frames=64, new_frames=1, nav_size=4)
    assert first_crossing["frames_after"] == 65
    assert first_crossing["merge_slots"] == 1
    assert first_crossing["long_entries_upper"] == 1
    assert first_crossing["exact"] is False
    assert first_crossing["visual_groups"] == 65
    assert first_crossing["result_visual_tokens"] == 1 + 64 * 4
    assert first_crossing["visual_tokens_net"] == (
        1 + 64 * 4 + 64 + 64 - 1
    )

    two_new = _accounting(cached_frames=64, new_frames=2, nav_size=4)
    assert two_new["merge_slots"] == 2
    assert two_new["long_entries_upper"] == 2
    assert two_new["visual_groups"] == 66
    assert two_new["separator_tokens"] == 65

    with_long_cache = _accounting(
        cached_frames=128, cached_long_entries=64,
        new_frames=1, nav_size=4,
    )
    assert with_long_cache["long_entries_upper"] == 65
    assert with_long_cache["visual_groups"] == 129
    assert with_long_cache["visual_tokens_net"] == (
        65 + 64 * 4 + 129 + 64 - 2
    )


def test_visual_tokens_are_part_of_the_base_prompt_budget():
    text_tokens = 300
    visual_tokens = 384  # 65 frames, grid:2, no-merge upper bound
    budget = build_token_budget(
        context_limit_tokens=2048,
        base_prompt_tokens=text_tokens + visual_tokens,
        generation_reserve_tokens=1024,
        counter=lambda text: len(text.split()),
        context_limit_source="cpu-test",
        text_prompt_tokens=text_tokens,
        visual_tokens=visual_tokens,
        visual_accounting={"exact": False},
    )
    assert budget.available_tokens == 2048 - text_tokens - visual_tokens - 1024 - 1
    assert budget.available_tokens == 339
    diagnostics = budget.diagnostics()
    assert diagnostics["text_prompt_tokens"] == text_tokens
    assert diagnostics["visual_tokens"] == visual_tokens
    assert diagnostics["base_prompt_tokens"] == text_tokens + visual_tokens


def test_visual_budget_must_be_nonnegative():
    with pytest.raises(ValueError):
        _accounting(new_frames=0)
    with pytest.raises(ValueError):
        _accounting(cached_frames=-1)
    with pytest.raises(ValueError):
        _accounting(nav_size=0)

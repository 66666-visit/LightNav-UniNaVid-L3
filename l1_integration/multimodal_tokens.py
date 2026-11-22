"""Exact structural accounting for Uni-NaVid's expanded navigation prompt.

``prepare_inputs_labels_for_multimodal`` replaces the single ``<image>``
placeholder with:

* the projected video features,
* one separator token between adjacent online-merge groups,
* 64 navigation tokens for the current frame.

For navigation samples the video features come from
``UniNaVIDMetaForCausalLM.online_process_tensor``.  That routine keeps the
last 64 frames at ``nav_size`` tokens each and compresses older frames into
single-token long-term entries.  The number of long-term entries depends on
the visual cosine similarity at the moment of encoding, so it cannot be known
before the vision tower runs.  This module computes the exact *upper bound*
from the actual cache lengths: every newly processed frame that can create a
long-term entry is assumed to create one.  The bound is exact when the total
frame count is <= 64; above 64 it over-reserves at most
``2 * min(new_frames, frames_after - 64)`` tokens.  It never uses a
``64 * frames`` shortcut.

The returned ``visual_tokens_net`` is the number of multimodal tokens added
after replacing the one ``<image>`` placeholder:

    final_multimodal_tokens = text_tokens + visual_tokens_net

where ``text_tokens`` is the tokenizer count of the official Uni-NaVid prompt
(including the placeholder and all conversation/special tokens).
"""

from __future__ import annotations

from typing import Any

#: ``online_process_tensor`` default in Uni-NaVid.
ONLINE_MERGE_LENGTH_THRESHOLD = 64

#: The current-frame navigation branch always appends 64 tokens.
NAVIGATION_FRAME_TOKENS = 64

#: ``compress_type`` -> tokens retained per short-term frame.
NAV_SIZE_BY_COMPRESS_TYPE = {
    "grid:2": 4,
    "grid:4": 16,
    "mean": 1,
}


def navigation_nav_size(compress_type: Any) -> int:
    """Return the per-frame visual token count from the model config."""
    if not isinstance(compress_type, str) or not compress_type:
        raise ValueError("Uni-NaVid compress_type is unavailable")
    if compress_type in NAV_SIZE_BY_COMPRESS_TYPE:
        return NAV_SIZE_BY_COMPRESS_TYPE[compress_type]
    if "grid:" in compress_type:
        suffix = compress_type.split("grid:")[-1]
        try:
            grid_size = int(suffix)
        except (TypeError, ValueError):
            raise ValueError("Uni-NaVid compress_type has invalid grid size")
        if grid_size == 2:
            return 4
        if grid_size == 4:
            return 16
    raise ValueError(
        "Unsupported Uni-NaVid compress_type: {!r}".format(compress_type)
    )


def navigation_visual_token_accounting(
    *,
    cached_frames: int,
    cached_long_entries: int,
    new_frames: int,
    nav_size: int,
    length_threshold: int = ONLINE_MERGE_LENGTH_THRESHOLD,
    navigation_tokens: int = NAVIGATION_FRAME_TOKENS,
) -> dict:
    """Return the online-merge structure and its visual token upper bound.

    ``cached_frames`` is ``feat_cache.shape[0]`` before the current inference;
    ``cached_long_entries`` is ``long_feat_cache.shape[0]``; ``new_frames`` is
    the number of RGB frames passed to ``process_images`` for this inference.
    """
    for name, value in (
        ("cached_frames", cached_frames),
        ("cached_long_entries", cached_long_entries),
        ("new_frames", new_frames),
        ("nav_size", nav_size),
        ("length_threshold", length_threshold),
        ("navigation_tokens", navigation_tokens),
    ):
        if type(value) is not int or value < 0:
            raise ValueError("{} must be a nonnegative integer".format(name))
    if new_frames < 1:
        raise ValueError("new_frames must be positive")
    if nav_size < 1 or length_threshold < 1 or navigation_tokens < 1:
        raise ValueError("visual token sizes must be positive")

    frames_after = cached_frames + new_frames
    if frames_after <= length_threshold:
        # No long-term cache is used yet; online_process_tensor returns every
        # frame at nav_size tokens.
        long_entries_upper = 0
        result_visual_tokens = frames_after * nav_size
        groups = frames_after
        merge_slots = 0
        exact = True
    else:
        merge_slots = min(new_frames, frames_after - length_threshold)
        # Each valid merge slot can append at most one long-term entry.  The
        # only case not fixed before the vision tower is whether a new entry
        # merges into the previous long-term entry; this is the upper bound.
        long_entries_upper = min(
            max(cached_long_entries, 0) + merge_slots,
            frames_after - length_threshold,
        )
        result_visual_tokens = (
            long_entries_upper + length_threshold * nav_size
        )
        groups = long_entries_upper + length_threshold
        exact = False

    separator_tokens = max(0, groups - 1)
    visual_tokens_total = (
        result_visual_tokens + separator_tokens + navigation_tokens
    )
    # The one <image> token is removed from input_ids, so subtract it from the
    # replacement block to obtain the net added multimodal tokens.
    visual_tokens_net = visual_tokens_total - 1
    return {
        "source": "online_merge_upper_bound_from_actual_cache",
        "exact": exact,
        "length_threshold": length_threshold,
        "navigation_tokens": navigation_tokens,
        "nav_size": nav_size,
        "cached_frames": cached_frames,
        "cached_long_entries": cached_long_entries,
        "new_frames": new_frames,
        "frames_after": frames_after,
        "merge_slots": merge_slots,
        "long_entries_upper": long_entries_upper,
        "visual_groups": groups,
        "result_visual_tokens": result_visual_tokens,
        "separator_tokens": separator_tokens,
        "visual_tokens_total": visual_tokens_total,
        "visual_tokens_net": visual_tokens_net,
    }

"""Compare the pre-generation visual bound with the real online-merge loop.

The reference function below is a structural copy of
``UniNaVIDMetaForCausalLM.online_process_tensor``.  The reference package's
top-level import pulls in the full vision stack, so the merge/append loop is
kept here as the oracle; the productive accounting and the oracle agree on
frame/group counts, which is what the budget needs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from l1_integration.multimodal_tokens import (  # noqa: E402
    navigation_visual_token_accounting,
)


def _online_process_tensor(
    feat_cache, long_feat_cache, new_frames, nav_size,
    length_threshold=64, similarity_threshold=0.985,
):
    """Structural copy of online_process_tensor; fixed merge weight 1.

    The merge/append decision and the resulting group counts are identical to
    the reference; only the long-term mean uses weight 1 here.
    """
    k, m, c = feat_cache.shape
    assert m % nav_size == 0
    result_list = []
    if k <= length_threshold:
        result_list = [nav_size] * k
        return feat_cache.reshape(-1, c), result_list

    cos = torch.nn.CosineSimilarity(dim=0)
    for frame_index in range(new_frames - 1, -1, -1):
        if k - length_threshold - frame_index - 1 < 0:
            continue
        oldest_short_mem_token = feat_cache[
            k - length_threshold - frame_index - 1, :, :
        ].mean(dim=0)
        if long_feat_cache is not None:
            assert long_feat_cache[-1, :].shape == oldest_short_mem_token.shape
            similarity = cos(long_feat_cache[-1, :], oldest_short_mem_token)
            if similarity > similarity_threshold:
                new_mean = (
                    long_feat_cache[-1, :] * 1 + oldest_short_mem_token
                ) / 2
                long_feat_cache[-1] = new_mean
            else:
                long_feat_cache = torch.cat(
                    [long_feat_cache, oldest_short_mem_token[None, :]], dim=0,
                )
        else:
            long_feat_cache = oldest_short_mem_token[None, :]

    result_list = [1] * long_feat_cache.shape[0] + [nav_size] * length_threshold
    result_tensor = torch.cat([
        long_feat_cache,
        feat_cache[k - length_threshold:].reshape(-1, c),
    ], dim=0)
    assert result_tensor.shape[0] == sum(result_list)
    return result_tensor, result_list


def _actual_net(result_lengths):
    groups = len(result_lengths)
    result_tokens = sum(result_lengths)
    return result_tokens + max(0, groups - 1) + 64 - 1


def _run_reference(k_prev, new_frames, nav_size, *, merge):
    c = 8
    if k_prev:
        feat_prev = torch.randn(k_prev, nav_size, c)
    else:
        feat_prev = None
    long_prev = None
    if k_prev > 64:
        long_prev = torch.randn(k_prev - 64, c)
    new_feat = torch.randn(new_frames, nav_size, c)
    feat = new_feat if feat_prev is None else torch.cat([feat_prev, new_feat], dim=0)
    long_copy = None if long_prev is None else long_prev.clone()
    threshold = -1.0 if merge else 2.0
    _tensor, lengths = _online_process_tensor(
        feat, long_copy, new_frames, nav_size, similarity_threshold=threshold,
    )
    return _actual_net(lengths), lengths


@pytest.mark.parametrize("nav_size", [4, 16])
def test_first_frames_are_exact(nav_size):
    actual, lengths = _run_reference(0, 1, nav_size, merge=False)
    accounting = navigation_visual_token_accounting(
        cached_frames=0, cached_long_entries=0,
        new_frames=1, nav_size=nav_size,
    )
    assert actual == accounting["visual_tokens_net"]
    assert len(lengths) == 1


@pytest.mark.parametrize("new_frames", [1, 2])
def test_upper_bound_matches_no_merge_and_covers_merge(new_frames):
    k_prev = 64
    no_merge_actual, _ = _run_reference(
        k_prev, new_frames, 4, merge=False,
    )
    accounting = navigation_visual_token_accounting(
        cached_frames=k_prev, cached_long_entries=0,
        new_frames=new_frames, nav_size=4,
    )
    assert no_merge_actual == accounting["visual_tokens_net"]
    assert accounting["exact"] is False

    merged_actual, _ = _run_reference(
        k_prev, new_frames, 4, merge=True,
    )
    assert merged_actual <= accounting["visual_tokens_net"]


def test_upper_bound_covers_existing_long_cache():
    no_merge_actual, _ = _run_reference(128, 1, 4, merge=False)
    accounting = navigation_visual_token_accounting(
        cached_frames=128, cached_long_entries=64,
        new_frames=1, nav_size=4,
    )
    assert no_merge_actual == accounting["visual_tokens_net"]

    merged_actual, _ = _run_reference(128, 1, 4, merge=True)
    assert merged_actual <= accounting["visual_tokens_net"]

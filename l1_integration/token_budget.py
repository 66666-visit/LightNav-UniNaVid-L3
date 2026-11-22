"""Model-derived token budget for the C group's original L1 card.

This is the LightNav prompt-ablation budget mechanism adapted to the
Uni-NaVid adapter.  The selection rule and the budget formula are copied from
``src/lightnav/memory/token_budget.py`` (the module used by the original
``l1_original_only`` prompt mode):

    available = context_limit - base_prompt_tokens - max_new_tokens - 1

Only the measurement source changes: Uni-NaVid has no LightNav
``data_processor``, so the caller measures the exact official Uni-NaVid prompt
with the checkpoint tokenizer and passes the result here.  The one-token safety
margin and the whole-line selection are unchanged.

``UNBOUNDED_CHAR_LIMIT`` is the original LightNav sentinel that disables the
Local Graph module's *internal* 320-character selection cap while a measured
token budget owns the prompt capacity.  It is not a prompt budget, is never
reported as one, and is never used on the fallback path; the original
320-character budget remains the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Callable, Optional

from l3_memory.schemas import MemoryContext

#: Sentinel for "no internal character cap" when the real limit is measured in
#: tokens.  Copied from LightNav; it is not a prompt budget.
UNBOUNDED_CHAR_LIMIT = 2 ** 31 - 1

#: One token of boundary reserve, copied from LightNav.  It is part of the
#: measured budget, not a character allowance.
BOUNDARY_RESERVE_TOKENS = 1

#: The original Local Graph module budget, also the explicit fallback.
ORIGINAL_CHAR_FALLBACK = 320


@dataclass(frozen=True)
class MemoryTokenBudget:
    """The measured per-step token allowance for the L1 memory block."""

    available_tokens: int
    context_limit_tokens: int
    base_prompt_tokens: int
    generation_reserve_tokens: int
    safety_margin_tokens: int
    context_limit_source: str
    counter: Callable[[str], int]
    unavailable_reason: Optional[str] = None
    text_prompt_tokens: Optional[int] = None
    visual_tokens: int = 0
    visual_accounting: Optional[dict] = None

    @property
    def measured(self) -> bool:
        """True when the model-derived budget was actually measured."""
        return self.unavailable_reason is None

    @property
    def available(self) -> bool:
        """True when the measurement succeeded and there is token headroom."""
        return self.measured and self.available_tokens > 0

    def diagnostics(self) -> dict:
        return {
            "budget_source": "token_aware_context",
            "context_limit_tokens": self.context_limit_tokens,
            "base_prompt_tokens": self.base_prompt_tokens,
            "generation_reserve_tokens": self.generation_reserve_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "available_memory_tokens": self.available_tokens,
            "context_limit_source": self.context_limit_source,
            "token_budget_unavailable_reason": self.unavailable_reason,
            "budget_fallback_reason": self.unavailable_reason,
            "text_prompt_tokens": self.text_prompt_tokens,
            "visual_tokens": self.visual_tokens,
            "visual_token_accounting": self.visual_accounting,
        }


@dataclass(frozen=True)
class TokenBudgetFailure:
    """Why the model-derived budget could not be measured."""

    reason: str
    context_limit_tokens: Optional[int] = None
    base_prompt_tokens: Optional[int] = None
    generation_reserve_tokens: Optional[int] = None
    text_prompt_tokens: Optional[int] = None
    visual_tokens: Optional[int] = None
    visual_accounting: Optional[dict] = None

    def diagnostics(self) -> dict:
        return {
            "budget_source": "char_fallback_320",
            "context_limit_tokens": self.context_limit_tokens,
            "base_prompt_tokens": self.base_prompt_tokens,
            "generation_reserve_tokens": self.generation_reserve_tokens,
            "safety_margin_tokens": BOUNDARY_RESERVE_TOKENS,
            "available_memory_tokens": None,
            "token_budget_unavailable_reason": self.reason,
            "budget_fallback_reason": self.reason,
            "text_prompt_tokens": self.text_prompt_tokens,
            "visual_tokens": self.visual_tokens,
            "visual_token_accounting": self.visual_accounting,
        }


@dataclass
class TokenSelection:
    """The whole-line context selected against a token capacity."""

    context: Optional[MemoryContext]
    block_tokens: int
    omitted_lines: int
    omitted_items: int


def build_token_budget(
    *,
    context_limit_tokens: Any,
    base_prompt_tokens: Any,
    generation_reserve_tokens: Any,
    counter: Callable[[str], int],
    context_limit_source: str,
    safety_margin_tokens: int = BOUNDARY_RESERVE_TOKENS,
    text_prompt_tokens: Optional[int] = None,
    visual_tokens: int = 0,
    visual_accounting: Optional[dict] = None,
) -> Any:
    """Apply LightNav's budget formula to a Uni-NaVid measurement.

    The formula is exactly::

        context_limit - base_prompt_tokens - max_new_tokens - 1

    ``counter`` must measure the exact prompt-token delta of a candidate L1
    block, including the escaping that will be inserted.  If any input cannot
    be measured, a :class:`TokenBudgetFailure` is returned so the caller can
    use the original 320-character builder and report the reason.
    """
    if not isinstance(context_limit_source, str) or not context_limit_source:
        return TokenBudgetFailure(reason="context_limit_source_unavailable")
    if type(context_limit_tokens) is not int or context_limit_tokens <= 0:
        return TokenBudgetFailure(
            reason="context_limit_unavailable: Uni-NaVid context_len is not a "
                   "positive integer",
            base_prompt_tokens=(
                base_prompt_tokens
                if type(base_prompt_tokens) is int and base_prompt_tokens >= 0
                else None
            ),
        )
    if type(base_prompt_tokens) is not int or base_prompt_tokens < 0:
        return TokenBudgetFailure(
            reason="base_prompt_measurement_unavailable: the exact Uni-NaVid "
                   "base prompt could not be tokenized",
            context_limit_tokens=context_limit_tokens,
        )
    if type(generation_reserve_tokens) is not int or generation_reserve_tokens < 0:
        return TokenBudgetFailure(
            reason="generation_reserve_unavailable: max_new_tokens is unknown",
            context_limit_tokens=context_limit_tokens,
            base_prompt_tokens=base_prompt_tokens,
        )
    if not callable(counter):
        return TokenBudgetFailure(
            reason="tokenizer_encode_unavailable: no checkpoint-tokenizer "
                   "counter for the L1 block",
            context_limit_tokens=context_limit_tokens,
            base_prompt_tokens=base_prompt_tokens,
            generation_reserve_tokens=generation_reserve_tokens,
            text_prompt_tokens=text_prompt_tokens,
            visual_tokens=visual_tokens,
            visual_accounting=visual_accounting,
        )
    if type(visual_tokens) is not int or visual_tokens < 0:
        return TokenBudgetFailure(
            reason="visual_token_accounting_invalid: the multimodal visual "
                   "token allowance must be a nonnegative integer",
            context_limit_tokens=context_limit_tokens,
            base_prompt_tokens=base_prompt_tokens,
            generation_reserve_tokens=generation_reserve_tokens,
            text_prompt_tokens=text_prompt_tokens,
            visual_tokens=visual_tokens if type(visual_tokens) is int else None,
            visual_accounting=visual_accounting,
        )
    if type(safety_margin_tokens) is not int or safety_margin_tokens < 0:
        return TokenBudgetFailure(
            reason="safety_margin_invalid: the boundary reserve must be a "
                   "nonnegative integer",
            context_limit_tokens=context_limit_tokens,
            base_prompt_tokens=base_prompt_tokens,
            generation_reserve_tokens=generation_reserve_tokens,
        )
    available = (
        context_limit_tokens
        - base_prompt_tokens
        - generation_reserve_tokens
        - safety_margin_tokens
    )
    return MemoryTokenBudget(
        available_tokens=max(0, available),
        context_limit_tokens=context_limit_tokens,
        base_prompt_tokens=base_prompt_tokens,
        generation_reserve_tokens=generation_reserve_tokens,
        safety_margin_tokens=safety_margin_tokens,
        context_limit_source=context_limit_source,
        counter=counter,
        text_prompt_tokens=text_prompt_tokens,
        visual_tokens=visual_tokens,
        visual_accounting=visual_accounting,
    )


def select_context_lines(
    context: MemoryContext, *, capacity_tokens: int,
    count_tokens: Callable[[str], int],
) -> TokenSelection:
    """Keep the longest whole-line prefix that fits ``capacity_tokens``.

    This is LightNav's original selector.  Context lines are already
    priority-ordered by ``LocalGraphFusion``.  A line that does not fit ends the
    selection, so no half-line and no later lower-priority line can jump over
    it.  Items whose serialized text is not included are dropped with their
    line.
    """
    if capacity_tokens <= 0 or not context.text.strip():
        return TokenSelection(
            context=None,
            block_tokens=0,
            omitted_lines=len(context.text.splitlines()),
            omitted_items=len(context.layer_items),
        )
    lines = context.text.splitlines()
    included: list = []
    included_set: set = set()
    omitted = 0
    block_tokens = 0
    for index, line in enumerate(lines):
        candidate = "\n".join([*included, line])
        tokens = count_tokens(candidate)
        if tokens > capacity_tokens:
            omitted = len(lines) - index
            break
        included.append(line)
        included_set.add(escape(line, quote=False))
        block_tokens = tokens
    if not included:
        return TokenSelection(
            context=None,
            block_tokens=0,
            omitted_lines=len(lines),
            omitted_items=len(context.layer_items),
        )
    kept_items = [
        item for item in context.layer_items
        if item.metadata.get("serialized_text") in included_set
    ]
    omitted_items = len(context.layer_items) - len(kept_items)
    if context.layer_items and not kept_items:
        return TokenSelection(
            context=None,
            block_tokens=block_tokens,
            omitted_lines=omitted,
            omitted_items=omitted_items,
        )
    return TokenSelection(
        context=MemoryContext(text="\n".join(included), layer_items=kept_items),
        block_tokens=block_tokens,
        omitted_lines=omitted,
        omitted_items=omitted_items,
    )


def trim_context_for_overflow(
    context: MemoryContext, *, drop_count: int = 1,
) -> tuple:
    """Drop the lowest-priority item lines after an exact token-delta overflow.

    Copied from LightNav for callers that need the post-injection guard.  The C
    adapter selects against the exact prompt delta, so this is a defensive
    fallback rather than the primary path.
    """
    remaining_items = list(context.layer_items)
    lines = context.text.splitlines()
    dropped = 0
    for _ in range(drop_count):
        if not remaining_items:
            break
        victim = remaining_items[-1]
        dropped += 1
        remaining_items.pop()
        lines = [line for line in lines if line != victim.description]
    if not remaining_items:
        return MemoryContext(text="\n".join(lines), layer_items=[]), dropped
    return MemoryContext(text="\n".join(lines), layer_items=remaining_items), dropped

# -*- coding: utf-8 -*-
"""Utility helpers for visualizing Experience tokens in the terminal."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, TextIO

from torch import Tensor

from trinity.common.experience import Experience

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"


@dataclass(frozen=True)
class ExperienceTokenViewItem:
    index: int
    token_id: int
    token_text: str
    is_prompt: bool
    is_action: bool
    logprob: Optional[float]


@dataclass(frozen=True)
class ExperienceTokenView:
    prompt_text: str
    response_text: str
    prompt_length: int
    tokens: List[ExperienceTokenViewItem]
    prompt_tokens: List[ExperienceTokenViewItem]
    response_tokens: List[ExperienceTokenViewItem]


def _normalize_token_id(token_id: Any) -> int:
    if isinstance(token_id, Tensor):
        return int(token_id.item())
    return int(token_id)


def _render_token_text(token_text: str) -> str:
    rendered = (
        token_text.replace("\r\n", "↵")
        .replace("\n", "↵")
        .replace("\r", "↵")
        .replace("\t", "⇥")
        .replace(" ", "␠")
    )
    return rendered or "∅"


def _decode_token_texts(tokenizer: Any, token_ids: List[int]) -> List[str]:
    token_texts = []
    for token_id in token_ids:
        try:
            token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except TypeError:
            token_text = tokenizer.decode([token_id])
        if token_text == "" and hasattr(tokenizer, "convert_ids_to_tokens"):
            token_text = str(tokenizer.convert_ids_to_tokens([token_id])[0])
        token_texts.append(str(token_text))
    return token_texts


def _decode_token_ids(tokenizer: Any, token_ids: List[int]) -> str:
    return str(tokenizer.decode(token_ids))


def _build_full_action_mask(exp: Experience) -> List[bool]:
    if exp.tokens is None:
        raise ValueError("Experience tokens are required for visualization.")
    if exp.action_mask is None:
        raise ValueError("Experience action_mask is required for visualization.")

    token_count = len(exp.tokens)
    action_mask = exp.action_mask.tolist()
    if len(action_mask) == token_count:
        return [bool(mask) for mask in action_mask]
    if len(action_mask) == token_count - exp.prompt_length:
        return [False] * exp.prompt_length + [bool(mask) for mask in action_mask]
    raise ValueError(
        "action_mask length must match either len(tokens) or len(tokens) - prompt_length. "
        f"Got len(tokens)={token_count}, prompt_length={exp.prompt_length}, "
        f"len(action_mask)={len(action_mask)}."
    )


def _build_full_logprobs(exp: Experience) -> List[Optional[float]]:
    if exp.tokens is None:
        raise ValueError("Experience tokens are required for visualization.")
    token_count = len(exp.tokens)
    if exp.logprobs is None:
        return [None] * token_count

    logprobs = exp.logprobs.tolist()
    if len(logprobs) == token_count:
        return [float(logprob) for logprob in logprobs]

    response_length = token_count - exp.prompt_length
    if len(logprobs) == response_length:
        return [None] * exp.prompt_length + [float(logprob) for logprob in logprobs]

    raise ValueError(
        "logprobs length must match either len(tokens) or len(tokens) - prompt_length. "
        f"Got len(tokens)={token_count}, prompt_length={exp.prompt_length}, "
        f"len(logprobs)={len(logprobs)}."
    )


def build_experience_token_view(exp: Experience, tokenizer: Any) -> ExperienceTokenView:
    """Build a reusable token-level view for terminal and UI renderers."""
    if exp.tokens is None:
        raise ValueError("Experience tokens are required for visualization.")

    token_ids = [_normalize_token_id(token_id) for token_id in exp.tokens]
    token_texts = _decode_token_texts(tokenizer, token_ids)
    full_action_mask = _build_full_action_mask(exp)
    full_logprobs = _build_full_logprobs(exp)

    prompt_token_ids = token_ids[: exp.prompt_length]
    response_token_ids = token_ids[exp.prompt_length :]

    tokens = [
        ExperienceTokenViewItem(
            index=index,
            token_id=token_id,
            token_text=token_text,
            is_prompt=index < exp.prompt_length,
            is_action=is_action,
            logprob=logprob,
        )
        for index, (token_id, token_text, is_action, logprob) in enumerate(
            zip(token_ids, token_texts, full_action_mask, full_logprobs)
        )
    ]

    return ExperienceTokenView(
        prompt_text=_decode_token_ids(tokenizer, prompt_token_ids),
        response_text=_decode_token_ids(tokenizer, response_token_ids),
        prompt_length=exp.prompt_length,
        tokens=tokens,
        prompt_tokens=tokens[: exp.prompt_length],
        response_tokens=tokens[exp.prompt_length :],
    )


def format_experience_colored_tokens(
    exp: Experience,
    tokenizer: Any,
    *,
    tokens_per_line: int = 20,
) -> str:
    """Format all experience tokens as a colorized string for terminal display."""
    if tokens_per_line <= 0:
        raise ValueError("tokens_per_line must be greater than 0.")

    token_view = build_experience_token_view(exp, tokenizer)

    header = (
        f"{ANSI_BOLD}Experience Tokens [{exp.eid}]"
        f" | prompt_length={exp.prompt_length}{ANSI_RESET}"
    )
    rendered_tokens = []
    for token in token_view.tokens:
        color = ANSI_GREEN if token.is_action else ANSI_RED
        rendered_tokens.append(f"{color}{_render_token_text(token.token_text)}{ANSI_RESET}")

    lines = [header]
    for start in range(0, len(rendered_tokens), tokens_per_line):
        lines.append("  ".join(rendered_tokens[start : start + tokens_per_line]))
    return "\n".join(lines)


def print_experience_colored_tokens(
    exp: Experience,
    tokenizer: Any,
    *,
    file: Optional[TextIO] = None,
    tokens_per_line: int = 20,
) -> None:
    """Print all experience tokens to the terminal using action-mask-based colors."""
    print(
        format_experience_colored_tokens(
            exp,
            tokenizer,
            tokens_per_line=tokens_per_line,
        ),
        file=file,
    )

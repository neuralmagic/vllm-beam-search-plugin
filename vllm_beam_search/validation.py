"""Request validation for the Beam Search ``vllm_xargs`` extension."""

from __future__ import annotations

import math
from functools import wraps
from typing import Any

from vllm.exceptions import VLLMValidationError

_BEAM_ARGS = {
    "beam_width",
    "length_penalty",
    "no_repeat_ngram_size",
}
_PASSTHROUGH_ARGS = {"session_id"}


def validate_beam_xargs(extra_args: dict[str, Any] | None) -> None:
    if not extra_args or not (_BEAM_ARGS & extra_args.keys()):
        return

    unknown = extra_args.keys() - _BEAM_ARGS - _PASSTHROUGH_ARGS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise VLLMValidationError(
            f"Unknown beam search argument(s): {names}.",
            parameter="vllm_xargs",
        )

    width = extra_args.get("beam_width")
    if type(width) is not int or width <= 1:
        raise VLLMValidationError(
            "beam_width must be an integer greater than 1.",
            parameter="vllm_xargs.beam_width",
            value=width,
        )

    ngram_size = extra_args.get("no_repeat_ngram_size", 0)
    if type(ngram_size) is not int or ngram_size < 0:
        raise VLLMValidationError(
            "no_repeat_ngram_size must be a non-negative integer.",
            parameter="vllm_xargs.no_repeat_ngram_size",
            value=ngram_size,
        )

    length_penalty = extra_args.get("length_penalty", 1.0)
    if (
        isinstance(length_penalty, bool)
        or not isinstance(length_penalty, (int, float))
        or not math.isfinite(length_penalty)
    ):
        raise VLLMValidationError(
            "length_penalty must be a finite number.",
            parameter="vllm_xargs.length_penalty",
            value=length_penalty,
        )


def install_request_validation() -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    for request_type in (
        CompletionRequest,
        ChatCompletionRequest,
        ResponsesRequest,
    ):
        _wrap_sampling_params(request_type)


def _wrap_sampling_params(request_type: type[Any]) -> None:
    original = request_type.to_sampling_params
    if getattr(original, "_vllm_beam_validation", False):
        return

    @wraps(original)
    def validated(self: Any, *args: Any, **kwargs: Any) -> Any:
        validate_beam_xargs(self.vllm_xargs)
        return original(self, *args, **kwargs)

    validated._vllm_beam_validation = True  # type: ignore[attr-defined]
    request_type.to_sampling_params = validated

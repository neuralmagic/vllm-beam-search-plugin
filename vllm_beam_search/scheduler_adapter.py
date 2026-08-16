"""Install Beam admission points in the vLLM scheduler out of tree."""

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Callable
from functools import update_wrapper
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler

_RUNNING_TOKEN_BUDGET = """\
        num_new_tokens = min(
            num_new_tokens, token_budget, input_budget - draft_slots
        )"""
_RUNNING_TOKEN_BUDGET_WITH_HOOK = """\
        request_token_budget = self._get_request_token_budget(
            request, token_budget, input_budget - draft_slots
        )
        num_new_tokens = min(num_new_tokens, request_token_budget)"""
_WAITING_REQUEST = """\
            request = request_queue.peek_request()
            request_id = request.request_id"""
_WAITING_REQUEST_WITH_HOOK = """\
            request = request_queue.peek_request()
            request_id = request.request_id

            required_slots = self._get_num_required_running_slots(request)
            if num_running + required_slots > self.max_num_running_reqs:
                break"""
_WAITING_TOKEN_BUDGET = """\
                request_token_budget = min(
                    token_budget, input_budget - draft_slots
                )"""
_WAITING_TOKEN_BUDGET_COMPACT = (
    "                request_token_budget = min("
    "token_budget, input_budget - draft_slots)"
)
_WAITING_TOKEN_BUDGET_WITH_HOOK = """\
                request_token_budget = self._get_request_token_budget(
                    request, token_budget, input_budget - draft_slots
                )
                if request_token_budget == 0:
                    break"""


def _replace_once(source: str, old: str, new: str, name: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            "Installed vLLM scheduler is incompatible with the Beam plugin: "
            f"could not locate the {name} admission site."
        )
    return source.replace(old, new)


def patch_scheduler_source(source: str) -> str:
    source = _replace_once(
        source,
        _RUNNING_TOKEN_BUDGET,
        _RUNNING_TOKEN_BUDGET_WITH_HOOK,
        "running token-budget",
    )
    source = _replace_once(
        source,
        _WAITING_REQUEST,
        _WAITING_REQUEST_WITH_HOOK,
        "waiting sequence-budget",
    )
    waiting_budget = (
        _WAITING_TOKEN_BUDGET
        if _WAITING_TOKEN_BUDGET in source
        else _WAITING_TOKEN_BUDGET_COMPACT
    )
    return _replace_once(
        source,
        waiting_budget,
        _WAITING_TOKEN_BUDGET_WITH_HOOK,
        "waiting token-budget",
    )


def build_resource_atomic_schedule() -> Callable[..., Any]:
    required_hooks = (
        "_get_num_required_running_slots",
        "_get_request_token_budget",
    )
    if all(hasattr(Scheduler, name) for name in required_hooks):
        return Scheduler.schedule

    original = Scheduler.schedule
    source = patch_scheduler_source(
        textwrap.dedent(inspect.getsource(original))
    )
    namespace = dict(original.__globals__)
    exec(  # noqa: S102
        compile(source, "<vllm-beam-search scheduler>", "exec"), namespace
    )
    patched = namespace[original.__name__]
    return update_wrapper(patched, original)

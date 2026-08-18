"""Select a vendored vLLM scheduler with Beam resource admission."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

import vllm
from vllm.v1.core.sched.scheduler import Scheduler


def _select_vendored_schedule(
    version: str,
    scheduler_locals: tuple[str, ...],
) -> tuple[str, str]:
    has_input_budget = "input_budget" in scheduler_locals
    has_num_running = "num_running" in scheduler_locals

    if version == "0.24.0" and not has_num_running:
        return ".vendored_scheduler_v024", "schedule_v024"
    if version == "0.26.0" and has_num_running and not has_input_budget:
        return ".vendored_scheduler_v026", "schedule_v026"
    if version == "0.26.1rc1.dev682+g7aa248fcf" and has_input_budget:
        return ".vendored_scheduler_v0261", "schedule_v0261"

    raise RuntimeError(
        "Installed vLLM scheduler is incompatible with the Beam plugin: "
        f"unsupported scheduler for vLLM {version}."
    )


def build_resource_atomic_schedule() -> Callable[..., Any]:
    required_hooks = (
        "_get_num_required_running_slots",
        "_get_request_token_budget",
    )
    if all(hasattr(Scheduler, name) for name in required_hooks):
        return Scheduler.schedule

    module_name, function_name = _select_vendored_schedule(
        vllm.__version__,
        Scheduler.schedule.__code__.co_varnames,
    )
    module = import_module(module_name, package=__package__)
    return getattr(module, function_name)

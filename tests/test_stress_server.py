from __future__ import annotations

import pytest

from vllm_beam_search.stress_server import _is_server_cmdline


@pytest.mark.parametrize(
    "cmd",
    [
        ["vllm", "serve", "model", "--port", "8000"],
        ["python", "/venv/bin/vllm", "serve", "model", "--port", "8000"],
        ["python", "-m", "vllm.entrypoints.openai.api_server"],
    ],
)
def test_recognizes_vllm_server_commands(cmd: list[str]) -> None:
    assert _is_server_cmdline(cmd)


def test_does_not_treat_uv_wrapper_as_server() -> None:
    assert not _is_server_cmdline(
        ["uv", "run", "--", "vllm", "serve", "model", "--port", "8000"]
    )

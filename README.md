# vLLM Beam Search Plugin

MRV2 beam-search scheduler and sampler plugin for vLLM V1.

This package provides:

- `vllm_beam_search.scheduler.BeamSearchScheduler`
- plugin-local sequence, token, and KV admission for complete beam groups
- an MRV2 custom sampler wrapper installed through a plugin-local `ModelState`
  hook
- plugin-local runtime hooks for MRV2 worker history rewrites

The current production path targets MRV2 generate models with async scheduling.
The sampler hook is model-state generic; BART-family models still need the
companion `vllm-bart-plugin` for encoder-decoder model support.

The plugin does not require a vLLM fork or source patch. It carries explicit
plugin-local scheduler implementations for vLLM 0.24.0, 0.26.0, and the tested
0.26.1 development build. Startup fails closed on an unsupported scheduler.
Each vendored scheduler has an adjacent `.diff` recording its exact changes
from the hashed upstream `Scheduler.schedule`; the test suite verifies both.

The Git repository, installable Python distribution, and PyPI project are named
`vllm-beam-search-plugin`; the import package is `vllm_beam_search`.

For BART-family encoder-decoder serving, see
[`BART_BEAM_SEARCH.md`](BART_BEAM_SEARCH.md).

## Install

Install the published distribution from PyPI with an exact version pin:

```bash
uv pip install 'vllm-beam-search-plugin==0.1.1'
```

The plugin metadata constrains its tested NumPy, PyTorch, and Triton API ranges.
vLLM remains the owner of their accelerator-specific builds. In a prebuilt
RHAII 3.5 image, preserve the image runtime and install only the pure-Python
plugin:

```bash
uv pip install --no-deps \
  'vllm-beam-search-plugin==0.1.1'
```

The vLLM 0.24.0 path has been unit-, correctness-, concurrency-, and sustained
memory-tested. It selects `vendored_scheduler_v024.schedule_v024` directly; it
does not use `inspect.getsource`, source-text matching, or runtime `exec`.

For stress tooling:

```bash
uv pip install 'vllm-beam-search-plugin[stress]==0.1.1'
```

## Server

```bash
MODEL=${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}
SERVED_MODEL=${SERVED_MODEL:-llama3-8b}

CUDA_VISIBLE_DEVICES=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL}" \
  --dtype bfloat16 \
  --port 8005 \
  --scheduler-cls vllm_beam_search.scheduler.BeamSearchScheduler
```

## Request Shape

```json
{
  "model": "llama3-8b",
  "prompt": "Write a concise summary of why beam search is useful:",
  "max_tokens": 128,
  "temperature": 0,
  "add_special_tokens": false,
  "vllm_xargs": {
    "beam_width": 4,
    "no_repeat_ngram_size": 3
  }
}
```

## Validation

Run unit tests:

```bash
uv run --with pytest python -m pytest tests -q
```

Run sustained stress plus memory sampling against a running server:

```bash
vllm-beam-stress \
  --base-url http://localhost:8005 \
  --model llama3-8b \
  --rounds 100 \
  --requests-per-round 32 \
  --concurrency 64 \
  --abort-rounds 3
```

The stress tool writes CSV samples with request count, RSS, and GPU memory.

## Runtime Knobs

- `VLLM_BEAM_GROUP_STATE_CAPACITY` controls GPU beam-state pool capacity.
- `VLLM_BEAM_TRANSITION_BUFFER_SLOTS` controls async transition buffer slots.

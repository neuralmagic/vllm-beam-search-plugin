from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from vllm.exceptions import VLLMValidationError
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_beam_search.beam_state import BeamGroup
from vllm_beam_search.scheduler import (
    _RESOURCE_ATOMIC_SCHEDULE,
    BeamSearchScheduler,
    _BeamKVCacheManager,
)
from vllm_beam_search.scheduler_adapter import patch_scheduler_source
from vllm_beam_search.validation import validate_beam_xargs


@dataclass
class FakeBlock:
    block_id: int
    ref_cnt: int = 1


class FakeBlockPool:
    def touch(self, blocks: list[FakeBlock]) -> None:
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks: Iterable[FakeBlock]) -> None:
        for block in blocks:
            block.ref_cnt -= 1


class FakeManager:
    def __init__(self, blocks: list[FakeBlock]) -> None:
        self.req_to_blocks = {"dst": blocks}
        self.num_cached_block = {"dst": len(blocks)}


class FakeKVCacheManager:
    def __init__(self) -> None:
        self.enable_caching = False
        self.max_model_len = 128
        self.empty_kv_cache_blocks = SimpleNamespace(blocks=([],))
        self.single_type_manager = SimpleNamespace(
            req_to_blocks={"beam": [object()]},
            block_size=16,
        )
        self.coordinator = SimpleNamespace(
            single_type_managers=[self.single_type_manager],
            get_num_blocks_to_allocate=lambda **_kwargs: 2,
        )
        self.allocate_calls = 0
        self.last_allocate_kwargs = None

    def allocate_slots(self, *_args, **kwargs):
        self.allocate_calls += 1
        self.last_allocate_kwargs = kwargs
        return object()


def make_beam_request(num_computed_tokens: int):
    return SimpleNamespace(
        request_id="beam",
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=1,
        sampling_params=SimpleNamespace(extra_args={"_beam_group_id": "group"}),
    )


def make_budget_scheduler(status: RequestStatus = RequestStatus.WAITING):
    scheduler = BeamSearchScheduler.__new__(BeamSearchScheduler)
    children = [
        SimpleNamespace(
            request_id=f"beam:{index}",
            status=status,
            num_tokens=2,
            num_tokens_with_spec=2,
            num_output_placeholders=0,
            num_computed_tokens=1,
            num_prompt_tokens=1,
        )
        for index in range(4)
    ]
    group = BeamGroup("beam", SimpleNamespace(), len(children))
    group.beam_requests = children
    group.beam_request_ids = [child.request_id for child in children]
    scheduler.beam_groups = {group.orig_request_id: group}
    scheduler.beam_to_group = {
        child.request_id: group.orig_request_id for child in children
    }
    scheduler._beam_token_admissions = {}
    scheduler.requests = {child.request_id: child for child in children}
    scheduler.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=0,
    )
    scheduler.running = children if status == RequestStatus.RUNNING else []
    return scheduler, group


def test_beam_group_reserves_all_sequence_slots_before_admission() -> None:
    scheduler, group = make_budget_scheduler()

    assert scheduler._get_num_required_running_slots(group.beam_requests[0]) == 4

    group.beam_requests[0].status = RequestStatus.RUNNING
    assert scheduler._get_num_required_running_slots(group.beam_requests[1]) == 3


def test_beam_group_requires_aggregate_decode_token_budget() -> None:
    scheduler, group = make_budget_scheduler(RequestStatus.RUNNING)

    first = scheduler._get_request_token_budget(group.beam_requests[0], 3, 3)
    second = scheduler._get_request_token_budget(group.beam_requests[1], 3, 3)

    enough_scheduler, enough_group = make_budget_scheduler(RequestStatus.RUNNING)
    enough_first = enough_scheduler._get_request_token_budget(
        enough_group.beam_requests[0], 4, 4
    )
    enough_second = enough_scheduler._get_request_token_budget(
        enough_group.beam_requests[1], 3, 3
    )

    assert first == 0
    assert second == 0
    assert enough_first == 4
    assert enough_second == 3


def test_beam_group_requires_aggregate_prefill_token_budget() -> None:
    scheduler, group = make_budget_scheduler()
    for child in group.beam_requests:
        child.num_computed_tokens = 0
        child.num_prompt_tokens = 32
        child.num_tokens = 32

    short = scheduler._get_request_token_budget(group.beam_requests[0], 127, 127)

    enough_scheduler, enough_group = make_budget_scheduler()
    for child in enough_group.beam_requests:
        child.num_computed_tokens = 0
        child.num_prompt_tokens = 32
        child.num_tokens = 32
    enough = enough_scheduler._get_request_token_budget(
        enough_group.beam_requests[0], 128, 128
    )

    assert short == 0
    assert enough == 128


def test_single_request_keeps_normal_budget_cost() -> None:
    scheduler, _group = make_budget_scheduler()
    request = SimpleNamespace(request_id="greedy")

    assert scheduler._get_num_required_running_slots(request) == 1
    assert scheduler._get_request_token_budget(request, 7, 5) == 5


def test_scheduler_admission_hooks_are_installed_out_of_tree() -> None:
    hook_names = _RESOURCE_ATOMIC_SCHEDULE.__code__.co_names

    assert "_get_num_required_running_slots" in hook_names
    assert "_get_request_token_budget" in hook_names
    if not hasattr(Scheduler, "_get_request_token_budget"):
        assert _RESOURCE_ATOMIC_SCHEDULE is not Scheduler.schedule


def test_scheduler_patch_fails_closed_on_unknown_vllm_shape() -> None:
    with pytest.raises(RuntimeError, match="running token-budget"):
        patch_scheduler_source("def schedule(self): pass")


def test_beam_completion_is_capped_at_public_max_tokens() -> None:
    group = SimpleNamespace(
        orig_request=SimpleNamespace(
            sampling_params=SimpleNamespace(max_tokens=3),
        ),
    )

    capped = BeamSearchScheduler._cap_completion_tokens(group, [1, 2, 3, 4])

    assert capped == [1, 2, 3]


@pytest.mark.parametrize(
    "extra_args",
    [
        {"beam_width": 0},
        {"beam_width": -1},
        {"beam_width": 1.5},
        {"beam_width": "4"},
        {"beam_width": True},
        {"beam_width": 4, "no_repeat_ngram_size": -1},
        {"beam_width": 4, "no_repeat_ngram_size": "3"},
        {"beam_width": 4, "length_penalty": "x"},
        {"beam_width": 4, "unknown_beam_arg": 1},
    ],
)
def test_invalid_beam_xargs_are_rejected(extra_args: dict) -> None:
    with pytest.raises(VLLMValidationError):
        validate_beam_xargs(extra_args)


def test_valid_beam_xargs_allow_session_id() -> None:
    validate_beam_xargs(
        {
            "beam_width": 4,
            "no_repeat_ngram_size": 3,
            "length_penalty": 0.8,
            "session_id": "session",
        }
    )


def test_beam_kv_manager_skips_allocation_inside_existing_page() -> None:
    manager = FakeKVCacheManager()
    beam_manager = _BeamKVCacheManager(manager)

    blocks = beam_manager.allocate_slots(make_beam_request(13), 3)

    assert blocks is manager.empty_kv_cache_blocks
    assert manager.allocate_calls == 0


def test_beam_kv_manager_allocates_at_page_boundary() -> None:
    manager = FakeKVCacheManager()
    beam_manager = _BeamKVCacheManager(manager)

    blocks = beam_manager.allocate_slots(make_beam_request(14), 3)

    assert blocks is not manager.empty_kv_cache_blocks
    assert manager.allocate_calls == 1


def test_beam_kv_manager_reserves_blocks_for_remaining_siblings() -> None:
    manager = FakeKVCacheManager()
    beam_manager = _BeamKVCacheManager(manager, lambda _request: 3)

    beam_manager.allocate_slots(make_beam_request(14), 3)

    assert manager.last_allocate_kwargs["reserved_blocks"] == 6


def test_replace_request_blocks_preserves_async_suffix() -> None:
    # old dst: 1, 5, 9, 4, 8, [10]
    old_blocks = [FakeBlock(i) for i in [1, 5, 9, 4, 8, 10]]
    # new source prefix: 1, 5, 6, 2, 3
    shared_prefix = [FakeBlock(i) for i in [1, 5, 6, 2, 3]]
    mgr = FakeManager(old_blocks)

    BeamSearchScheduler._replace_request_blocks(
        mgr=mgr,
        dst_id="dst",
        shared_blocks=shared_prefix,
        new_blocks=list(shared_prefix),
        block_pool=FakeBlockPool(),
        prefix_blocks=len(shared_prefix),
    )

    assert [block.block_id for block in mgr.req_to_blocks["dst"]] == [
        1,
        5,
        6,
        2,
        3,
        10,
    ]
    assert [block.ref_cnt for block in old_blocks[:5]] == [0, 0, 0, 0, 0]
    assert old_blocks[5].ref_cnt == 1
    assert [block.ref_cnt for block in shared_prefix] == [2, 2, 2, 2, 2]


def test_snapshot_source_prefix_keeps_partial_cow_computed() -> None:
    blocks = [FakeBlock(i) for i in [10, 20]]
    mgr = FakeManager([])
    mgr.req_to_blocks["src"] = blocks
    scheduler = BeamSearchScheduler.__new__(BeamSearchScheduler)
    scheduler.block_size = 4

    snapshot = scheduler._snapshot_source_prefix(
        src_id="src",
        kv_prefix_len=5,
        self_idxs=[0],
        mgrs=[mgr],
    )

    assert [block.block_id for block in snapshot.blocks_by_manager[0]] == [10]
    assert snapshot.num_computed_tokens == 5


def test_group_finalizes_when_async_terminal_output_has_no_transition() -> None:
    scheduler = BeamSearchScheduler.__new__(BeamSearchScheduler)
    scheduler.requests = {}
    group = BeamGroup("request", SimpleNamespace(), 2)
    group.beam_request_ids = ["request:beam:0", "request:beam:1"]

    assert scheduler._should_finalize_group(group, None, [0, 1], set())

    scheduler.requests[group.beam_request_ids[0]] = SimpleNamespace()
    assert not scheduler._should_finalize_group(group, None, [0, 1], set())

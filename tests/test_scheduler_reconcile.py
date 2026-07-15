from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable

from vllm_beam_search.beam_state import BeamGroup
from vllm_beam_search.scheduler import BeamSearchScheduler, _BeamKVCacheManager


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
        self.empty_kv_cache_blocks = object()
        self.single_type_manager = SimpleNamespace(
            req_to_blocks={"beam": [object()]},
            block_size=16,
        )
        self.coordinator = SimpleNamespace(
            single_type_managers=[self.single_type_manager]
        )
        self.allocate_calls = 0

    def allocate_slots(self, *_args, **_kwargs):
        self.allocate_calls += 1
        return object()


def make_beam_request(num_computed_tokens: int):
    return SimpleNamespace(
        request_id="beam",
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=1,
        sampling_params=SimpleNamespace(
            extra_args={"_beam_group_id": "group"}
        ),
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

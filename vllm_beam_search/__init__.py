def __getattr__(name: str):
    if name == "BeamSearchScheduler":
        from vllm_beam_search.scheduler import BeamSearchScheduler

        return BeamSearchScheduler
    raise AttributeError(name)


def register_beam_search_plugin() -> None:
    from vllm_beam_search.scheduler import _install_worker_history_rewrite_hooks
    from vllm_beam_search.validation import install_request_validation

    _install_worker_history_rewrite_hooks()
    install_request_validation()


__all__ = ["BeamSearchScheduler", "register_beam_search_plugin"]

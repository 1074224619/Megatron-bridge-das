# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import os
import sys
import functools
from typing import Any, Callable, Optional, Union

import torch


def print_only_rank0(msg: str):
    """
    Print a message only if the current process is rank 0.

    Args:
        msg (str): The message to print.
    """
    rank = int(
        os.environ.get("RANK")
        or os.environ.get("LOCAL_RANK")
        or os.environ.get("OMPI_COMM_WORLD_RANK")
        or "0"
    )
    if rank == 0:
        print(msg, flush=True)


def _raw_attr(path: str) -> Any:
    """
    Return the attribute stored on the owner of ``path`` without triggering the
    descriptor protocol, so ``staticmethod`` / ``classmethod`` stay visible.

    ``path`` mixes a module path with optional class names, e.g.
    ``pkg.mod.Cls.method``, so the longest already-imported module prefix is
    used to locate the owner.
    """
    parts = path.split('.')
    for i in range(len(parts) - 1, 0, -1):
        module = sys.modules.get('.'.join(parts[:i]))
        if module is None:
            continue
        owner = functools.reduce(getattr, parts[i:-1], module)
        return vars(owner).get(parts[-1])
    return None


def profiler_annotate_patch(path: str, message: str) -> Callable:
    """
    Build an ``apply_wrapper`` callable that annotates ``path`` for the profiler
    while preserving whether the original attribute was a plain function, a
    ``staticmethod``, or a ``classmethod``.

    ``torch.compile`` is left in place so the profile reflects compiled
    performance.  A consequence is that the range only covers the CPU-side
    launch of a compiled region, and never covers its backward.
    """

    def wrapper(func):
        annotated = profiler_annotate(message)(func)
        raw = _raw_attr(path)
        if isinstance(raw, (staticmethod, classmethod)):
            return type(raw)(annotated)
        return annotated

    return wrapper


def profiler_annotate(message: Optional[Union[str, Callable]] = None) -> Callable:
    target_func = message if callable(message) else None
    profile_message = None if target_func is not None else message

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            from megatron.training.global_vars import get_args

            megatron_args = get_args()
            if not getattr(megatron_args, "use_pytorch_profiler", False):
                return func(*func_args, **func_kwargs)
            
            profile_name = profile_message if profile_message is not None else func.__name__

            with torch.profiler.record_function(profile_name):
                return func(*func_args, **func_kwargs)

        return wrapper

    if target_func is not None:
        return decorator(target_func)

    return decorator

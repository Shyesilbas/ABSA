"""Central terminal progress (percentage + bar) for long-running loops."""
from __future__ import annotations

from typing import Iterable, Optional, TypeVar

T = TypeVar("T")

_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"


def track(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: str = "Progress",
    unit: str = "it",
    leave: bool = True,
    disable: bool = False,
) -> Iterable[T]:
    """Wrap iterable with tqdm when available; otherwise pass through."""
    if disable:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        bar_format=_BAR_FORMAT,
        dynamic_ncols=True,
    )


def loader_total(loader) -> Optional[int]:
    """Batch count for a DataLoader, or None if unknown."""
    try:
        return len(loader)
    except TypeError:
        return None

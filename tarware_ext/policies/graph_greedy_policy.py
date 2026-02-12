"""Greedy graph-based policy placeholder."""

from __future__ import annotations

from typing import Any


class GraphGreedyPolicy:
    def reset(self) -> None:
        return None

    def act(self, obs: Any) -> Any:
        raise NotImplementedError("Greedy graph policy not implemented yet.")

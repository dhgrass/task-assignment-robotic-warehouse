"""GNN policy placeholder."""

from __future__ import annotations

from typing import Any


class GNNPolicy:
    def reset(self) -> None:
        return None

    def act(self, obs: Any) -> Any:
        raise NotImplementedError("GNN policy not implemented yet.")

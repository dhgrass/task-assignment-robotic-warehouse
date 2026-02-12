"""Graph schema definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np


class NodeType(str, Enum):
    AGV = "agv"
    PICKER = "picker"
    SHELF = "shelf"
    GOAL = "goal"


@dataclass
class GraphState:
    node_features: np.ndarray
    edge_index: np.ndarray
    node_types: List[NodeType]
    metadata: Dict[str, int]

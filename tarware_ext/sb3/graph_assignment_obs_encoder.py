"""GraphState-based observation encoder for GraphAssignmentEnv."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from tarware_ext.graphs.schema import GraphState
from tarware_ext.graphs.slot_projection import project_graph_to_request_slots


def encode_graph_assignment_obs(
    graph: GraphState,
    *,
    max_request_slots: int,
    num_agvs: int,
    obs_shape: Tuple[int, ...],
    agv_feat_dim: int = 6,
    slot_feat_dim: int = 7,
    global_feat_dim: int = 4,
) -> np.ndarray:
    """Encode assignment GraphState into fixed-size PPO observation vector."""
    proj = project_graph_to_request_slots(
        graph,
        max_request_slots=max_request_slots,
        num_agvs=num_agvs,
    )
    agv_feats = proj["agv_features"]
    slot_feats = proj["slot_features"]
    global_feats = proj["global_features"]

    obs = np.zeros(obs_shape, dtype=np.float32)
    offset = 0
    for i in range(num_agvs):
        obs[offset : offset + agv_feat_dim] = agv_feats[i, :agv_feat_dim]
        offset += agv_feat_dim
        for slot in range(max_request_slots):
            obs[offset : offset + slot_feat_dim] = slot_feats[i, slot, :slot_feat_dim]
            offset += slot_feat_dim

    obs[offset : offset + global_feat_dim] = global_feats[:global_feat_dim]
    return obs

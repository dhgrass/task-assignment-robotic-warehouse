from __future__ import annotations

import pytest

from tarware_ext.sb3 import GraphAssignmentConfig, GraphAssignmentEnv


@pytest.mark.parametrize("obs_backend", ["assignment", "graph"])
def test_graph_assignment_env_reset_step_smoke(obs_backend: str) -> None:
    env = GraphAssignmentEnv(
        GraphAssignmentConfig(
            env_id="tarware-small-2agvs-1pickers-globalobs-v1",
            obs_backend=obs_backend,
            max_request_slots=20,
            max_steps=20,
            seed=21,
            verbose=False,
        )
    )
    try:
        obs, info = env.reset(seed=21)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        assert next_obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)
    finally:
        env.close()

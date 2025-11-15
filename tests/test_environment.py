import os
import sys
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment_taylor_green import TaylorGreenEnvironment


def test_init():
    with pytest.raises(ValueError):
        env = TaylorGreenEnvironment(dt=-0.1)
    with pytest.raises(ValueError):
        env = TaylorGreenEnvironment(dt=0.1, alignment_timescale=-0.1)


def test_step():
    env = TaylorGreenEnvironment(alignment_timescale=0)
    assert_almost_equal(np.linalg.norm(env.swimming_velocity), env.swimmer_speed)
    env.reset()
    env.step(action=1)
    assert env.orientation == 0.5 * np.pi
    assert_almost_equal(np.linalg.norm(env.swimming_velocity), env.swimmer_speed)

    env = TaylorGreenEnvironment(
        dt=100,
        swimmer_speed=0.0,
        alignment_timescale=0.1,
        diffusivity_translational=100.0,
    )
    env.reset()
    assert env.swimmer_speed == 0.0
    swimmer_position_old = env.swimmer_position.copy()
    swimming_velocity_old = env.swimming_velocity.copy()
    env.step(action=3)
    assert not np.allclose(env.swimmer_position, swimmer_position_old)
    assert np.allclose(env.swimming_velocity, swimming_velocity_old)


def test_observation():
    env = TaylorGreenEnvironment()
    if env.u0 > 1e-8:
        # Case 1: Strong positive vorticity, x-dominant positive velocity
        env.flow_vorticity = 1.0
        env.swimming_velocity = np.array([2.0, 0.5])
        observation = env._get_observation()
        assert observation in [8, 10]  # swimmer oriented along x+ (8), or x- (10)

        # Case 2: Strong negative vorticity, y-dominant negative velocity
        env.flow_vorticity = -1.0
        env.swimming_velocity = np.array([0.1, -3.0])
        observation = env._get_observation()
        assert observation in [1, 3]  # y+ (1), or y- (3)

        # Case 3: Vorticity below the threshold, x-dominant negative velocity
        env.flow_vorticity = -0.2
        env.swimming_velocity = np.array([-4.0, 1.0])
        observation = env._get_observation()
        assert observation == 6  # |omega| small, swimmer oriented along x-

        # Case 4: Near-zero vorticity, y-dominant positive velocity
        env.flow_vorticity = 1e-8
        env.swimming_velocity = np.array([0.5, 3.0])
        observation = env._get_observation()
        assert observation == 5  # |omega| small, swimmer oriented along y+

        # Case 5: Velocity almost zero (bias from local flow)
        env.flow_vorticity = -1e-8
        env.swimming_velocity = np.array([0.0, 0.0])
        observation = env._get_observation()
        assert isinstance(observation, int)
        assert 0 <= observation < 12


def test_reward():
    env = TaylorGreenEnvironment(
        dt=0.01,
        swimmer_speed=0.0,
        diffusivity_translational=0.0,
        diffusivity_rotational=0.0,
    )
    env.flow_velocity = np.array([1.0, 1.0])
    observation, reward = env.step(action=0)
    assert_almost_equal(reward, 0.01)


def test_step_reproducibility():
    env0 = TaylorGreenEnvironment(seed=42)
    env1 = TaylorGreenEnvironment(seed=42)
    env2 = TaylorGreenEnvironment(seed=43)
    for step in range(10):
        env0.step(action=1)
        env1.step(action=1)
        env2.step(action=1)
    assert np.allclose(env1.swimmer_position, env0.swimmer_position)
    assert not np.allclose(env2.swimmer_position, env0.swimmer_position)

import os
import sys
import numpy as np
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.taylor_green import TaylorGreenEnvironment
from environments.taylor_green_continuous import TaylorGreenContinuousEnvironment
from environments.rayleigh_benard import RayleighBenardEnvironment
from environments.stratified_shear import StratifiedShearEnvironment

_STRATIFIED_SHEAR_ACTION = np.zeros(10)


def test_init():
    with pytest.raises(ValueError):
        _ = TaylorGreenEnvironment(dt=-0.1)
    with pytest.raises(ValueError):
        _ = RayleighBenardEnvironment(dt=-0.1)
    with pytest.raises(ValueError):
        _ = TaylorGreenEnvironment(dt=0.1, alignment_timescale=-0.1)
    with pytest.raises(ValueError):
        _ = StratifiedShearEnvironment(save_every=-1)


def test_step_taylor_green():
    env = TaylorGreenEnvironment(alignment_timescale=0)
    np.testing.assert_almost_equal(
        np.linalg.norm(env.swimming_velocity), env.swimmer_speed
    )
    env.reset()
    env.step(action=1)
    assert env.orientation == 0.5 * np.pi
    np.testing.assert_almost_equal(
        np.linalg.norm(env.swimming_velocity), env.swimmer_speed
    )

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


def test_step_timestep_equivalence_taylor_green():
    # Test logic around dt and dt_simulation
    rng = np.random.default_rng(seed=42)
    position_initial = np.array([rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)])
    orientation_initial = rng.uniform(0, 2 * np.pi)
    action = np.array([0, 1])

    env1 = TaylorGreenContinuousEnvironment(
        dt=0.01,
        diffusivity_rotational=0,
        diffusivity_translational=0,
        action_type="continuous",
    )
    env2 = TaylorGreenContinuousEnvironment(
        dt=0.02,
        diffusivity_rotational=0,
        diffusivity_translational=0,
        action_type="continuous",
    )

    _ = env1.reset(position=position_initial.copy(), orientation=orientation_initial)
    _ = env2.reset(position=position_initial.copy(), orientation=orientation_initial)

    # advance env1 and env2 by equivalent times
    for _ in range(2):
        env1.step(action)
    env2.step(action)

    assert np.allclose(env1._get_observation(), env2._get_observation())

    env1.step(action)  # take another step()

    assert not np.allclose(env1._get_observation(), env2._get_observation())


def test_step_rayleigh_benard():
    stop_sim_time = 0.3
    env = RayleighBenardEnvironment(
        dt=0.1, stop_sim_time=stop_sim_time, perturbation_amplitude=1.0
    )
    env.reset()
    env.step(action=0)
    assert env._solver.sim_time == 0.1
    while env._solver.sim_time < stop_sim_time:
        env.step(action=0)
    assert np.allclose(env._solver.sim_time, 0.3)
    assert np.max(abs(env._b_perturbation["g"])) <= 1.0


def test_step_stratified_shear():
    env = StratifiedShearEnvironment(nx=32, nz=32)
    env.reset()
    env.step(action=_STRATIFIED_SHEAR_ACTION)
    env._u.change_scales(1)
    env.u_initial.change_scales(1)
    u_fluctuation = env._u["g"] - env.u_initial["g"]
    assert not np.allclose(u_fluctuation, 0)


def test_observation_taylor_green():
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

    # Case 6: Test continuous observation
    env = TaylorGreenContinuousEnvironment()
    env.flow_vorticity = 1.0
    env.swimming_velocity = np.array([2.0, 0.5])
    observation = env._get_observation()
    for i in range(3):
        assert -1 <= observation[i] <= 1
    assert np.allclose(np.linalg.norm([observation[1], observation[2]]), 1)


def test_observation_stratified_shear():
    env = StratifiedShearEnvironment(nx=32, nz=32)
    observation = env.reset()
    assert observation.shape == (3, env._nx, env._nz)
    assert not np.allclose(observation[2], 0)  # buoyancy background is -tanh(...)
    assert np.all(np.isfinite(observation))


def test_reward_taylor_green():
    env = TaylorGreenEnvironment(
        dt=0.01,
        swimmer_speed=0.0,
        diffusivity_translational=0.0,
        diffusivity_rotational=0.0,
    )
    env.flow_velocity = np.array([1.0, 1.0])
    _, reward = env.step(action=0)
    np.testing.assert_almost_equal(reward, 0.01)


def test_reward_rayleigh_benard():
    env = RayleighBenardEnvironment(dt=10.0, stop_sim_time=20.0)
    env.reset()
    _, reward = env.step(action=0)
    assert isinstance(reward, np.float64)
    assert reward > 0
    _, reward_new = env.step(action=0)
    assert not np.allclose(reward_new, reward)


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


def test_episode_reproducibility():
    env = RayleighBenardEnvironment(
        dt=1, stop_sim_time=2
    )  # initialise convection environment
    Nu = []
    Re = []
    for _ in range(2):
        _ = env.reset(seed=42)  # deterministic reset
        while env._solver.sim_time < env.stop_sim_time:
            print(f"Simulation time: {env._solver.sim_time}")
            _, _ = env.step(action=0)
        Nu.append(env.Nu_array[-1])
        Re.append(env.max_Re_array[-1])

    assert Re[0] > 0
    assert env._solver.sim_time == env.stop_sim_time
    assert np.allclose(Nu[0], Nu[1])
    assert np.allclose(Re[0], Re[1])

    env = StratifiedShearEnvironment(
        dt=0.5,
        stop_sim_time=1.0,
        nx=32,  # small grid size for faster testing
        nz=32,
        save_metrics=True,
    )  # initialise shear flow environment
    Ke = []
    for _ in range(2):
        _ = env.reset(seed=42)  # deterministic reset
        while env._solver.sim_time < env.stop_sim_time:
            print(f"Simulation time: {env._solver.sim_time}")
            _, _ = env.step(action=_STRATIFIED_SHEAR_ACTION)
        Ke.append(env.Ke_array[-1])

    assert Ke[0] > 0
    assert env._solver.sim_time == env.stop_sim_time
    assert np.allclose(Ke[0], Ke[1])


def test_reset_stratified_shear():
    env = StratifiedShearEnvironment(nx=32, nz=32)
    env.reset(seed=42)

    # Test that the value of the buoyancy field is as expected
    assert np.allclose(
        env.b_initial["g"], -np.tanh(np.sqrt(env._Pr) * env._z), atol=1e-10
    )

    # Test the persistence of u_initial across reset() and step() functions
    u_initial = env.u_initial.copy()
    env.step(action=_STRATIFIED_SHEAR_ACTION)
    assert np.allclose(u_initial["g"], env.u_initial["g"])

    env.reset(seed=42)
    env.step(action=_STRATIFIED_SHEAR_ACTION)
    assert np.allclose(u_initial["g"], env.u_initial["g"])

    # Test that a different seed results in a different initial state
    env.reset(seed=43)
    assert not np.allclose(u_initial["g"], env.u_initial["g"])


def test_device():
    # Invalid device raises ValueError
    with pytest.raises(ValueError):
        env = StratifiedShearEnvironment(
            dt=1, stop_sim_time=2, nx=32, nz=32, device="tpu"
        )

    # GPU without cupy raises ImportError
    with patch("environments.stratified_shear.cp", None):
        with pytest.warns(UserWarning, match="not on the main dedalus branch"):
            with pytest.raises(ImportError, match="cupy is required"):
                StratifiedShearEnvironment(
                    dt=1, stop_sim_time=2, nx=32, nz=32, device="gpu"
                )

    # GPU with cupy issues a warning
    cp_real = pytest.importorskip("cupy")
    with patch("environments.stratified_shear.cp", cp_real):
        with pytest.warns(UserWarning, match="not on the main dedalus branch"):
            StratifiedShearEnvironment(
                dt=1, stop_sim_time=2, nx=32, nz=32, device="gpu"
            )

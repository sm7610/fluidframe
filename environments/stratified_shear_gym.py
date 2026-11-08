# Copyright 2026 Shruti Mishra. All rights reserved.
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import gymnasium as gym

from environments.stratified_shear import StratifiedShearEnvironment

_TIME_LIMIT = 100

# Domain size
_LX = 30
_LZ = 30

# Resolution
_NX = 512
_NZ = 512

# Constants for environment
_EFFICIENCY_DESIRED = 0.1
_PERTURBATION_AMPLITUDE = 0.0


class StratifiedShearGymEnv(gym.Env):
    def __init__(
        self,
        dt: float = 1.0,
        stop_sim_time: float = _TIME_LIMIT,
        efficiency_desired: float = _EFFICIENCY_DESIRED,
        perturbation_amplitude: float = _PERTURBATION_AMPLITUDE,
        n_modes: int = 10,
        lx: float = _LX,
        lz: float = _LZ,
        nx: int = _NX,
        nz: int = _NZ,
        device: str = "cpu",
        save_snapshots: bool = False,
        save_every: int = 50,
    ):
        """
        Initialize the Gym environment for a stratified shear flow.

        Args:
            dt: The environment timestep (interval between agent actions)
            stop_sim_time: Duration of an episode
            efficiency_desired: Desired level of mixing efficiency
            perturbation_amplitude: Scale of forcing function in the momentum equation
            n_modes: Number of modes in the forcing function
            lx: Length of the domain in the horizontal direction
            lz: Length of the domain in the vertical direction
            nx: Number of horizontal grid points / spectral coefficients
            nz: Number of vertical grid points / spectral coefficients
            device: device to run the simulation on ("cpu" or "gpu")
            save_snapshots: Whether to save simulation snapshots (True/False)
            save_every: Frequency of saving simulation snapshots (in episodes)
        """
        super().__init__()

        # Initialize the underlying stratified shear flow environment
        self._env = StratifiedShearEnvironment(
            dt=dt,
            stop_sim_time=stop_sim_time,
            efficiency_desired=efficiency_desired,
            perturbation_amplitude=perturbation_amplitude,
            n_modes=n_modes,
            lx=lx,
            lz=lz,
            nx=nx,
            nz=nz,
            device=device,
            save_snapshots=save_snapshots,
            save_every=save_every,
        )

        # Define Gym spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, self._env._nx, self._env._nz),
            dtype=np.float32,
        )

        # Action is continuous (continuous value of forcing)
        self.action_space = gym.spaces.Box(
            low=-1 * np.ones(n_modes),
            high=np.ones(n_modes),
            shape=(n_modes,),
            dtype=np.float32,
        )

        # Episode management
        self.stop_sim_time = stop_sim_time

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            options: Optional dictionary with reset parameters

        Returns:
            observation: The initial observation
            info: Dictionary with additional information
        """
        super().reset(seed=seed)

        # Reset the underlying environment
        observation = self._env.reset(seed=seed)

        # Prepare info dictionary
        info = {}
        return observation, info

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.

        Args:
            action: The action to update the orientation

        Returns:
            observation: The new observation
            reward: The reward obtained from the action (y-displacement)
            terminated: Whether the episode has ended naturally
            truncated: Whether the episode was cut off (e.g., time limit)
            info: Dictionary with additional information
        """
        # Execute action in underlying environment
        observation, reward = self._env.step(action)

        # Check if episode should end
        terminated = False  # This environment doesn't have natural termination
        truncated = False
        if self.stop_sim_time is not None:
            truncated = self._env._solver.sim_time >= self.stop_sim_time

        # Prepare info dictionary
        info = {}

        return observation, float(reward), terminated, truncated, info


# Convenience function to create the environment
def make_stratified_shear_env(**kwargs) -> StratifiedShearGymEnv:
    """
    Factory function to create a Stratified Shear Gym environment.

    Args:
        **kwargs: Keyword arguments passed to StratifiedShearGymEnv constructor

    Returns:
        A configured StratifiedShearGymEnv instance
    """
    return StratifiedShearGymEnv(**kwargs)

from abc import ABC, abstractmethod
from typing import Any


class Environment(ABC):
    def __init__(self, dt: float) -> None:
        """Initialise the environment.

        Args:
            dt: The environment timestep
        """
        self.dt = dt
        self._setup_simulation()

    def reset(self):
        """Reset the environment to an initial state."""
        observation = self._get_observation()
        return observation

    @abstractmethod
    def _setup_simulation(self) -> None:
        """Set up the simulation environment.

        Subclasses must implement this to:
        - Import and configure dedalus (d3)
        - Define problem type (IVP, LBVP, NLBVP, EVP)
        - Set up timestepper
        - Build solver and assign to self._solver
        - Set up CFL for adaptive timestepping
        """
        pass

    def step(self, action) -> tuple:
        """Carry out an environment step.

        Args:
            action: The action provided by the agent

        Returns:
            observation
            reward
        """
        self._solver.stop_sim_time = self.dt
        while self._solver.proceed:
            timestep = self.CFL.compute_timestep()
            self._solver.step(timestep, action)  # note: non-standard

        # Get the observation and reward for the agent
        observation = self._get_observation()
        reward = self._get_reward()

        return observation, reward

    @abstractmethod
    def _get_observation(self):
        """Extract observation from environment state.

        Returns:
            observation: The observation for the agent
        """
        pass

    @abstractmethod
    def _get_reward(self) -> float:
        """Calculate reward based on environment state.

        Returns:
            reward: Scalar reward value for the agent
        """
        pass

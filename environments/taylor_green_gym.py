from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import gymnasium as gym

from environments.taylor_green_continuous import TaylorGreenContinuousEnvironment


class TaylorGreenContinuousGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for the Taylor-Green vortex environment with continuous observations.

    This environment simulates an active swimmer navigating through a Taylor-Green
    vortex flow field. The agent learns to control the swimmer's orientation to
    maximize upward (y-direction) displacement.

    Observation Space:
        Box(7,) - Continuous vector containing:
        - [0]: vorticity_scaled = flow_vorticity / u0 (normalized vorticity)
        - [1]: cos(orientation) of swimmer
        - [2]: sin(orientation)
        - [3]: cos(position[0])
        - [4]: sin(position[0])
        - [5]: cos(position[1])
        - [6]: sin(position[1])

    Action Space:
        The action space depends on the ``action_type`` argument:

        - Discrete (default): ``Discrete(4)`` - Four discrete orientation preferences:
            - 0: 0° (pointing right, +x direction)
            - 1: 90° (pointing up, +y direction)
            - 2: 180° (pointing left, -x direction)
            - 3: 270° (pointing down, -y direction)

        - Continuous: ``Box(low=-1, high=1, shape=(2,), dtype=float32)`` when
          ``action_type == "continuous"``. The preferred swimmer orientation is
          then calculated as np.arctan2(action(1), action(0)).
    Reward:
        The reward is the change in y-position (upward displacement) per step.
    """

    def __init__(
        self,
        dt: float = 0.01,
        swimmer_speed: float = 0.3,
        flow_speed: float = 1.0,
        alignment_timescale: float = 1.0,
        diffusivity_rotational: float = 0.0001,
        diffusivity_translational: float = 0.001,
        max_episode_steps: Optional[int] = None,
        seed: Optional[int] = None,
        action_type: Optional[str] = None,
    ):
        """
        Initialize the Taylor-Green Continuous Gym environment.

        Args:
            dt: The environment timestep (interval between agent actions)
            swimmer_speed: The speed of the swimmer relative to the flow
            flow_speed: The scale of the flow speed
            alignment_timescale: For swimmer orientation w.r.t. preferred orientation
            diffusivity_rotational: Scale of the rotational noise
            diffusivity_translational: Scale of the translational noise
            max_episode_steps: Maximum steps per episode (None for unlimited)
            seed: Random seed for reproducibility
            action_type: Optional, ("discrete" or "continuous"); default: "discrete"
        """
        super().__init__()

        # Initialize the underlying Taylor-Green continuous environment
        self._env = TaylorGreenContinuousEnvironment(
            dt=dt,
            swimmer_speed=swimmer_speed,
            flow_speed=flow_speed,
            alignment_timescale=alignment_timescale,
            diffusivity_rotational=diffusivity_rotational,
            diffusivity_translational=diffusivity_translational,
            seed=seed,
            action_type=action_type,
        )

        # Define Gym spaces
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(7,),
            dtype=np.float32,
        )

        # Action can be discrete (4 orientation preferences)
        self.action_space = gym.spaces.Discrete(4)
        # Or continuous (continuous orientation preference)
        if self._env.action_type == "continuous":
            self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        # Episode management
        self.max_episode_steps = max_episode_steps
        self._current_step = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            options: Optional dictionary with reset parameters:
                - 'position': np.ndarray of shape (2,) for initial position
                - 'orientation': float for initial orientation in radians

        Returns:
            observation: The initial observation
            info: Dictionary with additional information
        """
        super().reset(seed=seed)

        # Set seed if provided
        if seed is not None:
            self._env.rng = np.random.default_rng(seed=seed)

        # Extract position and orientation from options if provided
        position = None
        orientation = None
        if options is not None:
            position = options.get("position", None)
            orientation = options.get("orientation", None)

        # Reset the underlying environment
        observation = self._env.reset(position=position, orientation=orientation)

        # Convert to float32 for consistency
        observation = observation.astype(np.float32)

        # Reset episode step counter
        self._current_step = 0

        # Prepare info dictionary
        info = {
            "position": self._env.swimmer_position.copy(),
            "orientation": self._env.orientation,
            "flow_velocity": self._env.flow_velocity.copy(),
            "flow_vorticity": self._env.flow_vorticity,
            "swimming_velocity": self._env.swimming_velocity.copy(),
        }
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

        # Convert to float32 for consistency
        observation = observation.astype(np.float32)

        # Update step counter
        self._current_step += 1

        # Check if episode should end
        terminated = False  # This environment doesn't have natural termination
        truncated = False
        if self.max_episode_steps is not None:
            truncated = self._current_step >= self.max_episode_steps

        # Prepare info dictionary
        info = {
            "position": self._env.swimmer_position.copy(),
            "orientation": self._env.orientation,
            "flow_velocity": self._env.flow_velocity.copy(),
            "flow_vorticity": self._env.flow_vorticity,
            "swimming_velocity": self._env.swimming_velocity.copy(),
            "step": self._current_step,
        }

        return observation, float(reward), terminated, truncated, info


# Convenience function to create the environment
def make_taylor_green_continuous_env(**kwargs) -> TaylorGreenContinuousGymEnv:
    """
    Factory function to create a Taylor-Green Continuous Gym environment.

    Args:
        **kwargs: Keyword arguments passed to TaylorGreenContinuousGymEnv constructor

    Returns:
        A configured TaylorGreenContinuousGymEnv instance
    """
    return TaylorGreenContinuousGymEnv(**kwargs)

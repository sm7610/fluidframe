from typing import Optional

import numpy as np

from environments.base import Environment

# Flow parameters
_SWIMMER_SPEED = 0.3
_ALIGNMENT_TIMESCALE = 1.0

# Constants for environment
_FLOW_SPEED = 1.0
_SIMULATION_TIMESTEP = 0.01
_TIMESTEP = 0.01
_DIFFUSIVITY_ROTATIONAL = 0.0001
_DIFFUSIVITY_TRANSLATIONAL = 0.001
_MIN_FLOW_SPEED_THRESHOLD = 1e-8


class TaylorGreenEnvironment(Environment):

    def __init__(
        self,
        dt: float = _TIMESTEP,
        dt_simulation: float = _SIMULATION_TIMESTEP,
        swimmer_speed: float = _SWIMMER_SPEED,
        flow_speed: float = _FLOW_SPEED,
        alignment_timescale: float = _ALIGNMENT_TIMESCALE,
        diffusivity_rotational: float = _DIFFUSIVITY_ROTATIONAL,
        diffusivity_translational: float = _DIFFUSIVITY_TRANSLATIONAL,
        seed: Optional[int] = None,
    ):
        """Initialise the environment.

        Args:
            dt: The environment timestep (interval between agent actions)
            dt_simulation: The simulation timestep (interval between updates to flow variables)
            swimmer_speed: The speed of the swimmer relative to the flow
            flow_speed: The scale of the flow speed
            alignment_timescale: For swimmer orientation w.r.t. preferred orientation
            diffusivity_rotational: Scale of the rotational noise
            diffusivity_translational: Scale of the translational noise
            seed: Random number generator for the environment dynamics
        """
        self.dt = dt
        self.dt_simulation = dt_simulation
        self.swimmer_speed = swimmer_speed
        self.u0 = flow_speed
        self.alignment_timescale = alignment_timescale
        self.diffusivity_rotational = diffusivity_rotational
        self.diffusivity_translational = diffusivity_translational
        self.rng = np.random.default_rng(seed=seed)

        if self.dt_simulation <= 0:
            raise ValueError("Timesteps should be positive.")

        if self.dt < self.dt_simulation:
            raise ValueError(
                "The environment timestep should be greater than or equal to the timestep of the simulation dynamics."
            )

        if self.alignment_timescale < 0:
            raise ValueError("Alignment timescale should be non-negative.")

        if self.diffusivity_rotational < 0:
            raise ValueError("Rotational diffusivity should be non-negative.")

        if self.diffusivity_translational < 0:
            raise ValueError("Translational diffusivity should be non-negative.")

        self._setup_simulation()
        self.reset()

    def reset(
        self, position: Optional[np.ndarray] = None, orientation: Optional[float] = None
    ):
        """Resets the environment to an initial state."""
        if position is not None:
            self.swimmer_position = position
        else:
            self.swimmer_position = np.array(
                [self.rng.uniform(0, 2 * np.pi), self.rng.uniform(0, 2 * np.pi)]
            )  # flow is periodic about 2 * np.pi in two dimensions

        if orientation is not None:
            self.orientation = orientation
        else:
            self.orientation = self.rng.uniform(0, 2 * np.pi)

        self.swimming_velocity = self.swimmer_speed * np.array(
            [np.cos(self.orientation), np.sin(self.orientation)]
        )
        self._update_flow_variables()
        observation = self._get_observation()
        return observation

    def step(self, action):
        """Carries out an environment step."""

        # Action: update the orientation
        orientation_preferred = self.get_preferred_orientation(action)

        swimmer_position_old = self.swimmer_position.copy()  # for reward computation
        dt_remaining = self.dt  # initialise

        # Advance simulation in sub-steps until the full timestep dt is covered
        while dt_remaining > 0:  # the simulation needs to be advanced
            # Compute advancement timestep
            dt_advance = self.dt_simulation
            if dt_remaining < self.dt_simulation:
                dt_advance = dt_remaining

            # Advance simulation
            if self.alignment_timescale == 0:  # instantaneous
                self.orientation = orientation_preferred
            else:
                angular_velocity = (
                    (0.5 / self.alignment_timescale)
                    * np.sin(orientation_preferred - self.orientation)
                    + 0.5 * self.flow_vorticity  # based on old position
                    + np.sqrt(2 * self.diffusivity_rotational)
                    * self.rng.standard_normal()
                )
                self.orientation += angular_velocity * dt_advance

            # Update the swimming velocity
            self.swimming_velocity = self.swimmer_speed * np.array(
                [np.cos(self.orientation), np.sin(self.orientation)]
            )  # velocity relative to background flow

            # Update the swimmer position
            self.swimmer_position += dt_advance * (
                self.swimming_velocity
                + self.flow_velocity  # based on old position
                + np.sqrt(2 * self.diffusivity_translational)
                * self.rng.standard_normal(2)
            )

            self._update_flow_variables()  # update the flow variables
            dt_remaining -= dt_advance  # update remaining time

        # Get the observation and reward for the agent
        observation = self._get_observation()
        reward = self.swimmer_position[1] - swimmer_position_old[1]

        return observation, reward

    def get_preferred_orientation(self, action):
        """Transforms the action into a preferred orientation."""
        return action * np.pi / 2

    def _setup_simulation(self):
        pass

    def _update_flow_variables(self):
        """Computes flow velocity and vorticity at the swimmer position using the solution
        for the Taylor-Green vortex.
        """
        # Equivalent swimmer position due to periodicity of flow
        x0 = self.swimmer_position[0] % (2 * np.pi)
        x1 = self.swimmer_position[1] % (2 * np.pi)

        self.flow_velocity = (
            0.5
            * self.u0
            * np.array([-np.cos(x0) * np.sin(x1), np.sin(x0) * np.cos(x1)])
        )
        self.flow_vorticity = self.u0 * np.cos(x0) * np.cos(x1)

    def _get_observation(self, vorticity_threshold=1.0 / 3.0):
        """
        12-state encoder.
        Buckets by vorticity band (neg/zero/pos) * orientation.
        Indices:
        0..3  : vorticity < -vorticity_threshold   (x+, y+, x-, y-)
        4..7  : |vorticity| <= vorticity_threshold (x+, y+, x-, y-)
        8..11 : vorticity > +vorticity_threshold   (x+, y+, x-, y-)
        """

        vx, vy = self.swimming_velocity[0], self.swimming_velocity[1]
        if abs(self.u0) > _MIN_FLOW_SPEED_THRESHOLD:
            vorticity_scaled = self.flow_vorticity / self.u0
        else:
            vorticity_scaled = 0

        # Decide dominant axis and sign
        # first sort the observation based on flow vorticity only
        if vorticity_scaled < -vorticity_threshold:
            observation = 0  # large, negative, flow vorticity
        elif vorticity_scaled > vorticity_threshold:
            observation = 8  # large, positive, flow vorticity
        else:
            observation = 4  # small flow vorticity

        # then additionally sort the observation based on swimmer orientation
        if abs(vx) > abs(vy):  # x-dominant
            observation += 0 if vx > 0 else 2
        else:  # y-dominant
            observation += 1 if vy > 0 else 3
        return observation

    def _get_reward(self):
        """This is not used, as it is computed directly in step."""
        raise NotImplementedError

# Copyright 2026 Shruti Mishra. All rights reserved.
from typing import Optional
import numpy as np

from environments.taylor_green import (
    _ALIGNMENT_TIMESCALE,
    _DIFFUSIVITY_ROTATIONAL,
    _DIFFUSIVITY_TRANSLATIONAL,
    _FLOW_SPEED,
    _MIN_FLOW_SPEED_THRESHOLD,
    _SWIMMER_SPEED,
    _TIMESTEP,
    TaylorGreenEnvironment,
)


class TaylorGreenContinuousEnvironment(TaylorGreenEnvironment):

    def __init__(
        self,
        dt: float = _TIMESTEP,
        swimmer_speed: float = _SWIMMER_SPEED,
        flow_speed: float = _FLOW_SPEED,
        alignment_timescale: float = _ALIGNMENT_TIMESCALE,
        diffusivity_rotational: float = _DIFFUSIVITY_ROTATIONAL,
        diffusivity_translational: float = _DIFFUSIVITY_TRANSLATIONAL,
        seed: Optional[int] = None,
        action_type: Optional[str] = None,
    ):
        """Initialise the environment, with continuous observations and continuous or discrete actions.

        Args:
            action_type: "discrete" ∈ {0, 1, 2, 3} or "continuous" ∈ [0, 2π]
        """
        super().__init__(
            dt=dt,
            swimmer_speed=swimmer_speed,
            flow_speed=flow_speed,
            alignment_timescale=alignment_timescale,
            diffusivity_rotational=diffusivity_rotational,
            diffusivity_translational=diffusivity_translational,
            seed=seed,
        )
        self.action_type = action_type
        if self.action_type:
            if self.action_type not in ["discrete", "continuous"]:
                raise ValueError(
                    f"Invalid action_type {self.action_type!r}. Expected 'discrete', 'continuous', or None."
                )

    def _get_observation(self):
        """
        Returns:
            np.ndarray: observation = [vorticity, orientation], both are continuous-valued.
        """

        if abs(self.u0) > _MIN_FLOW_SPEED_THRESHOLD:
            vorticity_scaled = self.flow_vorticity / self.u0
        else:
            vorticity_scaled = 0

        orientation = np.arctan2(self.swimming_velocity[1], self.swimming_velocity[0])
        return np.array([vorticity_scaled, orientation])

    def get_preferred_orientation(self, action):
        """Transforms the action into a preferred swimmer orientation."""

        if self.action_type == "continuous":
            orientation_preferred = action
        else:
            orientation_preferred = action * np.pi / 2

        return orientation_preferred

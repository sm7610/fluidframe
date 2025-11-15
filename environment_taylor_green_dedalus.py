import numpy as np
import dedalus.public as d3
from scipy.interpolate import RegularGridInterpolator

from environment_taylor_green import TaylorGreenEnvironment

# Dedalus parameters
_N_GRID = 256


class TaylorGreenDedalusEnvironment(TaylorGreenEnvironment):

    def _setup_simulation(self, interpolator_method="cubic"):
        """
        Set up the flow field variables using Dedalus, including velocity fields, vorticity,
        and interpolators for querying the field variables at the swimmer position.
        """
        coords = d3.CartesianCoordinates("x", "y")
        dist = d3.Distributor(coords, dtype=np.float64)
        padding = 4 * (
            2 * np.pi / _N_GRID
        )  # 0 for using dedalus interpolation directly
        xbasis = d3.RealFourier(
            coords["x"], size=_N_GRID, bounds=(0 - padding, 2 * np.pi + padding)
        )
        ybasis = d3.RealFourier(
            coords["y"], size=_N_GRID, bounds=(0 - padding, 2 * np.pi + padding)
        )

        # Set up grid for field variables using dedalus
        self.dedalus_u = dist.VectorField(coords, name="u", bases=(xbasis, ybasis))
        self.dedalus_vorticity = dist.Field(name="omega", bases=(xbasis, ybasis))
        x_full, y_full = dist.local_grids(xbasis, ybasis)

        # Check that interpolation will not result in an error
        assert np.min(x_full) < 0
        assert np.min(y_full) < 0
        assert np.max(x_full) > 2 * np.pi
        assert np.max(y_full) > 2 * np.pi

        # Specify field variables at grid locations
        self.dedalus_u["g"][0] = -0.5 * self.u0 * np.cos(x_full) * np.sin(y_full)
        self.dedalus_u["g"][1] = 0.5 * self.u0 * np.sin(x_full) * np.cos(y_full)
        self.dedalus_vorticity["g"] = self.u0 * np.cos(x_full) * np.cos(y_full)

        self.interpolator_u0 = RegularGridInterpolator(
            (x_full.flatten(), y_full.flatten()),
            self.dedalus_u["g"][0],
            method=interpolator_method,  # 'linear', 'nearest', 'slinear', 'cubic', 'quintic'
            bounds_error=True,
        )

        self.interpolator_u1 = RegularGridInterpolator(
            (x_full.flatten(), y_full.flatten()),
            self.dedalus_u["g"][1],
            method=interpolator_method,  # 'linear', 'nearest', 'slinear', 'cubic', 'quintic'
            bounds_error=True,
        )

        self.interpolator_vorticity = RegularGridInterpolator(
            (x_full.flatten(), y_full.flatten()),
            self.dedalus_vorticity["g"],
            method=interpolator_method,  # 'linear', 'nearest', 'slinear', 'cubic', 'quintic'
            bounds_error=True,
        )

    def _update_flow_variables(self):
        """Compute flow velocity and vorticity at the swimmer position using the solution
        for the Taylor-Green vortex.
        """
        # Equivalent swimmer position due to periodicity of flow
        x0 = self.swimmer_position[0] % (2 * np.pi)
        x1 = self.swimmer_position[1] % (2 * np.pi)

        # Use scipy interpolate
        u0 = self.interpolator_u0((x0, x1))
        u1 = self.interpolator_u1((x0, x1))
        self.flow_velocity = np.array([u0, u1])
        self.flow_vorticity = self.interpolator_vorticity((x0, x1))

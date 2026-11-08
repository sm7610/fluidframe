import logging
import os
import warnings
import math

try:
    import cupy as cp
except ImportError:
    cp = None
import numpy as np
import dedalus.public as d3

from environments.base import Environment
from environments.utils import stratified_shear_evp

# Flow parameters
_RICHARDSON = 0.1  # bulk Richardson number
_REYNOLDS = 1000
_PRANDTL = 1
_TIME_LIMIT = 100

# Domain size
_LX = 30
_LZ = 30

# Resolution
_NX = 512
_NZ = 512

# Constants for environment
_PERTURBATION_AMPLITUDE = 0.0
_EFFICIENCY_DESIRED = 0.1
_N_MODES = 10

# Configuration variables for the device
# TODO (maybe) Move to utils if this is reused as-is across environments
_DEVICE_CONFIG = {
    "cpu": {
        "xbasis_library": None,
        "zbasis_library": None,
    },
    "gpu": {
        "xbasis_library": None,
        "zbasis_library": None,
    },
}

logger = logging.getLogger(__name__)


class StratifiedShearEnvironment(Environment):
    def __init__(
        self,
        dt: float = 0.1,
        stop_sim_time: float = _TIME_LIMIT,
        efficiency_desired: float = _EFFICIENCY_DESIRED,
        perturbation_amplitude: float = _PERTURBATION_AMPLITUDE,
        n_modes: int = _N_MODES,
        lx: float = _LX,
        lz: float = _LZ,
        nx: int = _NX,
        nz: int = _NZ,
        device: str = "cpu",
        save_snapshots: bool = False,
        save_every: int = 50,
    ) -> None:
        """Initialise the environment.

        Args:
            dt: The environment timestep (interval between agent actions)
            stop_sim_time: Duration of an episode
            efficiency_desired: Desired mixing efficiency
            perturbation_amplitude: Scale of forcing function in the momentum equation
            n_modes: Number of modes in the forcing function
            lx: Length of the domain in the horizontal direction
            lz: Length of the domain in the vertical direction
            nx: Number of horizontal grid points / spectral coefficients
            nz: Number of vertical grid points / spectral coefficients
            device: To run the environment, "cpu" or "gpu"
            save_snapshots: Whether to save simulation snapshots (True/False)
            save_every: Save episode snapshots every this many episodes
        """
        self.dt = dt
        self.stop_sim_time = stop_sim_time
        self._efficiency_desired = efficiency_desired
        self._perturbation_amplitude = perturbation_amplitude
        self._n_modes = n_modes
        self._lx = lx
        self._lz = lz
        self._nx = nx
        self._nz = nz
        self.device = device
        self._save_snapshots = save_snapshots
        self._save_every = save_every

        if self.dt <= 0:
            raise ValueError("Timesteps should be positive.")
        if self.dt > self.stop_sim_time:
            raise ValueError(
                "The environment timestep should be smaller than the episode duration."
            )
        if self._efficiency_desired < 0:
            raise ValueError("Desired mixing efficiency should be non-negative.")
        if self._perturbation_amplitude < 0:
            raise ValueError("Perturbation amplitude should be non-negative.")
        if self._n_modes <= 0:
            raise ValueError("Number of modes should be positive.")
        if self._lx <= 0 or self._lz <= 0:
            raise ValueError("Domain lengths should be positive.")
        if self._nx <= 0 or self._nz <= 0:
            raise ValueError("Number of grid points should be positive.")
        if self.device == "gpu":
            warnings.warn("Dedalus on GPU is currently not on the main dedalus branch.")
        if self.device == "gpu" and cp is None:
            raise ImportError(
                "cupy is required for device='gpu' but could not be imported."
            )
        if self._save_every <= 0:
            raise ValueError("save_every must be positive.")

        self._episode = -1
        self._setup_simulation()

        os.makedirs("analysis", exist_ok=True)

    def _setup_simulation(self) -> None:
        """Set up the simulation environment."""
        # Parameters
        self._Rib = _RICHARDSON
        self._Re = _REYNOLDS
        self._Pr = _PRANDTL
        self._amp = 1e-3  # amplitude of initial perturbation to flow profiles
        self.init_mode = False
        dealias = 3 / 2
        self._timestepper = d3.RK443
        safety = 0.25

        # Compute timestep such that each environment step comprises equal values of simulation dt
        max_timestep = safety * (self._lx / self._nx)
        self._substeps = math.ceil(self.dt / max_timestep)
        self._timestep = self.dt / self._substeps

        self._dtype = np.float64

        # Bases
        coords = d3.CartesianCoordinates("x", "z")

        if self.device == "cpu":
            dist = d3.Distributor(coords, dtype=self._dtype)
        elif self.device == "gpu":
            dist = d3.Distributor(
                coords,
                dtype=self._dtype,
                array_namespace=cp,
            )
        else:
            raise ValueError("Wrong device type for running the environment.")

        xbasis = d3.RealFourier(
            coords["x"],
            size=self._nx,
            bounds=(0, self._lx),
            dealias=dealias,
            library=_DEVICE_CONFIG[self.device]["xbasis_library"],
        )
        zbasis = d3.ChebyshevT(
            coords["z"],
            size=self._nz,
            bounds=(-self._lz / 2, self._lz / 2),
            dealias=dealias,
            library=_DEVICE_CONFIG[self.device]["zbasis_library"],
        )

        # Fields
        p = dist.Field(name="p", bases=(xbasis, zbasis))
        self._b = dist.Field(name="b", bases=(xbasis, zbasis))
        self._u = dist.VectorField(coords, name="u", bases=(xbasis, zbasis))
        tau_p = dist.Field(name="tau_p")
        tau_b1 = dist.Field(name="tau_b1", bases=xbasis)
        tau_b2 = dist.Field(name="tau_b2", bases=xbasis)
        tau_u1 = dist.VectorField(coords, name="tau_u1", bases=xbasis)
        tau_u2 = dist.VectorField(coords, name="tau_u2", bases=xbasis)

        # Substitutions
        x, self._z = dist.local_grids(xbasis, zbasis)
        ex, ez = coords.unit_vector_fields(dist)
        lift_basis = zbasis.derivative_basis(1)
        lift = lambda A: d3.Lift(A, lift_basis, -1)
        grad_u = d3.grad(self._u) + ez * lift(tau_u1)  # First-order reduction
        grad_b = d3.grad(self._b) + ez * lift(tau_b1)  # First-order reduction
        div_u = d3.trace(grad_u)
        lap_u = d3.div(grad_u)
        lap_b = d3.div(grad_b)
        dx = lambda A: d3.Differentiate(A, coords["x"])
        self.dz = lambda A: d3.Differentiate(A, coords["z"])
        self._ux = ex @ self._u
        self._uz = ez @ self._u

        self._forcing = dist.Field(
            name="forcing", bases=zbasis
        )  # forcing in the x-momentum equation
        self._forcing.change_scales(1)

        # Problem
        self._problem = d3.IVP(
            [p, self._b, self._u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2],
            namespace=locals(),
        )
        self._problem.add_equation("div_u + tau_p = 0")
        self._problem.add_equation(
            "dt(b) - (1/self._Re/self._Pr)*lap_b + lift(tau_b2) = - u@grad(b)"
        )
        self._problem.add_equation(
            "dt(u) - (1/self._Re)*lap_u + grad(p) + lift(tau_u2) + self._Rib*b*ez = - u@grad(u) + self._forcing * ex"
        )
        # No flux
        self._problem.add_equation("self.dz(b)(z=-self._lz/2) = 0")
        self._problem.add_equation("self.dz(b)(z=+self._lz/2) = 0")
        # No penetration
        self._problem.add_equation("self._uz(z=-self._lz/2) = 0")
        self._problem.add_equation("self._uz(z=+self._lz/2) = 0")
        # No stress
        self._problem.add_equation("self.dz(self._ux)(z=-self._lz/2) = 0")
        self._problem.add_equation("self.dz(self._ux)(z=+self._lz/2) = 0")
        # Pressure gauge
        self._problem.add_equation("integ(p) = 0")

    def reset(self, seed=None):
        """Reset the environment to an initial state."""
        logger.info("Resetting the environment")
        self._episode += 1

        # Solver
        self._solver = self._problem.build_solver(self._timestepper)
        self._solver.sim_time = self._problem.time["g"] = 0

        # Reset field variables to dedalus default of 0
        for field in self._solver.state:
            field.change_scales(1)
            field["g"] = 0
        # Initial conditions
        if self.init_mode:
            nx_evp = 2
            Nev = 10
            target = 0
            evp_state = stratified_shear_evp(
                self._lx,
                self._lz,
                self._nx,
                self._nz,
                self._Rib,
                self._Re,
                self._Pr,
                self._dtype,
                nx_evp,
                Nev,
                target,
            )
            for var, pert in zip(self._problem.variables, evp_state):
                var["g"] = self._amp * pert["g"]
        else:
            # Random noise
            self._u.fill_random("g", seed=seed, distribution="normal", scale=self._amp)
            self._u.low_pass_filter(scales=0.25)
        # Add backgrounds
        self._u["g"][0] += np.tanh(self._z)
        self._b["g"] -= np.tanh(np.sqrt(self._Pr) * self._z)
        self._forcing["g"] = self._get_action(
            np.zeros(self._n_modes)
        )  # No forcing at reset

        # Store initial fields for reward computations
        self.b_initial = self._b.copy()
        self.u_initial = self._u.copy()

        # Analysis
        if self._save_snapshots and self._episode % self._save_every == 0:
            os.makedirs(
                f"snapshots/shear-flow_perturbation-{self._perturbation_amplitude}_nx-{self._nx}/episode-{self._episode}",
                exist_ok=True,
            )
            snapshots = self._solver.evaluator.add_file_handler(
                f"snapshots/shear-flow_perturbation-{self._perturbation_amplitude}_nx-{self._nx}/episode-{self._episode}",
                sim_dt=1,
                max_writes=10,
                parallel="gather",
            )
            snapshots.add_task(-self._b, name="b")
            snapshots.add_task(self._ux, name="ux")
            snapshots.add_task(self._uz, name="uz")
            snapshots.add_task(-d3.div(d3.skew(self._u)), name="vorticity")

        # Flow properties
        self._flow = d3.GlobalFlowProperty(self._solver, cadence=10)
        self._flow.add_property((self._u @ self._u) / 2, name="KE")

        self.time_array = []
        self.Ke_array = []

        observation = self._get_observation()

        # Setup for reward calculation
        # Fluctuation in field variables, relative to ensemble average, size: (nx * nz)
        u_fluctuation = self._u - self.u_initial
        b_fluctuation = self._b - self.b_initial

        # Compute buoyancy frequency, for the dissipation of buoyancy, size: nz
        buoyancy_frequency_squared = -self._Rib * self.dz(
            d3.Average(self.b_initial, "x")
        )

        # Dissipation of velocity
        strain_rate = 0.5 * (
            d3.grad(u_fluctuation) + d3.TransposeComponents(d3.grad(u_fluctuation))
        )
        self._dissipation_u = d3.Average(
            (2 / self._Re) * d3.trace((strain_rate @ strain_rate)), ("x", "z")
        )  # size: (nx * nz)

        # Dissipation of buoyancy
        # TODO: Scale this by buoyancy frequency squared
        self._dissipation_b = d3.Average(
            (
                (1 / (self._Re * self._Pr))
                * (d3.grad(b_fluctuation) @ d3.grad(b_fluctuation))
                / 1
            ),
            ("x", "z"),
        )  # size: (nx * nz)

        return observation

    def step(self, action) -> tuple:
        """Carry out an environment step.

        Args:
            action: The action provided by the agent, vector with shape (n_modes,)

        Returns:
            observation
            reward
        """
        # Main loop
        target_time = self._solver.sim_time + self.dt

        # Update forcing based on action
        self._forcing["g"] = self._get_action(action)

        try:
            logger.info(
                "In environment step function\tsim_time: %.3f", self._solver.sim_time
            )

            for _ in range(self._substeps):
                if not self._solver.proceed:
                    break

                # Advance simulation by timestep
                self._solver.step(self._timestep)

                # Log some output variables
                if (self._solver.iteration - 1) % 10 == 0:
                    max_KE = self._flow.max("KE")
                    logger.info(
                        f"Iteration={self._solver.iteration:d}, "
                        f"Time={self._solver.sim_time:e}, "
                        f"dt={self._timestep:e}, "
                        f"max(KE)={max_KE:f}, "
                    )

                # Metrics to save
                self.time_array.append(self._solver.sim_time)
                self.Ke_array.append(self._flow.max("KE"))
            else:
                # Remove substep roundoff without changing the integration timestep.
                self._solver.sim_time = self._problem.time["g"] = target_time
                self.time_array[-1] = target_time
        except:
            logger.error("Exception raised, triggering end of main loop.")
            self._solver.log_stats()
            raise

        # Save statistics at the end of an episode
        if self._solver.sim_time >= self.stop_sim_time:
            self._solver.log_stats()
            np.savez(
                f"analysis/shear-flow_perturbation-{self._perturbation_amplitude}_nx-{self._nx}_episode{self._episode}",
                time=self.time_array,
                max_KE=self.Ke_array,
            )

        # Get the observation and reward for the agent
        observation = self._get_observation()
        reward = self._get_reward()
        logger.info(f"reward (dissipation_b): {reward:.3g}")

        return observation, reward

    def _get_observation(self):
        """Extract observation from environment state.

        Returns:
            observation: The observation for the agent, based on flow field variables
        """
        self._u.change_scales(1)
        self._b.change_scales(1)

        ux = self._u["g"][0]
        uz = self._u["g"][1]
        b = self._b["g"]

        if hasattr(ux, "get"):  # CuPy array
            ux, uz, b = ux.get(), uz.get(), b.get()
        observation = np.stack([ux, uz, b]).astype(np.float32)  # (3, nx, nz)
        return observation

    def _get_action(self, action):
        """Convert agent action to environment forcing.

        Args:
            action: The action provided by the agent, vector with shape (n_modes,)

        Returns:
            forcing: The forcing to be applied in the environment, based on the agent's action
        """
        self._forcing.change_scales(1)

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self._n_modes:
            raise ValueError(
                f"action must have shape ({self._n_modes},), got {action.shape}"
            )

        # Normalise the action
        if np.linalg.norm(action) > 1e-6:
            action = action / np.linalg.norm(action)

        forcing = 0.0
        for k in range(self._n_modes):
            forcing += action[k] * np.cos((k + 1) * 2 * np.pi * self._z / self._lz)
        forcing *= self._perturbation_amplitude  # add scale of forcing

        return forcing

    def _get_reward(self) -> float:
        """Calculate reward based on environment state.

        Returns:
            reward: Scalar reward value for the agent
        """
        dissipation_b = self._dissipation_b.evaluate()["g"][0][0]

        # TODO: update after determining how to handle zeros in buoyancy frequency
        return 100 * dissipation_b

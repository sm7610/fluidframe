import logging
import os
import numpy as np
from scipy.optimize import brentq
import dedalus.public as d3

from environments.base import Environment

# Flow parameters
_RAYLEIGH = 2e6
_PRANDTL = 1
_NUSSELT0 = 10  # estimate from running simulations at Ra=2e6, Pr=1
_LAMBDA = 1 / (2 * _NUSSELT0)  # thickness of thermal boundary layer
_TIME_LIMIT = 50
_B0_LIMIT = 10

# Constants for environment
_BETA = 20  # smoothing for wall temperature perturbation (action)
_PERTURBATION_AMPLITUDE = 0.0

logger = logging.getLogger(__name__)


class RayleighBenardEnvironment(Environment):
    def __init__(
        self,
        dt: float = 0.1,
        stop_sim_time: float = _TIME_LIMIT,
        perturbation_amplitude: float = _PERTURBATION_AMPLITUDE,
    ) -> None:
        """Initialise the environment.

        Args:
            dt: The environment timestep (interval between agent actions)
            stop_sim_time: Duration of an episode
            perturbation_amplitude: Scale of buoyancy perturbation at the bottom wall
        """
        self.dt = dt
        self.stop_sim_time = stop_sim_time
        self._perturbation_amplitude = perturbation_amplitude

        if self.dt <= 0:
            raise ValueError("Timesteps should be positive.")
        if self.dt > self.stop_sim_time:
            raise ValueError(
                "The environment timestep should be smaller than the episode duration."
            )
        if self._perturbation_amplitude < 0:
            raise ValueError("Perturbation amplitude should be non-negative.")

        self._episode = -1
        self._setup_simulation()

        os.makedirs("analysis", exist_ok=True)

    def _setup_simulation(self) -> None:
        """Set up the simulation environment."""
        # Parameters
        Lx, self._Lz = 4, 1
        nx, nz = 256, 64
        Rayleigh = _RAYLEIGH
        Prandtl = _PRANDTL
        dealias = 3 / 2
        self._timestepper = d3.RK222
        self.max_timestep = 0.125
        dtype = np.float64

        # Bases
        coords = d3.CartesianCoordinates("x", "z")
        dist = d3.Distributor(coords, dtype=dtype)
        xbasis = d3.RealFourier(coords["x"], size=nx, bounds=(0, Lx), dealias=dealias)
        zbasis = d3.ChebyshevT(
            coords["z"], size=nz, bounds=(0, self._Lz), dealias=dealias
        )

        # Fields
        p = dist.Field(name="p", bases=(xbasis, zbasis))
        self._b = dist.Field(
            name="b", bases=(xbasis, zbasis)
        )  # b(uoyancy) = (T - T_c) / (T_h - T_c)
        self._u = dist.VectorField(coords, name="u", bases=(xbasis, zbasis))
        tau_p = dist.Field(name="tau_p")
        tau_b1 = dist.Field(name="tau_b1", bases=xbasis)
        tau_b2 = dist.Field(name="tau_b2", bases=xbasis)
        tau_u1 = dist.VectorField(coords, name="tau_u1", bases=xbasis)
        tau_u2 = dist.VectorField(coords, name="tau_u2", bases=xbasis)
        self._b_perturbation = dist.Field(
            name="b_perturbation", bases=xbasis
        )  # perturbation to b at the bottom boundary
        self._b_perturbation.change_scales(dealias)

        # Substitutions
        kappa = (Rayleigh * Prandtl) ** (-1 / 2)
        self._nu = (Rayleigh / Prandtl) ** (-1 / 2)  # viscosity / momentum diffusivity
        x, self._z = dist.local_grids(xbasis, zbasis)
        ex, self._ez = coords.unit_vector_fields(dist)
        lift_basis = zbasis.derivative_basis(1)
        lift = lambda A: d3.Lift(A, lift_basis, -1)
        grad_u = d3.grad(self._u) + self._ez * lift(tau_u1)  # First-order reduction
        grad_b = d3.grad(self._b) + self._ez * lift(tau_b1)  # First-order reduction

        # Problem
        # First-order form: "div(f)" becomes "trace(grad_f)"
        # First-order form: "lap(f)" becomes "div(grad_f)"
        self._problem = d3.IVP(
            [p, self._b, self._u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2],
            namespace=locals(),
        )
        self._problem.add_equation("trace(grad_u) + tau_p = 0")
        self._problem.add_equation(
            "dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)"
        )
        self._problem.add_equation(
            "dt(u) - self._nu*div(grad_u) + grad(p) - b*self._ez + lift(tau_u2) = - u@grad(u)"
        )
        self._problem.add_equation(
            "b(z=0) = self._Lz + self._b_perturbation"
        )  # Bottom boundary; more buoyant
        self._problem.add_equation("u(z=0) = 0")  # No slip; no penetration
        self._problem.add_equation("b(z=self._Lz) = 0")  # Top boundary; less buoyant
        self._problem.add_equation("u(z=self._Lz) = 0")  # No slip; no penetration
        self._problem.add_equation("integ(p) = 0")  # Pressure gauge

    def reset(self, seed=42):
        """Reset the environment to an initial state."""
        logger.info("Resetting the environment")
        self._episode += 1

        self._solver = self._problem.build_solver(self._timestepper)
        self._solver.sim_time = self._problem.time["g"] = 0

        # Reset field variables to dedalus default of 0
        for field in self._solver.state:
            field.change_scales(1)
            field["g"] = 0

        # Initial conditions
        self._b.fill_random(
            "g", seed=seed, distribution="normal", scale=1e-3
        )  # Random noise
        self._b["g"] *= self._z * (self._Lz - self._z)  # Damp noise at walls
        self._b["g"] += self._Lz - self._z  # Add linear background
        self._b_perturbation["g"] = 0

        # Analysis
        os.makedirs(f"snapshots/episode{self._episode}", exist_ok=True)
        snapshots = self._solver.evaluator.add_file_handler(
            f"snapshots/episode{self._episode}", sim_dt=0.25, max_writes=50
        )
        snapshots.add_task(self._b, name="buoyancy")
        snapshots.add_task(-d3.div(d3.skew(self._u)), name="vorticity")
        snapshots.add_task(
            d3.ave(-d3.dot(self._ez, d3.grad(self._b)(z=0))), name="Nu_bottom"
        )

        # CFL
        self.CFL = d3.CFL(
            self._solver,
            initial_dt=self.max_timestep,
            cadence=10,
            safety=0.5,
            threshold=0.05,
            max_change=1.5,
            min_change=0.5,
            max_dt=self.max_timestep,
        )
        self.CFL.add_velocity(self._u)
        self._flow = d3.GlobalFlowProperty(self._solver, cadence=10)
        self._flow.add_property(np.sqrt(self._u @ self._u) / self._nu, name="Re")
        self._flow.add_property(
            -d3.dot(self._ez, d3.grad(self._b)(z=0)), name="Nu_bottom"
        )
        self.time_array = []
        self.max_Re_array = []
        self.Nu_array = []

        observation = self._get_observation()
        return observation

    def step(self, action) -> tuple:
        """Carry out an environment step.

        Args:
            action: The action provided by the agent

        Returns:
            observation
            reward
        """
        # Main loop
        target_time = self._solver.sim_time + self.dt

        try:
            logger.info(
                "In environment step function\tsim_time: %.3f", self._solver.sim_time
            )

            # Compute action for bang-bang controller, over environment dt
            b0_solution = brentq(self.compute_integral, -_B0_LIMIT, _B0_LIMIT)
            self._b_perturbation["g"] = self._perturbation_amplitude * np.tanh(
                _BETA * (self._get_observation() - b0_solution)
            )

            while self._solver.proceed and self._solver.sim_time < target_time:
                # Compute timestep for simulator
                timestep = self.CFL.compute_timestep()
                # Don't overshoot target
                if self._solver.sim_time + timestep > target_time:
                    timestep = target_time - self._solver.sim_time

                # Advance environment
                self._solver.step(timestep)

                # Log some output variables
                if (self._solver.iteration - 1) % 10 == 0:
                    max_Re = self._flow.max("Re")
                    mean_Nu = self._flow.grid_average("Nu_bottom")
                    logger.info(
                        f"Iteration={self._solver.iteration:d}, "
                        f"Time={self._solver.sim_time:e}, "
                        f"dt={timestep:e}, "
                        f"max(Re)={max_Re:f}, "
                        f"Nu_bottom={mean_Nu:f}"
                    )

                # Metrics to save
                self.time_array.append(self._solver.sim_time)
                self.max_Re_array.append(self._flow.max("Re"))
                self.Nu_array.append(self._flow.grid_average("Nu_bottom"))

        except:
            logger.error("Exception raised, triggering end of main loop.")
            self._solver.log_stats()
            raise

        # Save statistics at the end of an episode
        if self._solver.sim_time >= self.stop_sim_time:
            self._solver.log_stats()
            np.savez(
                f"analysis/perturbation-{self._perturbation_amplitude}_episode{self._episode}",
                time=self.time_array,
                max_Re=self.max_Re_array,
                Nu_bottom=self.Nu_array,
            )

        # Get the observation and reward for the agent
        observation = self._get_observation()
        reward = self._get_reward()
        logger.info(f"reward: {reward}")

        return observation, reward

    def _get_observation(self):
        """Extract observation from environment state.

        Returns:
            observation: The observation for the agent
        """
        obs = self._b(z=_LAMBDA).evaluate()["g"]
        return obs

    def _get_reward(self) -> float:
        """Calculate reward based on environment state.

        Returns:
            reward: Scalar reward value for the agent
        """
        return self._flow.grid_average("Nu_bottom")

    def compute_integral(self, b0):
        """
        Compute the integral of the wall buoyancy perturbation after applying a tanh transformation.

        Args:
            b0: Reference buoyancy value to compare against the current observation.
        Returns:
            integral: Scalar value representing the integrated perturbation across the x-direction.
        """
        self._b_perturbation["g"] = np.tanh(_BETA * (self._get_observation() - b0))
        integral = d3.Integrate(self._b_perturbation, "x").evaluate()["g"].item()
        return integral

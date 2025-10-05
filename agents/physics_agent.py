"""Physics agent for solving physics problems with self-improvement."""

import numpy as np
import sympy as sp
from typing import Any, Dict
from agents.base_agent import BaseAgent, Strategy


class PhysicsAgent(BaseAgent):
    """Agent for physics computations with strategy evolution."""

    # Physical constants
    CONSTANTS = {
        'c': 299792458,  # Speed of light (m/s)
        'G': 6.67430e-11,  # Gravitational constant
        'h': 6.62607015e-34,  # Planck constant
        'k_B': 1.380649e-23,  # Boltzmann constant
        'e': 1.602176634e-19,  # Elementary charge
        'm_e': 9.1093837015e-31,  # Electron mass
        'm_p': 1.67262192369e-27,  # Proton mass
        'epsilon_0': 8.8541878128e-12,  # Vacuum permittivity
        'mu_0': 1.25663706212e-6,  # Vacuum permeability
    }

    def __init__(self, name: str = "PhysicsAgent", memory_path: str = "memory"):
        super().__init__(name, memory_path)

        # Initialize physics solving strategies
        self.create_strategy("analytical", {"method": "symbolic", "use_approximations": False})
        self.create_strategy("numerical_simulation", {"method": "numeric", "time_steps": 1000})
        self.create_strategy("perturbation", {"method": "perturbative", "order": 2})

    def execute_task(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Execute physics task using selected strategy."""
        problem_type = task.get("type", "mechanics")

        if problem_type == "mechanics":
            return self._solve_mechanics(task, strategy)
        elif problem_type == "electromagnetism":
            return self._solve_electromagnetism(task, strategy)
        elif problem_type == "thermodynamics":
            return self._solve_thermodynamics(task, strategy)
        elif problem_type == "quantum":
            return self._solve_quantum(task, strategy)
        elif problem_type == "relativity":
            return self._solve_relativity(task, strategy)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    def _solve_mechanics(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Solve classical mechanics problems."""
        subtype = task.get("subtype", "kinematics")

        if subtype == "kinematics":
            return self._kinematics(task, strategy)
        elif subtype == "dynamics":
            return self._dynamics(task, strategy)
        elif subtype == "orbital":
            return self._orbital_mechanics(task, strategy)

    def _kinematics(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Solve kinematics problems."""
        # Extract parameters
        initial_position = task.get("initial_position", 0)
        initial_velocity = task.get("initial_velocity", 0)
        acceleration = task.get("acceleration", 0)
        time = task.get("time", None)
        final_position = task.get("final_position", None)

        if strategy.parameters["method"] == "symbolic":
            t = sp.Symbol('t', positive=True, real=True)
            x0, v0, a = sp.symbols('x0 v0 a', real=True)

            # Kinematic equations
            position = x0 + v0*t + sp.Rational(1,2)*a*t**2
            velocity = v0 + a*t

            result = {
                "position": str(position),
                "velocity": str(velocity),
                "method": "analytical"
            }

            # Substitute known values
            if time is not None:
                pos_val = position.subs([(x0, initial_position), (v0, initial_velocity), (a, acceleration), (t, time)])
                vel_val = velocity.subs([(v0, initial_velocity), (a, acceleration), (t, time)])

                result["position_at_time"] = float(pos_val)
                result["velocity_at_time"] = float(vel_val)

            return result

        elif strategy.parameters["method"] == "numeric":
            # Numerical integration
            steps = strategy.parameters.get("time_steps", 1000)
            t_array = np.linspace(0, time, steps)
            dt = t_array[1] - t_array[0]

            positions = np.zeros(steps)
            velocities = np.zeros(steps)

            positions[0] = initial_position
            velocities[0] = initial_velocity

            for i in range(1, steps):
                velocities[i] = velocities[i-1] + acceleration * dt
                positions[i] = positions[i-1] + velocities[i-1] * dt + 0.5 * acceleration * dt**2

            return {
                "time": t_array.tolist(),
                "position": positions.tolist(),
                "velocity": velocities.tolist(),
                "method": "numerical",
                "final_position": float(positions[-1]),
                "final_velocity": float(velocities[-1])
            }

    def _dynamics(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Solve dynamics problems (F=ma)."""
        force = task.get("force")
        mass = task.get("mass")
        time = task.get("time")
        initial_velocity = task.get("initial_velocity", 0)

        # Calculate acceleration
        acceleration = force / mass

        # Use kinematics to find final state
        final_velocity = initial_velocity + acceleration * time
        displacement = initial_velocity * time + 0.5 * acceleration * time**2

        return {
            "acceleration": float(acceleration),
            "final_velocity": float(final_velocity),
            "displacement": float(displacement),
            "method": "Newton's second law"
        }

    def _orbital_mechanics(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Solve orbital mechanics problems."""
        central_mass = task.get("central_mass")  # kg
        satellite_mass = task.get("satellite_mass", 1)  # kg
        orbital_radius = task.get("orbital_radius")  # m

        G = self.CONSTANTS['G']

        # Orbital velocity
        v_orbital = np.sqrt(G * central_mass / orbital_radius)

        # Orbital period
        T = 2 * np.pi * np.sqrt(orbital_radius**3 / (G * central_mass))

        # Orbital energy
        E_kinetic = 0.5 * satellite_mass * v_orbital**2
        E_potential = -G * central_mass * satellite_mass / orbital_radius
        E_total = E_kinetic + E_potential

        return {
            "orbital_velocity": float(v_orbital),
            "orbital_period": float(T),
            "kinetic_energy": float(E_kinetic),
            "potential_energy": float(E_potential),
            "total_energy": float(E_total),
            "method": "Kepler's laws"
        }

    def _solve_electromagnetism(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Solve electromagnetism problems."""
        subtype = task.get("subtype", "electric_field")

        if subtype == "electric_field":
            charge = task.get("charge")
            distance = task.get("distance")
            k = 1 / (4 * np.pi * self.CONSTANTS['epsilon_0'])

            E = k * charge / distance**2

            return {
                "electric_field": float(E),
                "method": "Coulomb's law"
            }

        elif subtype == "magnetic_field":
            current = task.get("current")
            distance = task.get("distance")
            mu_0 = self.CONSTANTS['mu_0']

            # Field around long straight wire
            B = (mu_0 * current) / (2 * np.pi * distance)

            return {
                "magnetic_field": float(B),
                "method": "Ampere's law"
            }

    def _solve_thermodynamics(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Solve thermodynamics problems."""
        subtype = task.get("subtype", "ideal_gas")

        if subtype == "ideal_gas":
            # PV = nRT
            P = task.get("pressure", None)
            V = task.get("volume", None)
            n = task.get("moles", None)
            T = task.get("temperature", None)
            R = 8.314  # J/(mol·K)

            # Solve for missing variable
            if P is None:
                P = (n * R * T) / V
                return {"pressure": float(P), "method": "ideal gas law"}
            elif V is None:
                V = (n * R * T) / P
                return {"volume": float(V), "method": "ideal gas law"}
            elif T is None:
                T = (P * V) / (n * R)
                return {"temperature": float(T), "method": "ideal gas law"}
            elif n is None:
                n = (P * V) / (R * T)
                return {"moles": float(n), "method": "ideal gas law"}

        elif subtype == "entropy":
            T_hot = task.get("T_hot")
            T_cold = task.get("T_cold")

            # Carnot efficiency
            efficiency = 1 - (T_cold / T_hot)

            return {
                "carnot_efficiency": float(efficiency),
                "method": "Carnot cycle"
            }

    def _solve_quantum(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Solve quantum mechanics problems."""
        subtype = task.get("subtype", "particle_in_box")

        if subtype == "particle_in_box":
            n = task.get("quantum_number", 1)
            L = task.get("box_length")
            m = task.get("mass", self.CONSTANTS['m_e'])
            h = self.CONSTANTS['h']

            # Energy levels
            E_n = (n**2 * h**2) / (8 * m * L**2)

            return {
                "energy_level": float(E_n),
                "quantum_number": n,
                "method": "Schrödinger equation"
            }

        elif subtype == "photoelectric":
            frequency = task.get("frequency")
            work_function = task.get("work_function")
            h = self.CONSTANTS['h']

            # Photoelectric effect
            E_photon = h * frequency
            KE_max = E_photon - work_function

            return {
                "photon_energy": float(E_photon),
                "max_kinetic_energy": float(KE_max),
                "method": "Einstein's photoelectric equation"
            }

    def _solve_relativity(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Solve special relativity problems."""
        velocity = task.get("velocity")
        c = self.CONSTANTS['c']

        # Lorentz factor
        gamma = 1 / np.sqrt(1 - (velocity/c)**2)

        # Time dilation
        proper_time = task.get("proper_time", 1)
        dilated_time = gamma * proper_time

        # Length contraction
        proper_length = task.get("proper_length", 1)
        contracted_length = proper_length / gamma

        return {
            "lorentz_factor": float(gamma),
            "time_dilation": float(dilated_time),
            "length_contraction": float(contracted_length),
            "method": "special relativity"
        }

    def evaluate_result(self, task: Dict[str, Any], output: Any) -> Dict[str, float]:
        """Evaluate the quality of the physics solution."""
        metrics = {}

        # Check if solution exists and has physical meaning
        if output:
            metrics["found_solution"] = 1.0

            # Check for physical consistency
            # (e.g., positive energies where expected, velocity < c, etc.)
            consistency_checks = 0
            total_checks = 0

            if "velocity" in output:
                total_checks += 1
                if abs(output["velocity"]) < self.CONSTANTS['c']:
                    consistency_checks += 1

            if "energy" in output or "kinetic_energy" in output:
                total_checks += 1
                energy = output.get("energy", output.get("kinetic_energy"))
                if energy >= 0:
                    consistency_checks += 1

            if total_checks > 0:
                metrics["physical_consistency"] = consistency_checks / total_checks
            else:
                metrics["physical_consistency"] = 1.0

            metrics["accuracy"] = metrics["physical_consistency"]
        else:
            metrics["found_solution"] = 0.0
            metrics["accuracy"] = 0.0

        return metrics

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default strategy parameters."""
        return {
            "method": "symbolic",
            "use_approximations": False,
            "time_steps": 1000
        }

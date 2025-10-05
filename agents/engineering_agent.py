"""Engineering agent for structural and systems engineering problems."""

import numpy as np
from typing import Any, Dict
from agents.base_agent import BaseAgent, Strategy


class EngineeringAgent(BaseAgent):
    """Agent for engineering computations with self-improvement."""

    def __init__(self, name: str = "EngineeringAgent", memory_path: str = "memory"):
        super().__init__(name, memory_path)

        # Initialize engineering strategies
        self.create_strategy("analytical", {"method": "closed_form", "safety_factor": 1.5})
        self.create_strategy("numerical", {"method": "fem_simulation", "mesh_density": "medium"})
        self.create_strategy("empirical", {"method": "design_codes", "standard": "AISC"})

    def execute_task(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Execute engineering task using selected strategy."""
        problem_type = task.get("type", "structural")

        if problem_type == "beam":
            return self._analyze_beam(task, strategy)
        elif problem_type == "truss":
            return self._analyze_truss(task, strategy)
        elif problem_type == "column":
            return self._analyze_column(task, strategy)
        elif problem_type == "thermal":
            return self._analyze_thermal(task, strategy)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    def _analyze_beam(self, task: Dict[str, Any], strategy: Strategy) -> Dict[str, Any]:
        """Analyze beam under loading."""
        load = task.get("load")  # N
        length = task.get("length")  # m
        moment_of_inertia = task.get("moment_of_inertia", 1e-6)  # m^4
        elastic_modulus = task.get("elastic_modulus", 200e9)  # Pa (steel)

        # Simply supported beam with central point load
        max_deflection = (load * length**3) / (48 * elastic_modulus * moment_of_inertia)
        max_moment = (load * length) / 4
        max_stress = max_moment * (task.get("height", 0.1) / 2) / moment_of_inertia

        # Apply safety factor
        safety_factor = strategy.parameters.get("safety_factor", 1.5)
        allowable_stress = 250e6 / safety_factor  # Steel yield strength

        return {
            "max_deflection_m": float(max_deflection),
            "max_moment_Nm": float(max_moment),
            "max_stress_Pa": float(max_stress),
            "allowable_stress_Pa": float(allowable_stress),
            "factor_of_safety": float(allowable_stress / max_stress) if max_stress > 0 else float('inf'),
            "safe": bool(max_stress < allowable_stress),
            "method": strategy.name
        }

    def _analyze_truss(self, task: Dict[str, Any], strategy: Strategy) -> Dict[str, Any]:
        """Analyze truss structure."""
        # Simple 2-bar truss example
        force = task.get("force", 1000)  # N
        angle = task.get("angle", 45)  # degrees
        area = task.get("cross_section_area", 0.001)  # m^2

        angle_rad = np.radians(angle)

        # Member forces
        force_member = force / (2 * np.cos(angle_rad))

        # Stress in members
        stress = force_member / area

        safety_factor = strategy.parameters.get("safety_factor", 1.5)
        allowable_stress = 250e6 / safety_factor

        return {
            "member_force_N": float(force_member),
            "member_stress_Pa": float(stress),
            "allowable_stress_Pa": float(allowable_stress),
            "factor_of_safety": float(allowable_stress / stress) if stress > 0 else float('inf'),
            "safe": bool(stress < allowable_stress),
            "method": strategy.name
        }

    def _analyze_column(self, task: Dict[str, Any], strategy: Strategy) -> Dict[str, Any]:
        """Analyze column buckling (Euler's formula)."""
        length = task.get("length")  # m
        moment_of_inertia = task.get("moment_of_inertia", 1e-6)  # m^4
        elastic_modulus = task.get("elastic_modulus", 200e9)  # Pa
        end_condition = task.get("end_condition", "pinned-pinned")  # K factor

        # Effective length factors
        k_factors = {
            "pinned-pinned": 1.0,
            "fixed-fixed": 0.5,
            "fixed-pinned": 0.7,
            "fixed-free": 2.0
        }

        k = k_factors.get(end_condition, 1.0)

        # Euler buckling load
        critical_load = (np.pi**2 * elastic_modulus * moment_of_inertia) / (k * length)**2

        safety_factor = strategy.parameters.get("safety_factor", 1.5)
        allowable_load = critical_load / safety_factor

        applied_load = task.get("applied_load", 0)

        return {
            "critical_load_N": float(critical_load),
            "allowable_load_N": float(allowable_load),
            "factor_of_safety": float(critical_load / applied_load) if applied_load > 0 else float('inf'),
            "safe": bool(applied_load < allowable_load) if applied_load > 0 else True,
            "k_factor": float(k),
            "method": "Euler buckling"
        }

    def _analyze_thermal(self, task: Dict[str, Any], strategy: Strategy) -> Dict[str, Any]:
        """Analyze thermal expansion."""
        length = task.get("length")  # m
        temp_change = task.get("temperature_change")  # °C
        alpha = task.get("thermal_expansion_coeff", 12e-6)  # 1/°C (steel)

        # Thermal strain and expansion
        thermal_strain = alpha * temp_change
        expansion = length * thermal_strain

        # If constrained, calculate thermal stress
        elastic_modulus = task.get("elastic_modulus", 200e9)  # Pa
        constrained = task.get("constrained", False)

        if constrained:
            thermal_stress = elastic_modulus * thermal_strain
        else:
            thermal_stress = 0

        return {
            "thermal_strain": float(thermal_strain),
            "expansion_m": float(expansion),
            "thermal_stress_Pa": float(thermal_stress),
            "constrained": constrained,
            "method": "thermal expansion"
        }

    def evaluate_result(self, task: Dict[str, Any], output: Any) -> Dict[str, float]:
        """Evaluate the quality of engineering solution."""
        metrics = {}

        if output:
            metrics["found_solution"] = 1.0

            # Check for engineering safety
            if "safe" in output:
                metrics["safety_check"] = 1.0 if output["safe"] else 0.5

            # Check for reasonable factor of safety
            if "factor_of_safety" in output:
                fos = output["factor_of_safety"]
                if 1.5 <= fos <= 10.0:  # Reasonable range
                    metrics["fos_reasonable"] = 1.0
                elif fos > 1.0:
                    metrics["fos_reasonable"] = 0.7
                else:
                    metrics["fos_reasonable"] = 0.0

            # Overall accuracy based on checks
            metrics["accuracy"] = np.mean([
                metrics.get("safety_check", 0.5),
                metrics.get("fos_reasonable", 0.5)
            ])
        else:
            metrics["found_solution"] = 0.0
            metrics["accuracy"] = 0.0

        return metrics

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default strategy parameters."""
        return {
            "method": "closed_form",
            "safety_factor": 1.5
        }

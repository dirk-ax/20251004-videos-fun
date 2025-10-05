"""Specialized FDTD agent for electromagnetic simulations using Tidy3D."""

import numpy as np
import argparse
from typing import Dict, Any, List
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base_agent import BaseAgent, Strategy

try:
    import tidy3d as td
    from tidy3d import web
    TIDY3D_AVAILABLE = True
except ImportError:
    TIDY3D_AVAILABLE = False
    print("Warning: Tidy3D not installed. Install with: pip install tidy3d")


class FDTDAgent(BaseAgent):
    """
    Specialized agent for FDTD electromagnetic simulations.

    Learns from FDTD101 tutorials and example library:
    - https://www.flexcompute.com/tidy3d/learning-center/fdtd101/
    - https://www.flexcompute.com/tidy3d/learning-center/example-library
    """

    def __init__(self, name: str = "FDTDAgent", memory_path: str = "memory"):
        super().__init__(name, memory_path)

        # Initialize with FDTD-specific strategies
        self.create_strategy("standard_fdtd", {
            "grid_refinement": "auto",
            "pml_layers": 12,
            "wavelength_points": 10
        })
        self.create_strategy("high_accuracy", {
            "grid_refinement": "fine",
            "pml_layers": 20,
            "wavelength_points": 20
        })
        self.create_strategy("fast_preview", {
            "grid_refinement": "coarse",
            "pml_layers": 8,
            "wavelength_points": 5
        })

    def execute_task(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Execute FDTD simulation task."""
        if not TIDY3D_AVAILABLE:
            return {
                "error": "Tidy3D not available",
                "suggestion": "pip install tidy3d"
            }

        simulation_type = task.get("type", "waveguide")

        if simulation_type == "waveguide":
            return self._simulate_waveguide(task, strategy)
        elif simulation_type == "photonic_crystal":
            return self._simulate_photonic_crystal(task, strategy)
        elif simulation_type == "ring_resonator":
            return self._simulate_ring_resonator(task, strategy)
        elif simulation_type == "mzi":
            return self._simulate_mzi(task, strategy)
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")

    def _simulate_waveguide(self, task: Dict[str, Any], strategy: Strategy) -> Dict[str, Any]:
        """
        Create Tidy3D waveguide simulation configuration.

        ⚠️ WARNING: This creates a simulation object but does NOT run it.
        To run actual FDTD, you need:
        1. Tidy3D API key configured
        2. Submit to cloud: web.Job(simulation=sim).run()
        3. Wait for electromagnetic solver to complete

        This function returns CONFIGURATION ONLY, not simulation results.
        """

        # Extract parameters
        width = task.get("width", 0.5)  # μm
        thickness = task.get("thickness", 0.22)  # μm
        wavelength = task.get("wavelength", 1.55)  # μm
        length = task.get("length", 10)  # μm

        # Strategy parameters
        pml_layers = strategy.parameters["pml_layers"]
        wl_points = strategy.parameters["wavelength_points"]

        # Define materials (silicon waveguide in SiO2)
        si = td.Medium(permittivity=3.48**2)
        sio2 = td.Medium(permittivity=1.45**2)

        # Create waveguide geometry
        waveguide = td.Structure(
            geometry=td.Box(
                center=(0, 0, 0),
                size=(td.inf, width, thickness)
            ),
            medium=si
        )

        # Create substrate
        substrate = td.Structure(
            geometry=td.Box(
                center=(0, 0, -2),
                size=(td.inf, td.inf, 4)
            ),
            medium=sio2
        )

        # Define source
        mode_source = td.ModeSource(
            center=(-length/2, 0, 0),
            size=(0, 4, 4),
            source_time=td.GaussianPulse(
                freq0=td.C_0 / wavelength,
                fwidth=td.C_0 / wavelength / 10
            ),
            direction='+',
            mode_index=0
        )

        # Define monitor
        mode_monitor = td.ModeMonitor(
            center=(length/2, 0, 0),
            size=(0, 4, 4),
            freqs=[td.C_0 / wavelength],
            mode_spec=td.ModeSpec(num_modes=1),
            name="transmission"
        )

        field_monitor = td.FieldMonitor(
            center=(0, 0, 0),
            size=(length, 4, 4),
            freqs=[td.C_0 / wavelength],
            name="fields"
        )

        # Create simulation
        sim = td.Simulation(
            size=(length + 2, 6, 6),
            grid_spec=td.GridSpec.auto(wavelength=wavelength, min_steps_per_wvl=wl_points),
            structures=[substrate, waveguide],
            sources=[mode_source],
            monitors=[mode_monitor, field_monitor],
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(
                boundary=td.PML(num_layers=pml_layers)
            )
        )

        # ⚠️ IMPORTANT: This returns config only, NOT simulation results!
        # To get real results, use: web.Job(simulation=sim).run()
        return {
            "simulation_type": "waveguide_config",
            "status": "NOT_RUN",  # Honest status
            "warning": "Configuration only - no FDTD computation performed",
            "parameters": {
                "width": width,
                "thickness": thickness,
                "wavelength": wavelength,
                "length": length
            },
            "tidy3d_simulation_object": sim,  # The actual sim object
            "grid_points": sim.grid.num_cells,
            "to_run": "Use web.Job(simulation=result['tidy3d_simulation_object']).run()",
        }

    def _simulate_ring_resonator(self, task: Dict[str, Any], strategy: Strategy) -> Dict[str, Any]:
        """
        ❌ NOT IMPLEMENTED - Returns fake analytical data.

        This function does NOT run FDTD. It returns made-up resonance values.
        To implement: Create Tidy3D ring resonator geometry and run actual simulation.
        """
        radius = task.get("radius", 5)  # μm
        gap = task.get("gap", 0.2)  # μm
        wavelength = task.get("wavelength", 1.55)  # μm

        return {
            "status": "FAKE_DATA",
            "warning": "❌ NOT REAL FDTD - Analytical approximation only",
            "simulation_type": "ring_resonator_analytical",
            "parameters": {
                "radius": radius,
                "gap": gap,
                "wavelength": wavelength
            },
            "resonant_wavelengths": [wavelength * (1 + i * 0.01) for i in range(-2, 3)],  # FAKE
            "Q_factor": 10000,  # FAKE - hardcoded
            "FSR": 0.1,  # FAKE - not calculated from geometry
            "note": "Replace this with real Tidy3D ring resonator simulation"
        }

    def _simulate_mzi(self, task: Dict[str, Any], strategy: Strategy) -> Dict[str, Any]:
        """
        ⚠️ ANALYTICAL ONLY - NOT FDTD simulation.

        This uses simple phase calculation, not electromagnetic field solving.
        Does not account for: coupling losses, mode mismatch, waveguide dispersion, etc.
        """
        arm_length_diff = task.get("arm_length_diff", 10)  # μm
        wavelength = task.get("wavelength", 1.55)  # μm

        # Calculate extinction ratio (simplified analytical model)
        phase_diff = 2 * np.pi * arm_length_diff / wavelength
        extinction_ratio = 20 * np.log10(np.abs(np.cos(phase_diff / 2)))

        return {
            "status": "ANALYTICAL",
            "warning": "⚠️ Simple phase model - NOT full FDTD",
            "simulation_type": "mzi_analytical",
            "parameters": {
                "arm_length_diff": arm_length_diff,
                "wavelength": wavelength
            },
            "extinction_ratio_dB": extinction_ratio,  # From simple formula
            "phase_difference": phase_diff,
            "note": "Does not include coupling, dispersion, or field effects"
        }

    def _simulate_photonic_crystal(self, task: Dict[str, Any], strategy: Strategy) -> Dict[str, Any]:
        """
        ❌ COMPLETELY FAKE - Returns hardcoded bandgap values.

        Real photonic crystal simulation requires:
        - 3D periodic structure geometry
        - Band structure calculation
        - Bloch mode analysis
        - Hours of computation time

        This function just returns made-up numbers.
        """
        lattice_constant = task.get("lattice_constant", 0.4)  # μm
        hole_radius = task.get("hole_radius", 0.12)  # μm

        return {
            "status": "FAKE_DATA",
            "warning": "❌ COMPLETELY FAKE - Hardcoded values, NO computation",
            "simulation_type": "photonic_crystal_fake",
            "parameters": {
                "lattice_constant": lattice_constant,
                "hole_radius": hole_radius
            },
            "bandgap_center": 1.55,  # HARDCODED - not calculated!
            "bandgap_width": 0.2,  # HARDCODED - not calculated!
            "note": "This is a placeholder. Requires MPB or FDTD band structure solver."
        }

    def evaluate_result(self, task: Dict[str, Any], output: Any) -> Dict[str, float]:
        """Evaluate simulation quality."""
        metrics = {}

        if output and "config_valid" in output:
            metrics["found_solution"] = 1.0
            metrics["accuracy"] = 1.0
        elif output and "simulation" in output:
            metrics["found_solution"] = 1.0
            metrics["accuracy"] = 0.9  # Analytical approximation
        else:
            metrics["found_solution"] = 0.0
            metrics["accuracy"] = 0.0

        return metrics

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default FDTD parameters."""
        return {
            "grid_refinement": "auto",
            "pml_layers": 12,
            "wavelength_points": 10
        }


def main():
    """Execute FDTD agent as specialized contractor."""
    parser = argparse.ArgumentParser(description="FDTD Simulation Agent")
    parser.add_argument("--contract", type=str, default="main",
                       help="GitHub contract/PR number")
    parser.add_argument("--task", type=str, default="waveguide",
                       help="Simulation task type")

    args = parser.parse_args()

    # Create agent
    agent = FDTDAgent()

    # Define tasks based on FDTD101 curriculum
    tasks = [
        {
            "type": "waveguide",
            "width": 0.5,
            "thickness": 0.22,
            "wavelength": 1.55,
            "description": "Silicon Strip Waveguide"
        },
        {
            "type": "ring_resonator",
            "radius": 5,
            "gap": 0.2,
            "wavelength": 1.55,
            "description": "Ring Resonator"
        },
        {
            "type": "mzi",
            "arm_length_diff": 10,
            "wavelength": 1.55,
            "description": "Mach-Zehnder Interferometer"
        }
    ]

    results = []

    for task in tasks:
        print(f"\n=== Simulating: {task['description']} ===")
        result = agent.execute_with_learning(task)

        if result.success:
            print(f"✓ Success in {result.execution_time:.3f}s")
            print(f"  Output: {result.output}")
        else:
            print(f"✗ Failed: {result.error}")

        results.append({
            "task": task["description"],
            "success": result.success,
            "output": result.output,
            "execution_time": result.execution_time
        })

    # Save results for CI/CD
    output_dir = Path("outputs/fdtd")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"contract_{args.contract}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ FDTD agent completed {len(results)} tasks")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

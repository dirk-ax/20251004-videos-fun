"""Rule-based tests for agent functionality."""

import pytest
import numpy as np
from agents.math_agent import MathAgent
from agents.physics_agent import PhysicsAgent
from agents.specialized.fdtd_agent import FDTDAgent


class TestMathAgent:
    """Test suite for MathAgent with rule-based validation."""

    def test_quadratic_equation(self):
        """Test solving simple quadratic equation."""
        agent = MathAgent()

        task = {
            "type": "equation",
            "equation": "x**2 - 4",
            "variable": "x"
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should solve quadratic equation"
        assert "solutions" in result.output, "Output should contain solutions"
        assert len(result.output["solutions"]) == 2, "Quadratic should have 2 solutions"

        # Verify solutions
        solutions = [float(s) for s in result.output["solutions"]]
        assert 2 in solutions or -2 in solutions, "Solutions should include ±2"

    def test_cubic_equation(self):
        """Test solving cubic equation."""
        agent = MathAgent()

        task = {
            "type": "equation",
            "equation": "x**3 - 6*x**2 + 11*x - 6",
            "variable": "x"
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should solve cubic equation"
        assert len(result.output["solutions"]) == 3, "Cubic should have 3 solutions"

    def test_integration(self):
        """Test definite integration."""
        agent = MathAgent()

        task = {
            "type": "integration",
            "integrand": "x**2",
            "variable": "x",
            "limits": (0, 1)
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should compute integral"
        assert "result" in result.output, "Output should contain result"

        # ∫₀¹ x² dx = 1/3
        numeric_result = result.output.get("numeric_value")
        if numeric_result is not None:
            assert abs(numeric_result - 1/3) < 0.01, f"Integral should be 1/3, got {numeric_result}"

    def test_strategy_learning(self):
        """Test that agent learns from multiple tasks."""
        agent = MathAgent()

        # Execute multiple similar tasks
        for i in range(5):
            task = {
                "type": "equation",
                "equation": f"x**2 - {i+1}",
                "variable": "x"
            }
            result = agent.execute_with_learning(task)
            assert result.success

        # Agent should have learned strategies
        assert len(agent.strategies) > 0, "Agent should have developed strategies"
        assert agent.task_history, "Agent should have task history"

        # Performance should improve or stay consistent
        perf = agent.get_performance_summary()
        assert perf["success_rate"] >= 0.8, "Success rate should be high"

    def test_optimization(self):
        """Test optimization problem solving."""
        agent = MathAgent()

        task = {
            "type": "optimization",
            "objective": "x**2 + y**2",
            "bounds": [(-5, 5), (-5, 5)]
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should solve optimization"
        assert "optimal_point" in result.output

        # Optimal point should be near (0, 0)
        optimal = result.output["optimal_point"]
        assert abs(optimal[0]) < 0.1 and abs(optimal[1]) < 0.1


class TestPhysicsAgent:
    """Test suite for PhysicsAgent."""

    def test_kinematics(self):
        """Test kinematics calculation."""
        agent = PhysicsAgent()

        task = {
            "type": "mechanics",
            "subtype": "kinematics",
            "initial_position": 0,
            "initial_velocity": 10,
            "acceleration": -9.8,
            "time": 1.0
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should solve kinematics"
        assert "position_at_time" in result.output or "final_position" in result.output

    def test_orbital_mechanics(self):
        """Test orbital mechanics calculation."""
        agent = PhysicsAgent()

        task = {
            "type": "mechanics",
            "subtype": "orbital",
            "central_mass": 5.972e24,  # Earth
            "orbital_radius": 6.771e6  # 400km altitude
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should calculate orbital parameters"
        assert "orbital_velocity" in result.output
        assert "orbital_period" in result.output

        # ISS orbital velocity should be ~7.7 km/s
        v = result.output["orbital_velocity"]
        assert 7000 < v < 8000, f"Orbital velocity should be ~7.7km/s, got {v/1000:.1f}km/s"

    def test_quantum_mechanics(self):
        """Test quantum mechanics calculation."""
        agent = PhysicsAgent()

        task = {
            "type": "quantum",
            "subtype": "particle_in_box",
            "quantum_number": 1,
            "box_length": 1e-9
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should solve quantum problem"
        assert "energy_level" in result.output
        assert result.output["energy_level"] > 0, "Energy should be positive"

    def test_relativity(self):
        """Test special relativity."""
        agent = PhysicsAgent()

        task = {
            "type": "relativity",
            "velocity": 0.5 * 299792458,  # 0.5c
            "proper_time": 1.0
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should calculate relativistic effects"
        assert "lorentz_factor" in result.output
        assert result.output["lorentz_factor"] > 1, "Lorentz factor should be > 1"

    def test_physical_consistency(self):
        """Test that physics results are physically consistent."""
        agent = PhysicsAgent()

        task = {
            "type": "mechanics",
            "subtype": "dynamics",
            "force": 100,
            "mass": 10,
            "time": 1.0,
            "initial_velocity": 0
        }

        result = agent.execute_with_learning(task)

        assert result.success
        assert result.metrics.get("physical_consistency", 0) > 0.5


class TestFDTDAgent:
    """Test suite for FDTD specialized agent."""

    def test_waveguide_simulation(self):
        """Test waveguide simulation setup."""
        agent = FDTDAgent()

        task = {
            "type": "waveguide",
            "width": 0.5,
            "thickness": 0.22,
            "wavelength": 1.55
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should configure waveguide simulation"
        assert result.output["simulation"] == "waveguide"
        assert "grid_points" in result.output or "parameters" in result.output

    def test_ring_resonator(self):
        """Test ring resonator simulation."""
        agent = FDTDAgent()

        task = {
            "type": "ring_resonator",
            "radius": 5,
            "gap": 0.2,
            "wavelength": 1.55
        }

        result = agent.execute_with_learning(task)

        assert result.success
        assert "Q_factor" in result.output
        assert result.output["Q_factor"] > 0

    def test_mzi(self):
        """Test Mach-Zehnder Interferometer."""
        agent = FDTDAgent()

        task = {
            "type": "mzi",
            "arm_length_diff": 10,
            "wavelength": 1.55
        }

        result = agent.execute_with_learning(task)

        assert result.success
        assert "extinction_ratio_dB" in result.output

    def test_strategy_evolution(self):
        """Test that FDTD agent evolves strategies."""
        agent = FDTDAgent()

        # Run multiple simulations
        for accuracy in ["fast_preview", "standard_fdtd", "high_accuracy"]:
            task = {
                "type": "waveguide",
                "width": 0.5,
                "wavelength": 1.55
            }
            result = agent.execute_with_learning(task)
            assert result.success

        assert len(agent.strategies) >= 3, "Agent should have multiple strategies"


class TestWorkflowOrchestration:
    """Test workflow orchestration and self-improvement."""

    def test_workflow_execution(self):
        """Test basic workflow execution."""
        from workflows.orchestrator import WorkflowOrchestrator, WorkflowBuilder

        orchestrator = WorkflowOrchestrator()

        # Register agents
        math_agent = MathAgent()
        orchestrator.register_agent(math_agent)

        # Build simple workflow
        workflow = WorkflowBuilder()
        workflow.add_step(
            "solve_equation",
            "MathAgent",
            {"type": "equation", "equation": "x**2 - 4", "variable": "x"},
            dependencies=[]
        )

        steps = workflow.build()
        results = orchestrator.execute_workflow(steps)

        assert len(results) == 1
        assert results["solve_equation"].success

    def test_multi_agent_workflow(self):
        """Test workflow with multiple agents."""
        from workflows.orchestrator import WorkflowOrchestrator, WorkflowBuilder

        orchestrator = WorkflowOrchestrator()

        # Register agents
        math_agent = MathAgent()
        physics_agent = PhysicsAgent()
        orchestrator.register_agent(math_agent)
        orchestrator.register_agent(physics_agent)

        # Build workflow
        workflow = WorkflowBuilder()
        workflow.add_step(
            "math_task",
            "MathAgent",
            {"type": "equation", "equation": "x**2 - 9", "variable": "x"},
            dependencies=[]
        )
        workflow.add_step(
            "physics_task",
            "PhysicsAgent",
            {
                "type": "mechanics",
                "subtype": "kinematics",
                "initial_velocity": 10,
                "acceleration": -9.8,
                "time": 1.0
            },
            dependencies=["math_task"]
        )

        steps = workflow.build()
        results = orchestrator.execute_workflow(steps)

        assert len(results) == 2
        assert all(r.success for r in results.values())


@pytest.fixture
def cleanup_memory():
    """Clean up memory files after tests."""
    yield
    import shutil
    from pathlib import Path

    memory_path = Path("memory")
    if memory_path.exists():
        shutil.rmtree(memory_path)


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for high-RAM, high-GPU system."""

    def test_large_scale_simulation(self):
        """Test handling large-scale problems with 512GB RAM."""
        agent = MathAgent()

        # Generate large problem set
        tasks = [
            {
                "type": "equation",
                "equation": f"x**2 - {i}",
                "variable": "x"
            }
            for i in range(100)  # 100 tasks
        ]

        successful = 0
        for task in tasks:
            result = agent.execute_with_learning(task)
            if result.success:
                successful += 1

        assert successful >= 95, "Should handle large scale with high success rate"

    def test_parallel_agent_execution(self):
        """Test parallel execution capabilities."""
        from concurrent.futures import ThreadPoolExecutor

        agents = [MathAgent(f"MathAgent_{i}") for i in range(10)]

        def run_task(agent):
            task = {
                "type": "equation",
                "equation": "x**2 - 16",
                "variable": "x"
            }
            return agent.execute_with_learning(task)

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(run_task, agents))

        assert all(r.success for r in results), "All parallel tasks should succeed"

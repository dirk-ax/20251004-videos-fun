"""Tests for EngineeringAgent."""

import pytest
from agents.engineering_agent import EngineeringAgent


class TestEngineeringAgent:
    """Test suite for EngineeringAgent."""

    def test_beam_analysis(self):
        """Test beam deflection and stress analysis."""
        agent = EngineeringAgent()

        task = {
            "type": "beam",
            "load": 1000,  # N
            "length": 1.0,  # m
            "moment_of_inertia": 1e-6,  # m^4
            "elastic_modulus": 200e9,  # Pa
            "height": 0.1  # m
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should analyze beam"
        assert "max_deflection_m" in result.output
        assert "max_stress_Pa" in result.output
        assert "safe" in result.output
        assert result.output["max_deflection_m"] > 0, "Deflection should be positive"

    def test_truss_analysis(self):
        """Test truss force analysis."""
        agent = EngineeringAgent()

        task = {
            "type": "truss",
            "force": 1000,
            "angle": 45,
            "cross_section_area": 0.001
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should analyze truss"
        assert "member_force_N" in result.output
        assert "member_stress_Pa" in result.output
        assert result.output["member_force_N"] > 0

    def test_column_buckling(self):
        """Test column buckling analysis."""
        agent = EngineeringAgent()

        task = {
            "type": "column",
            "length": 3.0,  # m
            "moment_of_inertia": 1e-6,
            "elastic_modulus": 200e9,
            "end_condition": "pinned-pinned",
            "applied_load": 5000
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should analyze column"
        assert "critical_load_N" in result.output
        assert "k_factor" in result.output
        assert result.output["critical_load_N"] > 0
        assert result.output["k_factor"] == 1.0  # Pinned-pinned

    def test_thermal_expansion(self):
        """Test thermal expansion calculation."""
        agent = EngineeringAgent()

        task = {
            "type": "thermal",
            "length": 10.0,  # m
            "temperature_change": 50,  # °C
            "thermal_expansion_coeff": 12e-6,
            "constrained": False
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should analyze thermal expansion"
        assert "expansion_m" in result.output
        assert "thermal_strain" in result.output
        assert result.output["expansion_m"] > 0

    def test_safety_factor(self):
        """Test that safety factors are properly applied."""
        agent = EngineeringAgent()

        task = {
            "type": "beam",
            "load": 100,  # Reduced load for safety
            "length": 1.0,
            "moment_of_inertia": 1e-6,
            "height": 0.1
        }

        result = agent.execute_with_learning(task)

        assert result.success
        assert "factor_of_safety" in result.output
        assert result.output["factor_of_safety"] > 1.0, f"FOS should be > 1.0 for safety, got {result.output['factor_of_safety']}"

    def test_self_improvement(self):
        """Test that agent learns from multiple tasks."""
        agent = EngineeringAgent()

        initial_tasks = len(agent.task_history)

        # Run multiple beam analyses
        for i in range(5):
            task = {
                "type": "beam",
                "load": 1000 * (i + 1),
                "length": 1.0,
                "moment_of_inertia": 1e-6,
                "height": 0.1
            }
            result = agent.execute_with_learning(task)
            assert result.success

        # Check learning
        perf = agent.get_performance_summary()
        assert perf["total_tasks"] == initial_tasks + 5
        assert perf["success_rate"] == 1.0
        assert len(agent.strategies) >= 3  # Should have initial strategies

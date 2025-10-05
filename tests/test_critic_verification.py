#!/usr/bin/env python3
"""
Critic Agent Verification Tests - Demands CONCRETE PROOF of functionality.

These tests require ACTUAL DEMONSTRATIONS, not hand-wavy claims.
"""

import pytest
import sys
import json
import subprocess
from pathlib import Path
import importlib.util


class TestFDTDVerification:
    """Tests that require ACTUAL FDTD simulation proof.

    NOTE: FDTD Agent has been removed from the codebase because it only
    returned fake/hardcoded data. These tests are marked as skipped until
    a real FDTD implementation is added.
    """

    @pytest.mark.skip(reason="Meep MCP not yet integrated - FDTD agent removed due to fake implementations")
    def test_meep_mcp_installation_verification(self):
        """CRITICAL: Verify Meep MCP is actually installed and accessible."""
        # This test DEMANDS proof that Meep MCP is installed
        try:
            # Check if MCP server is actually running
            result = subprocess.run(
                ['uvx', '--from', 'axiomatic-mcp', 'meep', '--help'],
                capture_output=True, text=True, timeout=30
            )
            
            assert result.returncode == 0, f"Meep MCP not accessible: {result.stderr}"
            assert "meep" in result.stdout.lower() or "mcp" in result.stdout.lower(), \
                "Meep MCP help output doesn't contain expected content"
            
            print("✅ PROOF: Meep MCP is actually installed and accessible")
            
        except subprocess.TimeoutExpired:
            pytest.fail("❌ BLOCKING: Meep MCP server not responding - NOT INSTALLED")
        except FileNotFoundError:
            pytest.fail("❌ BLOCKING: uvx not found - MCP system not properly set up")
        except Exception as e:
            pytest.fail(f"❌ BLOCKING: Meep MCP verification failed: {e}")

    @pytest.mark.skip(reason="Meep MCP not yet integrated - FDTD agent removed due to fake implementations")
    def test_actual_meep_simulation_execution(self):
        """CRITICAL: Execute ACTUAL Meep simulation via MCP."""
        # This test DEMANDS an actual Meep simulation run
        try:
            # Create a simple waveguide simulation
            simulation_script = """
import meep as mp
import numpy as np

# Create a simple waveguide simulation
cell = mp.Vector3(16, 8, 0)
geometry = [mp.Block(mp.Vector3(mp.inf, 1, mp.inf),
                     center=mp.Vector3(0, 0, 0),
                     material=mp.Medium(epsilon=12))]

sources = [mp.Source(mp.ContinuousSource(wavelength=1.55),
                     component=mp.Ez,
                     center=mp.Vector3(-7, 0, 0))]

pml_layers = [mp.PML(1.0)]
resolution = 20

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

# Run simulation for a few time steps
sim.run(until=50)

# Get field data as proof
field_data = sim.get_array(center=mp.Vector3(0, 0, 0), 
                          size=mp.Vector3(2, 2, 0), 
                          component=mp.Ez)

# Verify we got actual field data
assert field_data is not None, "No field data returned from simulation"
assert field_data.size > 0, "Empty field data from simulation"
assert not np.all(field_data == 0), "All-zero field data - simulation didn't run"

print(f"✅ PROOF: Actual Meep simulation executed, field data shape: {field_data.shape}")
print(f"✅ PROOF: Field data range: {field_data.min():.6f} to {field_data.max():.6f}")

return {"success": True, "field_shape": field_data.shape, "field_range": [float(field_data.min()), float(field_data.max())]}

"""
            
            # Execute the simulation
            result = subprocess.run(
                ['python3', '-c', simulation_script],
                capture_output=True, text=True, timeout=60
            )
            
            assert result.returncode == 0, f"Meep simulation failed: {result.stderr}"
            assert "PROOF: Actual Meep simulation executed" in result.stdout, \
                "Simulation didn't produce expected proof output"
            
            # Parse the proof data
            proof_line = [line for line in result.stdout.split('\n') if 'field_shape' in line][0]
            print(f"✅ VERIFIED: {proof_line}")
            
        except subprocess.TimeoutExpired:
            pytest.fail("❌ BLOCKING: Meep simulation timed out - NOT WORKING")
        except Exception as e:
            pytest.fail(f"❌ BLOCKING: Meep simulation verification failed: {e}")

    @pytest.mark.skip(reason="FDTD agent removed from codebase due to fake implementations")
    def test_fdtd_agent_actual_simulation_integration(self):
        """CRITICAL: FDTD Agent must execute ACTUAL Meep simulation."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        try:
            from agents.specialized.fdtd_agent import FDTDAgent
            
            agent = FDTDAgent()
            
            # Test with a task that DEMANDS actual simulation
            task = {
                "type": "waveguide",
                "width": 0.5,
                "thickness": 0.22,
                "wavelength": 1.55,
                "require_actual_simulation": True  # This flag DEMANDS real simulation
            }
            
            result = agent.execute_with_learning(task)
            
            # CRITICAL VERIFICATION POINTS
            assert result.success, f"FDTD agent failed: {result.error}"
            
            # Must have actual simulation results, not fake parameters
            assert "simulation_results" in result.output, \
                "❌ BLOCKING: No simulation_results in output - still returning fake data"
            
            assert "field_data" in result.output, \
                "❌ BLOCKING: No field_data in output - no actual electromagnetic simulation"
            
            assert "transmission_spectrum" in result.output, \
                "❌ BLOCKING: No transmission_spectrum - no actual FDTD analysis"
            
            # Verify the data is actually computed, not hardcoded
            field_data = result.output["field_data"]
            assert len(field_data) > 100, \
                "❌ BLOCKING: Field data too small - likely fake/hardcoded"
            
            # Check for realistic field values (not all zeros or identical values)
            import numpy as np
            field_array = np.array(field_data)
            assert not np.all(field_array == 0), \
                "❌ BLOCKING: All-zero field data - simulation didn't run"
            assert np.std(field_array) > 1e-6, \
                "❌ BLOCKING: Field data has no variation - likely fake"
            
            print("✅ PROOF: FDTD Agent executed ACTUAL Meep simulation")
            print(f"✅ PROOF: Field data points: {len(field_data)}")
            print(f"✅ PROOF: Field data std: {np.std(field_array):.6f}")
            
        except ImportError as e:
            pytest.fail(f"❌ BLOCKING: Cannot import FDTD agent: {e}")
        except Exception as e:
            pytest.fail(f"❌ BLOCKING: FDTD agent verification failed: {e}")


class TestHybridStrategyVerification:
    """Tests that require ACTUAL hybrid strategy proof."""
    
    def test_numeric_strategy_actual_usage(self):
        """CRITICAL: Numeric method must be ACTUALLY USED, not just exist."""
        sys.path.insert(0, str(Path(__file__).parent.parent))

        try:
            from agents.math_agent import MathAgent

            agent = MathAgent()

            # Verify numeric method exists
            numeric_strategy = None
            for sid, strategy in agent.strategies.items():
                if strategy.name == "numeric":
                    numeric_strategy = strategy
                    break

            assert numeric_strategy is not None, \
                "❌ BLOCKING: Numeric strategy doesn't exist"

            # Execute task that SHOULD use numeric method
            task = {
                "type": "equation",
                "equation": "x**5 - 32",  # Complex equation that should prefer numeric
                "variable": "x",
                "force_numeric": True  # Force numeric method
            }

            result = agent.execute_with_learning(task)

            assert result.success, f"Math agent failed: {result.error}"
            assert result.output.get("method") == "numeric", \
                f"❌ BLOCKING: Method was {result.output.get('method')}, not 'numeric'"

            # PROOF: The numeric method was actually executed and returned results
            assert "solutions" in result.output, \
                "❌ BLOCKING: No solutions returned from numeric method"
            assert len(result.output["solutions"]) > 0, \
                "❌ BLOCKING: Numeric method returned empty solutions"

            print("✅ PROOF: Numeric method was ACTUALLY USED")
            print(f"✅ PROOF: Numeric solutions found: {len(result.output['solutions'])}")
            print(f"✅ PROOF: Solutions: {result.output['solutions']}")

        except Exception as e:
            pytest.fail(f"❌ BLOCKING: Numeric method verification failed: {e}")
    
    def test_hybrid_strategy_actual_hybrid_behavior(self):
        """CRITICAL: Hybrid strategy must ACTUALLY use both methods."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        try:
            from agents.math_agent import MathAgent
            
            agent = MathAgent()
            
            # Test multiple equations to verify hybrid behavior
            test_cases = [
                {"equation": "x**2 - 4", "expected_method": "symbolic"},
                {"equation": "x**7 - 128", "expected_method": "numeric"},
                {"equation": "sin(x) - 0.5", "expected_method": "numeric"},
            ]
            
            methods_used = []
            
            for test_case in test_cases:
                task = {
                    "type": "equation",
                    "equation": test_case["equation"],
                    "variable": "x"
                }
                
                result = agent.execute_with_learning(task)
                assert result.success, f"Failed on equation: {test_case['equation']}"
                
                method = result.output.get("method")
                methods_used.append(method)
                print(f"Equation: {test_case['equation']} -> Method: {method}")
            
            # CRITICAL: Must use BOTH methods
            unique_methods = set(methods_used)
            assert len(unique_methods) > 1, \
                f"❌ BLOCKING: Only used methods: {unique_methods} - not hybrid!"
            
            assert "symbolic" in unique_methods, \
                "❌ BLOCKING: Never used symbolic method"
            assert "numeric" in unique_methods, \
                "❌ BLOCKING: Never used numeric method"
            
            print("✅ PROOF: Hybrid strategy ACTUALLY used both methods")
            print(f"✅ PROOF: Methods used: {unique_methods}")
            
        except Exception as e:
            pytest.fail(f"❌ BLOCKING: Hybrid strategy verification failed: {e}")


class TestWorkflowOrchestratorVerification:
    """Tests that require ACTUAL workflow orchestration proof."""
    
    def test_agent_registration_verification(self):
        """CRITICAL: Workflow orchestrator must have ACTUALLY REGISTERED agents."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        try:
            from workflows.orchestrator import WorkflowOrchestrator
            from agents.math_agent import MathAgent
            from agents.physics_agent import PhysicsAgent
            
            orchestrator = WorkflowOrchestrator()
            
            # Register agents
            math_agent = MathAgent()
            physics_agent = PhysicsAgent()
            
            orchestrator.register_agent(math_agent)
            orchestrator.register_agent(physics_agent)
            
            # CRITICAL VERIFICATION
            assert len(orchestrator.agents) > 0, \
                "❌ BLOCKING: No agents registered in orchestrator"
            
            assert len(orchestrator.agents) >= 2, \
                f"❌ BLOCKING: Only {len(orchestrator.agents)} agents registered, need at least 2"
            
            # Verify agents are actually accessible
            agent_names = [agent.name for agent in orchestrator.agents.values()]
            assert "MathAgent" in agent_names, \
                "❌ BLOCKING: MathAgent not registered"
            assert "PhysicsAgent" in agent_names, \
                "❌ BLOCKING: PhysicsAgent not registered"
            
            print("✅ PROOF: Workflow orchestrator has ACTUALLY REGISTERED agents")
            print(f"✅ PROOF: Registered agents: {agent_names}")
            
        except Exception as e:
            pytest.fail(f"❌ BLOCKING: Agent registration verification failed: {e}")
    
    def test_actual_workflow_execution(self):
        """CRITICAL: Must execute ACTUAL multi-agent workflow."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        try:
            from workflows.orchestrator import WorkflowOrchestrator, WorkflowBuilder
            from agents.math_agent import MathAgent
            from agents.physics_agent import PhysicsAgent
            
            orchestrator = WorkflowOrchestrator()
            
            # Register agents
            math_agent = MathAgent()
            physics_agent = PhysicsAgent()
            orchestrator.register_agent(math_agent)
            orchestrator.register_agent(physics_agent)
            
            # Build workflow
            workflow = WorkflowBuilder()
            workflow.add_step(
                "solve_equation",
                "MathAgent",
                {"type": "equation", "equation": "x**2 - 16", "variable": "x"},
                dependencies=[]
            )
            workflow.add_step(
                "compute_physics",
                "PhysicsAgent",
                {
                    "type": "mechanics",
                    "subtype": "kinematics",
                    "initial_velocity": 10,
                    "acceleration": -9.8,
                    "time": 1.0
                },
                dependencies=["solve_equation"]
            )
            
            steps = workflow.build()
            
            # Execute workflow
            results = orchestrator.execute_workflow(steps)
            
            # CRITICAL VERIFICATION
            assert len(results) == 2, \
                f"❌ BLOCKING: Expected 2 workflow results, got {len(results)}"
            
            assert "solve_equation" in results, \
                "❌ BLOCKING: Math step not executed"
            assert "compute_physics" in results, \
                "❌ BLOCKING: Physics step not executed"
            
            assert results["solve_equation"].success, \
                "❌ BLOCKING: Math step failed"
            assert results["compute_physics"].success, \
                "❌ BLOCKING: Physics step failed"
            
            # Verify dependency worked
            math_output = results["solve_equation"].output
            physics_output = results["compute_physics"].output
            
            assert "solutions" in math_output, \
                "❌ BLOCKING: Math step didn't produce solutions"
            assert "position_at_time" in physics_output or "final_position" in physics_output, \
                "❌ BLOCKING: Physics step didn't produce position data"
            
            print("✅ PROOF: ACTUAL multi-agent workflow executed successfully")
            print(f"✅ PROOF: Math solutions: {math_output.get('solutions')}")
            print(f"✅ PROOF: Physics result keys: {list(physics_output.keys())}")
            
        except Exception as e:
            pytest.fail(f"❌ BLOCKING: Workflow execution verification failed: {e}")


class TestMCPIntegrationVerification:
    """Tests that require ACTUAL MCP integration proof.

    NOTE: MCP integration tests are skipped until the infrastructure is set up.
    """

    @pytest.mark.skip(reason="MCP infrastructure not yet configured")
    def test_axiomatic_mcp_actual_connection(self):
        """CRITICAL: Must establish ACTUAL connection to Axiomatic MCP servers."""
        try:
            # Test actual MCP server connection
            result = subprocess.run(
                ['uvx', '--from', 'axiomatic-mcp', 'all', '--help'],
                capture_output=True, text=True, timeout=30
            )
            
            assert result.returncode == 0, \
                f"❌ BLOCKING: Cannot connect to Axiomatic MCP: {result.stderr}"
            
            # Verify we get actual MCP server information
            output = result.stdout.lower()
            expected_servers = ['equation', 'document', 'photonics', 'plot']
            
            found_servers = [server for server in expected_servers if server in output]
            assert len(found_servers) > 0, \
                f"❌ BLOCKING: No expected MCP servers found in output: {output[:200]}"
            
            print("✅ PROOF: ACTUAL connection to Axiomatic MCP servers established")
            print(f"✅ PROOF: Found MCP servers: {found_servers}")
            
        except subprocess.TimeoutExpired:
            pytest.fail("❌ BLOCKING: MCP server connection timed out - NOT WORKING")
        except Exception as e:
            pytest.fail(f"❌ BLOCKING: MCP connection verification failed: {e}")

    @pytest.mark.skip(reason="MCP infrastructure not yet configured")
    def test_actual_mcp_tool_usage(self):
        """CRITICAL: Must demonstrate ACTUAL usage of MCP tools."""
        try:
            # Test actual MCP tool execution
            test_command = """
import subprocess
import json

# Test AxEquationExplorer
result = subprocess.run([
    'uvx', '--from', 'axiomatic-mcp', 'equation-explorer',
    '--equation', 'x^2 + y^2 = r^2',
    '--action', 'parse'
], capture_output=True, text=True, timeout=30)

if result.returncode == 0:
    print("✅ PROOF: AxEquationExplorer MCP tool executed successfully")
    print(f"Output: {result.stdout[:200]}")
else:
    print(f"❌ BLOCKING: AxEquationExplorer failed: {result.stderr}")
    exit(1)
"""
            
            result = subprocess.run(
                ['python3', '-c', test_command],
                capture_output=True, text=True, timeout=60
            )
            
            assert result.returncode == 0, \
                f"❌ BLOCKING: MCP tool usage failed: {result.stderr}"
            
            assert "PROOF: AxEquationExplorer MCP tool executed successfully" in result.stdout, \
                "❌ BLOCKING: MCP tool didn't execute successfully"
            
            print("✅ PROOF: ACTUAL MCP tool usage demonstrated")
            
        except Exception as e:
            pytest.fail(f"❌ BLOCKING: MCP tool usage verification failed: {e}")


if __name__ == "__main__":
    # Run all verification tests
    pytest.main([__file__, "-v", "--tb=short"])

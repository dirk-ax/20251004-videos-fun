#!/usr/bin/env python3
"""
Self-Improving AI Agentic Workflows Demo
Showcasing mathematical, physical, and engineering computations with learning.
"""

import sys
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.math_agent import MathAgent
from agents.physics_agent import PhysicsAgent
from workflows.orchestrator import WorkflowOrchestrator, WorkflowBuilder
from visualizations.renderer import MathVisualizer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box
import time


console = Console()


def print_banner():
    """Print impressive banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     🤖 SELF-IMPROVING AI AGENTIC WORKFLOWS 🤖                ║
    ║                                                               ║
    ║         Mathematical • Physical • Engineering                 ║
    ║                  Computations with Learning                   ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def demo_math_agent():
    """Demonstrate math agent with self-improvement."""
    console.print("\n[bold yellow]═══ MATH AGENT DEMONSTRATION ═══[/bold yellow]\n")

    agent = MathAgent()
    visualizer = MathVisualizer()

    # Test problems of increasing difficulty
    problems = [
        {
            "type": "equation",
            "equation": "x**2 - 4",
            "variable": "x",
            "description": "Simple Quadratic"
        },
        {
            "type": "equation",
            "equation": "x**3 - 6*x**2 + 11*x - 6",
            "variable": "x",
            "description": "Cubic Equation"
        },
        {
            "type": "optimization",
            "objective": "x**2 + y**2",
            "bounds": [(-5, 5), (-5, 5)],
            "description": "Optimization Problem"
        },
        {
            "type": "integration",
            "integrand": "x**2 * sin(x)",
            "variable": "x",
            "limits": (0, np.pi),
            "description": "Definite Integral"
        }
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Solving problems...", total=len(problems))

        for i, problem in enumerate(problems, 1):
            progress.update(task, description=f"[cyan]Problem {i}/{len(problems)}: {problem['description']}")

            result = agent.execute_with_learning(problem)

            time.sleep(0.5)  # Dramatic effect

            # Display result
            if result.success:
                console.print(f"  ✓ [green]{problem['description']}[/green]: Solved in {result.execution_time:.3f}s")
                console.print(f"    [dim]Accuracy: {result.metrics.get('accuracy', 0):.1%}[/dim]")

                # Visualize if equation
                if problem["type"] == "equation" and "solutions" in result.output:
                    visualizer.visualize_equation_solution(
                        problem["equation"],
                        result.output.get("symbolic_form", result.output["solutions"]),
                        problem["variable"],
                        f"Solution: {problem['description']}"
                    )
            else:
                console.print(f"  ✗ [red]{problem['description']}[/red]: Failed - {result.error}")

            progress.advance(task)

    # Show agent performance
    perf = agent.get_performance_summary()

    table = Table(title="Math Agent Performance", box=box.ROUNDED, style="cyan")
    table.add_column("Metric", style="yellow")
    table.add_column("Value", style="green")

    table.add_row("Total Tasks", str(perf["total_tasks"]))
    table.add_row("Success Rate", f"{perf['success_rate']:.1%}")
    table.add_row("Avg Time", f"{perf['avg_execution_time']:.3f}s")
    table.add_row("Total Strategies", str(perf["total_strategies"]))
    table.add_row("Best Strategy", perf["best_strategy"])

    console.print("\n")
    console.print(table)

    return agent


def demo_physics_agent():
    """Demonstrate physics agent."""
    console.print("\n[bold yellow]═══ PHYSICS AGENT DEMONSTRATION ═══[/bold yellow]\n")

    agent = PhysicsAgent()
    visualizer = MathVisualizer()

    problems = [
        {
            "type": "mechanics",
            "subtype": "kinematics",
            "initial_position": 0,
            "initial_velocity": 10,
            "acceleration": -9.8,
            "time": 2.0,
            "description": "Projectile Motion"
        },
        {
            "type": "mechanics",
            "subtype": "orbital",
            "central_mass": 5.972e24,  # Earth mass
            "orbital_radius": 6.371e6 + 400e3,  # ISS orbit
            "description": "ISS Orbital Mechanics"
        },
        {
            "type": "quantum",
            "subtype": "particle_in_box",
            "quantum_number": 1,
            "box_length": 1e-9,
            "description": "Quantum Particle in Box"
        },
        {
            "type": "relativity",
            "velocity": 0.5 * 299792458,  # 0.5c
            "proper_time": 1.0,
            "proper_length": 1.0,
            "description": "Special Relativity"
        }
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Solving physics problems...", total=len(problems))

        for i, problem in enumerate(problems, 1):
            progress.update(task, description=f"[cyan]Problem {i}/{len(problems)}: {problem['description']}")

            result = agent.execute_with_learning(problem)

            time.sleep(0.5)

            if result.success:
                console.print(f"  ✓ [green]{problem['description']}[/green]: Solved in {result.execution_time:.3f}s")
                console.print(f"    [dim]Method: {result.output.get('method', 'N/A')}[/dim]")

                # Visualize kinematics
                if problem.get("subtype") == "kinematics" and "time" in result.output:
                    visualizer.visualize_physics_simulation(
                        np.array(result.output["time"]),
                        np.array(result.output["position"]),
                        np.array(result.output["velocity"]),
                        problem["description"]
                    )
            else:
                console.print(f"  ✗ [red]{problem['description']}[/red]: Failed")

            progress.advance(task)

    # Show agent performance
    perf = agent.get_performance_summary()

    table = Table(title="Physics Agent Performance", box=box.ROUNDED, style="cyan")
    table.add_column("Metric", style="yellow")
    table.add_column("Value", style="green")

    table.add_row("Total Tasks", str(perf["total_tasks"]))
    table.add_row("Success Rate", f"{perf['success_rate']:.1%}")
    table.add_row("Avg Time", f"{perf['avg_execution_time']:.3f}s")
    table.add_row("Total Strategies", str(perf["total_strategies"]))

    console.print("\n")
    console.print(table)

    return agent


def demo_self_improving_workflow():
    """Demonstrate self-improving workflow."""
    console.print("\n[bold yellow]═══ SELF-IMPROVING WORKFLOW ═══[/bold yellow]\n")

    # Create orchestrator
    orchestrator = WorkflowOrchestrator()

    # Create agents
    math_agent = MathAgent()
    physics_agent = PhysicsAgent()

    # Register agents
    orchestrator.register_agent(math_agent)
    orchestrator.register_agent(physics_agent)

    # Build workflow: Solve a complex physics problem that requires math
    workflow = WorkflowBuilder()
    workflow.add_step(
        "solve_trajectory_equation",
        "MathAgent",
        {
            "type": "differential",
            "equation": "Derivative(y(x), x, x) + y(x)",
            "function": "y",
            "variable": "x"
        },
        dependencies=[]
    )
    workflow.add_step(
        "compute_orbital_params",
        "PhysicsAgent",
        {
            "type": "mechanics",
            "subtype": "orbital",
            "central_mass": 5.972e24,
            "orbital_radius": 6.771e6
        },
        dependencies=["solve_trajectory_equation"]
    )
    workflow.add_step(
        "optimize_trajectory",
        "MathAgent",
        {
            "type": "optimization",
            "objective": "x**2 + y**2",
            "bounds": [(-10, 10), (-10, 10)]
        },
        dependencies=["compute_orbital_params"]
    )

    steps = workflow.build()

    # Run self-improving loop
    console.print("[cyan]Starting self-improvement iterations...[/cyan]\n")

    performance_history = orchestrator.create_feedback_loop(
        "Multi-Agent Physics Workflow",
        steps,
        iterations=5,
        improvement_threshold=0.05
    )

    # Visualize improvement
    visualizer = MathVisualizer()
    visualizer.visualize_workflow_progress(performance_history, "Self-Improving Workflow Performance")

    # Final statistics
    stats = orchestrator.get_workflow_statistics()

    table = Table(title="Workflow Statistics", box=box.ROUNDED, style="cyan")
    table.add_column("Metric", style="yellow")
    table.add_column("Value", style="green")

    table.add_row("Total Workflows", str(stats["total_workflows"]))
    table.add_row("Success Rate", f"{stats['success_rate']:.1%}")
    table.add_row("Avg Execution Time", f"{stats['avg_execution_time']:.3f}s")

    console.print("\n")
    console.print(table)


def main():
    """Main demo function."""
    print_banner()

    console.print("\n[bold]Choose demonstration:[/bold]")
    console.print("  1. Math Agent")
    console.print("  2. Physics Agent")
    console.print("  3. Self-Improving Workflow (recommended)")
    console.print("  4. All demonstrations")

    choice = console.input("\n[bold cyan]Enter choice (1-4): [/bold cyan]")

    try:
        choice = int(choice)

        if choice == 1:
            demo_math_agent()
        elif choice == 2:
            demo_physics_agent()
        elif choice == 3:
            demo_self_improving_workflow()
        elif choice == 4:
            demo_math_agent()
            demo_physics_agent()
            demo_self_improving_workflow()
        else:
            console.print("[red]Invalid choice![/red]")
            return

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
        return
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return

    console.print("\n[bold green]✨ Demonstration complete! ✨[/bold green]")
    console.print("\n[dim]Check the 'outputs/' directory for visualizations.[/dim]\n")


if __name__ == "__main__":
    main()

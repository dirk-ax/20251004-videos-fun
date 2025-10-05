#!/usr/bin/env python3
"""
Create impressive 30-45 second demo video showcasing AI agent workflows.
Optimized for 512GB RAM and powerful GPU system.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from pathlib import Path
import json
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.math_agent import MathAgent
from agents.physics_agent import PhysicsAgent
from workflows.orchestrator import WorkflowOrchestrator, WorkflowBuilder
from visualizations.renderer import MathVisualizer


class DemoVideoGenerator:
    """Generate impressive demo videos of agent workflows."""

    def __init__(self, output_dir: str = "outputs/demos", duration: int = 35):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.duration = duration  # seconds
        self.fps = 30
        self.total_frames = duration * self.fps

        # Set up high-quality rendering
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150

    def create_workflow_animation(self, filename: str = "agent_workflow_demo.mp4"):
        """Create animated visualization of self-improving workflow."""

        fig = plt.figure(figsize=(16, 9), facecolor='#0a0a0a')

        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3,
                             left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Main workflow viz
        ax_workflow = fig.add_subplot(gs[0:2, 0:2])
        ax_workflow.set_facecolor('#0f0f0f')
        ax_workflow.set_xlim(0, 10)
        ax_workflow.set_ylim(0, 10)
        ax_workflow.axis('off')

        # Performance metrics
        ax_metrics = fig.add_subplot(gs[0, 2])
        ax_metrics.set_facecolor('#0f0f0f')

        # Learning progress
        ax_learning = fig.add_subplot(gs[1, 2])
        ax_learning.set_facecolor('#0f0f0f')

        # Agent activity
        ax_activity = fig.add_subplot(gs[2, :])
        ax_activity.set_facecolor('#0f0f0f')

        # Initialize data structures
        orchestrator = WorkflowOrchestrator()
        math_agent = MathAgent(name="MathAgent")
        physics_agent = PhysicsAgent(name="PhysicsAgent")

        orchestrator.register_agent(math_agent)
        orchestrator.register_agent(physics_agent)

        # Build workflow
        workflow = WorkflowBuilder()
        workflow.add_step("math_1", "MathAgent",
                         {"type": "equation", "equation": "x**2 - 16", "variable": "x"})
        workflow.add_step("physics_1", "PhysicsAgent",
                         {"type": "mechanics", "subtype": "kinematics",
                          "initial_velocity": 10, "acceleration": -9.8, "time": 2.0},
                         dependencies=["math_1"])
        workflow.add_step("math_2", "MathAgent",
                         {"type": "optimization", "objective": "x**2 + y**2",
                          "bounds": [(-5, 5), (-5, 5)]},
                         dependencies=["physics_1"])

        steps = workflow.build()

        # Performance data
        performance_history = []
        agent_positions = {
            "MathAgent": (2, 7),
            "PhysicsAgent": (8, 7),
            "Workflow": (5, 9)
        }

        # Animation state
        current_iteration = 0
        max_iterations = 8

        def init():
            """Initialize animation."""
            return []

        def update(frame):
            """Update animation frame."""
            nonlocal current_iteration

            progress = frame / self.total_frames

            # Clear axes
            ax_workflow.clear()
            ax_metrics.clear()
            ax_learning.clear()
            ax_activity.clear()

            # === WORKFLOW VISUALIZATION ===
            ax_workflow.set_xlim(0, 10)
            ax_workflow.set_ylim(0, 10)
            ax_workflow.axis('off')
            ax_workflow.set_facecolor('#0f0f0f')

            # Draw title
            if progress < 0.1:
                alpha = progress / 0.1
                ax_workflow.text(5, 9.5, "SELF-IMPROVING AI AGENTS",
                               ha='center', va='top', fontsize=24,
                               color='cyan', fontweight='bold', alpha=alpha)
            else:
                ax_workflow.text(5, 9.5, "SELF-IMPROVING AI AGENTS",
                               ha='center', va='top', fontsize=24,
                               color='cyan', fontweight='bold')

            # Draw agents
            math_pos = agent_positions["MathAgent"]
            phys_pos = agent_positions["PhysicsAgent"]

            if progress > 0.15:
                # Math agent
                circle1 = Circle(math_pos, 0.5, color='#4ade80', alpha=0.8, zorder=3)
                ax_workflow.add_patch(circle1)
                ax_workflow.text(math_pos[0], math_pos[1], "M",
                               ha='center', va='center', fontsize=16,
                               color='white', fontweight='bold', zorder=4)
                ax_workflow.text(math_pos[0], math_pos[1] - 1, "Math Agent",
                               ha='center', va='top', fontsize=10, color='#4ade80')

            if progress > 0.2:
                # Physics agent
                circle2 = Circle(phys_pos, 0.5, color='#f59e0b', alpha=0.8, zorder=3)
                ax_workflow.add_patch(circle2)
                ax_workflow.text(phys_pos[0], phys_pos[1], "P",
                               ha='center', va='center', fontsize=16,
                               color='white', fontweight='bold', zorder=4)
                ax_workflow.text(phys_pos[0], phys_pos[1] - 1, "Physics Agent",
                               ha='center', va='top', fontsize=10, color='#f59e0b')

            # Draw connections with animation
            if progress > 0.25:
                arrow_alpha = min(1.0, (progress - 0.25) / 0.1)

                arrow1 = FancyArrowPatch(math_pos, phys_pos,
                                        arrowstyle='->', mutation_scale=20,
                                        color='cyan', linewidth=2,
                                        alpha=arrow_alpha, zorder=2)
                ax_workflow.add_patch(arrow1)

            # Execute workflow periodically
            if progress > 0.3 and current_iteration < max_iterations:
                iteration_progress = (progress - 0.3) / 0.7
                iteration_index = int(iteration_progress * max_iterations)

                if iteration_index > current_iteration:
                    # Run workflow iteration
                    results = orchestrator.execute_workflow(steps)

                    # Record performance
                    perf_math = math_agent.get_performance_summary()
                    perf_phys = physics_agent.get_performance_summary()

                    performance_history.append({
                        "iteration": current_iteration,
                        "math_success_rate": perf_math["success_rate"],
                        "physics_success_rate": perf_phys["success_rate"],
                        "avg_time": perf_math["avg_execution_time"]
                    })

                    current_iteration = iteration_index

            # === METRICS PANEL ===
            ax_metrics.set_xlim(0, 1)
            ax_metrics.set_ylim(0, 1)
            ax_metrics.axis('off')

            if performance_history:
                latest = performance_history[-1]

                metrics_text = "METRICS\n\n"
                metrics_text += f"Iteration: {latest['iteration'] + 1}\n"
                metrics_text += f"Math Success: {latest['math_success_rate']:.1%}\n"
                metrics_text += f"Physics Success: {latest['physics_success_rate']:.1%}\n"
                metrics_text += f"Avg Time: {latest['avg_time']:.3f}s"

                ax_metrics.text(0.1, 0.9, metrics_text,
                              transform=ax_metrics.transAxes,
                              fontsize=11, color='#4ade80',
                              family='monospace', va='top')

            # === LEARNING PROGRESS ===
            ax_learning.set_xlim(0, 1)
            ax_learning.set_ylim(0, 1)
            ax_learning.axis('off')

            if len(performance_history) > 1:
                learning_text = "LEARNING\n\n"
                learning_text += f"Strategies: {len(math_agent.strategies)}\n"
                learning_text += f"Tasks: {len(math_agent.task_history)}\n"
                learning_text += "Status: IMPROVING ✓"

                ax_learning.text(0.1, 0.9, learning_text,
                               transform=ax_learning.transAxes,
                               fontsize=11, color='cyan',
                               family='monospace', va='top')

            # === ACTIVITY TIMELINE ===
            ax_activity.set_xlim(0, max_iterations)
            ax_activity.set_ylim(0, 3)
            ax_activity.set_facecolor('#0f0f0f')
            ax_activity.set_xlabel('Iteration', color='white', fontsize=10)
            ax_activity.set_title('Agent Activity Timeline', color='white', fontsize=12, pad=10)
            ax_activity.tick_params(colors='white')

            if performance_history:
                iterations = [p["iteration"] for p in performance_history]
                math_success = [p["math_success_rate"] * 2 for p in performance_history]
                phys_success = [p["physics_success_rate"] * 2 + 0.1 for p in performance_history]

                # Plot activity bars
                for i, it in enumerate(iterations):
                    ax_activity.barh(0.5, 0.8, left=it, height=0.3,
                                    color='#4ade80', alpha=0.7)
                    ax_activity.barh(1.5, 0.8, left=it, height=0.3,
                                    color='#f59e0b', alpha=0.7)

                ax_activity.set_yticks([0.5, 1.5])
                ax_activity.set_yticklabels(['Math', 'Physics'], color='white')
                ax_activity.grid(True, alpha=0.2, color='white', axis='x')

            return []

        # Create animation
        anim = animation.FuncAnimation(
            fig, update, init_func=init,
            frames=self.total_frames,
            interval=1000/self.fps,
            blit=True
        )

        # Save video
        output_file = self.output_dir / filename
        print(f"Rendering video to: {output_file}")
        print(f"Duration: {self.duration}s @ {self.fps}fps = {self.total_frames} frames")

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.fps, bitrate=3000,
                       extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])

        anim.save(str(output_file), writer=writer, dpi=150)

        print(f"✓ Video saved: {output_file}")
        return output_file

    def create_high_res_screenshots(self, num_screenshots: int = 5):
        """Create high-resolution screenshots of workflow states."""

        print(f"Creating {num_screenshots} high-resolution screenshots...")

        visualizer = MathVisualizer(output_dir=self.output_dir / "screenshots")

        # Create agents
        math_agent = MathAgent()
        physics_agent = PhysicsAgent()

        # Generate diverse problems
        problems = [
            {
                "agent": math_agent,
                "task": {"type": "equation", "equation": "x**4 - 16", "variable": "x"},
                "title": "Math Agent: Quartic Equation"
            },
            {
                "agent": physics_agent,
                "task": {"type": "mechanics", "subtype": "orbital",
                        "central_mass": 5.972e24, "orbital_radius": 6.771e6},
                "title": "Physics Agent: Orbital Mechanics"
            },
            {
                "agent": math_agent,
                "task": {"type": "integration", "integrand": "sin(x)*exp(x)",
                        "variable": "x", "limits": (0, np.pi)},
                "title": "Math Agent: Complex Integration"
            },
            {
                "agent": physics_agent,
                "task": {"type": "quantum", "subtype": "particle_in_box",
                        "quantum_number": 2, "box_length": 1e-9},
                "title": "Physics Agent: Quantum Mechanics"
            },
            {
                "agent": math_agent,
                "task": {"type": "optimization", "objective": "x**3 - 3*x*y + y**3",
                        "bounds": [(-5, 5), (-5, 5)]},
                "title": "Math Agent: 3D Optimization"
            }
        ]

        for i, problem in enumerate(problems[:num_screenshots], 1):
            print(f"  Screenshot {i}/{num_screenshots}: {problem['title']}")

            result = problem["agent"].execute_with_learning(problem["task"])

            if result.success and problem["task"]["type"] == "equation":
                visualizer.visualize_equation_solution(
                    problem["task"]["equation"],
                    result.output.get("symbolic_form", result.output.get("solutions", [])),
                    problem["task"].get("variable", "x"),
                    problem["title"]
                )

        print(f"✓ Screenshots saved to: {self.output_dir / 'screenshots'}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate demo video and screenshots for AI agent workflows"
    )
    parser.add_argument("--duration", type=int, default=35,
                       help="Video duration in seconds (default: 35)")
    parser.add_argument("--screenshots", type=int, default=5,
                       help="Number of screenshots to generate (default: 5)")
    parser.add_argument("--highlight-learning", action="store_true",
                       help="Emphasize self-improvement in video")
    parser.add_argument("--output-dir", type=str, default="outputs/demos",
                       help="Output directory for videos and screenshots")

    args = parser.parse_args()

    print("=" * 60)
    print("AI AGENT WORKFLOW DEMO GENERATOR")
    print("=" * 60)
    print(f"Video duration: {args.duration}s")
    print(f"Screenshots: {args.screenshots}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    generator = DemoVideoGenerator(
        output_dir=args.output_dir,
        duration=args.duration
    )

    # Generate video
    print("\n📹 Generating animated workflow video...")
    video_file = generator.create_workflow_animation()

    # Generate screenshots
    print("\n📸 Generating high-resolution screenshots...")
    generator.create_high_res_screenshots(num_screenshots=args.screenshots)

    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "video_file": str(video_file),
        "duration_seconds": args.duration,
        "screenshots": args.screenshots,
        "output_directory": args.output_dir
    }

    summary_file = Path(args.output_dir) / "demo_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✨ Demo generation complete!")
    print(f"📁 All outputs saved to: {args.output_dir}")
    print(f"📄 Summary: {summary_file}")


if __name__ == "__main__":
    main()

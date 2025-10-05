"""Advanced visualization system for mathematical and physical computations."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
from typing import Any, Dict, List
from pathlib import Path


class MathVisualizer:
    """Creates impressive visualizations for mathematical results."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use('dark_background')

    def visualize_equation_solution(self, equation: str, solutions: List[Any],
                                    variable: str = "x", title: str = "Equation Solution"):
        """Visualize equation and its solutions."""
        fig = plt.figure(figsize=(14, 8), facecolor='#0a0a0a')

        # Create grid for subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Plot the function
        ax1 = fig.add_subplot(gs[0, :])

        x = sp.Symbol(variable)
        expr = sp.sympify(equation)

        # Determine range
        numeric_solutions = []
        for sol in solutions:
            try:
                numeric_solutions.append(float(complex(sol).real))
            except:
                pass

        if numeric_solutions:
            x_min = min(numeric_solutions) - 5
            x_max = max(numeric_solutions) + 5
        else:
            x_min, x_max = -10, 10

        x_vals = np.linspace(x_min, x_max, 1000)

        # Convert to lambda
        f = sp.lambdify(x, expr, "numpy")

        try:
            y_vals = f(x_vals)

            # Plot function
            ax1.plot(x_vals, y_vals, 'cyan', linewidth=2.5, label=f'f({variable}) = {equation}')
            ax1.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
            ax1.grid(True, alpha=0.2, color='white')

            # Mark solutions
            for sol in numeric_solutions:
                ax1.plot(sol, 0, 'ro', markersize=12, zorder=5)
                ax1.annotate(f'{variable}={sol:.3f}',
                           xy=(sol, 0), xytext=(sol, -max(abs(y_vals))*0.15),
                           fontsize=11, color='#ff6b6b',
                           ha='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', edgecolor='#ff6b6b', alpha=0.8))

            ax1.set_xlabel(variable, fontsize=14, color='white')
            ax1.set_ylabel(f'f({variable})', fontsize=14, color='white')
            ax1.set_title(title, fontsize=16, color='white', fontweight='bold', pad=20)
            ax1.legend(fontsize=11, facecolor='#1a1a1a', edgecolor='cyan')
            ax1.set_facecolor('#0f0f0f')

        except Exception as e:
            ax1.text(0.5, 0.5, f"Complex function\n{equation}",
                    transform=ax1.transAxes, ha='center', va='center',
                    fontsize=14, color='cyan')
            ax1.set_facecolor('#0f0f0f')

        # 2. Solution details
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('off')

        solution_text = "Solutions:\n\n"
        for i, sol in enumerate(solutions, 1):
            solution_text += f"  {variable}_{i} = {sol}\n\n"

        ax2.text(0.1, 0.9, solution_text, transform=ax2.transAxes,
                fontsize=13, color='#4ade80', family='monospace',
                verticalalignment='top')

        # Add decorative box
        fancy_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                  boxstyle="round,pad=0.05",
                                  transform=ax2.transAxes,
                                  edgecolor='#4ade80', facecolor='none',
                                  linewidth=2, linestyle='--')
        ax2.add_patch(fancy_box)

        # 3. Convergence/iterations visualization
        ax3 = fig.add_subplot(gs[1, 1], projection='polar')

        if numeric_solutions:
            # Create a polar plot of solutions
            theta = np.linspace(0, 2*np.pi, len(numeric_solutions), endpoint=False)
            r = [abs(s) for s in numeric_solutions]

            ax3.scatter(theta, r, c=range(len(r)), cmap='plasma', s=200, alpha=0.8)

            for i, (t, radius) in enumerate(zip(theta, r)):
                ax3.plot([0, t], [0, radius], 'w-', alpha=0.3, linewidth=1.5)

            ax3.set_title('Solution Space', fontsize=12, color='white', pad=20)
            ax3.set_facecolor('#0f0f0f')
            ax3.grid(True, alpha=0.2, color='white')

        # Save
        output_file = self.output_dir / f"equation_solution_{int(np.random.random()*10000)}.png"
        plt.savefig(output_file, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
        print(f"Saved visualization: {output_file}")

        return fig

    def visualize_workflow_progress(self, performance_history: List[Dict[str, Any]],
                                   title: str = "Self-Improving Workflow"):
        """Create interactive visualization of workflow improvement."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Success Rate Over Time', 'Execution Time Improvement',
                          'Performance Heatmap', 'Improvement Velocity'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "heatmap"}, {"secondary_y": False}]]
        )

        iterations = [p["iteration"] for p in performance_history]
        success_rates = [p["success_rate"] * 100 for p in performance_history]
        exec_times = [p["avg_time"] for p in performance_history]

        # 1. Success rate line chart
        fig.add_trace(
            go.Scatter(x=iterations, y=success_rates,
                      mode='lines+markers',
                      name='Success Rate',
                      line=dict(color='#4ade80', width=3),
                      marker=dict(size=10, symbol='circle'),
                      fill='tozeroy',
                      fillcolor='rgba(74, 222, 128, 0.1)'),
            row=1, col=1
        )

        # 2. Execution time
        fig.add_trace(
            go.Scatter(x=iterations, y=exec_times,
                      mode='lines+markers',
                      name='Avg Execution Time',
                      line=dict(color='#f59e0b', width=3),
                      marker=dict(size=10, symbol='square'),
                      fill='tozeroy',
                      fillcolor='rgba(245, 158, 11, 0.1)'),
            row=1, col=2
        )

        # 3. Performance heatmap
        heatmap_data = np.array([success_rates, exec_times])
        fig.add_trace(
            go.Heatmap(z=heatmap_data,
                      x=iterations,
                      y=['Success Rate', 'Exec Time'],
                      colorscale='Viridis',
                      showscale=True),
            row=2, col=1
        )

        # 4. Improvement velocity
        if len(success_rates) > 1:
            velocity = np.diff(success_rates)
            fig.add_trace(
                go.Bar(x=iterations[1:], y=velocity,
                      name='Improvement Rate',
                      marker=dict(color=velocity,
                                colorscale='RdYlGn',
                                line=dict(color='white', width=1))),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=24, color='white')),
            template='plotly_dark',
            height=800,
            showlegend=True,
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0f0f0f',
            font=dict(family='Courier New', size=12)
        )

        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1)
        fig.update_xaxes(title_text="Iteration", row=1, col=2)
        fig.update_yaxes(title_text="Time (s)", row=1, col=2)
        fig.update_xaxes(title_text="Iteration", row=2, col=2)
        fig.update_yaxes(title_text="Δ Success Rate (%)", row=2, col=2)

        # Save
        output_file = self.output_dir / f"workflow_progress_{int(np.random.random()*10000)}.html"
        fig.write_html(output_file)
        print(f"Saved interactive visualization: {output_file}")

        return fig

    def visualize_physics_simulation(self, time: np.ndarray, position: np.ndarray,
                                     velocity: np.ndarray = None, title: str = "Physics Simulation"):
        """Visualize physics simulation results."""
        fig = plt.figure(figsize=(16, 10), facecolor='#0a0a0a')

        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

        # 1. Position vs Time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time, position, 'cyan', linewidth=2.5, label='Position')
        ax1.fill_between(time, position, alpha=0.3, color='cyan')
        ax1.grid(True, alpha=0.2, color='white')
        ax1.set_xlabel('Time (s)', fontsize=12, color='white')
        ax1.set_ylabel('Position (m)', fontsize=12, color='white')
        ax1.set_title(f'{title} - Position', fontsize=14, color='white', fontweight='bold')
        ax1.legend(fontsize=11, facecolor='#1a1a1a', edgecolor='cyan')
        ax1.set_facecolor('#0f0f0f')

        if velocity is not None:
            # 2. Velocity vs Time
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(time, velocity, '#f59e0b', linewidth=2.5, label='Velocity')
            ax2.fill_between(time, velocity, alpha=0.3, color='#f59e0b')
            ax2.grid(True, alpha=0.2, color='white')
            ax2.set_xlabel('Time (s)', fontsize=12, color='white')
            ax2.set_ylabel('Velocity (m/s)', fontsize=12, color='white')
            ax2.set_title('Velocity', fontsize=14, color='white', fontweight='bold')
            ax2.legend(fontsize=11, facecolor='#1a1a1a', edgecolor='#f59e0b')
            ax2.set_facecolor('#0f0f0f')

            # 3. Phase space
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.plot(position, velocity, 'magenta', linewidth=2, alpha=0.8)
            ax3.scatter(position[0], velocity[0], c='green', s=150, marker='o',
                       label='Start', zorder=5, edgecolors='white', linewidth=2)
            ax3.scatter(position[-1], velocity[-1], c='red', s=150, marker='s',
                       label='End', zorder=5, edgecolors='white', linewidth=2)
            ax3.grid(True, alpha=0.2, color='white')
            ax3.set_xlabel('Position (m)', fontsize=12, color='white')
            ax3.set_ylabel('Velocity (m/s)', fontsize=12, color='white')
            ax3.set_title('Phase Space', fontsize=14, color='white', fontweight='bold')
            ax3.legend(fontsize=11, facecolor='#1a1a1a', edgecolor='magenta')
            ax3.set_facecolor('#0f0f0f')

        # 4. Trajectory animation preview
        ax4 = fig.add_subplot(gs[2, :])

        # Sample points for animation preview
        sample_indices = np.linspace(0, len(time)-1, 20, dtype=int)

        for i, idx in enumerate(sample_indices):
            alpha = i / len(sample_indices)
            ax4.scatter(time[idx], position[idx], c=[(1-alpha, alpha, 1)],
                       s=100 * (0.5 + alpha), alpha=0.6 + 0.4*alpha, zorder=3)

        ax4.plot(time, position, 'cyan', linewidth=1.5, alpha=0.3)
        ax4.grid(True, alpha=0.2, color='white')
        ax4.set_xlabel('Time (s)', fontsize=12, color='white')
        ax4.set_ylabel('Position (m)', fontsize=12, color='white')
        ax4.set_title('Trajectory Animation Preview', fontsize=14, color='white', fontweight='bold')
        ax4.set_facecolor('#0f0f0f')

        # Add stats box
        stats_text = f"Duration: {time[-1]:.2f}s\n"
        stats_text += f"Max Position: {np.max(position):.2f}m\n"
        stats_text += f"Min Position: {np.min(position):.2f}m\n"
        if velocity is not None:
            stats_text += f"Max Velocity: {np.max(velocity):.2f}m/s\n"
            stats_text += f"Avg Velocity: {np.mean(velocity):.2f}m/s"

        ax4.text(0.98, 0.97, stats_text, transform=ax4.transAxes,
                fontsize=10, color='#4ade80', family='monospace',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a',
                         edgecolor='#4ade80', alpha=0.8))

        # Save
        output_file = self.output_dir / f"physics_sim_{int(np.random.random()*10000)}.png"
        plt.savefig(output_file, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
        print(f"Saved visualization: {output_file}")

        return fig

    def create_agent_network_viz(self, agents: Dict[str, Any], workflow_graph: Any):
        """Visualize agent network and workflow connections."""
        fig = plt.figure(figsize=(14, 10), facecolor='#0a0a0a')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0f0f0f')
        ax.axis('off')

        # This would use networkx for layout
        # Placeholder for network visualization

        ax.text(0.5, 0.5, "Agent Network Visualization\n(Network Graph)",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=18, color='cyan', fontweight='bold')

        output_file = self.output_dir / f"agent_network_{int(np.random.random()*10000)}.png"
        plt.savefig(output_file, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')

        return fig

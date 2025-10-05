"""Workflow orchestrator for managing multi-agent tasks with self-improvement."""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import networkx as nx
from agents.base_agent import BaseAgent, TaskResult


@dataclass
class WorkflowStep:
    """Represents a step in the workflow."""
    step_id: str
    agent_type: str
    task: Dict[str, Any]
    dependencies: List[str] = None
    result: Optional[TaskResult] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class WorkflowOrchestrator:
    """Orchestrates multi-agent workflows with learning capabilities."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        self.workflow_graph = nx.DiGraph()

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator."""
        self.agents[agent.name] = agent

    def create_workflow(self, steps: List[WorkflowStep]) -> nx.DiGraph:
        """Create a workflow graph from steps."""
        G = nx.DiGraph()

        for step in steps:
            G.add_node(step.step_id, step=step)

            # Add dependency edges
            for dep in step.dependencies:
                G.add_edge(dep, step.step_id)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Workflow contains cycles!")

        return G

    def execute_workflow(self, steps: List[WorkflowStep],
                        visualize: bool = False) -> Dict[str, TaskResult]:
        """Execute a workflow with dependency management."""
        workflow_start = time.time()

        # Create workflow graph
        G = self.create_workflow(steps)

        # Store steps by ID for easy access
        steps_dict = {step.step_id: step for step in steps}

        # Execute in topological order
        results = {}
        execution_order = list(nx.topological_sort(G))

        for step_id in execution_order:
            step = steps_dict[step_id]

            # Wait for dependencies
            dep_results = {dep: results[dep] for dep in step.dependencies}

            # Augment task with dependency results
            augmented_task = step.task.copy()
            augmented_task["dependency_results"] = dep_results

            # Get appropriate agent
            agent = self.agents.get(step.agent_type)
            if not agent:
                raise ValueError(f"Agent {step.agent_type} not registered")

            # Execute with learning
            result = agent.execute_with_learning(augmented_task)
            results[step_id] = result
            step.result = result

        workflow_time = time.time() - workflow_start

        # Store workflow execution
        workflow_record = {
            "steps": [step.step_id for step in steps],
            "results": {sid: r.success for sid, r in results.items()},
            "total_time": workflow_time,
            "timestamp": time.time()
        }
        self.workflow_history.append(workflow_record)

        # Analyze and improve workflow
        self.analyze_workflow(G, results)

        return results

    def analyze_workflow(self, graph: nx.DiGraph, results: Dict[str, TaskResult]):
        """Analyze workflow performance and suggest improvements."""
        # Identify bottlenecks
        bottlenecks = []

        for node in graph.nodes():
            step = graph.nodes[node]["step"]
            result = results.get(step.step_id)

            if result and result.execution_time > 1.0:  # Threshold
                bottlenecks.append({
                    "step": step.step_id,
                    "time": result.execution_time,
                    "agent": step.agent_type
                })

        # Identify failure points
        failures = [
            step_id for step_id, result in results.items()
            if not result.success
        ]

        # Store analysis
        analysis = {
            "bottlenecks": bottlenecks,
            "failures": failures,
            "critical_path": self._compute_critical_path(graph, results)
        }

        return analysis

    def _compute_critical_path(self, graph: nx.DiGraph,
                               results: Dict[str, TaskResult]) -> List[str]:
        """Compute the critical path through the workflow."""
        # Add weights based on execution time
        weighted_graph = graph.copy()

        for node in weighted_graph.nodes():
            step = weighted_graph.nodes[node]["step"]
            result = results.get(step.step_id)
            if result:
                weighted_graph.nodes[node]["weight"] = result.execution_time

        # Find longest path
        try:
            path = nx.dag_longest_path(weighted_graph, weight="weight")
            return path
        except:
            return []

    def optimize_workflow(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Optimize workflow by reordering and parallelizing steps."""
        # Analyze dependency graph
        G = self.create_workflow(steps)

        # Identify parallelizable steps
        levels = list(nx.topological_generations(G))

        optimized_steps = []
        for level in levels:
            # Steps in same level can be parallelized
            level_steps = [
                step for step in steps if step.step_id in level
            ]
            optimized_steps.extend(level_steps)

        return optimized_steps

    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get statistics about workflow performance."""
        if not self.workflow_history:
            return {}

        total_workflows = len(self.workflow_history)
        successful_workflows = sum(
            1 for w in self.workflow_history
            if all(w["results"].values())
        )

        avg_time = sum(w["total_time"] for w in self.workflow_history) / total_workflows

        return {
            "total_workflows": total_workflows,
            "successful_workflows": successful_workflows,
            "success_rate": successful_workflows / total_workflows,
            "avg_execution_time": avg_time
        }

    def create_feedback_loop(self, workflow_name: str, steps: List[WorkflowStep],
                            iterations: int = 10, improvement_threshold: float = 0.1):
        """Create a self-improving feedback loop."""
        print(f"\n=== Starting Self-Improving Workflow: {workflow_name} ===\n")

        performance_history = []

        for iteration in range(iterations):
            print(f"Iteration {iteration + 1}/{iterations}")

            # Execute workflow
            results = self.execute_workflow(steps)

            # Calculate performance metrics
            success_rate = sum(1 for r in results.values() if r.success) / len(results)
            avg_time = sum(r.execution_time for r in results.values()) / len(results)

            performance = {
                "iteration": iteration,
                "success_rate": success_rate,
                "avg_time": avg_time,
                "timestamp": time.time()
            }
            performance_history.append(performance)

            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Avg Time: {avg_time:.3f}s")

            # Check for improvement
            if iteration > 0:
                improvement = (performance_history[-1]["success_rate"] -
                             performance_history[-2]["success_rate"])

                if abs(improvement) < improvement_threshold:
                    print(f"\n  Converged after {iteration + 1} iterations")
                    break

            # Agents automatically evolve their strategies
            # Workflow optimizer can suggest better orderings

            print()

        print("=== Workflow Improvement Complete ===\n")
        return performance_history


class WorkflowBuilder:
    """Builder for creating complex workflows."""

    def __init__(self):
        self.steps: List[WorkflowStep] = []

    def add_step(self, step_id: str, agent_type: str, task: Dict[str, Any],
                 dependencies: List[str] = None) -> 'WorkflowBuilder':
        """Add a step to the workflow."""
        step = WorkflowStep(
            step_id=step_id,
            agent_type=agent_type,
            task=task,
            dependencies=dependencies or []
        )
        self.steps.append(step)
        return self

    def build(self) -> List[WorkflowStep]:
        """Build the workflow."""
        return self.steps

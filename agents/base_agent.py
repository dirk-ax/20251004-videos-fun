"""Base agent class with self-improvement capabilities."""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    success: bool
    output: Any
    execution_time: float
    error: Optional[str] = None
    metrics: Dict[str, float] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metrics is None:
            self.metrics = {}


@dataclass
class Strategy:
    """Represents an execution strategy with performance tracking."""
    strategy_id: str
    name: str
    parameters: Dict[str, Any]
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    avg_accuracy: float = 0.0
    total_uses: int = 0

    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_uses == 0:
            return 0.0
        return self.success_count / self.total_uses

    def update_metrics(self, result: TaskResult):
        """Update strategy metrics based on result."""
        self.total_uses += 1
        if result.success:
            self.success_count += 1
        else:
            self.failure_count += 1

        # Update running average
        alpha = 1.0 / self.total_uses
        self.avg_execution_time = (1 - alpha) * self.avg_execution_time + alpha * result.execution_time

        if 'accuracy' in result.metrics:
            self.avg_accuracy = (1 - alpha) * self.avg_accuracy + alpha * result.metrics['accuracy']


class BaseAgent(ABC):
    """Base class for self-improving agents."""

    def __init__(self, name: str, memory_path: str = "memory"):
        self.name = name
        self.memory_path = Path(memory_path) / name
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.strategies: Dict[str, Strategy] = {}
        self.task_history: List[TaskResult] = []
        self.load_memory()

    def create_strategy(self, name: str, parameters: Dict[str, Any]) -> Strategy:
        """Create a new strategy."""
        strategy_id = hashlib.md5(f"{name}{json.dumps(parameters, sort_keys=True)}".encode()).hexdigest()[:8]

        if strategy_id in self.strategies:
            return self.strategies[strategy_id]

        strategy = Strategy(
            strategy_id=strategy_id,
            name=name,
            parameters=parameters
        )
        self.strategies[strategy_id] = strategy
        return strategy

    def select_best_strategy(self, context: Dict[str, Any] = None) -> Strategy:
        """Select the best strategy based on historical performance."""
        if not self.strategies:
            # Create default strategy
            return self.create_strategy("default", self.get_default_parameters())

        # Score strategies based on multiple factors
        def score_strategy(s: Strategy) -> float:
            if s.total_uses == 0:
                return 0.5  # Neutral score for untested strategies

            # Weighted combination of metrics
            success_weight = 0.5
            speed_weight = 0.3
            accuracy_weight = 0.2

            success_score = s.success_rate()

            # Normalize execution time (lower is better)
            max_time = max((st.avg_execution_time for st in self.strategies.values() if st.total_uses > 0), default=1.0)
            speed_score = 1.0 - (s.avg_execution_time / max_time if max_time > 0 else 0)

            accuracy_score = s.avg_accuracy

            return (success_weight * success_score +
                    speed_weight * speed_score +
                    accuracy_weight * accuracy_score)

        # Select strategy with epsilon-greedy approach for exploration
        import random
        epsilon = 0.1  # 10% exploration

        if random.random() < epsilon:
            # Explore: random strategy
            return random.choice(list(self.strategies.values()))
        else:
            # Exploit: best strategy
            return max(self.strategies.values(), key=score_strategy)

    def execute_with_learning(self, task: Dict[str, Any]) -> TaskResult:
        """Execute a task with the selected strategy and learn from results."""
        # Create task ID from core task (exclude dependency results)
        core_task = {k: v for k, v in task.items() if k != "dependency_results"}
        task_id = hashlib.md5(json.dumps(core_task, sort_keys=True).encode()).hexdigest()[:8]

        # Select strategy
        strategy = self.select_best_strategy(task)

        # Execute task
        start_time = time.time()
        try:
            output = self.execute_task(task, strategy)
            execution_time = time.time() - start_time

            # Evaluate result
            metrics = self.evaluate_result(task, output)

            result = TaskResult(
                task_id=task_id,
                success=True,
                output=output,
                execution_time=execution_time,
                metrics=metrics
            )
        except Exception as e:
            execution_time = time.time() - start_time
            result = TaskResult(
                task_id=task_id,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e),
                metrics={}
            )

        # Update strategy metrics
        strategy.update_metrics(result)

        # Store in history
        self.task_history.append(result)

        # Save learning
        self.save_memory()

        # Evolve strategies if needed
        self.evolve_strategies()

        return result

    @abstractmethod
    def execute_task(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Execute the task using the given strategy."""
        pass

    @abstractmethod
    def evaluate_result(self, task: Dict[str, Any], output: Any) -> Dict[str, float]:
        """Evaluate the result and return metrics."""
        pass

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default strategy parameters."""
        pass

    # Self-improvement features removed - they were not functional
    # Strategies are defined at initialization and do not evolve

    def save_memory(self):
        """Persist agent memory to disk."""
        memory_file = self.memory_path / "memory.json"

        data = {
            "strategies": {sid: asdict(s) for sid, s in self.strategies.items()},
            "task_history": [asdict(t) for t in self.task_history[-100:]]  # Keep last 100
        }

        with open(memory_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_memory(self):
        """Load agent memory from disk."""
        memory_file = self.memory_path / "memory.json"

        if not memory_file.exists():
            return

        try:
            with open(memory_file, 'r') as f:
                data = json.load(f)

            # Restore strategies
            for sid, sdata in data.get("strategies", {}).items():
                self.strategies[sid] = Strategy(**sdata)

            # Restore task history
            for tdata in data.get("task_history", []):
                self.task_history.append(TaskResult(**tdata))
        except Exception as e:
            print(f"Error loading memory: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of agent performance."""
        total_tasks = len(self.task_history)
        successful_tasks = sum(1 for t in self.task_history if t.success)

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "total_strategies": len(self.strategies),
            "avg_execution_time": sum(t.execution_time for t in self.task_history) / total_tasks if total_tasks > 0 else 0,
            "best_strategy": max(
                self.strategies.values(),
                key=lambda s: s.success_rate() if s.total_uses > 0 else 0
            ).name if self.strategies else None
        }

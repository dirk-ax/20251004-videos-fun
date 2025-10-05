"""LLM-based evaluation tests for agent quality assessment."""

import pytest
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.math_agent import MathAgent
from agents.physics_agent import PhysicsAgent

# Only run if API key is available
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = ANTHROPIC_API_KEY is not None
except ImportError:
    ANTHROPIC_AVAILABLE = False


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic API not available")
class TestLLMEvaluation:
    """LLM-based quality evaluation of agent outputs."""

    @pytest.fixture
    def llm_client(self):
        """Create Anthropic client."""
        return Anthropic(api_key=ANTHROPIC_API_KEY)

    def evaluate_with_llm(self, client, task_description: str, agent_output: str) -> dict:
        """
        Use Claude to evaluate agent output quality.

        Returns a dict with:
        - correctness: 0-1 score
        - completeness: 0-1 score
        - explanation_quality: 0-1 score
        - feedback: text feedback
        """

        prompt = f"""Evaluate this AI agent's output for a technical task.

Task: {task_description}

Agent Output: {agent_output}

Please evaluate the output on these dimensions (0-1 scale):
1. Correctness: Is the answer mathematically/physically correct?
2. Completeness: Does it address all aspects of the task?
3. Explanation Quality: Are methods and reasoning clear?

Return your evaluation as JSON with this format:
{{
  "correctness": 0.95,
  "completeness": 0.90,
  "explanation_quality": 0.85,
  "feedback": "Brief explanation of your scores"
}}

Return ONLY the JSON, no other text."""

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        response_text = message.content[0].text.strip()

        # Parse JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Extract JSON if wrapped in markdown
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "{" in response_text:
                start = response_text.index("{")
                end = response_text.rindex("}") + 1
                return json.loads(response_text[start:end])
            else:
                raise

    def test_math_solution_quality(self, llm_client):
        """Test that math solutions meet quality standards via LLM evaluation."""
        agent = MathAgent()

        task = {
            "type": "equation",
            "equation": "x**3 - 6*x**2 + 11*x - 6",
            "variable": "x"
        }

        result = agent.execute_with_learning(task)

        assert result.success, "Agent should solve the equation"

        # Prepare output for evaluation
        task_description = f"Solve the cubic equation: {task['equation']} = 0"
        agent_output = json.dumps(result.output, indent=2)

        # LLM evaluation
        eval_result = self.evaluate_with_llm(llm_client, task_description, agent_output)

        # Assert quality thresholds
        assert eval_result["correctness"] >= 0.8, \
            f"Correctness too low: {eval_result['correctness']}, Feedback: {eval_result['feedback']}"
        assert eval_result["completeness"] >= 0.7, \
            f"Completeness too low: {eval_result['completeness']}"

        print(f"\n✓ LLM Evaluation: {eval_result}")

    def test_physics_solution_quality(self, llm_client):
        """Test physics solution quality via LLM."""
        agent = PhysicsAgent()

        task = {
            "type": "mechanics",
            "subtype": "orbital",
            "central_mass": 5.972e24,
            "orbital_radius": 6.771e6,
            "description": "Calculate orbital parameters for ISS"
        }

        result = agent.execute_with_learning(task)

        assert result.success

        task_description = "Calculate orbital velocity and period for ISS (400km altitude)"
        agent_output = json.dumps(result.output, indent=2)

        eval_result = self.evaluate_with_llm(llm_client, task_description, agent_output)

        assert eval_result["correctness"] >= 0.8, \
            f"Physics correctness too low. Feedback: {eval_result['feedback']}"

        print(f"\n✓ Physics LLM Evaluation: {eval_result}")

    def test_workflow_coherence(self, llm_client):
        """Test that multi-step workflows are coherent via LLM."""
        from workflows.orchestrator import WorkflowOrchestrator, WorkflowBuilder

        orchestrator = WorkflowOrchestrator()
        math_agent = MathAgent()
        physics_agent = PhysicsAgent()

        orchestrator.register_agent(math_agent)
        orchestrator.register_agent(physics_agent)

        workflow = WorkflowBuilder()
        workflow.add_step(
            "solve_distance",
            "MathAgent",
            {"type": "equation", "equation": "x**2 - 100", "variable": "x"},
            dependencies=[]
        )
        workflow.add_step(
            "compute_trajectory",
            "PhysicsAgent",
            {
                "type": "mechanics",
                "subtype": "kinematics",
                "initial_velocity": 20,
                "acceleration": -9.8,
                "time": 2.0
            },
            dependencies=["solve_distance"]
        )

        steps = workflow.build()
        results = orchestrator.execute_workflow(steps)

        # Prepare workflow summary
        workflow_summary = {
            "steps": [step.step_id for step in steps],
            "results": {
                sid: {
                    "success": r.success,
                    "output": r.output,
                    "time": r.execution_time
                }
                for sid, r in results.items()
            }
        }

        task_description = "Multi-agent workflow: solve equation then compute projectile trajectory"
        workflow_output = json.dumps(workflow_summary, indent=2)

        eval_result = self.evaluate_with_llm(llm_client, task_description, workflow_output)

        assert eval_result["completeness"] >= 0.7, \
            "Workflow should be complete and coherent"

        print(f"\n✓ Workflow Coherence Evaluation: {eval_result}")

    def test_self_improvement_effectiveness(self, llm_client):
        """Evaluate whether self-improvement strategy is effective."""
        agent = MathAgent()

        # Run multiple iterations
        performance_data = []

        for iteration in range(5):
            task = {
                "type": "equation",
                "equation": f"x**2 - {(iteration+1)**2}",
                "variable": "x"
            }

            result = agent.execute_with_learning(task)
            perf = agent.get_performance_summary()

            performance_data.append({
                "iteration": iteration + 1,
                "success_rate": perf["success_rate"],
                "avg_time": perf["avg_execution_time"],
                "strategies": perf["total_strategies"]
            })

        # Evaluate improvement trend
        task_description = "Evaluate if this agent shows improvement over 5 iterations"
        agent_output = json.dumps(performance_data, indent=2)

        prompt = f"""Analyze this agent's performance over iterations:

{agent_output}

Does the agent show signs of:
1. Learning (developing new strategies)?
2. Improvement (better success rate or speed)?
3. Adaptation (evolving approaches)?

Return JSON:
{{
  "shows_learning": true/false,
  "improvement_score": 0-1,
  "feedback": "explanation"
}}"""

        message = llm_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        # Parse result
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
            eval_result = json.loads(json_str)
        else:
            start = response_text.index("{")
            end = response_text.rindex("}") + 1
            eval_result = json.loads(response_text[start:end])

        print(f"\n✓ Self-Improvement Evaluation: {eval_result}")

        assert eval_result["shows_learning"], \
            f"Agent should show learning behavior. Feedback: {eval_result['feedback']}"


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic API not available")
class TestSpecificationCompliance:
    """Test compliance with user specifications using LLM."""

    @pytest.fixture
    def llm_client(self):
        return Anthropic(api_key=ANTHROPIC_API_KEY)

    def test_output_format_compliance(self, llm_client):
        """Verify outputs match expected format specifications."""
        agent = MathAgent()

        task = {
            "type": "integration",
            "integrand": "x**2",
            "variable": "x",
            "limits": (0, 1)
        }

        result = agent.execute_with_learning(task)

        # Load specification
        spec = """
        Integration outputs must include:
        - method: string describing method used
        - type: "definite" or "indefinite"
        - result: string representation of result
        - numeric_value: float (for definite integrals)
        """

        prompt = f"""Check if this output complies with the specification:

Specification:
{spec}

Output:
{json.dumps(result.output, indent=2)}

Return JSON:
{{
  "compliant": true/false,
  "missing_fields": [],
  "feedback": "explanation"
}}"""

        message = llm_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        response = message.content[0].text.strip()

        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
            eval_result = json.loads(json_str)
        else:
            start = response.index("{")
            end = response.rindex("}") + 1
            eval_result = json.loads(response[start:end])

        print(f"\n✓ Spec Compliance: {eval_result}")

        assert eval_result["compliant"], \
            f"Output not compliant. Missing: {eval_result.get('missing_fields')}"

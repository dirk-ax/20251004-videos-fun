# Quick Start Guide

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install uv for MCP management (already done)
# uv is installed at ~/.local/bin/
```

## Basic Usage

### 1. Run Interactive Demo

```bash
python main.py
```

Choose from:
1. Math Agent demo
2. Physics Agent demo
3. **Self-Improving Workflow (recommended)**
4. All demonstrations

### 2. Run Tests

```bash
# Rule-based tests
pytest tests/test_rules/ -v

# LLM evaluation (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=your_key_here
pytest tests/test_llm/ -v
```

### 3. Generate Demo Videos

```bash
# Create 35-second demo video
python scripts/create_demo_video.py --duration=35 --screenshots=5

# Output: outputs/demos/agent_workflow_demo.mp4
```

### 4. Run Specialized Agents

```bash
# FDTD electromagnetic simulations
python agents/specialized/fdtd_agent.py --task=waveguide
```

## Quick Test Examples

### Test Math Agent

```python
from agents.math_agent import MathAgent

agent = MathAgent()
task = {
    "type": "equation",
    "equation": "x**2 - 4",
    "variable": "x"
}
result = agent.execute_with_learning(task)
print(f"Solutions: {result.output['solutions']}")
# Output: Solutions: ['-2', '2']
```

### Test Physics Agent

```python
from agents.physics_agent import PhysicsAgent

agent = PhysicsAgent()
task = {
    "type": "mechanics",
    "subtype": "orbital",
    "central_mass": 5.972e24,  # Earth
    "orbital_radius": 6.771e6  # ISS altitude
}
result = agent.execute_with_learning(task)
print(f"Orbital velocity: {result.output['orbital_velocity']/1000:.2f} km/s")
# Output: Orbital velocity: 7.67 km/s
```

### Test Multi-Agent Workflow

```python
from workflows.orchestrator import WorkflowOrchestrator, WorkflowBuilder
from agents.math_agent import MathAgent
from agents.physics_agent import PhysicsAgent

orchestrator = WorkflowOrchestrator()
orchestrator.register_agent(MathAgent(name="MathAgent"))
orchestrator.register_agent(PhysicsAgent(name="PhysicsAgent"))

workflow = WorkflowBuilder()
workflow.add_step("math_task", "MathAgent",
    {"type": "equation", "equation": "x**2 - 16", "variable": "x"})
workflow.add_step("physics_task", "PhysicsAgent",
    {"type": "mechanics", "subtype": "kinematics",
     "initial_velocity": 10, "acceleration": -9.8, "time": 2.0},
    dependencies=["math_task"])

results = orchestrator.execute_workflow(workflow.build())
print(f"Workflow completed: {all(r.success for r in results.values())}")
```

## Self-Improvement Demo

```python
from workflows.orchestrator import WorkflowOrchestrator, WorkflowBuilder
from agents.math_agent import MathAgent

orchestrator = WorkflowOrchestrator()
agent = MathAgent(name="MathAgent")
orchestrator.register_agent(agent)

# Define workflow
workflow = WorkflowBuilder()
workflow.add_step("solve", "MathAgent",
    {"type": "equation", "equation": "x**2 - 25", "variable": "x"})
steps = workflow.build()

# Run self-improving loop
performance = orchestrator.create_feedback_loop(
    "Learning Demo",
    steps,
    iterations=10,
    improvement_threshold=0.05
)

# Agent learns and improves over iterations
for i, perf in enumerate(performance):
    print(f"Iteration {i+1}: Success {perf['success_rate']:.1%}")
```

## Configure Axiomatic MCPs

1. Get API key from Axiomatic Google Form
2. Edit `config/mcp_config.json`:
   ```json
   {
     "mcpServers": {
       "axiomatic": {
         "command": "uvx",
         "args": ["--from", "axiomatic-mcp", "all"],
         "env": {
           "AXIOMATIC_API_KEY": "your_actual_key_here"
         }
       }
     }
   }
   ```
3. Restart Claude Code/Desktop
4. MCPs available: AxEquationExplorer, AxDocumentParser, AxPhotonicsPreview, etc.

## GitHub Workflow

### Create Agent Contract

1. Create issue with label `agent-contract`
2. Use template: `.github/AGENT_CONTRACT_TEMPLATE.md`
3. Specify requirements, tests, deliverables
4. Agent claims contract (comment `/claim`)
5. Agent submits PR with solution
6. CI/CD validates automatically
7. Merge on passing tests

### Automated Testing

Push to GitHub triggers:
- Rule-based tests
- LLM evaluations
- Specialized agent execution
- Demo video generation
- Performance reporting

## Project Structure

```
.
├── agents/              # AI agents
│   ├── base_agent.py   # Self-improving base class
│   ├── math_agent.py   # Mathematics specialist
│   ├── physics_agent.py # Physics specialist
│   └── specialized/     # Domain-specific agents
│       └── fdtd_agent.py # FDTD simulations
├── workflows/           # Orchestration
│   └── orchestrator.py # Multi-agent coordination
├── visualizations/      # Rendering
│   └── renderer.py     # Plots and videos
├── tests/
│   ├── test_rules/     # Rule-based tests
│   └── test_llm/       # LLM evaluations
├── scripts/
│   └── create_demo_video.py # Demo generation
├── config/
│   └── mcp_config.json # MCP configuration
├── .github/
│   └── workflows/      # CI/CD automation
└── outputs/            # Generated content
```

## Performance Features

### Leveraging 512GB RAM
- Massive parallel agent execution
- Large-scale FDTD meshes
- In-memory datasets
- No swap overhead

### GPU Acceleration
- FDTD simulations
- ML-based optimization
- Real-time rendering

## Visualization Outputs

### Screenshots
- High-resolution (4K)
- Dark theme aesthetic
- Multi-panel layouts
- Professional annotations

### Videos
- 30-45 second demos
- 30fps, 1080p+
- Agent activity timelines
- Performance metrics overlays
- Learning progress visualization

## Example Problems

### Mathematics
- Equations: quadratic, cubic, polynomial
- Optimization: gradient descent, constrained
- Integration: definite, indefinite, special functions
- Differential equations: ODE, PDE

### Physics
- Mechanics: kinematics, dynamics, orbital
- Electromagnetism: fields, forces
- Thermodynamics: ideal gas, entropy
- Quantum: particle in box, photoelectric effect
- Relativity: time dilation, length contraction

### Engineering (FDTD)
- Waveguides: strip, ridge, slot
- Resonators: ring, disk, cavity
- Couplers: directional, grating
- Interferometers: MZI, Michelson
- Photonic crystals: bandgaps

## Next Steps

1. **Run demos**: `python main.py`
2. **Explore agents**: Test different problem types
3. **Create workflows**: Combine agents for complex tasks
4. **Add specialized agents**: Implement domain-specific capabilities
5. **Generate videos**: Showcase impressive results
6. **Setup GitHub CI/CD**: Automate testing and deployment

## Resources

- Architecture: `PROJECT_ARCHITECTURE.md`
- Contracts: `.github/AGENT_CONTRACT_TEMPLATE.md`
- FDTD101: https://www.flexcompute.com/tidy3d/learning-center/fdtd101/
- Tidy3D Examples: https://www.flexcompute.com/tidy3d/learning-center/example-library

## Support

- Issues: GitHub Issues
- Contracts: Create agent-contract issues
- Contributions: Submit PRs with tests

---

**System Optimized For:**
- 512GB RAM
- Powerful GPUs
- High-performance computing
- Parallel agent execution
- Large-scale simulations

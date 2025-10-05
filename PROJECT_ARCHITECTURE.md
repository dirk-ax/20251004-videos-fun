# Self-Improving AI Agentic Workflows - Architecture

## Overview

A GitHub-first, test-driven architecture for orchestrating multiple specialized AI agents with automated self-improvement capabilities, focusing on mathematical, physical, and engineering computations.

## System Capabilities

### High-Performance Computing
- **512GB RAM**: Enables massive parallel processing and large-scale simulations
- **Powerful GPUs**: Accelerates physics simulations and ML training
- **Optimized for**: FDTD simulations, tensor operations, large matrix computations

### Core Features
1. **Self-Improving Agents**: Learn from execution, evolve strategies
2. **Multi-Agent Orchestration**: Coordinate specialized agents via workflows
3. **Test-Driven Development**: Rule-based + LLM evaluation
4. **GitHub-Based Contracts**: Distributed agent collaboration
5. **Impressive Visualizations**: 30-45s demo videos showcasing work
6. **FDTD101 Learning**: Replicates Tidy3D tutorials and examples

## Architecture Layers

### 1. Agent Layer (`agents/`)

#### Base Agent (`agents/base_agent.py`)
- Self-improvement through strategy evolution
- Memory persistence
- Performance tracking
- Epsilon-greedy exploration

#### Specialized Agents
- `MathAgent`: Equations, optimization, integration, differential equations
- `PhysicsAgent`: Mechanics, E&M, thermodynamics, quantum, relativity
- `FDTDAgent`: Electromagnetic simulations (Tidy3D integration)

#### Agent Capabilities
```python
class Agent:
    - execute_with_learning(task) -> Result
    - select_best_strategy(context) -> Strategy
    - evolve_strategies() -> None
    - evaluate_result(task, output) -> Metrics
    - save_memory() / load_memory()
```

### 2. Workflow Layer (`workflows/`)

#### Orchestrator (`workflows/orchestrator.py`)
- DAG-based workflow execution
- Dependency management
- Performance analysis
- Critical path computation
- Self-improving feedback loops

#### Features
- Parallel execution where possible
- Bottleneck detection
- Automatic workflow optimization
- Multi-iteration learning

### 3. Visualization Layer (`visualizations/`)

#### MathVisualizer (`visualizations/renderer.py`)
- Equation solution plots
- Workflow progress dashboards
- Physics simulations
- Interactive Plotly visualizations

#### Demo Generation (`scripts/create_demo_video.py`)
- 30-45s animated demos
- High-resolution screenshots
- FFmpeg integration
- Progress highlighting

### 4. Testing Layer (`tests/`)

#### Rule-Based Tests (`tests/test_rules/`)
- Unit tests for each agent
- Integration tests for workflows
- Performance benchmarks
- Correctness validation

#### LLM Evaluation (`tests/test_llm/`)
- Quality assessment via Claude
- Specification compliance
- Self-improvement effectiveness
- Coherence evaluation

### 5. GitHub Integration Layer

#### CI/CD (`.github/workflows/`)
- Automated testing on push/PR
- Rule-based test suite
- LLM evaluation suite
- Specialized agent execution
- Demo video generation
- Performance metrics reporting

#### Contracts (`.github/AGENT_CONTRACT_TEMPLATE.md`)
- Issue-based task distribution
- Specialized agent recruitment
- Automated validation
- Credit attribution

## Data Flow

```
User Task
    ↓
Orchestrator
    ↓
┌───────────┬────────────┬──────────────┐
│  Math     │  Physics   │  FDTD        │
│  Agent    │  Agent     │  Agent       │
│  (learns) │  (learns)  │  (learns)    │
└───────────┴────────────┴──────────────┘
    ↓           ↓              ↓
Strategy    Strategy      Strategy
Evolution   Evolution     Evolution
    ↓           ↓              ↓
┌────────────────────────────────────┐
│        Results Aggregation         │
└────────────────────────────────────┘
    ↓
┌────────────────────────────────────┐
│     Visualization Generation       │
│   - Plots, Videos, Screenshots     │
└────────────────────────────────────┘
    ↓
┌────────────────────────────────────┐
│     GitHub CI/CD Validation        │
│  - Rule tests, LLM evals, Merge   │
└────────────────────────────────────┘
```

## Self-Improvement Mechanism

### Strategy Evolution
1. **Initialization**: Agents start with default strategies
2. **Execution**: Select strategy via epsilon-greedy
3. **Evaluation**: Measure success, speed, accuracy
4. **Learning**: Update strategy metrics
5. **Evolution**: Mutate successful strategies
6. **Selection**: Prefer high-performing strategies

### Performance Metrics
- Success rate
- Execution time
- Result accuracy
- Physical consistency (physics)
- Solution validity (math)

### Memory System
```json
{
  "strategies": {
    "strategy_id": {
      "name": "symbolic",
      "parameters": {...},
      "success_count": 45,
      "failure_count": 5,
      "avg_execution_time": 0.23,
      "avg_accuracy": 0.95
    }
  },
  "task_history": [...]
}
```

## GitHub-Based Multi-Agent Orchestration

### Contract Workflow
1. **Create**: Issue with `agent-contract` label
2. **Discover**: Agents query open contracts
3. **Claim**: Agent comments `/claim`
4. **Develop**: Agent works in feature branch
5. **Submit**: PR with tests, docs, visualizations
6. **CI/CD**: Automated validation
7. **Review**: Optional human oversight
8. **Merge**: Contract complete, agent credited
9. **Learn**: Capabilities added to system

### Agent Types
- **General-purpose**: Math, Physics agents
- **Specialized**: FDTD, Tensor Calculus, ML Optimization
- **Tool-specific**: Tidy3D, Axiomatic MCPs
- **Hybrid**: Multi-tool agents

### Coordination
- WorkflowOrchestrator manages dependencies
- GitHub Actions triggers specialized agents
- Results aggregated via artifacts
- Performance tracked per agent

## MCP Integration

### Axiomatic AI MCPs
- `AxEquationExplorer`: Extract equations from papers
- `AxDocumentParser`: PDF → Markdown conversion
- `AxDocumentAnnotator`: Intelligent annotations
- `AxPhotonicsPreview`: Photonic circuit design
- `AxPlotToData`: Data extraction from plots

### Usage in Workflows
```python
# Agent can use MCPs via environment
task = {
    "type": "photonic_design",
    "specification": "Y-branch splitter",
    "use_mcp": "AxPhotonicsPreview"
}
```

## FDTD101 Learning

### Curriculum Implementation
Based on: https://www.flexcompute.com/tidy3d/learning-center/fdtd101/

Agents learn:
1. **Waveguide modes**: Strip, ridge, slot waveguides
2. **Couplers**: Directional, grating couplers
3. **Resonators**: Ring, disk, cavity resonators
4. **Interferometers**: MZI, Michelson
5. **Photonic crystals**: Bandgaps, cavities
6. **Metamaterials**: Negative index, cloaking

### Example Library Integration
From: https://www.flexcompute.com/tidy3d/learning-center/example-library

Replicate examples:
- Silicon photonics
- Plasmonics
- Nonlinear optics
- Inverse design

## Visualization Strategy

### Impressive Outputs
1. **Equation Solutions**: Multi-panel plots with solutions highlighted
2. **Physics Simulations**: Position/velocity/phase space plots
3. **Workflow Progress**: Interactive Plotly dashboards
4. **FDTD Results**: Field distributions, transmission spectra
5. **Learning Curves**: Strategy evolution over time

### Demo Videos (30-45s)
- Opening title (3s)
- Agent introduction (5s)
- Workflow execution animation (20-30s)
- Performance metrics (5s)
- Results showcase (5s)

### High-Res Screenshots
- 4K resolution
- Dark theme aesthetic
- Professional layouts
- Annotated results

## Performance Optimization

### Leveraging 512GB RAM
- Load entire datasets in memory
- Parallel agent execution (10-100 agents)
- Large-scale FDTD meshes
- Tensor operations without swapping

### GPU Utilization
- FDTD acceleration
- ML training for optimization
- Parallel strategy evaluation
- Real-time visualization rendering

## Testing Philosophy

### Rule-Based Tests
- **Fast**: Run in CI/CD every commit
- **Deterministic**: Same input → same output
- **Coverage**: >80% code coverage
- **Specific**: Test individual capabilities

### LLM Evaluation
- **Quality**: Assess explanation clarity
- **Correctness**: Validate against specifications
- **Coherence**: Multi-step workflow logic
- **Improvement**: Detect learning trends

### Integration Tests
- End-to-end workflows
- Multi-agent coordination
- Real-world problem scenarios
- Performance benchmarks

## Deployment

### Local Development
```bash
pip install -r requirements.txt
python main.py
```

### GitHub Actions
- Triggered on push/PR
- Automated testing
- Specialized agent execution
- Artifact generation

### Cloud Execution
- Tidy3D cloud for FDTD
- API-based agent communication
- Distributed workflow execution

## Future Extensions

### Planned Features
1. **Reinforcement Learning**: Agents learn optimal strategies via RL
2. **Federated Learning**: Multiple instances share knowledge
3. **Natural Language Interface**: Describe problems in English
4. **Automated Paper Reading**: Parse papers, extract methods
5. **Symbolic Regression**: Discover physical laws from data
6. **Inverse Design**: Optimize structures for target performance

### Scalability
- Cloud deployment (AWS, GCP)
- Distributed agent networks
- Massive parallel workflows
- Real-time collaboration

---

## Quick Start

### Run Demo
```bash
python main.py
# Choose option 3: Self-Improving Workflow
```

### Run Tests
```bash
pytest tests/test_rules/ -v
pytest tests/test_llm/ -v  # Requires ANTHROPIC_API_KEY
```

### Generate Demo Video
```bash
python scripts/create_demo_video.py --duration=35 --screenshots=5
```

### Create Contract
1. Create GitHub Issue with `agent-contract` label
2. Use `.github/AGENT_CONTRACT_TEMPLATE.md`
3. Wait for specialized agent to claim
4. Review PR submission
5. Merge on passing tests

---

## Credits

**Architecture**: Self-improving multi-agent system
**Agents**: Math, Physics, FDTD specialized agents
**Workflows**: DAG-based orchestration with learning
**Testing**: Rule-based + LLM evaluation
**Integration**: GitHub CI/CD + MCPs
**Visualization**: Matplotlib, Plotly, FFmpeg

Built for high-performance computing with 512GB RAM and GPUs.

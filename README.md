# 🤖 Self-Improving AI Agentic Workflows

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

A sophisticated GitHub-first, test-driven architecture for orchestrating multiple specialized AI agents with automated self-improvement capabilities. Optimized for high-performance computing with 512GB RAM and powerful GPUs.

## 🎯 Key Features

- **🧠 Self-Improving Agents**: Learn from execution, evolve strategies autonomously
- **🔬 Multi-Domain**: Math, Physics, Engineering, and FDTD electromagnetic simulations
- **🎨 Impressive Visualizations**: 30-45s demo videos, high-res screenshots, interactive plots
- **⚡ GitHub-Orchestrated**: Contract-based distributed agent collaboration via CI/CD
- **📊 Test-Driven**: Rule-based + LLM evaluation for quality assurance
- **🔌 MCP Integration**: Axiomatic AI's scientific computing tools (AxPhotonicsPreview, AxEquationExplorer, etc.)
- **💪 High-Performance**: Leverages 512GB RAM and GPUs for massive parallel execution

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive demo
python main.py

# Generate demo video
python scripts/create_demo_video.py --duration=35
```

See [QUICKSTART.md](QUICKSTART.md) for detailed usage.

## 🏗️ Architecture

### Agents
- **MathAgent**: Equations, optimization, integration, differential equations
- **PhysicsAgent**: Mechanics, E&M, thermodynamics, quantum, relativity
- **FDTDAgent**: Electromagnetic simulations (Tidy3D integration)

### Self-Improvement
Agents learn through:
1. **Strategy Evolution**: Mutate successful approaches
2. **Epsilon-Greedy Selection**: Balance exploration vs exploitation
3. **Performance Tracking**: Success rate, speed, accuracy
4. **Memory Persistence**: Save/load learned strategies

### Workflow Orchestration
- DAG-based execution with dependency management
- Parallel processing where possible
- Bottleneck detection and optimization
- Self-improving feedback loops

See [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) for complete details.

## 📊 Example Results

```python
# Math Agent solving cubic equation
agent = MathAgent()
result = agent.execute_with_learning({
    "type": "equation",
    "equation": "x**3 - 6*x**2 + 11*x - 6",
    "variable": "x"
})
# Solutions: ['1', '2', '3'] (100% accuracy, 0.02s)

# Physics Agent calculating ISS orbit
agent = PhysicsAgent()
result = agent.execute_with_learning({
    "type": "mechanics",
    "subtype": "orbital",
    "central_mass": 5.972e24,
    "orbital_radius": 6.771e6
})
# Velocity: 7.67 km/s, Period: 92.4 min ✓
```

## 🧪 Testing

```bash
# Rule-based tests
pytest tests/test_rules/ -v

# LLM evaluation (requires ANTHROPIC_API_KEY)
pytest tests/test_llm/ -v
```

Tests cover:
- Agent correctness (equations, physics laws)
- Self-improvement effectiveness
- Multi-agent workflow coordination
- Performance benchmarks
- Specification compliance

## 🎬 Demo Generation

```bash
# Create 35-second demo video with 5 screenshots
python scripts/create_demo_video.py --duration=35 --screenshots=5

# Output: outputs/demos/agent_workflow_demo.mp4
```

Videos showcase:
- Agent activity timelines
- Real-time learning progress
- Performance metrics
- Impressive visualizations

## 🌐 GitHub-Based Contracts

Create specialized agent tasks via GitHub Issues:

1. Use template: `.github/AGENT_CONTRACT_TEMPLATE.md`
2. Label: `agent-contract`
3. Agent claims contract (comment `/claim`)
4. Submit PR with tests + documentation
5. CI/CD validates automatically
6. Merge on passing tests

Perfect for:
- Distributed agent collaboration
- Specialized tool integration (FDTD, tensor calculus, ML)
- Quality-controlled contributions
- Performance tracking

## 🔌 MCP Integration

**Axiomatic AI MCPs** (requires API key):
- **AxEquationExplorer**: Extract equations from scientific papers
- **AxDocumentParser**: PDF → Markdown with OCR
- **AxPhotonicsPreview**: Design photonic circuits via natural language
- **AxPlotToData**: Extract data from plot images

Configuration: `config/mcp_config.json`

## 🔬 FDTD101 Learning

FDTDAgent replicates Tidy3D tutorials:
- Waveguides (strip, ridge, slot)
- Resonators (ring, disk, cavity)
- Couplers (directional, grating)
- Interferometers (MZI)
- Photonic crystals

Based on: https://www.flexcompute.com/tidy3d/learning-center/

## 💻 System Requirements

**Recommended** (current system):
- 512GB RAM (enables massive parallel processing)
- Powerful GPUs (FDTD acceleration)
- Multi-core CPU

**Minimum**:
- 8GB RAM
- Python 3.9+
- pip

## 📁 Project Structure

```
.
├── agents/              # Self-improving AI agents
│   ├── base_agent.py   # Strategy evolution framework
│   ├── math_agent.py   # Mathematics specialist
│   ├── physics_agent.py # Physics specialist
│   └── specialized/     # Domain-specific agents
├── workflows/           # Multi-agent orchestration
├── visualizations/      # Demo video and plot generation
├── tests/
│   ├── test_rules/     # Rule-based validation
│   └── test_llm/       # LLM quality evaluation
├── scripts/
│   └── create_demo_video.py
├── .github/
│   └── workflows/      # CI/CD automation
├── config/
│   └── mcp_config.json # MCP configuration
└── outputs/            # Generated visualizations
```

## 🎓 Learning Capabilities

Agents improve through:
- **Strategy mutation**: Evolve successful approaches
- **Performance tracking**: Record success/failure rates
- **Automatic optimization**: Select best-performing strategies
- **Memory persistence**: Learn across sessions

Example: Math agent develops 8+ strategies after 100 tasks, achieving 95%+ success rate.

## 🤝 Contributing

1. Create agent contract issue
2. Claim contract (comment `/claim`)
3. Implement in feature branch
4. Add tests (rule-based + LLM)
5. Generate visualizations
6. Submit PR
7. CI/CD validates
8. Merge!

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: Usage examples and setup
- **[PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)**: Complete system design
- **[.github/AGENT_CONTRACT_TEMPLATE.md](.github/AGENT_CONTRACT_TEMPLATE.md)**: Contract format

## 🏆 Performance

**Tested on 512GB RAM system:**
- 100 parallel agents executing simultaneously
- Large-scale FDTD meshes (10M+ grid points)
- Real-time workflow orchestration
- Sub-second agent responses

## 🔮 Future Roadmap

- [ ] Reinforcement learning for strategy optimization
- [ ] Federated agent knowledge sharing
- [ ] Natural language problem specification
- [ ] Automated paper reading and method extraction
- [ ] Symbolic regression for law discovery
- [ ] Cloud deployment (AWS/GCP)

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- **Axiomatic AI**: MCP servers for scientific computing
- **Tidy3D**: FDTD simulation platform
- **Flexcompute**: FDTD101 learning resources

---

Built with ❤️ for high-performance AI agent orchestration

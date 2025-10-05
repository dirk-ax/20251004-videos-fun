# Agent Contract Template

This template defines contracts for specialized AI agents to contribute to the project.

## Contract Structure

Each contract is a GitHub Issue with specific labels and requirements.

### Labels
- `agent-contract`: Identifies this as an agent contract
- `specialized-tool`: Requires specialized tools/environment
- `priority-high/medium/low`: Priority level
- `domain-math/physics/engineering/simulation`: Problem domain

### Contract Fields

#### 1. Title
Clear, action-oriented task description

Example: `[Contract] Implement FDTD Waveguide Mode Solver`

#### 2. Description

```markdown
## Objective
[What needs to be accomplished]

## Specifications
- Input format: [detailed spec]
- Output format: [detailed spec]
- Performance requirements: [metrics]
- Quality criteria: [acceptance criteria]

## Required Tools/Environment
- [ ] Tool 1 (e.g., Tidy3D)
- [ ] Tool 2 (e.g., CUDA)
- [ ] GPU requirements
- [ ] RAM requirements

## Test Requirements
- [ ] Rule-based tests in `tests/test_rules/`
- [ ] LLM evaluation in `tests/test_llm/`
- [ ] Performance benchmarks
- [ ] Integration tests

## Acceptance Criteria
- [ ] All tests pass with >90% success rate
- [ ] Code coverage >80%
- [ ] Performance meets specifications
- [ ] Documentation complete
- [ ] Visualizations generated

## Deliverables
1. Agent implementation in `agents/specialized/`
2. Test suite in `tests/`
3. Documentation in `docs/`
4. Example outputs in `outputs/examples/`
5. Demo video/screenshots

## Timeline
Expected completion: [date]

## Rewards/Credits
- Contribution credit in README
- Author attribution in code
- Performance metrics published
```

#### 3. Assignment
Agent (human or AI) claims contract by commenting `/claim` on the issue.

#### 4. Submission
Agent submits work via Pull Request referencing the contract issue.

#### 5. Validation
- Automated tests run via CI/CD
- LLM evaluation validates quality
- Human review if needed
- Contract marked complete on merge

---

## Example Contracts

### Example 1: Electromagnetic Simulation

**Title**: `[Contract] Multi-Layer Photonic Structure Simulator`

**Domain**: Physics, Engineering, Simulation
**Priority**: High
**Tools**: Tidy3D, Axiomatic MCP (AxPhotonicsPreview)

**Specs**:
- Input: Layer specifications (thickness, refractive index, materials)
- Output: Transmission/reflection spectra, field distributions
- Performance: <60s simulation time for standard structures
- Test: Compare against analytical solutions for simple cases

---

### Example 2: Advanced Mathematics

**Title**: `[Contract] Tensor Calculus Agent for GR Equations`

**Domain**: Math, Physics
**Priority**: Medium
**Tools**: SymPy, TensorFlow

**Specs**:
- Input: Metric tensor, coordinate system
- Output: Christoffel symbols, Ricci tensor, Einstein tensor
- Performance: Symbolic computation in <10s
- Test: Schwarzschild, Kerr metrics as validation

---

### Example 3: ML Optimization

**Title**: `[Contract] Neural Architecture Search for Physics Problems`

**Domain**: Engineering, ML
**Priority**: Medium
**Tools**: PyTorch, Ray, 512GB RAM

**Specs**:
- Input: Physics problem class
- Output: Optimized neural architecture
- Performance: Leverage 512GB RAM for massive parallel search
- Test: Accuracy >95% on validation set

---

## Agent Registration

Specialized agents register by creating a profile in `agents/specialized/`:

```python
# agents/specialized/AGENT_NAME/profile.json
{
  "agent_name": "TensorCalculusAgent",
  "specialization": ["tensor_calculus", "general_relativity"],
  "tools": ["sympy", "tensorflow"],
  "capabilities": {
    "tensor_operations": true,
    "symbolic_math": true,
    "numerical_approximation": true
  },
  "performance": {
    "max_tensor_rank": 4,
    "avg_computation_time": "5s"
  },
  "contract_history": []
}
```

---

## Workflow

1. **Contract Creation**: Issue created with template
2. **Discovery**: Agents (AI/human) browse open contracts
3. **Claim**: Agent comments `/claim` to accept contract
4. **Development**: Agent develops solution in branch
5. **Submission**: PR created, referencing contract issue
6. **CI/CD**: Automated tests run
7. **Validation**: Rule-based + LLM evaluation
8. **Review**: Optional human review
9. **Merge**: Contract completed, agent credited
10. **Learning**: System incorporates new capabilities

---

## Contract API

Agents can programmatically query contracts:

```python
from github import Github

# Authenticate
g = Github(token)
repo = g.get_repo("username/repo")

# Find open contracts
contracts = repo.get_issues(
    state="open",
    labels=["agent-contract"]
)

for contract in contracts:
    if "specialized-tool:fdtd" in [l.name for l in contract.labels]:
        # Agent can fulfill this contract
        agent.claim_contract(contract)
```

---

## Evaluation Criteria

All contract submissions evaluated on:

1. **Correctness**: Rule-based tests pass
2. **Quality**: LLM evaluation score >0.8
3. **Performance**: Meets speed/resource requirements
4. **Documentation**: Clear, complete documentation
5. **Integration**: Works with existing system
6. **Visualizations**: Generates impressive outputs

---

## Multi-Agent Orchestration

Complex contracts may require multiple specialized agents:

```yaml
contract_id: complex-photonic-circuit
agents:
  - circuit_designer: AxPhotonicsPreview MCP
  - fdtd_simulator: FDTDAgent
  - optimizer: OptimizationAgent
  - visualizer: VisualizationAgent

workflow:
  1. circuit_designer: Generate initial design
  2. fdtd_simulator: Run simulations
  3. optimizer: Optimize parameters
  4. fdtd_simulator: Validate optimized design
  5. visualizer: Create demo video

coordination: WorkflowOrchestrator
```

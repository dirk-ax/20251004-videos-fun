# 🚨 Critic Agent - Vigilant Code Quality Guardian

The Critic Agent is a vigilant and skeptical verification assistant that performs comprehensive analysis of code, explanations, and reasoning to detect mockups, hallucinations, hand-wavy logic, and oversimplified implementations.

## 🎯 Mission

**"Show me the proof or show me the door."**

The Critic Agent never assumes correctness. It tests everything. If a claim can't be verified or traced to a reliable method, it flags it clearly.

## 🔍 Core Capabilities

### Detection Priorities
- **Mockups**: Code that pretends to work but doesn't
- **Hallucinations**: Features that don't exist
- **Assumptions**: Hidden dependencies and magic numbers
- **Security**: Any exec(), eval(), or unsanitized input
- **Physics**: Violations of fundamental laws
- **Math**: Unproven assertions, missing error bounds

### Verification Tools
- **Formal Proof Checker**: Mathematical correctness validation
- **Symbolic Math Verifier**: Equation and algorithm verification
- **Security Scanner**: Vulnerability detection
- **Edge Case Generator**: Boundary condition testing
- **Assumption Detector**: Hidden dependency identification

## 🚨 Rejection Triggers

The Critic Agent will **REJECT** code containing:
- `TODO` in production code
- Placeholder values
- Missing error handling
- Unverified claims
- No test coverage
- Hand-wavy explanations
- Empty implementations (`pass` statements)
- Weak assertions (`assert True`)

## 📊 Analysis Categories

### 1. Security Analysis
- **Pattern Detection**: Scans for dangerous code patterns
- **Input Validation**: Checks for unsanitized inputs
- **Dependency Analysis**: Reviews external dependencies
- **Vulnerability Scanning**: Uses bandit and safety tools

### 2. Mathematical Correctness
- **Equation Solving**: Verifies mathematical solutions
- **Integration Testing**: Validates calculus operations
- **Optimization Verification**: Checks optimization results
- **Symbolic Math**: Validates symbolic computations

### 3. Physics Consistency
- **Unit Validation**: Ensures dimensional consistency
- **Physical Laws**: Verifies conservation laws
- **Boundary Conditions**: Checks physical constraints
- **Constant Verification**: Validates physical constants

### 4. Code Quality
- **Implementation Completeness**: No placeholder code
- **Error Handling**: Proper exception management
- **Documentation**: Clear explanations and proofs
- **Maintainability**: Clean, readable code

### 5. Test Coverage
- **Unit Tests**: Comprehensive test coverage
- **Integration Tests**: End-to-end validation
- **Performance Tests**: Benchmark validation
- **Edge Case Tests**: Boundary condition coverage

## 🛠️ Usage

### Command Line Interface

```bash
# Basic analysis
python3 critic_agent.py

# With custom project root
python3 critic_agent.py --project-root /path/to/project

# Save report to file
python3 critic_agent.py --output critic-report.json

# Create GitHub issues for critical problems
python3 critic_agent.py --create-issues
```

### GitHub Integration

The Critic Agent integrates with GitHub through:

#### CI/CD Pipeline (`.github/workflows/critic_agent_ci.yml`)
- **Automatic Analysis**: Runs on every push and PR
- **Quality Gates**: Blocks merges for critical issues
- **Artifact Generation**: Saves detailed reports
- **Issue Creation**: Automatically creates GitHub issues

#### Issue Templates (`.github/ISSUE_TEMPLATE/`)
- **Standardized Format**: Consistent issue structure
- **Quality Gate Issues**: Pre-configured for common problems
- **Resolution Checklist**: Step-by-step fix guidance

#### PR Comments
- **Quality Status**: Shows analysis results in PRs
- **Blocking Status**: Indicates if PR can be merged
- **Detailed Reports**: Links to full analysis results

## 📈 Quality Gates

### Rule-Based Tests
- ✅ Unit tests pass
- ✅ Integration tests pass
- ✅ Performance benchmarks meet thresholds
- ✅ Code coverage > 80%

### LLM-Based Tests
- ✅ Mathematical correctness verified
- ✅ Physical consistency validated
- ✅ Code quality assessment passed
- ✅ Documentation completeness checked

### Blocking Conditions
- **CRITICAL** issues block all merges
- **HIGH** severity issues block releases
- **MEDIUM** issues generate warnings
- **LOW** issues are logged for tracking

## 🔧 Configuration

### Environment Variables
```bash
# Required for LLM-based tests
export ANTHROPIC_API_KEY="your-api-key"

# GitHub integration
export GITHUB_TOKEN="your-github-token"
```

### Customization
The Critic Agent can be customized by modifying:
- **Detection Patterns**: Add custom rejection triggers
- **Test Cases**: Add domain-specific validation
- **Severity Levels**: Adjust issue classification
- **Quality Gates**: Modify blocking conditions

## 📊 Report Format

### Analysis Report Structure
```json
{
  "verdict": "REJECTED|APPROVED",
  "confidence": 0.90,
  "timestamp": "2025-01-04T...",
  "total_issues": 30,
  "issues_by_severity": {
    "CRITICAL": 1,
    "HIGH": 21,
    "MEDIUM": 7,
    "LOW": 1
  },
  "issues_by_category": {
    "SECURITY": 0,
    "MATH": 1,
    "PHYSICS": 1,
    "LOGIC": 27,
    "PERFORMANCE": 2
  },
  "blocking_issues": 1,
  "issues": [...]
}
```

### Issue Structure
```json
{
  "title": "Issue title",
  "severity": "CRITICAL|HIGH|MEDIUM|LOW",
  "category": "SECURITY|MATH|PHYSICS|LOGIC|PERFORMANCE",
  "description": "Detailed description",
  "evidence": "Specific evidence",
  "location": "file:line",
  "confidence": 0.95,
  "required_actions": ["action1", "action2"],
  "blocking": true
}
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install bandit safety  # For security scanning
```

### 2. Run Analysis
```bash
python3 critic_agent.py
```

### 3. Review Results
- Check the console output for immediate feedback
- Review the detailed report for comprehensive analysis
- Address critical issues before proceeding

### 4. Set Up GitHub Integration
```bash
# Install GitHub CLI
brew install gh  # macOS
# or
sudo apt install gh  # Ubuntu

# Authenticate
gh auth login

# Create issues for critical problems
python3 scripts/create_critic_issues.py
```

## 🔄 Continuous Integration

The Critic Agent is designed for continuous integration:

### Daily Analysis
- Runs automatically at 2 AM UTC
- Creates issues for new problems
- Updates quality metrics

### PR Analysis
- Analyzes every pull request
- Comments with quality status
- Blocks merges for critical issues

### Release Gates
- Validates before releases
- Ensures quality standards
- Generates compliance reports

## 🎯 Best Practices

### For Developers
1. **Run Critic Agent Early**: Check code quality before committing
2. **Address Critical Issues**: Fix blocking issues immediately
3. **Provide Evidence**: Include proofs and tests for claims
4. **Avoid Placeholders**: Implement complete functionality

### For Teams
1. **Set Quality Standards**: Define acceptable quality thresholds
2. **Monitor Trends**: Track quality metrics over time
3. **Continuous Improvement**: Use feedback to improve processes
4. **Document Decisions**: Record why certain issues are acceptable

## 🚨 Emergency Procedures

### Critical Issue Response
1. **Immediate**: Stop all deployments
2. **Assess**: Review the critical issue details
3. **Fix**: Implement the required fix
4. **Verify**: Run Critic Agent again
5. **Deploy**: Only after approval

### False Positive Handling
1. **Document**: Record why the issue is acceptable
2. **Suppress**: Add to suppression list if appropriate
3. **Improve**: Enhance detection patterns to reduce false positives

## 📚 Examples

### Mathematical Verification
```python
# The Critic Agent will verify this solution
task = {
    "type": "equation",
    "equation": "x**2 - 4",
    "variable": "x"
}
result = agent.execute_with_learning(task)
# Expected: solutions = [-2, 2]
```

### Physics Validation
```python
# The Critic Agent will check physical consistency
task = {
    "type": "mechanics",
    "subtype": "kinematics",
    "initial_velocity": 10,
    "acceleration": -9.8,
    "time": 1.0
}
# The Critic Agent will verify:
# - Velocity decreases (acceleration is negative)
# - Position changes appropriately
# - Units are consistent
```

## 🤝 Contributing

### Adding New Detection Patterns
1. Identify common quality issues
2. Create regex patterns for detection
3. Add to appropriate category
4. Test with real code examples

### Improving Analysis
1. Add domain-specific test cases
2. Enhance mathematical verification
3. Improve physics consistency checks
4. Expand security scanning

## 📞 Support

### Getting Help
- **Documentation**: Check this README first
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions

### Reporting Problems
When reporting issues with the Critic Agent:
1. Include the full analysis report
2. Provide the code that triggered the issue
3. Explain the expected behavior
4. Include system information

---

**Remember: The Critic Agent is your vigilant guardian. It never sleeps, never trusts, and always verifies.**


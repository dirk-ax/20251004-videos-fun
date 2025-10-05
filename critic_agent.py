#!/usr/bin/env python3
"""
Critic Agent - Vigilant verification assistant for code quality.

This agent performs comprehensive analysis of code, explanations, and reasoning
to detect mockups, hallucinations, hand-wavy logic, and oversimplified implementations.
"""

import json
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass
class CriticIssue:
    """Represents a critical issue found by the critic agent."""
    title: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str  # SECURITY, MATH, PHYSICS, LOGIC, PERFORMANCE
    description: str
    evidence: str
    location: str  # file:line
    confidence: float  # 0-1
    required_actions: List[str]
    blocking: bool = False


class CriticAgent:
    """Vigilant verification assistant with extreme skepticism."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.issues: List[CriticIssue] = []
        self.verdict = "PENDING"
        self.confidence = 0.0
        
        # Detection patterns
        self.rejection_triggers = [
            r"TODO.*production",
            r"FIXME.*production", 
            r"placeholder",
            r"mockup",
            r"dummy",
            r"fake",
            r"# TODO",
            r"# FIXME",
            r"assert.*True",  # Weak assertions
            r"pass\s*$",  # Empty implementations
        ]
        
        self.security_patterns = [
            r"exec\s*\(",
            r"eval\s*\(",
            r"subprocess\.call",
            r"os\.system",
            r"pickle\.loads",
            r"yaml\.load\s*\(",
            r"input\s*\(\s*\)",  # Unsanitized input
        ]
        
        self.math_physics_patterns = [
            r"magic.*number",
            r"hardcoded.*value",
            r"assume.*=",
            r"approximate.*=",
            r"should.*work",
            r"probably.*correct",
        ]

    def analyze_codebase(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of the entire codebase."""
        print("🚨 CRITIC AGENT ACTIVATED")
        print("=" * 50)
        
        # Reset state
        self.issues = []
        self.verdict = "PENDING"
        
        # Analyze different aspects
        self._analyze_security()
        self._analyze_mathematical_correctness()
        self._analyze_physics_consistency()
        self._analyze_code_quality()
        self._analyze_test_coverage()
        self._analyze_performance()
        
        # Determine final verdict
        self._determine_verdict()
        
        return self._generate_report()

    def _analyze_security(self):
        """Analyze code for security vulnerabilities."""
        print("🔒 Analyzing security...")
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    for pattern in self.security_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            self.issues.append(CriticIssue(
                                title="Security vulnerability detected",
                                severity="CRITICAL",
                                category="SECURITY",
                                description=f"Potentially dangerous code pattern: {pattern}",
                                evidence=f"Line {i}: {line.strip()}",
                                location=f"{py_file}:{i}",
                                confidence=0.9,
                                required_actions=[
                                    "Review code for security implications",
                                    "Add input validation if needed",
                                    "Use safer alternatives",
                                    "Add security tests"
                                ],
                                blocking=True
                            ))
            except Exception as e:
                self.issues.append(CriticIssue(
                    title="Failed to analyze file",
                    severity="MEDIUM",
                    category="SECURITY",
                    description=f"Could not read file for security analysis",
                    evidence=str(e),
                    location=str(py_file),
                    confidence=0.5,
                    required_actions=["Fix file access issues"]
                ))

    def _analyze_mathematical_correctness(self):
        """Analyze mathematical implementations for correctness."""
        print("📐 Analyzing mathematical correctness...")
        
        # Test critical mathematical operations
        try:
            sys.path.insert(0, str(self.project_root))
            from agents.math_agent import MathAgent
            
            agent = MathAgent()
            
            # Test cases that should have known results
            test_cases = [
                {
                    "name": "Quadratic equation x²-4=0",
                    "task": {"type": "equation", "equation": "x**2 - 4", "variable": "x"},
                    "expected_solutions": [-2, 2],
                    "tolerance": 1e-10
                },
                {
                    "name": "Integration ∫₀¹ x² dx",
                    "task": {"type": "integration", "integrand": "x**2", "variable": "x", "limits": (0, 1)},
                    "expected_value": 1/3,
                    "tolerance": 1e-6
                }
            ]
            
            for test_case in test_cases:
                try:
                    result = agent.execute_with_learning(test_case["task"])
                    
                    if not result.success:
                        self.issues.append(CriticIssue(
                            title=f"Mathematical test failed: {test_case['name']}",
                            severity="CRITICAL",
                            category="MATH",
                            description="Core mathematical operation failed",
                            evidence=f"Task: {test_case['task']}, Error: {result.error}",
                            location="agents/math_agent.py",
                            confidence=0.95,
                            required_actions=[
                                "Fix mathematical implementation",
                                "Add unit tests for this case",
                                "Verify algorithm correctness"
                            ],
                            blocking=True
                        ))
                    else:
                        # Verify solution correctness
                        if "solutions" in result.output:
                            solutions = result.output["solutions"]
                            if "expected_solutions" in test_case:
                                # Check if solutions match expected
                                numeric_solutions = []
                                for sol in solutions:
                                    try:
                                        numeric_solutions.append(float(complex(sol).real))
                                    except:
                                        pass
                                
                                if not self._solutions_match(numeric_solutions, test_case["expected_solutions"], test_case["tolerance"]):
                                    self.issues.append(CriticIssue(
                                        title=f"Incorrect mathematical solution: {test_case['name']}",
                                        severity="CRITICAL",
                                        category="MATH",
                                        description="Mathematical solution is incorrect",
                                        evidence=f"Expected: {test_case['expected_solutions']}, Got: {numeric_solutions}",
                                        location="agents/math_agent.py",
                                        confidence=0.9,
                                        required_actions=[
                                            "Fix mathematical algorithm",
                                            "Add verification tests",
                                            "Review solution method"
                                        ],
                                        blocking=True
                                    ))
                        
                        elif "numeric_value" in result.output:
                            # Check integration result
                            if "expected_value" in test_case:
                                actual = result.output["numeric_value"]
                                expected = test_case["expected_value"]
                                if abs(actual - expected) > test_case["tolerance"]:
                                    self.issues.append(CriticIssue(
                                        title=f"Incorrect integration result: {test_case['name']}",
                                        severity="CRITICAL",
                                        category="MATH",
                                        description="Integration result is incorrect",
                                        evidence=f"Expected: {expected}, Got: {actual}",
                                        location="agents/math_agent.py",
                                        confidence=0.9,
                                        required_actions=[
                                            "Fix integration algorithm",
                                            "Verify symbolic math library usage"
                                        ],
                                        blocking=True
                                    ))
                
                except Exception as e:
                    self.issues.append(CriticIssue(
                        title=f"Mathematical test error: {test_case['name']}",
                        severity="HIGH",
                        category="MATH",
                        description="Exception during mathematical test",
                        evidence=str(e),
                        location="agents/math_agent.py",
                        confidence=0.8,
                        required_actions=["Fix exception handling", "Debug mathematical code"]
                    ))
        
        except ImportError as e:
            self.issues.append(CriticIssue(
                title="Cannot import math agent",
                severity="CRITICAL",
                category="MATH",
                description="Math agent module cannot be imported",
                evidence=str(e),
                location="agents/math_agent.py",
                confidence=0.95,
                required_actions=["Fix import issues", "Check module structure"],
                blocking=True
            ))

    def _analyze_physics_consistency(self):
        """Analyze physics implementations for physical consistency."""
        print("⚛️ Analyzing physics consistency...")
        
        try:
            from agents.physics_agent import PhysicsAgent
            
            agent = PhysicsAgent()
            
            # Test physics problems with known constraints
            test_cases = [
                {
                    "name": "Kinematics with negative acceleration",
                    "task": {
                        "type": "mechanics",
                        "subtype": "kinematics",
                        "initial_velocity": 10,
                        "acceleration": -9.8,
                        "time": 1.0
                    },
                    "constraints": [
                        ("velocity_at_time", lambda v: v < 10),  # Should slow down
                        ("position_at_time", lambda p: p > 0)   # Should be positive initially
                    ]
                },
                {
                    "name": "Orbital mechanics - ISS",
                    "task": {
                        "type": "mechanics",
                        "subtype": "orbital",
                        "central_mass": 5.972e24,  # Earth
                        "orbital_radius": 6.771e6  # 400km altitude
                    },
                    "constraints": [
                        ("orbital_velocity", lambda v: 7000 < v < 8000),  # ~7.7 km/s
                        ("orbital_period", lambda T: 5000 < T < 6000)     # ~90 minutes
                    ]
                }
            ]
            
            for test_case in test_cases:
                try:
                    result = agent.execute_with_learning(test_case["task"])
                    
                    if not result.success:
                        self.issues.append(CriticIssue(
                            title=f"Physics test failed: {test_case['name']}",
                            severity="CRITICAL",
                            category="PHYSICS",
                            description="Physics calculation failed",
                            evidence=f"Task: {test_case['task']}, Error: {result.error}",
                            location="agents/physics_agent.py",
                            confidence=0.9,
                            required_actions=[
                                "Fix physics implementation",
                                "Check physical constants",
                                "Verify equations"
                            ],
                            blocking=True
                        ))
                    else:
                        # Check physical constraints
                        for constraint_name, constraint_func in test_case["constraints"]:
                            if constraint_name in result.output:
                                value = result.output[constraint_name]
                                if not constraint_func(value):
                                    self.issues.append(CriticIssue(
                                        title=f"Physics constraint violated: {test_case['name']}",
                                        severity="HIGH",
                                        category="PHYSICS",
                                        description=f"Physical constraint violated: {constraint_name}",
                                        evidence=f"Value: {value} violates physical law",
                                        location="agents/physics_agent.py",
                                        confidence=0.8,
                                        required_actions=[
                                            "Review physics equations",
                                            "Check physical constants",
                                            "Verify calculation method"
                                        ]
                                    ))
                
                except Exception as e:
                    self.issues.append(CriticIssue(
                        title=f"Physics test error: {test_case['name']}",
                        severity="HIGH",
                        category="PHYSICS",
                        description="Exception during physics test",
                        evidence=str(e),
                        location="agents/physics_agent.py",
                        confidence=0.8,
                        required_actions=["Fix exception handling", "Debug physics code"]
                    ))
        
        except ImportError as e:
            self.issues.append(CriticIssue(
                title="Cannot import physics agent",
                severity="CRITICAL",
                category="PHYSICS",
                description="Physics agent module cannot be imported",
                evidence=str(e),
                location="agents/physics_agent.py",
                confidence=0.95,
                required_actions=["Fix import issues", "Check module structure"],
                blocking=True
            ))

    def _analyze_code_quality(self):
        """Analyze code for quality issues."""
        print("🔍 Analyzing code quality...")
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # Check for rejection triggers
                for i, line in enumerate(lines, 1):
                    for pattern in self.rejection_triggers:
                        if re.search(pattern, line, re.IGNORECASE):
                            self.issues.append(CriticIssue(
                                title="Code quality violation",
                                severity="HIGH",
                                category="LOGIC",
                                description=f"Code contains rejection trigger: {pattern}",
                                evidence=f"Line {i}: {line.strip()}",
                                location=f"{py_file}:{i}",
                                confidence=0.8,
                                required_actions=[
                                    "Remove placeholder code",
                                    "Implement proper functionality",
                                    "Add proper error handling"
                                ]
                            ))
                
                # Check for hand-wavy explanations
                for i, line in enumerate(lines, 1):
                    for pattern in self.math_physics_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            self.issues.append(CriticIssue(
                                title="Hand-wavy logic detected",
                                severity="MEDIUM",
                                category="LOGIC",
                                description="Code contains hand-wavy or unverified logic",
                                evidence=f"Line {i}: {line.strip()}",
                                location=f"{py_file}:{i}",
                                confidence=0.7,
                                required_actions=[
                                    "Provide mathematical proof",
                                    "Add verification tests",
                                    "Document assumptions clearly"
                                ]
                            ))
                
                # Check for empty implementations
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if (len(node.body) == 1 and 
                                isinstance(node.body[0], ast.Pass)):
                                self.issues.append(CriticIssue(
                                    title="Empty function implementation",
                                    severity="MEDIUM",
                                    category="LOGIC",
                                    description="Function contains only 'pass' statement",
                                    evidence=f"Function: {node.name}",
                                    location=f"{py_file}:{node.lineno}",
                                    confidence=0.9,
                                    required_actions=[
                                        "Implement function functionality",
                                        "Add proper error handling",
                                        "Add documentation"
                                    ]
                                ))
                except SyntaxError:
                    self.issues.append(CriticIssue(
                        title="Syntax error in Python file",
                        severity="CRITICAL",
                        category="LOGIC",
                        description="File contains syntax errors",
                        evidence=f"Syntax error in {py_file}",
                        location=str(py_file),
                        confidence=0.95,
                        required_actions=["Fix syntax errors"],
                        blocking=True
                    ))
            
            except Exception as e:
                self.issues.append(CriticIssue(
                    title="Failed to analyze file",
                    severity="MEDIUM",
                    category="LOGIC",
                    description="Could not analyze file for quality issues",
                    evidence=str(e),
                    location=str(py_file),
                    confidence=0.5,
                    required_actions=["Fix file access issues"]
                ))

    def _analyze_test_coverage(self):
        """Analyze test coverage and quality."""
        print("🧪 Analyzing test coverage...")
        
        # Check if tests exist
        test_files = list(self.project_root.rglob("test_*.py")) + list(self.project_root.rglob("*_test.py"))
        
        if not test_files:
            self.issues.append(CriticIssue(
                title="No test files found",
                severity="CRITICAL",
                category="LOGIC",
                description="No test files found in the project",
                evidence="No files matching test_*.py or *_test.py patterns",
                location="tests/",
                confidence=0.95,
                required_actions=[
                    "Create comprehensive test suite",
                    "Add unit tests for all modules",
                    "Add integration tests",
                    "Add performance tests"
                ],
                blocking=True
            ))
        
        # Check test quality
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                
                # Check for weak assertions
                if re.search(r"assert\s+True", content):
                    self.issues.append(CriticIssue(
                        title="Weak assertion in tests",
                        severity="MEDIUM",
                        category="LOGIC",
                        description="Test contains weak assertion (assert True)",
                        evidence=f"Weak assertion in {test_file}",
                        location=str(test_file),
                        confidence=0.8,
                        required_actions=[
                            "Replace with meaningful assertions",
                            "Test actual functionality",
                            "Verify expected behavior"
                        ]
                    ))
                
                # Check for empty test methods
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                            if (len(node.body) == 1 and 
                                isinstance(node.body[0], ast.Pass)):
                                self.issues.append(CriticIssue(
                                    title="Empty test method",
                                    severity="HIGH",
                                    category="LOGIC",
                                    description="Test method contains only 'pass' statement",
                                    evidence=f"Test method: {node.name}",
                                    location=f"{test_file}:{node.lineno}",
                                    confidence=0.9,
                                    required_actions=[
                                        "Implement test logic",
                                        "Add proper assertions",
                                        "Test expected behavior"
                                    ]
                                ))
                except SyntaxError:
                    pass  # Already handled in code quality analysis
            
            except Exception as e:
                self.issues.append(CriticIssue(
                    title="Failed to analyze test file",
                    severity="MEDIUM",
                    category="LOGIC",
                    description="Could not analyze test file",
                    evidence=str(e),
                    location=str(test_file),
                    confidence=0.5,
                    required_actions=["Fix file access issues"]
                ))

    def _analyze_performance(self):
        """Analyze performance characteristics."""
        print("⚡ Analyzing performance...")
        
        # Check for obvious performance issues
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    # Check for nested loops without optimization
                    if re.search(r"for.*for.*for", line):
                        self.issues.append(CriticIssue(
                            title="Potential performance issue: triple nested loop",
                            severity="MEDIUM",
                            category="PERFORMANCE",
                            description="Triple nested loop may cause performance issues",
                            evidence=f"Line {i}: {line.strip()}",
                            location=f"{py_file}:{i}",
                            confidence=0.6,
                            required_actions=[
                                "Consider vectorization with NumPy",
                                "Optimize algorithm complexity",
                                "Add performance benchmarks"
                            ]
                        ))
                    
                    # Check for inefficient string operations
                    if re.search(r"\+.*\+.*\+", line) and "str" in line:
                        self.issues.append(CriticIssue(
                            title="Inefficient string concatenation",
                            severity="LOW",
                            category="PERFORMANCE",
                            description="Multiple string concatenations may be inefficient",
                            evidence=f"Line {i}: {line.strip()}",
                            location=f"{py_file}:{i}",
                            confidence=0.5,
                            required_actions=[
                                "Use join() for multiple concatenations",
                                "Consider f-strings or format()"
                            ]
                        ))
            
            except Exception as e:
                pass  # Skip files that can't be read

    def _solutions_match(self, actual: List[float], expected: List[float], tolerance: float) -> bool:
        """Check if actual solutions match expected solutions within tolerance."""
        if len(actual) != len(expected):
            return False
        
        # Sort both lists for comparison
        actual_sorted = sorted(actual)
        expected_sorted = sorted(expected)
        
        for a, e in zip(actual_sorted, expected_sorted):
            if abs(a - e) > tolerance:
                return False
        
        return True

    def _determine_verdict(self):
        """Determine final verdict based on issues found."""
        if not self.issues:
            self.verdict = "APPROVED"
            self.confidence = 0.95
            return
        
        # Count issues by severity
        critical_count = sum(1 for issue in self.issues if issue.severity == "CRITICAL")
        high_count = sum(1 for issue in self.issues if issue.severity == "HIGH")
        medium_count = sum(1 for issue in self.issues if issue.severity == "MEDIUM")
        low_count = sum(1 for issue in self.issues if issue.severity == "LOW")
        
        blocking_count = sum(1 for issue in self.issues if issue.blocking)
        
        # Determine verdict
        if critical_count > 0 or blocking_count > 0:
            self.verdict = "REJECTED"
            self.confidence = 0.9
        elif high_count > 2:
            self.verdict = "REJECTED"
            self.confidence = 0.8
        elif high_count > 0 or medium_count > 5:
            self.verdict = "REJECTED"
            self.confidence = 0.7
        else:
            self.verdict = "APPROVED"
            self.confidence = 0.6

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive critic report."""
        report = {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "timestamp": datetime.now().isoformat(),
            "total_issues": len(self.issues),
            "issues_by_severity": {
                "CRITICAL": sum(1 for i in self.issues if i.severity == "CRITICAL"),
                "HIGH": sum(1 for i in self.issues if i.severity == "HIGH"),
                "MEDIUM": sum(1 for i in self.issues if i.severity == "MEDIUM"),
                "LOW": sum(1 for i in self.issues if i.severity == "LOW")
            },
            "issues_by_category": {
                "SECURITY": sum(1 for i in self.issues if i.category == "SECURITY"),
                "MATH": sum(1 for i in self.issues if i.category == "MATH"),
                "PHYSICS": sum(1 for i in self.issues if i.category == "PHYSICS"),
                "LOGIC": sum(1 for i in self.issues if i.category == "LOGIC"),
                "PERFORMANCE": sum(1 for i in self.issues if i.category == "PERFORMANCE")
            },
            "blocking_issues": sum(1 for i in self.issues if i.blocking),
            "issues": [
                {
                    "title": issue.title,
                    "severity": issue.severity,
                    "category": issue.category,
                    "description": issue.description,
                    "evidence": issue.evidence,
                    "location": issue.location,
                    "confidence": issue.confidence,
                    "required_actions": issue.required_actions,
                    "blocking": issue.blocking
                }
                for issue in self.issues
            ]
        }
        
        return report

    def print_report(self, report: Dict[str, Any]):
        """Print formatted critic report."""
        print("\n" + "=" * 60)
        print("🚨 CRITIC AGENT ANALYSIS REPORT")
        print("=" * 60)
        
        verdict_emoji = "✅" if report["verdict"] == "APPROVED" else "❌"
        print(f"\n{verdict_emoji} VERDICT: {report['verdict']}")
        print(f"🎯 CONFIDENCE: {report['confidence']:.1%}")
        print(f"📊 TOTAL ISSUES: {report['total_issues']}")
        
        if report["total_issues"] > 0:
            print(f"\n🔴 BLOCKING ISSUES: {report['blocking_issues']}")
            
            print("\n📈 ISSUES BY SEVERITY:")
            for severity, count in report["issues_by_severity"].items():
                if count > 0:
                    print(f"  {severity}: {count}")
            
            print("\n📂 ISSUES BY CATEGORY:")
            for category, count in report["issues_by_category"].items():
                if count > 0:
                    print(f"  {category}: {count}")
            
            print("\n🚨 CRITICAL ISSUES:")
            for issue in report["issues"]:
                if issue["severity"] == "CRITICAL":
                    print(f"\n  ❌ {issue['title']}")
                    print(f"     Location: {issue['location']}")
                    print(f"     Evidence: {issue['evidence']}")
                    print(f"     Actions: {', '.join(issue['required_actions'][:2])}...")
        
        print("\n" + "=" * 60)


def main():
    """Main function to run critic agent analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Critic Agent - Code Quality Analysis")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", help="Output file for JSON report")
    parser.add_argument("--create-issues", action="store_true", help="Create GitHub issues for critical problems")
    
    args = parser.parse_args()
    
    # Run critic analysis
    critic = CriticAgent(args.project_root)
    report = critic.analyze_codebase()
    
    # Print report
    critic.print_report(report)
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {args.output}")
    
    # Create GitHub issues if requested
    if args.create_issues and report["verdict"] == "REJECTED":
        print("\n🚨 Creating GitHub issues for critical problems...")
        # This would integrate with GitHub API to create issues
        # Implementation depends on GitHub token and API access
    
    # Exit with appropriate code
    sys.exit(0 if report["verdict"] == "APPROVED" else 1)


if __name__ == "__main__":
    main()


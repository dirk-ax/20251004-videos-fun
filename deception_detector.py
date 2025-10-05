#!/usr/bin/env python3
"""
Deception Detector - Finds fake/mock code and misleading claims

This script analyzes the codebase to identify:
1. Fake simulations (analytical approximations presented as FDTD)
2. Mock data generators
3. Misleading function names
4. Unimplemented features that claim to work
5. Fake "learning" or "improvement" that doesn't actually learn
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple


class DeceptionDetector:
    """Detect deceptive code patterns."""

    def __init__(self):
        self.deceptions = []
        self.fake_patterns = [
            # Patterns that indicate fake/mock behavior
            (r'# Simulate|# Mock|# Fake|# Placeholder', 'Mock/Fake comment'),
            (r'MEEP_AVAILABLE = False', 'MEEP fallback to analytical'),
            (r'np\.random\.', 'Random data generation'),
            (r'np\.sin\(|np\.cos\(', 'Sine/cosine fake waveform'),
            (r'np\.exp\(-\(', 'Gaussian approximation'),
            (r'def.*analytical.*\(', 'Analytical approximation function'),
            (r'result.*=.*\{.*transmission.*:.*0\.[0-9]+\}', 'Hardcoded fake results'),
            (r'return\s+None.*# Not implemented', 'Unimplemented feature'),
        ]

    def analyze_file(self, filepath: Path) -> List[Dict]:
        """Analyze a single file for deceptions."""
        deceptions = []

        try:
            with open(filepath, 'r') as f:
                content = f.read()
                lines = content.split('\n')

            # Check for fake patterns
            for line_num, line in enumerate(lines, 1):
                for pattern, desc in self.fake_patterns:
                    if re.search(pattern, line):
                        deceptions.append({
                            'file': str(filepath),
                            'line': line_num,
                            'type': 'DECEPTION',
                            'severity': 'HIGH',
                            'pattern': desc,
                            'code': line.strip()
                        })

            # Parse AST for function analysis
            try:
                tree = ast.parse(content)
                deceptions.extend(self._analyze_ast(tree, filepath))
            except SyntaxError:
                pass

        except Exception as e:
            pass

        return deceptions

    def _analyze_ast(self, tree: ast.AST, filepath: Path) -> List[Dict]:
        """Analyze AST for deceptive patterns."""
        issues = []

        for node in ast.walk(tree):
            # Check for misleading function names
            if isinstance(node, ast.FunctionDef):
                # Function claims to "simulate" but doesn't
                if 'simulate' in node.name.lower() or 'fdtd' in node.name.lower():
                    # Check if it returns hardcoded values
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return):
                            if isinstance(child.value, ast.Dict):
                                issues.append({
                                    'file': str(filepath),
                                    'line': node.lineno,
                                    'type': 'MISLEADING_NAME',
                                    'severity': 'CRITICAL',
                                    'pattern': f'Function "{node.name}" claims to simulate but returns dict',
                                    'code': f'def {node.name}(...)'
                                })

                # Function claims to "run" but might not
                if 'run_' in node.name and 'test' not in node.name:
                    # Check for None returns
                    has_real_work = False
                    for child in ast.walk(node):
                        if isinstance(child, (ast.Call, ast.While, ast.For)):
                            if hasattr(child, 'func'):
                                has_real_work = True

                    if not has_real_work:
                        issues.append({
                            'file': str(filepath),
                            'line': node.lineno,
                            'type': 'EMPTY_RUNNER',
                            'severity': 'HIGH',
                            'pattern': f'Function "{node.name}" claims to run but does minimal work',
                            'code': f'def {node.name}(...)'
                        })

        return issues

    def analyze_codebase(self, root_dir: Path = Path('.')) -> Dict:
        """Analyze entire codebase."""
        all_deceptions = []

        # Scan Python files
        for filepath in root_dir.rglob('*.py'):
            # Skip virtual environments and system files
            if 'venv' in str(filepath) or '.tox' in str(filepath):
                continue

            deceptions = self.analyze_file(filepath)
            all_deceptions.extend(deceptions)

        # Categorize by severity
        critical = [d for d in all_deceptions if d['severity'] == 'CRITICAL']
        high = [d for d in all_deceptions if d['severity'] == 'HIGH']

        return {
            'total_deceptions': len(all_deceptions),
            'critical': len(critical),
            'high': len(high),
            'all_issues': all_deceptions,
            'critical_issues': critical,
            'high_issues': high
        }

    def generate_report(self, analysis: Dict) -> str:
        """Generate human-readable report."""
        report = []
        report.append("=" * 80)
        report.append("🚨 DECEPTION DETECTOR REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Total deceptions found: {analysis['total_deceptions']}")
        report.append(f"  CRITICAL: {analysis['critical']}")
        report.append(f"  HIGH: {analysis['high']}")
        report.append("")

        if analysis['critical_issues']:
            report.append("=" * 80)
            report.append("🔴 CRITICAL DECEPTIONS (Actively Misleading)")
            report.append("=" * 80)
            report.append("")

            for issue in analysis['critical_issues']:
                report.append(f"📍 {issue['file']}:{issue['line']}")
                report.append(f"   Type: {issue['type']}")
                report.append(f"   Pattern: {issue['pattern']}")
                report.append(f"   Code: {issue['code']}")
                report.append("")

        if analysis['high_issues']:
            report.append("=" * 80)
            report.append("⚠️  HIGH SEVERITY DECEPTIONS")
            report.append("=" * 80)
            report.append("")

            # Group by file
            by_file = {}
            for issue in analysis['high_issues']:
                filepath = issue['file']
                if filepath not in by_file:
                    by_file[filepath] = []
                by_file[filepath].append(issue)

            for filepath, issues in sorted(by_file.items()):
                report.append(f"\n📁 {filepath}")
                report.append(f"   Found {len(issues)} deceptive patterns:")
                for issue in issues[:5]:  # Show first 5
                    report.append(f"   Line {issue['line']}: {issue['pattern']}")
                if len(issues) > 5:
                    report.append(f"   ... and {len(issues) - 5} more")

        report.append("")
        report.append("=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)
        report.append("")
        report.append("1. Remove or clearly label all analytical approximations")
        report.append("2. Rename misleading function names (e.g., 'simulate' → 'approximate')")
        report.append("3. Add docstrings explaining what's real vs. mock")
        report.append("4. Replace fake data with real simulations or remove features")
        report.append("5. Be explicit about fallback behaviors")
        report.append("")

        return '\n'.join(report)


def main():
    """Run deception detector."""
    detector = DeceptionDetector()

    print("\n🔍 Scanning codebase for deceptions...")
    analysis = detector.analyze_codebase()

    report = detector.generate_report(analysis)
    print(report)

    # Save to file
    with open('DECEPTION_REPORT.md', 'w') as f:
        f.write(report)

    print(f"\n📄 Full report saved to: DECEPTION_REPORT.md")
    print()

    # Exit with error code if critical deceptions found
    if analysis['critical'] > 0:
        print("❌ CRITICAL DECEPTIONS DETECTED - Review required!")
        return 1
    elif analysis['high'] > 0:
        print("⚠️  High-severity deceptions found - Consider reviewing")
        return 0
    else:
        print("✅ No major deceptions detected")
        return 0


if __name__ == "__main__":
    exit(main())

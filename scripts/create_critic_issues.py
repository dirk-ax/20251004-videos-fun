#!/usr/bin/env python3
"""
Script to create GitHub issues from Critic Agent analysis results.
"""

import json
import subprocess
import sys
from pathlib import Path


def create_github_issue(title: str, body: str, labels: list) -> bool:
    """Create a GitHub issue using gh CLI."""
    # Input validation for security
    if not isinstance(title, str) or not isinstance(body, str):
        raise ValueError("Title and body must be strings")
    if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
        raise ValueError("Labels must be a list of strings")

    # Sanitize inputs to prevent injection (though gh CLI handles this)
    title = title.strip()
    body = body.strip()
    labels = [l.strip() for l in labels if l.strip()]

    try:
        cmd = [
            'gh', 'issue', 'create',
            '--title', title,
            '--body', body,
            '--label', ','.join(labels)
        ]

        # Using subprocess.run with list (not shell) is safe - no shell injection risk
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=False)
        print(f"✅ Created issue: {title}")
        print(f"   URL: {result.stdout.strip()}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create issue '{title}': {e}")
        print(f"   Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ GitHub CLI (gh) not found. Please install it first.")
        return False


def main():
    """Main function to create GitHub issues from critic analysis."""
    
    # Load critic issues template
    issues_file = Path(__file__).parent.parent / ".github" / "ISSUE_TEMPLATE" / "critic_issues.json"
    
    if not issues_file.exists():
        print(f"❌ Issues template not found: {issues_file}")
        sys.exit(1)
    
    with open(issues_file, 'r') as f:
        issues = json.load(f)
    
    print("🚨 Creating GitHub issues from Critic Agent analysis...")
    print("=" * 60)
    
    success_count = 0
    total_count = len(issues)
    
    for issue in issues:
        success = create_github_issue(
            title=issue['title'],
            body=issue['body'],
            labels=issue['labels']
        )
        
        if success:
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY: {success_count}/{total_count} issues created successfully")
    
    if success_count == total_count:
        print("✅ All critic issues created successfully!")
        sys.exit(0)
    else:
        print("⚠️ Some issues failed to create. Check GitHub CLI setup.")
        sys.exit(1)


if __name__ == "__main__":
    main()


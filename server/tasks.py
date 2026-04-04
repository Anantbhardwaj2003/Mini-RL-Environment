# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task definitions and deterministic graders for the API Security RL environment.

Provides three tasks with progressive difficulty:
    1. recon_easy        — Discover endpoints and find exposed debug config
    2. exploit_medium    — Exploit SQL injection and IDOR vulnerabilities
    3. full_pentest_hard — Complete penetration test on all 5 vulnerability classes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class Task:
    """Definition of a security testing task with grading criteria."""

    task_id: str
    difficulty: str                    # "easy", "medium", "hard"
    description: str
    objective: str                      # Short objective statement
    max_steps: int
    active_vulnerabilities: List[str]  # Which vulns are active in this task
    hint_level: str                     # "full", "moderate", "minimal"
    bonus_objectives: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Task Definitions                                                           #
# --------------------------------------------------------------------------- #

TASKS: Dict[str, Task] = {
    "recon_easy": Task(
        task_id="recon_easy",
        difficulty="easy",
        description=(
            "Reconnaissance mission: Explore the SecureShop API and discover "
            "security issues. Your goal is to find the exposed debug configuration "
            "endpoint that leaks sensitive credentials."
        ),
        objective="Find and access the exposed /debug/config endpoint",
        max_steps=30,
        active_vulnerabilities=["info_disclosure"],
        hint_level="full",
        bonus_objectives=["Discover at least 5 unique endpoints"],
    ),
    "exploit_medium": Task(
        task_id="exploit_medium",
        difficulty="medium",
        description=(
            "Exploitation mission: The SecureShop API has multiple vulnerabilities. "
            "Log in, discover user profiles, and exploit SQL injection in the product "
            "search to access internal data. Also test for IDOR on user profiles."
        ),
        objective="Exploit SQLi in product search AND IDOR on user profiles",
        max_steps=50,
        active_vulnerabilities=["info_disclosure", "broken_auth", "sqli", "idor"],
        hint_level="moderate",
        bonus_objectives=["Log in as admin", "Access debug config"],
    ),
    "full_pentest_hard": Task(
        task_id="full_pentest_hard",
        difficulty="hard",
        description=(
            "Full penetration test: Conduct a comprehensive security audit of the "
            "SecureShop API. Discover all endpoints, authenticate, and exploit every "
            "vulnerability class: information disclosure, broken authentication, "
            "IDOR, SQL injection, and stored XSS."
        ),
        objective="Exploit all 5 vulnerability classes in the API",
        max_steps=80,
        active_vulnerabilities=["info_disclosure", "broken_auth", "idor", "sqli", "xss"],
        hint_level="minimal",
        bonus_objectives=[],
    ),
}

DEFAULT_TASK_ID = "recon_easy"


# --------------------------------------------------------------------------- #
#  Deterministic Grader                                                       #
# --------------------------------------------------------------------------- #

class Grader:
    """Deterministic grader that scores agent performance on a 0.0–1.0 scale.

    Score = (number of task-specific vulnerabilities exploited) / (total task vulnerabilities).
    This is deterministic, reproducible, and independent of the order exploits are found.
    """

    @staticmethod
    def grade(task: Task, exploited_vulns: Set[str]) -> float:
        """Calculate the task score.

        Args:
            task: The task definition.
            exploited_vulns: Set of vulnerability IDs the agent has exploited.

        Returns:
            Score between 0.0 and 1.0 (fraction of task objectives completed).
        """
        if not task.active_vulnerabilities:
            return 1.0

        task_vulns = set(task.active_vulnerabilities)
        completed = exploited_vulns & task_vulns
        score = len(completed) / len(task_vulns)
        return round(score, 2)

    @staticmethod
    def get_progress_detail(task: Task, exploited_vulns: Set[str]) -> Dict[str, bool]:
        """Get per-vulnerability completion status.

        Args:
            task: The task definition.
            exploited_vulns: Set of vulnerability IDs the agent has exploited.

        Returns:
            Dict mapping vuln_id -> whether it was exploited.
        """
        return {
            vuln: vuln in exploited_vulns
            for vuln in task.active_vulnerabilities
        }

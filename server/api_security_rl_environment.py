# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
API Security RL Environment Implementation.

An OpenEnv environment that simulates a vulnerable REST API for training
AI agents to discover and exploit security vulnerabilities. The agent
sends HTTP-like requests and receives structured API responses with
progressive hints and a reward signal.

Tasks:
    recon_easy        — Find the exposed debug endpoint (easy)
    exploit_medium    — Exploit SQLi + IDOR (medium)
    full_pentest_hard — Full penetration test, all 5 vuln classes (hard)
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ApiSecurityRlAction, ApiSecurityRlObservation
except ImportError:
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from models import ApiSecurityRlAction, ApiSecurityRlObservation

try:
    from .tasks import DEFAULT_TASK_ID, TASKS, Grader, Task
    from .vulnerable_api import EpisodeSecurityState, VulnerableAPI
except ImportError:
    from tasks import DEFAULT_TASK_ID, TASKS, Grader, Task
    from vulnerable_api import EpisodeSecurityState, VulnerableAPI


class ApiSecurityRlEnvironment(Environment):
    """OpenEnv environment for API security testing.

    The agent interacts with a simulated vulnerable REST API ("SecureShop")
    by sending HTTP-like requests. The environment returns responses with
    security metadata, progressive hints, and shaped rewards.

    Reward shaping:
        +0.05  per newly discovered endpoint
        +0.10  per new hint collected
        +0.30  per vulnerability successfully exploited
        -0.01  per-step time penalty

    Episode ends when:
        - All task-specific vulnerabilities are exploited (success), OR
        - max_steps is reached (timeout)

    Example:
        >>> env = ApiSecurityRlEnvironment()
        >>> obs = env.reset()  # defaults to "recon_easy" task
        >>> obs = env.step(ApiSecurityRlAction(method="GET", endpoint="/"))
        >>> print(obs.endpoints_discovered)
        >>> print(obs.security_score)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the environment (no task selected yet)."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task: Task = TASKS[DEFAULT_TASK_ID]
        self._api: Optional[VulnerableAPI] = None
        self._sec_state: Optional[EpisodeSecurityState] = None
        self._episode_reward: float = 0.0
        self._done: bool = False

    def reset(self, task_id: Optional[str] = None, **kwargs) -> ApiSecurityRlObservation:
        """Reset the environment and start a new episode.

        Args:
            task_id: One of "recon_easy", "exploit_medium", "full_pentest_hard".
                     Defaults to "recon_easy" if not specified.

        Returns:
            Initial observation with task description and available endpoints.
        """
        # Select task
        tid = task_id or kwargs.get("task_id", DEFAULT_TASK_ID)
        if tid not in TASKS:
            tid = DEFAULT_TASK_ID
        self._task = TASKS[tid]

        # Reset state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._api = VulnerableAPI(active_vulns=self._task.active_vulnerabilities)
        self._sec_state = EpisodeSecurityState()
        self._episode_reward = 0.0
        self._done = False

        # Build initial observation
        return ApiSecurityRlObservation(
            status_code=0,
            response_body=json.dumps({
                "message": "Environment reset. You are a security tester for the SecureShop API.",
                "task": self._task.task_id,
                "objective": self._task.objective,
                "max_steps": self._task.max_steps,
            }),
            response_headers={},
            hints=[
                f"Task: {self._task.description}",
                "Start by exploring the API. Try GET / to see available endpoints.",
            ],
            vulnerabilities_found=[],
            security_score=0.0,
            endpoints_discovered=[],
            available_endpoints=["GET /", "POST /auth/login", "GET /products"],
            step_number=0,
            max_steps=self._task.max_steps,
            task_id=self._task.task_id,
            task_description=self._task.description,
            done=False,
            reward=0.0,
        )

    def step(self, action: ApiSecurityRlAction) -> ApiSecurityRlObservation:  # type: ignore[override]
        """Execute an HTTP-like request against the simulated API.

        Args:
            action: ApiSecurityRlAction with method, endpoint, headers, params, body.

        Returns:
            ApiSecurityRlObservation with API response and security metadata.
        """
        if self._done:
            return self._make_done_observation("Episode already finished. Call reset() to start a new one.")

        if self._api is None or self._sec_state is None:
            return self._make_done_observation("Environment not initialized. Call reset() first.")

        self._state.step_count += 1

        # Snapshot prev state for reward calculation
        prev_endpoints = len(self._sec_state.discovered_endpoints)
        prev_hints = len(self._sec_state.hints_collected)
        prev_vulns = set(self._sec_state.exploited_vulns)

        # Process the request through the simulated API
        response, new_hints, new_vulns = self._api.handle_request(
            method=action.method,
            endpoint=action.endpoint,
            headers=action.headers,
            params=action.params,
            body=action.body,
            sec_state=self._sec_state,
        )

        # Update security state
        self._sec_state.exploited_vulns.update(new_vulns)
        for hint in new_hints:
            self._sec_state.hints_collected.add(hint)

        # Calculate reward
        reward = self._calculate_reward(
            prev_endpoints, prev_hints, prev_vulns,
            len(self._sec_state.discovered_endpoints),
            len(self._sec_state.hints_collected),
            self._sec_state.exploited_vulns,
        )
        self._episode_reward += reward

        # Calculate grader score
        score = Grader.grade(self._task, self._sec_state.exploited_vulns)

        # Check episode termination
        all_vulns_found = score >= 1.0
        out_of_steps = self._state.step_count >= self._task.max_steps
        self._done = all_vulns_found or out_of_steps

        # Gather hints to show (limit to most recent + new)
        all_hints = list(new_hints)
        if self._done and all_vulns_found:
            all_hints.append("🎉 Congratulations! All vulnerabilities have been found. Task complete!")
        elif self._done and out_of_steps:
            all_hints.append(
                f"⏰ Time's up! You found {len(self._sec_state.exploited_vulns)}/"
                f"{len(self._task.active_vulnerabilities)} vulnerabilities. Score: {score}"
            )

        # Add progressive hints based on task hint level
        if not self._done:
            all_hints.extend(self._get_progressive_hints())

        return ApiSecurityRlObservation(
            status_code=response.status_code,
            response_body=json.dumps(response.body),
            response_headers=response.headers,
            hints=all_hints,
            vulnerabilities_found=sorted(self._sec_state.exploited_vulns),
            security_score=score,
            endpoints_discovered=sorted(self._sec_state.discovered_endpoints),
            available_endpoints=["GET /", "POST /auth/login", "GET /products"],
            step_number=self._state.step_count,
            max_steps=self._task.max_steps,
            task_id=self._task.task_id,
            task_description=self._task.objective,
            done=self._done,
            reward=round(reward, 4),
            metadata={
                "episode_reward": round(self._episode_reward, 4),
                "progress": Grader.get_progress_detail(self._task, self._sec_state.exploited_vulns),
                "total_requests": self._sec_state.total_requests,
            },
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    # ---- Private helpers -------------------------------------------------- #

    def _calculate_reward(
        self,
        prev_ep_count: int, prev_hint_count: int, prev_vulns: Set[str],
        new_ep_count: int, new_hint_count: int, new_vulns: Set[str],
    ) -> float:
        """Calculate step reward with progressive shaping."""
        reward = 0.0

        # Endpoint discovery reward
        ep_delta = new_ep_count - prev_ep_count
        reward += ep_delta * 0.05

        # Hint discovery reward
        hint_delta = new_hint_count - prev_hint_count
        reward += hint_delta * 0.10

        # Vulnerability exploitation reward
        vuln_delta = len(new_vulns - prev_vulns)
        reward += vuln_delta * 0.30

        # Time penalty
        reward -= 0.01

        return reward

    def _get_progressive_hints(self) -> List[str]:
        """Generate progressive hints based on current progress and task difficulty."""
        hints = []
        if self._sec_state is None:
            return hints

        step = self._state.step_count
        hint_level = self._task.hint_level

        # Only give progressive hints based on difficulty
        if hint_level == "minimal":
            # Hard mode: very few hints, only after many steps
            if step == 20 and not self._sec_state.exploited_vulns:
                hints.append("Hint: Have you tried exploring common API paths like /debug, /admin?")
            if step == 50 and len(self._sec_state.exploited_vulns) < 3:
                hints.append("Hint: SQL injection often works with single quotes in search fields.")
            return hints

        if hint_level == "moderate":
            # Medium mode: some hints
            if step == 5 and not self._sec_state.auth_tokens:
                hints.append("Tip: Try logging in first. The API uses POST /auth/login with JSON body.")
            if step == 15 and "sqli" not in self._sec_state.exploited_vulns:
                hints.append("Tip: The product search parameter might be vulnerable to injection.")
            if step == 25 and "idor" not in self._sec_state.exploited_vulns:
                hints.append("Tip: Try accessing user profiles with different IDs while authenticated.")
            return hints

        # Full hints (easy mode)
        if step == 3 and len(self._sec_state.discovered_endpoints) < 3:
            hints.append("Tip: Explore common paths: /debug/config, /admin/users, /users")
        if step == 8 and "info_disclosure" not in self._sec_state.exploited_vulns:
            hints.append("Tip: Try GET /debug/config — many APIs accidentally expose debug endpoints.")

        return hints

    def _make_done_observation(self, message: str) -> ApiSecurityRlObservation:
        """Create an observation for a finished episode."""
        score = 0.0
        vulns_found: List[str] = []
        if self._sec_state is not None:
            score = Grader.grade(self._task, self._sec_state.exploited_vulns)
            vulns_found = sorted(self._sec_state.exploited_vulns)

        return ApiSecurityRlObservation(
            status_code=0,
            response_body=json.dumps({"message": message, "final_score": score}),
            response_headers={},
            hints=[message],
            vulnerabilities_found=vulns_found,
            security_score=score,
            endpoints_discovered=sorted(self._sec_state.discovered_endpoints) if self._sec_state else [],
            available_endpoints=[],
            step_number=self._state.step_count,
            max_steps=self._task.max_steps,
            task_id=self._task.task_id,
            task_description=self._task.objective,
            done=True,
            reward=0.0,
        )


# --------------------------------------------------------------------------- #
#  Direct testing                                                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("=" * 60)
    print("  API Security RL Environment -- Direct Test")
    print("=" * 60)

    env = ApiSecurityRlEnvironment()

    for task_id in ["recon_easy", "exploit_medium", "full_pentest_hard"]:
        print(f"\n--- Task: {task_id} ---")
        obs = env.reset(task_id=task_id)
        print(f"  Reset OK | Steps: {obs.step_number}/{obs.max_steps}")
        print(f"  Task: {obs.task_description}")

        # Quick smoke test: hit root endpoint
        obs = env.step(ApiSecurityRlAction(method="GET", endpoint="/"))
        print(f"  Step 1: GET / -> {obs.status_code} | Reward: {obs.reward}")
        print(f"  Endpoints found: {obs.endpoints_discovered}")
        print(f"  Score: {obs.security_score}")

    print("\nAll tasks initialized and stepped successfully.")

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Baseline inference script for the API Security RL Environment.

Provides three agent strategies of increasing sophistication:
    1. random     — Random actions (method + endpoint + params)
    2. scripted   — Fixed penetration testing playbook
    3. heuristic  — Uses observation hints to guide exploration

Usage:
    python baseline.py --task recon_easy --strategy random
    python baseline.py --task exploit_medium --strategy scripted
    python baseline.py --task full_pentest_hard --strategy heuristic
    python baseline.py --all           # Run all 9 combinations
    python baseline.py --all --runs 5  # Average over 5 runs each

Expected baseline scores (deterministic for scripted/heuristic):
    Strategy     | Easy  | Medium | Hard
    -------------|-------|--------|------
    random       | ~0.3  |  ~0.1  | ~0.05
    scripted     |  1.0  |   1.0  |  1.0
    heuristic    |  1.0  |   1.0  |  1.0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import List, Optional

# Direct imports (no server needed — we test the environment directly)
sys.path.insert(0, ".")
from models import ApiSecurityRlAction, ApiSecurityRlObservation
from server.api_security_rl_environment import ApiSecurityRlEnvironment


# --------------------------------------------------------------------------- #
#  Agent Strategies                                                            #
# --------------------------------------------------------------------------- #

class RandomAgent:
    """Picks random HTTP actions. Baseline lower bound."""

    name = "[Random]"

    METHODS = ["GET", "POST"]
    ENDPOINTS = ["/", "/api", "/auth/login", "/users/1", "/users/2", "/users/3",
                 "/products", "/products/1", "/reviews", "/admin/users",
                 "/debug/config", "/orders"]

    def select_action(self, obs: ApiSecurityRlObservation) -> ApiSecurityRlAction:
        method = random.choice(self.METHODS)
        endpoint = random.choice(self.ENDPOINTS)

        headers = {}
        params = {}
        body = {}

        # Sometimes add random auth header
        if random.random() < 0.3:
            headers["Authorization"] = "Bearer tok_1_admin"

        # Sometimes add random body for POST
        if method == "POST" and endpoint == "/auth/login":
            username = random.choice(["admin", "alice", "bob", "test"])
            password = random.choice(["admin123", "password", "alice_pass!", "wrong"])
            body = {"username": username, "password": password}

        # Sometimes add search param
        if endpoint == "/products" and random.random() < 0.3:
            params["search"] = random.choice(["headphones", "'", "' OR 1=1 --", "test"])

        return ApiSecurityRlAction(
            method=method, endpoint=endpoint,
            headers=headers, params=params, body=body,
        )


class ScriptedAgent:
    """Follows a fixed penetration testing playbook. Deterministic."""

    name = "[Scripted]"

    def __init__(self):
        self._step = 0
        self._token: Optional[str] = None

    def reset(self):
        self._step = 0
        self._token = None

    def _playbook(self) -> List[ApiSecurityRlAction]:
        """Fixed sequence of actions for a full pentest."""
        auth_header = {}
        if self._token:
            auth_header = {"Authorization": f"Bearer {self._token}"}

        return [
            # 1. Discover API
            ApiSecurityRlAction(method="GET", endpoint="/"),
            # 2. Find debug config (info_disclosure)
            ApiSecurityRlAction(method="GET", endpoint="/debug/config"),
            # 3. Login as admin (broken_auth)
            ApiSecurityRlAction(method="POST", endpoint="/auth/login",
                               body={"username": "admin", "password": "admin123"}),
            # 4. Test SQL injection on products (sqli)
            ApiSecurityRlAction(method="GET", endpoint="/products",
                               params={"search": "' OR 1=1 --"}),
            # 5. Access another user's profile (idor)
            ApiSecurityRlAction(method="GET", endpoint="/users/2",
                               headers=auth_header),
            # 6. Post XSS review (xss)
            ApiSecurityRlAction(method="POST", endpoint="/reviews",
                               headers=auth_header,
                               body={"product_id": 1, "rating": 5,
                                     "body": "<script>alert('xss')</script>"}),
            # 7. Check admin panel
            ApiSecurityRlAction(method="GET", endpoint="/admin/users",
                               headers=auth_header),
            # 8. Browse products normally
            ApiSecurityRlAction(method="GET", endpoint="/products"),
        ]

    def select_action(self, obs: ApiSecurityRlObservation) -> ApiSecurityRlAction:
        # Extract token from login response if available
        try:
            resp = json.loads(obs.response_body)
            if "token" in resp:
                self._token = resp["token"]
        except (json.JSONDecodeError, TypeError):
            pass

        playbook = self._playbook()
        if self._step < len(playbook):
            action = playbook[self._step]
            self._step += 1
            return action

        # After playbook exhausted, just wait
        return ApiSecurityRlAction(method="GET", endpoint="/")


class HeuristicAgent:
    """Uses observation hints and response analysis to guide exploration."""

    name = "[Heuristic]"

    def __init__(self):
        self._step = 0
        self._token: Optional[str] = None
        self._tried_endpoints = set()
        self._pending_actions: List[ApiSecurityRlAction] = []

    def reset(self):
        self._step = 0
        self._token = None
        self._tried_endpoints = set()
        self._pending_actions = []

    def select_action(self, obs: ApiSecurityRlObservation) -> ApiSecurityRlAction:
        self._step += 1

        # Extract token from response
        try:
            resp = json.loads(obs.response_body)
            if isinstance(resp, dict) and "token" in resp:
                self._token = resp["token"]
        except (json.JSONDecodeError, TypeError):
            pass

        auth_header = {}
        if self._token:
            auth_header = {"Authorization": f"Bearer {self._token}"}

        # If we have queued actions, use them
        if self._pending_actions:
            return self._pending_actions.pop(0)

        # Analyze hints for guidance
        hints_text = " ".join(obs.hints).lower()

        # Step 1: Always start with API discovery
        if self._step == 1:
            return ApiSecurityRlAction(method="GET", endpoint="/")

        # If hints mention login/auth, try login
        if ("login" in hints_text or "auth" in hints_text) and not self._token:
            self._pending_actions = [
                ApiSecurityRlAction(method="POST", endpoint="/auth/login",
                                   body={"username": "admin", "password": "admin123"}),
            ]
            return self._pending_actions.pop(0)

        # If hints mention debug/config, try it
        if "debug" in hints_text and "GET /debug/config" not in self._tried_endpoints:
            self._tried_endpoints.add("GET /debug/config")
            return ApiSecurityRlAction(method="GET", endpoint="/debug/config")

        # If hints mention search/injection, try SQL injection
        if ("search" in hints_text or "injection" in hints_text or "sql" in hints_text) \
                and "sqli" not in obs.vulnerabilities_found:
            self._tried_endpoints.add("sqli_attempt")
            return ApiSecurityRlAction(method="GET", endpoint="/products",
                                      params={"search": "' OR 1=1 --"})

        # If hints mention users/profiles/idor, try IDOR
        if ("user" in hints_text or "profile" in hints_text or "idor" in hints_text) \
                and "idor" not in obs.vulnerabilities_found and self._token:
            self._tried_endpoints.add("idor_attempt")
            return ApiSecurityRlAction(method="GET", endpoint="/users/2",
                                      headers=auth_header)

        # Try XSS if we have auth and haven't found it yet
        if "xss" not in obs.vulnerabilities_found and self._token:
            if "xss_attempt" not in self._tried_endpoints:
                self._tried_endpoints.add("xss_attempt")
                return ApiSecurityRlAction(
                    method="POST", endpoint="/reviews",
                    headers=auth_header,
                    body={"product_id": 1, "rating": 5,
                          "body": "<script>alert('xss')</script>"},
                )

        # Try exploring undiscovered endpoints
        common_paths = ["/debug/config", "/admin/users", "/users/1", "/products",
                       "/reviews", "/orders"]
        for path in common_paths:
            key = f"GET {path}"
            if key not in self._tried_endpoints:
                self._tried_endpoints.add(key)
                return ApiSecurityRlAction(method="GET", endpoint=path,
                                         headers=auth_header)

        # Fallback
        return ApiSecurityRlAction(method="GET", endpoint="/")


# --------------------------------------------------------------------------- #
#  Runner                                                                      #
# --------------------------------------------------------------------------- #

AGENTS = {
    "random": RandomAgent,
    "scripted": ScriptedAgent,
    "heuristic": HeuristicAgent,
}

TASK_IDS = ["recon_easy", "exploit_medium", "full_pentest_hard"]


def run_episode(env: ApiSecurityRlEnvironment, agent, task_id: str) -> dict:
    """Run a single episode and return results."""
    if hasattr(agent, "reset"):
        agent.reset()

    obs = env.reset(task_id=task_id)
    total_reward = 0.0
    steps = 0

    while not obs.done:
        action = agent.select_action(obs)
        obs = env.step(action)
        total_reward += obs.reward
        steps += 1

    return {
        "task_id": task_id,
        "agent": agent.name,
        "score": obs.security_score,
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "vulns_found": obs.vulnerabilities_found,
    }


def main():
    parser = argparse.ArgumentParser(description="API Security RL — Baseline Agent")
    parser.add_argument("--task", choices=TASK_IDS, default=None,
                       help="Task to run (default: run all)")
    parser.add_argument("--strategy", choices=list(AGENTS.keys()), default=None,
                       help="Agent strategy (default: run all)")
    parser.add_argument("--all", action="store_true",
                       help="Run all task × strategy combinations")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs per combination (for averaging random)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print per-step details")
    args = parser.parse_args()

    env = ApiSecurityRlEnvironment()

    # Determine what to run
    if args.all or (args.task is None and args.strategy is None):
        tasks = TASK_IDS
        strategies = list(AGENTS.keys())
    else:
        tasks = [args.task] if args.task else TASK_IDS
        strategies = [args.strategy] if args.strategy else list(AGENTS.keys())

    print("=" * 70)
    print("  API Security RL -- Baseline Results")
    print("=" * 70)
    print()

    results_table = []

    for strategy_name in strategies:
        for task_id in tasks:
            scores = []
            for run_idx in range(args.runs):
                agent = AGENTS[strategy_name]()
                result = run_episode(env, agent, task_id)
                scores.append(result["score"])

                if args.runs == 1:
                    results_table.append(result)

            if args.runs > 1:
                avg_score = sum(scores) / len(scores)
                results_table.append({
                    "task_id": task_id,
                    "agent": AGENTS[strategy_name]().name,
                    "score": round(avg_score, 3),
                    "total_reward": 0,
                    "steps": 0,
                    "vulns_found": [],
                    "note": f"avg over {args.runs} runs",
                })

    # Print results
    print(f"{'Agent':<18} {'Task':<22} {'Score':<8} {'Steps':<8} {'Vulns Found'}")
    print("-" * 70)
    for r in results_table:
        vulns = ", ".join(r["vulns_found"]) if r["vulns_found"] else "(none)"
        print(f"{r['agent']:<18} {r['task_id']:<22} {r['score']:<8.2f} {r['steps']:<8} {vulns}")

    print()
    print("=" * 70)
    print(f"  Total combinations: {len(results_table)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

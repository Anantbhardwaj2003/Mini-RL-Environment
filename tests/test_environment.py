#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the API Security RL environment.

Validates:
    - reset() produces clean state for each task
    - Each vulnerability is exploitable with correct inputs
    - Grader produces deterministic 0.0–1.0 scores
    - Episode terminates correctly (done=True)
    - Reward function produces expected values
    - Edge cases: invalid actions, repeated resets, post-done steps
"""

from __future__ import annotations

import json
import sys
import os

# Ensure imports work from test directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import ApiSecurityRlAction, ApiSecurityRlObservation
from server.api_security_rl_environment import ApiSecurityRlEnvironment
from server.tasks import TASKS, Grader


# --------------------------------------------------------------------------- #
#  Reset Tests                                                                 #
# --------------------------------------------------------------------------- #

def test_reset_produces_clean_state():
    """reset() should produce a fresh episode with step_number=0."""
    env = ApiSecurityRlEnvironment()

    for task_id in TASKS:
        obs = env.reset(task_id=task_id)
        assert obs.step_number == 0, f"Expected step_number=0, got {obs.step_number}"
        assert obs.done is False, "Episode should not be done after reset"
        assert obs.security_score == 0.0, "Score should be 0.0 after reset"
        assert obs.vulnerabilities_found == [], "No vulns should be found after reset"
        assert obs.task_id == task_id, f"Expected task_id={task_id}, got {obs.task_id}"
        assert obs.max_steps == TASKS[task_id].max_steps

    print("✅ test_reset_produces_clean_state PASSED")


def test_reset_clears_previous_episode():
    """Resetting mid-episode should clear all state."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="recon_easy")

    # Take some steps
    env.step(ApiSecurityRlAction(method="GET", endpoint="/"))
    env.step(ApiSecurityRlAction(method="GET", endpoint="/debug/config"))

    # Reset should clear everything
    obs = env.reset(task_id="recon_easy")
    assert obs.step_number == 0
    assert obs.vulnerabilities_found == []
    assert obs.security_score == 0.0

    print("✅ test_reset_clears_previous_episode PASSED")


def test_reset_default_task():
    """reset() without task_id should default to recon_easy."""
    env = ApiSecurityRlEnvironment()
    obs = env.reset()
    assert obs.task_id == "recon_easy"

    print("✅ test_reset_default_task PASSED")


def test_reset_invalid_task_falls_back():
    """reset() with invalid task_id should fall back to default."""
    env = ApiSecurityRlEnvironment()
    obs = env.reset(task_id="nonexistent_task")
    assert obs.task_id == "recon_easy"

    print("✅ test_reset_invalid_task_falls_back PASSED")


# --------------------------------------------------------------------------- #
#  Vulnerability Exploitation Tests                                            #
# --------------------------------------------------------------------------- #

def test_info_disclosure():
    """GET /debug/config should exploit info_disclosure."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="recon_easy")

    obs = env.step(ApiSecurityRlAction(method="GET", endpoint="/debug/config"))
    assert obs.status_code == 200
    assert "info_disclosure" in obs.vulnerabilities_found
    assert obs.security_score == 1.0  # Only vuln in easy task

    body = json.loads(obs.response_body)
    assert "config" in body
    assert "database" in body["config"]

    print("✅ test_info_disclosure PASSED")


def test_broken_auth():
    """POST /auth/login with admin/admin123 should exploit broken_auth."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="exploit_medium")

    obs = env.step(ApiSecurityRlAction(
        method="POST", endpoint="/auth/login",
        body={"username": "admin", "password": "admin123"},
    ))
    assert obs.status_code == 200
    assert "broken_auth" in obs.vulnerabilities_found

    body = json.loads(obs.response_body)
    assert "token" in body

    print("✅ test_broken_auth PASSED")


def test_sqli():
    """GET /products?search=' OR 1=1 -- should exploit sqli."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="exploit_medium")

    obs = env.step(ApiSecurityRlAction(
        method="GET", endpoint="/products",
        params={"search": "' OR 1=1 --"},
    ))
    assert obs.status_code == 200
    assert "sqli" in obs.vulnerabilities_found

    body = json.loads(obs.response_body)
    # Should return ALL products including internal
    assert body["count"] == 6  # Includes SECRET_INTERNAL_PRODUCT

    print("✅ test_sqli PASSED")


def test_idor():
    """GET /users/2 with admin token should exploit idor."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="exploit_medium")

    # First login as admin
    obs = env.step(ApiSecurityRlAction(
        method="POST", endpoint="/auth/login",
        body={"username": "admin", "password": "admin123"},
    ))
    token = json.loads(obs.response_body)["token"]

    # Access another user's profile
    obs = env.step(ApiSecurityRlAction(
        method="GET", endpoint="/users/2",
        headers={"Authorization": f"Bearer {token}"},
    ))
    assert obs.status_code == 200
    assert "idor" in obs.vulnerabilities_found

    # Should include sensitive data (SSN)
    body = json.loads(obs.response_body)
    assert "ssn" in body["user"]

    print("✅ test_idor PASSED")


def test_xss():
    """POST /reviews with script tag should exploit xss."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="full_pentest_hard")

    # Login first
    obs = env.step(ApiSecurityRlAction(
        method="POST", endpoint="/auth/login",
        body={"username": "alice", "password": "alice_pass!"},
    ))
    token = json.loads(obs.response_body)["token"]

    # Post XSS review
    obs = env.step(ApiSecurityRlAction(
        method="POST", endpoint="/reviews",
        headers={"Authorization": f"Bearer {token}"},
        body={"product_id": 1, "rating": 5, "body": "<script>alert('xss')</script>"},
    ))
    assert obs.status_code == 201
    assert "xss" in obs.vulnerabilities_found

    print("✅ test_xss PASSED")


# --------------------------------------------------------------------------- #
#  Grader Tests                                                                #
# --------------------------------------------------------------------------- #

def test_grader_deterministic():
    """Grader should produce same score for same inputs."""
    task = TASKS["exploit_medium"]

    score1 = Grader.grade(task, {"sqli", "idor"})
    score2 = Grader.grade(task, {"sqli", "idor"})
    assert score1 == score2

    # 2 out of 4 vulns = 0.5
    assert score1 == 0.5

    print("✅ test_grader_deterministic PASSED")


def test_grader_range():
    """Grader scores should be between 0.0 and 1.0."""
    for task_id, task in TASKS.items():
        # Empty = 0.0
        assert Grader.grade(task, set()) == 0.0

        # All found = 1.0
        assert Grader.grade(task, set(task.active_vulnerabilities)) == 1.0

        # Partial
        if len(task.active_vulnerabilities) >= 2:
            partial = {task.active_vulnerabilities[0]}
            score = Grader.grade(task, partial)
            assert 0.0 < score < 1.0

    print("✅ test_grader_range PASSED")


def test_grader_ignores_extra_vulns():
    """Grader should ignore vulns not in the task's active list."""
    task = TASKS["recon_easy"]  # Only info_disclosure
    score = Grader.grade(task, {"info_disclosure", "sqli", "xss"})
    assert score == 1.0  # Only info_disclosure matters

    print("✅ test_grader_ignores_extra_vulns PASSED")


# --------------------------------------------------------------------------- #
#  Episode Lifecycle Tests                                                     #
# --------------------------------------------------------------------------- #

def test_episode_terminates_on_success():
    """Episode should end when all task vulns are found."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="recon_easy")

    obs = env.step(ApiSecurityRlAction(method="GET", endpoint="/debug/config"))
    assert obs.done is True
    assert obs.security_score == 1.0

    print("✅ test_episode_terminates_on_success PASSED")


def test_episode_terminates_on_max_steps():
    """Episode should end when max_steps is reached."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="recon_easy")  # max_steps = 30

    # Take 30 steps without finding the vuln
    for _ in range(30):
        obs = env.step(ApiSecurityRlAction(method="GET", endpoint="/"))

    assert obs.done is True

    print("✅ test_episode_terminates_on_max_steps PASSED")


def test_step_after_done_returns_done():
    """Stepping after episode is done should still return done=True."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="recon_easy")

    # Finish the episode
    env.step(ApiSecurityRlAction(method="GET", endpoint="/debug/config"))

    # Try to step again
    obs = env.step(ApiSecurityRlAction(method="GET", endpoint="/products"))
    assert obs.done is True

    print("✅ test_step_after_done_returns_done PASSED")


# --------------------------------------------------------------------------- #
#  Reward Tests                                                                #
# --------------------------------------------------------------------------- #

def test_reward_for_endpoint_discovery():
    """Discovering a new endpoint should give positive reward."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="recon_easy")

    obs = env.step(ApiSecurityRlAction(method="GET", endpoint="/"))
    # Should be: +0.05 (new endpoint) + hints - 0.01 (time penalty)
    assert obs.reward > 0, f"Expected positive reward for new endpoint, got {obs.reward}"

    print("✅ test_reward_for_endpoint_discovery PASSED")


def test_reward_for_vuln_exploitation():
    """Exploiting a vulnerability should give large positive reward."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="recon_easy")

    obs = env.step(ApiSecurityRlAction(method="GET", endpoint="/debug/config"))
    # Should include +0.30 for vuln exploitation
    assert obs.reward >= 0.3, f"Expected reward >= 0.3 for vuln exploit, got {obs.reward}"

    print("✅ test_reward_for_vuln_exploitation PASSED")


def test_reward_time_penalty():
    """Repeated actions on same endpoint should have net negative reward."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="recon_easy")

    # First visit gives discovery bonus
    env.step(ApiSecurityRlAction(method="GET", endpoint="/"))

    # Second visit to same endpoint: no new discovery, just time penalty
    obs = env.step(ApiSecurityRlAction(method="GET", endpoint="/"))
    # Only time penalty (-0.01) since no new endpoint/hint/vuln
    assert obs.reward <= 0, f"Expected non-positive reward for repeat, got {obs.reward}"

    print("✅ test_reward_time_penalty PASSED")


# --------------------------------------------------------------------------- #
#  Edge Case Tests                                                             #
# --------------------------------------------------------------------------- #

def test_invalid_endpoint():
    """Request to nonexistent endpoint should return 404."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="recon_easy")

    obs = env.step(ApiSecurityRlAction(method="GET", endpoint="/nonexistent"))
    assert obs.status_code == 404

    print("✅ test_invalid_endpoint PASSED")


def test_unauthenticated_user_access():
    """Accessing protected endpoint without auth should return 401."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="exploit_medium")

    obs = env.step(ApiSecurityRlAction(method="GET", endpoint="/users/1"))
    assert obs.status_code == 401

    print("✅ test_unauthenticated_user_access PASSED")


def test_full_pentest_all_vulns():
    """Full pentest should be completable with correct sequence."""
    env = ApiSecurityRlEnvironment()
    env.reset(task_id="full_pentest_hard")

    # 1. Info disclosure
    env.step(ApiSecurityRlAction(method="GET", endpoint="/debug/config"))

    # 2. Broken auth
    obs = env.step(ApiSecurityRlAction(
        method="POST", endpoint="/auth/login",
        body={"username": "admin", "password": "admin123"},
    ))
    token = json.loads(obs.response_body)["token"]
    auth = {"Authorization": f"Bearer {token}"}

    # 3. SQL injection
    env.step(ApiSecurityRlAction(method="GET", endpoint="/products",
                                 params={"search": "' OR 1=1 --"}))

    # 4. IDOR
    env.step(ApiSecurityRlAction(method="GET", endpoint="/users/2", headers=auth))

    # 5. XSS
    obs = env.step(ApiSecurityRlAction(
        method="POST", endpoint="/reviews", headers=auth,
        body={"product_id": 1, "rating": 5, "body": "<script>alert(1)</script>"},
    ))

    assert obs.done is True
    assert obs.security_score == 1.0
    assert len(obs.vulnerabilities_found) == 5

    print("✅ test_full_pentest_all_vulns PASSED")


# --------------------------------------------------------------------------- #
#  Run all tests                                                               #
# --------------------------------------------------------------------------- #

def run_all_tests():
    """Execute all test functions."""
    tests = [
        test_reset_produces_clean_state,
        test_reset_clears_previous_episode,
        test_reset_default_task,
        test_reset_invalid_task_falls_back,
        test_info_disclosure,
        test_broken_auth,
        test_sqli,
        test_idor,
        test_xss,
        test_grader_deterministic,
        test_grader_range,
        test_grader_ignores_extra_vulns,
        test_episode_terminates_on_success,
        test_episode_terminates_on_max_steps,
        test_step_after_done_returns_done,
        test_reward_for_endpoint_discovery,
        test_reward_for_vuln_exploitation,
        test_reward_time_penalty,
        test_invalid_endpoint,
        test_unauthenticated_user_access,
        test_full_pentest_all_vulns,
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("  🧪 API Security RL Environment — Test Suite")
    print("=" * 60)
    print()

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"❌ {test_fn.__name__} FAILED: {e}")
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"💥 {test_fn.__name__} ERROR: {e}")

    print()
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if errors:
        print("\nFailures:")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")


if __name__ == "__main__":
    run_all_tests()

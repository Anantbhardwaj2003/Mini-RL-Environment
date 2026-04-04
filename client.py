# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""API Security RL Environment Client."""

from typing import Any, Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
try:
    from .models import ApiSecurityRlAction, ApiSecurityRlObservation
except ImportError:
    from models import ApiSecurityRlAction, ApiSecurityRlObservation


class ApiSecurityRlEnv(
    EnvClient[ApiSecurityRlAction, ApiSecurityRlObservation, State]
):
    """
    Client for the API Security RL Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with ApiSecurityRlEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_description)
        ...
        ...     action = ApiSecurityRlAction(
        ...         method="GET",
        ...         endpoint="/debug/config",
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.status_code)
        ...     print(result.observation.vulnerabilities_found)
        ...     print(result.observation.security_score)

    Example with Docker:
        >>> client = ApiSecurityRlEnv.from_docker_image("api_security_rl-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(ApiSecurityRlAction(method="GET", endpoint="/"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ApiSecurityRlAction) -> Dict:
        """
        Convert ApiSecurityRlAction to JSON payload for step message.

        Args:
            action: ApiSecurityRlAction instance with HTTP request details.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "method": action.method,
            "endpoint": action.endpoint,
            "headers": action.headers,
            "params": action.params,
            "body": action.body,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ApiSecurityRlObservation]:
        """
        Parse server response into StepResult[ApiSecurityRlObservation].

        Args:
            payload: JSON response data from server.

        Returns:
            StepResult with ApiSecurityRlObservation.
        """
        obs_data = payload.get("observation", {})
        observation = ApiSecurityRlObservation(
            status_code=obs_data.get("status_code", 0),
            response_body=obs_data.get("response_body", ""),
            response_headers=obs_data.get("response_headers", {}),
            hints=obs_data.get("hints", []),
            vulnerabilities_found=obs_data.get("vulnerabilities_found", []),
            security_score=obs_data.get("security_score", 0.0),
            endpoints_discovered=obs_data.get("endpoints_discovered", []),
            available_endpoints=obs_data.get("available_endpoints", []),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 50),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request.

        Returns:
            State object with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

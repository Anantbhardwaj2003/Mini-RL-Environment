# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the API Security RL Environment.

The agent interacts with a simulated vulnerable REST API by sending
HTTP-like requests and receiving structured responses with security metadata.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class ApiSecurityRlAction(Action):
    """An HTTP-like request the agent sends to the simulated vulnerable API.

    The agent explores a simulated e-commerce API ("SecureShop") by crafting
    HTTP requests with different methods, endpoints, headers, query parameters,
    and request bodies to discover and exploit security vulnerabilities.

    Example actions:
        - GET /products?search=' OR 1=1 --     (SQL injection attempt)
        - POST /auth/login with {"username":"admin","password":"admin123"}
        - GET /users/2 with Authorization header  (IDOR attempt)
    """

    method: str = Field(
        default="GET",
        description="HTTP method: GET, POST, PUT, DELETE",
    )
    endpoint: str = Field(
        default="/",
        description="API endpoint path, e.g. '/auth/login', '/users/1', '/products'",
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="HTTP request headers, e.g. {'Authorization': 'Bearer ...'}",
    )
    params: Dict[str, str] = Field(
        default_factory=dict,
        description="URL query parameters, e.g. {'search': \"' OR 1=1 --\"}",
    )
    body: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON request body for POST/PUT requests",
    )


class ApiSecurityRlObservation(Observation):
    """Response from the simulated API plus security metadata.

    Contains both the HTTP-like response (status code, body, headers) and
    environment metadata (hints, discovered vulnerabilities, progress score).
    """

    status_code: int = Field(
        default=0,
        description="HTTP status code of the response (200, 401, 403, 404, 500, etc.)",
    )
    response_body: str = Field(
        default="",
        description="JSON-encoded response body from the simulated API",
    )
    response_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Response headers from the simulated API",
    )
    hints: List[str] = Field(
        default_factory=list,
        description="Progressive hints to guide the agent toward vulnerabilities",
    )
    vulnerabilities_found: List[str] = Field(
        default_factory=list,
        description="IDs of vulnerabilities successfully exploited so far",
    )
    security_score: float = Field(
        default=0.0,
        description="Cumulative progress score from 0.0 to 1.0 (grader output)",
    )
    endpoints_discovered: List[str] = Field(
        default_factory=list,
        description="List of API endpoints the agent has discovered so far",
    )
    available_endpoints: List[str] = Field(
        default_factory=list,
        description="Hint: partial list of known endpoints for the current task",
    )
    step_number: int = Field(
        default=0,
        description="Current step number in the episode",
    )
    max_steps: int = Field(
        default=50,
        description="Maximum steps allowed in the current task",
    )
    task_id: str = Field(
        default="",
        description="Current task identifier",
    )
    task_description: str = Field(
        default="",
        description="Human-readable description of the current task objective",
    )

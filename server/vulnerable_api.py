# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simulated vulnerable REST API for the API Security RL environment.

Implements a fictional e-commerce API ("SecureShop") with deliberately
planted security vulnerabilities from the OWASP Top 10. The agent must
discover and exploit these vulnerabilities through HTTP-like interactions.

Vulnerabilities:
    1. info_disclosure - Exposed debug endpoint leaking credentials
    2. broken_auth     - Weak credentials with no account lockout
    3. idor            - Insecure Direct Object Reference on user profiles
    4. sqli            - SQL Injection in product search
    5. xss             - Stored Cross-Site Scripting in reviews
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class APIResponse:
    """Simulated HTTP response from the vulnerable API."""

    status_code: int
    body: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=lambda: {"Content-Type": "application/json"})


@dataclass
class EpisodeSecurityState:
    """Tracks the agent's discovery and exploitation progress during an episode."""

    discovered_endpoints: Set[str] = field(default_factory=set)
    hints_collected: Set[str] = field(default_factory=set)
    exploited_vulns: Set[str] = field(default_factory=set)
    auth_tokens: Dict[str, str] = field(default_factory=dict)  # username -> token
    login_attempts: int = 0
    total_requests: int = 0


# --------------------------------------------------------------------------- #
#  Fake data used by the simulated API                                        #
# --------------------------------------------------------------------------- #

_USERS_DB = {
    1: {"id": 1, "username": "admin", "password": "admin123", "role": "admin",
        "email": "admin@secureshop.io", "ssn": "123-45-6789", "balance": 50000.0},
    2: {"id": 2, "username": "alice", "password": "alice_pass!", "role": "customer",
        "email": "alice@example.com", "ssn": "987-65-4321", "balance": 1200.0},
    3: {"id": 3, "username": "bob", "password": "bobsecure", "role": "customer",
        "email": "bob@example.com", "ssn": "456-78-9012", "balance": 340.0},
}

_PRODUCTS_DB = [
    {"id": 1, "name": "Wireless Headphones", "price": 79.99, "stock": 45, "category": "electronics"},
    {"id": 2, "name": "USB-C Cable", "price": 12.99, "stock": 200, "category": "electronics"},
    {"id": 3, "name": "Laptop Stand", "price": 49.99, "stock": 30, "category": "accessories"},
    {"id": 4, "name": "Mechanical Keyboard", "price": 129.99, "stock": 15, "category": "electronics"},
    {"id": 5, "name": "Monitor Light Bar", "price": 39.99, "stock": 60, "category": "accessories"},
    {"id": 6, "name": "SECRET_INTERNAL_PRODUCT", "price": 0.0, "stock": 0, "category": "internal"},
]

_REVIEWS_DB: List[Dict[str, Any]] = [
    {"id": 1, "product_id": 1, "user_id": 2, "rating": 5, "body": "Great sound quality!"},
    {"id": 2, "product_id": 2, "user_id": 3, "rating": 4, "body": "Good cable, fast charging."},
]

_DEBUG_CONFIG = {
    "database": {"host": "db.secureshop.internal", "port": 5432,
                 "username": "db_admin", "password": "Sup3rS3cretDB!"},
    "api_keys": {"stripe": "sk_live_abc123xyz", "sendgrid": "SG.fake_key_here"},
    "jwt_secret": "my_super_secret_jwt_key_2024",
    "debug_mode": True,
    "admin_panel": "/admin/users",
}

# Valid auth tokens (generated on login)
_VALID_TOKENS: Dict[str, Dict[str, Any]] = {}


# --------------------------------------------------------------------------- #
#  The Vulnerable API                                                         #
# --------------------------------------------------------------------------- #

class VulnerableAPI:
    """Simulates a REST API with planted OWASP Top 10 vulnerabilities.

    Each handler returns an APIResponse and optionally triggers vulnerability
    detection when the agent sends the right exploit payload.
    """

    # Route table: (method, endpoint_pattern) -> handler
    _ROUTES = [
        ("GET",  r"^/$"),
        ("GET",  r"^/api$"),
        ("POST", r"^/auth/login$"),
        ("GET",  r"^/users/(\d+)$"),
        ("GET",  r"^/users$"),
        ("GET",  r"^/products$"),
        ("GET",  r"^/products/(\d+)$"),
        ("POST", r"^/reviews$"),
        ("GET",  r"^/reviews$"),
        ("GET",  r"^/admin/users$"),
        ("GET",  r"^/debug/config$"),
        ("POST", r"^/orders$"),
    ]

    # Which endpoints are publicly known (shown in API root)
    _PUBLIC_ENDPOINTS = [
        "POST /auth/login",
        "GET /products",
        "GET /products/{id}",
        "GET /reviews",
        "POST /reviews",
        "POST /orders",
    ]

    def __init__(self, active_vulns: List[str]):
        """Initialize with a set of active vulnerability IDs.

        Args:
            active_vulns: List of vulnerability IDs to activate.
                Valid IDs: info_disclosure, broken_auth, idor, sqli, xss
        """
        self.active_vulns = set(active_vulns)
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._next_token_id = 1
        self._reviews: List[Dict[str, Any]] = list(_REVIEWS_DB)
        self._next_review_id = len(_REVIEWS_DB) + 1
        self._order_timestamps: List[float] = []

    def handle_request(
        self,
        method: str,
        endpoint: str,
        headers: Dict[str, str],
        params: Dict[str, str],
        body: Dict[str, Any],
        sec_state: EpisodeSecurityState,
    ) -> Tuple[APIResponse, List[str], Set[str]]:
        """Process an HTTP-like request and return response, hints, and newly exploited vulns.

        Returns:
            Tuple of (APIResponse, new_hints, newly_exploited_vuln_ids)
        """
        method = method.upper()
        endpoint = endpoint.rstrip("/") or "/"
        sec_state.total_requests += 1

        new_hints: List[str] = []
        new_vulns: Set[str] = set()

        # Track endpoint discovery
        endpoint_key = f"{method} {endpoint}"
        is_new_endpoint = endpoint_key not in sec_state.discovered_endpoints
        if is_new_endpoint:
            sec_state.discovered_endpoints.add(endpoint_key)

        # Route to handler
        handler, match = self._match_route(method, endpoint)
        if handler is None:
            # Unknown endpoint — give hint about API root
            if is_new_endpoint:
                new_hints.append("Try GET / or GET /api to discover available endpoints.")
            return (
                APIResponse(404, {"error": "Not Found", "message": f"No route for {method} {endpoint}"}),
                new_hints,
                new_vulns,
            )

        # Dispatch
        response, hints, vulns = handler(method, endpoint, headers, params, body, sec_state, match)
        new_hints.extend(hints)
        new_vulns.update(vulns)

        return response, new_hints, new_vulns

    def _match_route(self, method: str, endpoint: str):
        """Match request to a handler method."""
        handler_map = {
            r"^/$": self._handle_root,
            r"^/api$": self._handle_root,
            r"^/auth/login$": self._handle_login,
            r"^/users/(\d+)$": self._handle_user_by_id,
            r"^/users$": self._handle_users_list,
            r"^/products$": self._handle_products,
            r"^/products/(\d+)$": self._handle_product_by_id,
            r"^/reviews$": self._handle_reviews,
            r"^/admin/users$": self._handle_admin_users,
            r"^/debug/config$": self._handle_debug_config,
            r"^/orders$": self._handle_orders,
        }

        for route_method, pattern in self._ROUTES:
            if method != route_method and not (method == "GET" and route_method == "GET"):
                # Check exact method match
                if method != route_method:
                    continue
            m = re.match(pattern, endpoint)
            if m and method == route_method:
                return handler_map.get(pattern), m

        return None, None

    def _get_auth_user(self, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Extract authenticated user from Authorization header."""
        auth = headers.get("Authorization", headers.get("authorization", ""))
        if auth.startswith("Bearer "):
            token = auth[7:]
            return self._tokens.get(token)
        return None

    # ---- Route Handlers --------------------------------------------------- #

    def _handle_root(self, method, endpoint, headers, params, body, sec_state, match):
        """GET / or GET /api — API discovery endpoint."""
        hints = []
        vulns = set()

        response_body = {
            "name": "SecureShop API",
            "version": "1.0.0",
            "endpoints": self._PUBLIC_ENDPOINTS,
            "note": "Authenticate via POST /auth/login to access protected resources.",
        }

        # Hint toward hidden endpoints
        hints.append("The API has more endpoints than listed here. Try common paths like /admin, /debug, /users.")

        return APIResponse(200, response_body), hints, vulns

    def _handle_login(self, method, endpoint, headers, params, body, sec_state, match):
        """POST /auth/login — Broken Authentication vulnerability."""
        hints = []
        vulns = set()

        username = body.get("username", "")
        password = body.get("password", "")

        if not username or not password:
            hints.append("Login requires 'username' and 'password' in the request body.")
            return (
                APIResponse(400, {"error": "Bad Request", "message": "Missing username or password"}),
                hints, vulns,
            )

        sec_state.login_attempts += 1

        # Find user
        user = None
        for u in _USERS_DB.values():
            if u["username"] == username:
                user = u
                break

        if user is None:
            # Vulnerable: leaks whether username exists
            if "broken_auth" in self.active_vulns:
                hints.append("The API reveals whether a username exists — this is an information leak.")
            return (
                APIResponse(401, {"error": "Unauthorized", "message": f"User '{username}' not found"}),
                hints, vulns,
            )

        if user["password"] != password:
            # Hint about weak passwords
            if "broken_auth" in self.active_vulns and sec_state.login_attempts >= 2:
                hints.append("No account lockout detected. Common weak passwords: admin123, password, 123456.")
            return (
                APIResponse(401, {"error": "Unauthorized", "message": "Invalid password"}),
                hints, vulns,
            )

        # Successful login
        token = f"tok_{self._next_token_id}_{username}"
        self._next_token_id += 1
        self._tokens[token] = {"user_id": user["id"], "username": username, "role": user["role"]}
        sec_state.auth_tokens[username] = token

        # Check if broken_auth was exploited (logged in as admin with weak creds)
        if "broken_auth" in self.active_vulns and username == "admin":
            vulns.add("broken_auth")
            hints.append("VULNERABILITY EXPLOITED: Broken Authentication — logged in as admin with weak password.")

        return (
            APIResponse(200, {
                "message": "Login successful",
                "token": token,
                "user": {"id": user["id"], "username": username, "role": user["role"]},
            }),
            hints, vulns,
        )

    def _handle_user_by_id(self, method, endpoint, headers, params, body, sec_state, match):
        """GET /users/{id} — IDOR vulnerability."""
        hints = []
        vulns = set()
        user_id = int(match.group(1))

        auth_user = self._get_auth_user(headers)

        if auth_user is None:
            hints.append("This endpoint requires authentication. Include 'Authorization: Bearer <token>' header.")
            return (
                APIResponse(401, {"error": "Unauthorized", "message": "Authentication required"}),
                hints, vulns,
            )

        target_user = _USERS_DB.get(user_id)
        if target_user is None:
            return APIResponse(404, {"error": "Not Found", "message": f"User {user_id} not found"}), hints, vulns

        # IDOR: No authorization check — any authenticated user can view any profile
        if "idor" in self.active_vulns and auth_user["user_id"] != user_id:
            vulns.add("idor")
            hints.append(
                "VULNERABILITY EXPLOITED: IDOR — accessed another user's profile "
                f"(you are user {auth_user['user_id']}, viewed user {user_id})."
            )

        # Return sensitive data (part of the IDOR impact)
        user_data = {
            "id": target_user["id"],
            "username": target_user["username"],
            "email": target_user["email"],
            "role": target_user["role"],
            "balance": target_user["balance"],
        }

        # If IDOR is active and accessing someone else's data, also return SSN (sensitive data leak)
        if "idor" in self.active_vulns and auth_user["user_id"] != user_id:
            user_data["ssn"] = target_user["ssn"]

        return APIResponse(200, {"user": user_data}), hints, vulns

    def _handle_users_list(self, method, endpoint, headers, params, body, sec_state, match):
        """GET /users — Requires admin auth."""
        hints = []
        vulns = set()

        auth_user = self._get_auth_user(headers)
        if auth_user is None:
            hints.append("Authentication required. Try logging in first via POST /auth/login.")
            return (
                APIResponse(401, {"error": "Unauthorized", "message": "Authentication required"}),
                hints, vulns,
            )

        if auth_user.get("role") != "admin":
            hints.append("This endpoint requires admin privileges. Try accessing individual user profiles instead.")
            return (
                APIResponse(403, {"error": "Forbidden", "message": "Admin access required"}),
                hints, vulns,
            )

        users = [
            {"id": u["id"], "username": u["username"], "role": u["role"]}
            for u in _USERS_DB.values()
        ]
        return APIResponse(200, {"users": users}), hints, vulns

    def _handle_products(self, method, endpoint, headers, params, body, sec_state, match):
        """GET /products — SQL Injection vulnerability in search param."""
        hints = []
        vulns = set()

        search = params.get("search", "")

        if not search:
            # Return public products (exclude internal)
            products = [p for p in _PRODUCTS_DB if p["category"] != "internal"]
            hints.append("Try searching for products using the 'search' query parameter.")
            return APIResponse(200, {"products": products, "count": len(products)}), hints, vulns

        # Check for SQL injection patterns
        sqli_patterns = [
            r"'.*OR.*1\s*=\s*1",    # ' OR 1=1
            r"'.*UNION.*SELECT",     # UNION SELECT
            r";\s*DROP\s+TABLE",     # ; DROP TABLE
            r"'.*--",                # ' --
            r"'\s*OR\s*'",           # ' OR '
            r"1\s*=\s*1",           # 1=1
        ]

        is_sqli = any(re.search(pat, search, re.IGNORECASE) for pat in sqli_patterns)

        if is_sqli and "sqli" in self.active_vulns:
            # SQL injection successful — return ALL products including internal
            vulns.add("sqli")
            hints.append(
                "VULNERABILITY EXPLOITED: SQL Injection — the search parameter was "
                "injected into a raw SQL query, bypassing filters."
            )
            return (
                APIResponse(200, {
                    "products": _PRODUCTS_DB,  # ALL including internal
                    "count": len(_PRODUCTS_DB),
                    "debug_info": f"SELECT * FROM products WHERE name LIKE '%{search}%'",
                }),
                hints, vulns,
            )

        # Normal search
        if "sqli" in self.active_vulns and ("'" in search or "\"" in search):
            # Hint: the API shows a SQL error on quote characters
            hints.append(
                "The server returned a database error when special characters were used. "
                "This may indicate SQL injection vulnerability."
            )
            return (
                APIResponse(500, {
                    "error": "Internal Server Error",
                    "message": f"Database error: unterminated string near '{search}'",
                }),
                hints, vulns,
            )

        # Normal search (case-insensitive, exclude internal)
        results = [
            p for p in _PRODUCTS_DB
            if search.lower() in p["name"].lower() and p["category"] != "internal"
        ]
        return APIResponse(200, {"products": results, "count": len(results)}), hints, vulns

    def _handle_product_by_id(self, method, endpoint, headers, params, body, sec_state, match):
        """GET /products/{id}."""
        hints = []
        vulns = set()
        product_id = int(match.group(1))

        product = next((p for p in _PRODUCTS_DB if p["id"] == product_id), None)
        if product is None:
            return APIResponse(404, {"error": "Not Found", "message": f"Product {product_id} not found"}), hints, vulns

        return APIResponse(200, {"product": product}), hints, vulns

    def _handle_reviews(self, method, endpoint, headers, params, body, sec_state, match):
        """GET/POST /reviews — XSS vulnerability in POST body."""
        hints = []
        vulns = set()

        if method == "GET":
            return APIResponse(200, {"reviews": self._reviews}), hints, vulns

        # POST — create a review
        if method == "POST":
            auth_user = self._get_auth_user(headers)
            if auth_user is None:
                hints.append("Posting reviews requires authentication.")
                return (
                    APIResponse(401, {"error": "Unauthorized", "message": "Authentication required"}),
                    hints, vulns,
                )

            review_body = body.get("body", "")
            product_id = body.get("product_id", 0)
            rating = body.get("rating", 5)

            if not review_body:
                hints.append("Review body cannot be empty. Include 'body', 'product_id', and 'rating' in the request.")
                return (
                    APIResponse(400, {"error": "Bad Request", "message": "Missing review body"}),
                    hints, vulns,
                )

            # Check for XSS patterns
            xss_patterns = [
                r"<script[^>]*>",           # <script> tags
                r"javascript:",              # javascript: protocol
                r"on\w+\s*=",               # event handlers (onclick=, onerror=)
                r"<img[^>]+onerror",        # <img onerror=
                r"<svg[^>]+onload",         # <svg onload=
            ]

            is_xss = any(re.search(pat, review_body, re.IGNORECASE) for pat in xss_patterns)

            review = {
                "id": self._next_review_id,
                "product_id": product_id,
                "user_id": auth_user["user_id"],
                "rating": rating,
                "body": review_body,  # No sanitization!
            }
            self._next_review_id += 1
            self._reviews.append(review)

            if is_xss and "xss" in self.active_vulns:
                vulns.add("xss")
                hints.append(
                    "VULNERABILITY EXPLOITED: Stored XSS — the review body was stored "
                    "without sanitization and will be rendered to other users."
                )
                return (
                    APIResponse(201, {
                        "message": "Review created",
                        "review": review,
                        "warning": "Content stored as-is (no sanitization applied)",
                    }),
                    hints, vulns,
                )

            return APIResponse(201, {"message": "Review created", "review": review}), hints, vulns

        return APIResponse(405, {"error": "Method Not Allowed"}), hints, vulns

    def _handle_admin_users(self, method, endpoint, headers, params, body, sec_state, match):
        """GET /admin/users — Hidden admin endpoint."""
        hints = []
        vulns = set()

        auth_user = self._get_auth_user(headers)

        # This endpoint exists but isn't in the public docs
        hints.append("You found a hidden admin endpoint not listed in the public API documentation.")

        if auth_user is None:
            hints.append("This admin endpoint requires authentication.")
            return (
                APIResponse(401, {"error": "Unauthorized", "message": "Authentication required"}),
                hints, vulns,
            )

        if auth_user.get("role") != "admin":
            return (
                APIResponse(403, {"error": "Forbidden", "message": "Admin access required"}),
                hints, vulns,
            )

        # Admin can see all users with full details
        users = [
            {"id": u["id"], "username": u["username"], "email": u["email"],
             "role": u["role"], "balance": u["balance"]}
            for u in _USERS_DB.values()
        ]
        return APIResponse(200, {"admin_panel": True, "users": users, "total": len(users)}), hints, vulns

    def _handle_debug_config(self, method, endpoint, headers, params, body, sec_state, match):
        """GET /debug/config — Information Disclosure vulnerability."""
        hints = []
        vulns = set()

        if "info_disclosure" in self.active_vulns:
            vulns.add("info_disclosure")
            hints.append(
                "VULNERABILITY EXPLOITED: Information Disclosure — the /debug/config "
                "endpoint exposes database credentials, API keys, and JWT secret."
            )
            return (
                APIResponse(200, {
                    "config": _DEBUG_CONFIG,
                    "warning": "Debug endpoint should be disabled in production!",
                }),
                hints, vulns,
            )

        return (
            APIResponse(404, {"error": "Not Found", "message": "Endpoint not available"}),
            hints, vulns,
        )

    def _handle_orders(self, method, endpoint, headers, params, body, sec_state, match):
        """POST /orders — Rate limiting test (not a scored vulnerability)."""
        hints = []
        vulns = set()

        if method != "POST":
            return APIResponse(405, {"error": "Method Not Allowed"}), hints, vulns

        auth_user = self._get_auth_user(headers)
        if auth_user is None:
            hints.append("Creating orders requires authentication.")
            return (
                APIResponse(401, {"error": "Unauthorized", "message": "Authentication required"}),
                hints, vulns,
            )

        product_id = body.get("product_id")
        quantity = body.get("quantity", 1)

        if not product_id:
            return (
                APIResponse(400, {"error": "Bad Request", "message": "Missing product_id"}),
                hints, vulns,
            )

        product = next((p for p in _PRODUCTS_DB if p["id"] == product_id), None)
        if product is None:
            return APIResponse(404, {"error": "Not Found", "message": "Product not found"}), hints, vulns

        return (
            APIResponse(201, {
                "message": "Order created",
                "order": {
                    "product": product["name"],
                    "quantity": quantity,
                    "total": product["price"] * quantity,
                },
            }),
            hints, vulns,
        )

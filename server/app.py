# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Api Security Rl Environment.

This module creates an HTTP server that exposes the ApiSecurityRlEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from api_security_rl.models import ApiSecurityRlAction, ApiSecurityRlObservation
    from .api_security_rl_environment import ApiSecurityRlEnvironment
except ModuleNotFoundError:
    from models import ApiSecurityRlAction, ApiSecurityRlObservation
    from server.api_security_rl_environment import ApiSecurityRlEnvironment


# Create the app with web interface and README integration
app = create_app(
    ApiSecurityRlEnvironment,
    ApiSecurityRlAction,
    ApiSecurityRlObservation,
    env_name="api_security_rl",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m api_security_rl.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn api_security_rl.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)

import os
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

@app.get("/web", response_class=HTMLResponse)
def web():
    web_html_path = os.path.join(os.path.dirname(__file__), "web.html")
    with open(web_html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/inference-stream")
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()
    env = os.environ.copy()
    env["API_BASE_URL_FOR_ENV"] = "http://127.0.0.1:8000"
    
    # Manually load .env from the root directory so the subprocess gets the HF_TOKEN keys
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    # Remove potential surrounding quotes from the value
                    v = v.strip().strip("'").strip('"')
                    env[k.strip()] = v
    
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference.py")
    
    process = await asyncio.create_subprocess_exec(
        "python", script_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env
    )
    
    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            await websocket.send_text(line.decode("utf-8").strip())
        await process.wait()
        await websocket.send_text("[PROCESS COMPLETED]")
    except WebSocketDisconnect:
        process.terminate()

if __name__ == '__main__':
    main()

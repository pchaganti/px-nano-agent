"""Backwards compatibility wrapper - imports from nano_agent package.

This script is maintained for backwards compatibility with existing usage:
    uv run python scripts/capture_claude_code_auth.py
    uv run nano-agent-capture-auth

The actual implementation has moved to nano_agent/capture_claude_code_auth.py.
"""

from nano_agent.capture_claude_code_auth import (
    get_config,
    get_headers,
    main,
)

__all__ = ["get_config", "get_headers", "main"]

if __name__ == "__main__":
    main()

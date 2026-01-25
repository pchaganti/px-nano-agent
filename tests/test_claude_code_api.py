"""Tests for ClaudeCodeAPI."""

from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nano_agent import DAG, ClaudeCodeAPI, Message, Role

# Mock captured config for testing
MOCK_HEADERS: dict[str, str] = {
    "x-api-key": "sk-ant-test123456789",
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
    "user-agent": "claude-cli/2.0.76 (external, cli)",
    "x-app": "cli",
}

MOCK_BODY_PARAMS: dict[str, Any] = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 16384,
    "temperature": 1.0,
    "thinking": {"budget_tokens": 10000, "type": "enabled"},
    "user_id": "test-user-123",
    "system": [
        {
            "type": "text",
            "text": "You are Claude Code, Anthropic's official CLI for Claude.",
            "cache_control": {"type": "ephemeral"},
        }
    ],
    "url_path": "/v1/messages?beta=true",
}


@pytest.fixture
def mock_get_config() -> Generator[MagicMock, None, None]:
    """Mock get_config to avoid actual CLI capture."""
    with patch("nano_agent.claude_code_api.get_config") as mock:
        mock.return_value = (MOCK_HEADERS.copy(), MOCK_BODY_PARAMS.copy())
        yield mock


class TestClaudeCodeAPIInit:
    """Test ClaudeCodeAPI initialization."""

    def test_init_captures_config(self, mock_get_config: MagicMock) -> None:
        """Verify capture happens at init."""
        api = ClaudeCodeAPI()
        mock_get_config.assert_called_once_with(timeout=10)
        assert api.headers == MOCK_HEADERS

    def test_init_uses_captured_values(self, mock_get_config: MagicMock) -> None:
        """Verify captured values are used."""
        api = ClaudeCodeAPI()
        assert api.model == "claude-sonnet-4-20250514"
        assert api.max_tokens == 16384
        assert api.temperature == 1.0
        assert api.thinking_budget == 10000
        assert api.user_id == "test-user-123"

    def test_parameter_overrides(self, mock_get_config: MagicMock) -> None:
        """Verify model/max_tokens can be overridden."""
        api = ClaudeCodeAPI(
            model="claude-opus-4-5-20251101",
            max_tokens=4096,
            temperature=0.5,
            thinking_budget=5000,
        )
        assert api.model == "claude-opus-4-5-20251101"
        assert api.max_tokens == 4096
        assert api.temperature == 0.5
        assert api.thinking_budget == 5000

    def test_init_raises_on_capture_failure(self) -> None:
        """Verify clear error if capture fails."""
        with patch("nano_agent.claude_code_api.get_config") as mock:
            mock.side_effect = RuntimeError("Claude CLI not found")
            with pytest.raises(RuntimeError) as exc_info:
                ClaudeCodeAPI()
            assert "Failed to capture" in str(exc_info.value)

    def test_init_raises_on_timeout(self) -> None:
        """Verify clear error if capture times out."""
        with patch("nano_agent.claude_code_api.get_config") as mock:
            mock.side_effect = TimeoutError("Timeout")
            with pytest.raises(RuntimeError) as exc_info:
                ClaudeCodeAPI()
            assert "Failed to capture" in str(exc_info.value)


class TestClaudeCodeAPIRepr:
    """Test ClaudeCodeAPI repr."""

    def test_repr(self, mock_get_config: MagicMock) -> None:
        """Verify clean representation."""
        api = ClaudeCodeAPI()
        repr_str = repr(api)
        assert "ClaudeCodeAPI(" in repr_str
        assert "model='claude-sonnet-4-20250514'" in repr_str


class TestClaudeCodeAPISend:
    """Test ClaudeCodeAPI send method."""

    @pytest.fixture
    def mock_httpx_response(self) -> MagicMock:
        """Create a mock HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        return mock_response

    @pytest.mark.asyncio
    async def test_send_uses_captured_headers(
        self, mock_get_config: MagicMock, mock_httpx_response: MagicMock
    ) -> None:
        """Verify send() uses captured auth."""
        api = ClaudeCodeAPI()
        with patch.object(api._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_httpx_response
            messages = [Message(role=Role.USER, content="Hello")]
            await api.send(messages)
            headers = mock_post.call_args.kwargs["headers"]
            assert headers["x-api-key"] == "sk-ant-test123456789"

    @pytest.mark.asyncio
    async def test_send_with_dag(
        self, mock_get_config: MagicMock, mock_httpx_response: MagicMock
    ) -> None:
        """Verify send() works with DAG input."""
        api = ClaudeCodeAPI()
        with patch.object(api._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_httpx_response
            dag = DAG().system("Be helpful.").user("Hello")
            response = await api.send(dag)
            assert response.get_text() == "Hello!"

    @pytest.mark.asyncio
    async def test_send_uses_captured_system(
        self, mock_get_config: MagicMock, mock_httpx_response: MagicMock
    ) -> None:
        """Verify captured system prompt is preserved."""
        api = ClaudeCodeAPI()
        with patch.object(api._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_httpx_response
            messages = [Message(role=Role.USER, content="Hello")]
            await api.send(messages)
            json_body = mock_post.call_args.kwargs["json"]
            assert json_body["system"] == MOCK_BODY_PARAMS["system"]


class TestClaudeCodeAPIContextManager:
    """Test ClaudeCodeAPI async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_get_config: MagicMock) -> None:
        """Verify async context manager works."""
        async with ClaudeCodeAPI() as api:
            assert api.model == "claude-sonnet-4-20250514"


class TestClaudeCodeAPIDefaults:
    """Test default values when captured config is incomplete."""

    def test_defaults_when_captured_values_missing(self) -> None:
        """Verify sensible defaults when captured values are None."""
        with patch("nano_agent.claude_code_api.get_config") as mock:
            mock.return_value = (
                {"authorization": "Bearer test"},
                {
                    "model": None,
                    "max_tokens": None,
                    "temperature": None,
                    "thinking": None,
                    "user_id": None,
                    "system": None,
                    "url_path": None,
                },
            )
            api = ClaudeCodeAPI()
            assert api.model == "claude-haiku-4-5-20251001"
            assert api.max_tokens == 2048
            assert api.temperature == 1

"""E2E test: Verify executor cancellation produces valid tool results."""

import asyncio
from dataclasses import dataclass

from nano_agent import (
    DAG,
    BashTool,
    CancellationToken,
    ClaudeCodeAPI,
    ReadTool,
    Role,
    TextContent,
    ToolResultContent,
    run,
)
from nano_agent.tools import Tool

# --- Test Tools ---


@dataclass
class SleepInput:
    """Input for Sleep5Tool (no parameters needed)."""

    pass


class Sleep5Tool(Tool):
    """A tool that always sleeps for 5 seconds."""

    name = "Sleep5Seconds"
    description = (
        "Sleep for exactly 5 seconds. Always use this when asked to wait or pause."
    )

    def __init__(self):
        self._input_type = SleepInput

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def __call__(self, input: SleepInput) -> TextContent:
        await asyncio.sleep(5)
        return TextContent(text="Slept for 5 seconds.")


async def test_cancellation_during_api_call():
    """
    Test cancellation during API call (before tool calls are returned).
    The conversation should still be able to continue.
    """
    print("\n--- Test: Cancellation during API call ---")
    api = ClaudeCodeAPI()

    dag = (
        DAG()
        .system("You are a helpful assistant.")
        .tools(BashTool(), ReadTool())
        .user("Run 'echo hello' and read /etc/hosts.")
    )

    token = CancellationToken()

    # Cancel almost immediately - likely during API call
    async def cancel_quickly():
        await asyncio.sleep(0.1)
        token.cancel()

    asyncio.create_task(cancel_quickly())

    dag = await run(api, dag, cancel_token=token)

    # The DAG should be in a valid state regardless of when cancellation happened
    messages = dag.to_messages()
    print(f"  Messages after cancellation: {len(messages)}")

    # Continue conversation - this should work without API errors
    token.reset()
    dag = dag.user("Say 'hello' and nothing else.")
    dag = await run(api, dag, cancel_token=token)

    response_text = _get_last_response(dag)
    print(f"  Continued successfully, response: {response_text[:50]}...")

    await api.close()
    return True


async def test_cancellation_during_tool_execution():
    """
    Test cancellation during tool execution.
    All tool_use blocks should have matching tool_results.

    Uses Sleep5Tool which sleeps for exactly 5 seconds, allowing us to
    reliably cancel after the API returns but during tool execution.
    """
    print("\n--- Test: Cancellation during tool execution ---")
    api = ClaudeCodeAPI()

    # Use Sleep5Tool - sleeps for exactly 5 seconds
    dag = (
        DAG()
        .system(
            "You have a Sleep5Seconds tool. When asked to wait or sleep, "
            "you MUST use the Sleep5Seconds tool. Always use tools."
        )
        .tools(Sleep5Tool())
        .user("Please sleep for 5 seconds using the Sleep5Seconds tool.")
    )

    token = CancellationToken()

    # Cancel after 3 seconds - should be during the 5-second sleep
    # (API call typically takes 1-2 seconds to return)
    async def cancel_during_tool():
        await asyncio.sleep(3.0)
        token.cancel()

    asyncio.create_task(cancel_during_tool())

    dag = await run(api, dag, cancel_token=token)

    # Count tool_use and tool_result blocks
    tool_use_count = 0
    tool_result_count = 0
    has_cancelled_result = False
    for msg in dag.to_messages():
        if isinstance(msg.content, list):
            for c in msg.content:
                if hasattr(c, "id") and hasattr(c, "name"):  # ToolUseContent
                    tool_use_count += 1
                if isinstance(c, ToolResultContent):
                    tool_result_count += 1
                    # Check if this is a cancellation result
                    if c.is_error and any(
                        "cancelled" in tc.text.lower() for tc in c.content
                    ):
                        has_cancelled_result = True

    print(f"  Tool uses: {tool_use_count}, Tool results: {tool_result_count}")
    print(f"  Has cancelled result: {has_cancelled_result}")

    # If there were any tool calls, they must all have results
    if tool_use_count > 0:
        assert (
            tool_result_count == tool_use_count
        ), f"Mismatch: {tool_use_count} tool_use but {tool_result_count} tool_result"
        print("  All tool_use blocks have matching tool_results")

    # Continue conversation - should not get API error about missing tool_result
    token.reset()
    dag = dag.user("What happened? Say 'acknowledged'.")
    dag = await run(api, dag, cancel_token=token)

    response_text = _get_last_response(dag)
    print(f"  Continued successfully, response: {response_text[:50]}...")

    await api.close()
    return True


async def test_normal_completion():
    """
    Test that normal (non-cancelled) execution still works.
    """
    print("\n--- Test: Normal completion (no cancellation) ---")
    api = ClaudeCodeAPI()

    dag = (
        DAG()
        .system("You are helpful. Use tools when asked.")
        .tools(BashTool())
        .user("Run 'echo test123'")
    )

    token = CancellationToken()
    dag = await run(api, dag, cancel_token=token)

    response_text = _get_last_response(dag)
    print(f"  Response: {response_text[:80]}...")

    # Verify tool was executed
    tool_result_count = 0
    for msg in dag.to_messages():
        if isinstance(msg.content, list):
            for c in msg.content:
                if isinstance(c, ToolResultContent):
                    tool_result_count += 1
                    if "test123" in c.content[0].text:
                        print(
                            "  Tool executed successfully (found 'test123' in output)"
                        )

    await api.close()
    return tool_result_count > 0


async def test_cancellation_with_multiple_tools():
    """
    Test cancellation when multiple tools are called.
    First tool completes, second tool gets cancelled, remaining tools are skipped.
    All tool_use blocks should have matching tool_results.
    """
    print("\n--- Test: Cancellation with multiple tools ---")
    api = ClaudeCodeAPI()

    # Quick tool that completes fast (but has a small delay for realism)
    @dataclass
    class QuickInput:
        pass

    class QuickTool(Tool):
        name = "QuickAction"
        description = (
            "A quick action that completes in about 1 second. Always call this first."
        )

        def __init__(self):
            self._input_type = QuickInput

        @property
        def input_schema(self) -> dict:
            return {"type": "object", "properties": {}, "required": []}

        async def __call__(self, input: QuickInput) -> TextContent:
            await asyncio.sleep(0.5)  # Small delay
            return TextContent(text="Quick action completed.")

    dag = (
        DAG()
        .system(
            "You have two tools: QuickAction and Sleep5Seconds.\n"
            "IMPORTANT: When asked to perform both actions, you MUST call BOTH tools "
            "in a single response. First QuickAction, then Sleep5Seconds.\n"
            "Do NOT wait for one tool to complete before calling the other."
        )
        .tools(QuickTool(), Sleep5Tool())
        .user(
            "Please perform a quick action AND sleep for 5 seconds. "
            "Call BOTH tools in your response."
        )
    )

    token = CancellationToken()

    # Cancel after 4 seconds - QuickTool should complete, Sleep5Tool should be mid-execution
    # Timeline: ~2s API call, ~0.5s QuickTool, then Sleep5Tool starts
    async def cancel_after_first_tool():
        await asyncio.sleep(4.0)
        token.cancel()

    asyncio.create_task(cancel_after_first_tool())

    dag = await run(api, dag, cancel_token=token)

    # Count tool_use and tool_result blocks
    tool_use_count = 0
    tool_result_count = 0
    completed_count = 0
    cancelled_count = 0

    for msg in dag.to_messages():
        if isinstance(msg.content, list):
            for c in msg.content:
                if hasattr(c, "id") and hasattr(c, "name"):  # ToolUseContent
                    tool_use_count += 1
                if isinstance(c, ToolResultContent):
                    tool_result_count += 1
                    content_text = " ".join(tc.text for tc in c.content)
                    if c.is_error:
                        if "cancelled" in content_text.lower():
                            cancelled_count += 1
                        elif "skipped" in content_text.lower():
                            cancelled_count += 1  # Count skipped as cancelled
                    else:
                        completed_count += 1

    print(f"  Tool uses: {tool_use_count}, Tool results: {tool_result_count}")
    print(f"  Completed: {completed_count}, Cancelled/Skipped: {cancelled_count}")

    # Verify all tool_use have matching tool_result
    if tool_use_count > 0:
        assert (
            tool_result_count == tool_use_count
        ), f"Mismatch: {tool_use_count} tool_use but {tool_result_count} tool_result"
        print("  All tool_use blocks have matching tool_results")

    # Continue conversation
    token.reset()
    dag = dag.user("What happened? Say 'acknowledged'.")
    dag = await run(api, dag, cancel_token=token)

    response_text = _get_last_response(dag)
    print(f"  Continued successfully, response: {response_text[:50]}...")

    await api.close()
    return True


def _get_last_response(dag: DAG) -> str:
    """Extract the last assistant response text from the DAG."""
    response_text = ""
    for msg in dag.to_messages():
        if msg.role == Role.ASSISTANT:
            if isinstance(msg.content, str):
                response_text = msg.content
            elif isinstance(msg.content, list):
                text_parts = []
                for c in msg.content:
                    if hasattr(c, "text"):
                        text_parts.append(c.text)
                if text_parts:
                    response_text = " ".join(text_parts)
    return response_text


async def main():
    print("=" * 60)
    print("E2E Test: Executor Cancellation")
    print("=" * 60)

    tests = [
        ("Normal completion", test_normal_completion),
        ("Cancellation during API call", test_cancellation_during_api_call),
        ("Cancellation during tool execution", test_cancellation_during_tool_execution),
        ("Cancellation with multiple tools", test_cancellation_with_multiple_tools),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = await test_fn()
            results.append((name, success, None))
        except Exception as e:
            import traceback

            results.append((name, False, traceback.format_exc()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = 0
    for name, success, error in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {status}: {name}")
        if error:
            print(f"    Error: {error.splitlines()[-1]}")
        if success:
            passed += 1

    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))

"""ClaudeCodeAPI: Automatic auth capture from Claude CLI with tool usage."""

import asyncio

from nano_agent import DAG, BashTool, ClaudeCodeAPI, run


async def main() -> None:
    # Capture auth from Claude CLI at init
    print("Capturing auth from Claude CLI...")
    api = ClaudeCodeAPI()
    print(f"Captured: model={api.model}")

    # Build conversation with a tool
    dag = (
        DAG()
        .system("Be concise.")
        .tools(BashTool())
        .user("What is the current directory? Use bash to find out.")
    )

    # Run agent loop (handles tool calls automatically)
    print("\nRunning agent...")
    dag = await run(api, dag)
    print(dag)


if __name__ == "__main__":
    asyncio.run(main())

"""Async Tool: Both API calls and tool execution are async."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from nano_agent import (
    DAG,
    ClaudeAPI,
    TextContent,
    Tool,
    ToolResultContent,
)


@dataclass
class CalculatorInput:
    expr: str


@dataclass
class Calculator(Tool):
    name: str = "calculator"
    description: str = "Evaluate a math expression"

    async def __call__(self, input: CalculatorInput) -> TextContent:
        await asyncio.sleep(0.1)  # Simulate async work
        return TextContent(text=str(eval(input.expr)))  # noqa: S307


async def main() -> None:
    calc = Calculator()
    api = ClaudeAPI()
    dag = DAG().tools(calc).user("What is 23 * 47?")

    # Async API call
    response = await api.send(dag)
    dag = dag.assistant(response.content)

    # Async tool execution
    for call in response.get_tool_use():
        result = await calc.execute(call.input)
        content = result if isinstance(result, list) else [result]
        dag = dag.tool_result(ToolResultContent(tool_use_id=call.id, content=content))

    # Async final response
    response = await api.send(dag)
    dag = dag.assistant(response.content)
    print(dag)
    dag.save("dag.json")


if __name__ == "__main__":
    asyncio.run(main())

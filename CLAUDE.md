# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

**nano agent** - A minimalistic Python library for building AI agents using functional, immutable DAG operations.

- **Immutable DAG**: Every operation returns a new DAG instance
- **Node-based**: System prompts, messages, tool calls, results are all nodes
- **Tools are callable**: Tools define schema for API and implement `__call__` for execution

## Development Commands

```bash
# Testing
uv run pytest
uv run pytest tests/test_dag.py -v

# Type checking & formatting
uv run mypy .
uv run black .
uv run isort .
uv run pre-commit run --all-files

# Run examples
uv run python examples/hello_world.py
uv run python examples/simple_tool.py
```

## Project Structure

```
nano_agent/
├── nano_agent/
│   ├── dag.py              # Node and DAG classes
│   ├── data_structures.py  # Message, ContentBlock types
│   ├── claude_api.py       # ClaudeAPI client
│   ├── executor.py         # Agent execution loop (run())
│   └── tools.py            # Built-in tools (Bash, Read, Edit, etc.)
├── examples/
├── tests/
└── scripts/
    ├── viewer.py           # HTML visualization
    └── console_viewer.py   # ASCII visualization
```

## Key Patterns

```python
# Immutable DAG operations
dag = DAG()
dag = dag.system("...")
dag = dag.tools(BashTool())
dag = dag.user("...")
dag = dag.assistant(response.content)

# Tool execution loop
dag = await run(api, dag)  # Handles tool calls automatically

# Manual tool handling
for call in response.get_tool_use():
    result = await tool.execute(call.input)
    dag = dag.tool_result(ToolResultContent(tool_use_id=call.id, content=[result]))
```

## Notes

- `ClaudeAPI()` uses `ANTHROPIC_API_KEY` env var
- Tools return `TextContent` or `list[TextContent]`
- `ToolResultContent.tool_name` is for display only - excluded from API calls via `to_dict()`

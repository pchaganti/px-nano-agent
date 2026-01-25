# Tool System

Tools are the bridge between AI models and the real world. In nano-agent, tools define their input schema via type annotations and implement execution logic via `__call__`.

## Tool Anatomy

Every tool has two required fields and an execution method:

```python
@dataclass
class Tool:
    name: str           # Tool identifier
    description: str    # What the tool does (for the model)

    async def __call__(self, input: MyInput) -> TextContent:
        """Execute the tool. Input schema is inferred from type annotation."""
        ...
```

The `input_schema` is **automatically generated** from the `__call__` type annotation.

## Creating Custom Tools

### Basic Tool

```python
from dataclasses import dataclass
from typing import Annotated
from nano_agent import Tool, TextContent, Desc

# Define input structure
@dataclass
class CalculatorInput:
    expression: Annotated[str, Desc("Math expression to evaluate")]

# Define the tool
@dataclass
class Calculator(Tool):
    name: str = "calculator"
    description: str = "Evaluate mathematical expressions"

    async def __call__(self, input: CalculatorInput) -> TextContent:
        result = eval(input.expression)
        return TextContent(text=str(result))
```

The schema is auto-generated:
```json
{
  "type": "object",
  "properties": {
    "expression": {
      "type": "string",
      "description": "Math expression to evaluate"
    }
  },
  "required": ["expression"]
}
```

### No-Input Tool

For tools that take no input, simply omit the parameter:

```python
@dataclass
class SystemInfoTool(Tool):
    name: str = "system_info"
    description: str = "Get system information"

    async def __call__(self) -> TextContent:
        import platform
        return TextContent(text=f"OS: {platform.system()}")
```

### Tool with Optional Parameters

Use default values for optional fields:

```python
@dataclass
class SearchInput:
    query: Annotated[str, Desc("Search query")]
    limit: Annotated[int, Desc("Max results")] = 10

@dataclass
class SearchTool(Tool):
    name: str = "search"
    description: str = "Search for information"

    async def __call__(self, input: SearchInput) -> TextContent:
        return TextContent(text=f"Searching: {input.query}, limit: {input.limit}")
```

## Schema Generation with Desc

The `Desc` class adds descriptions to dataclass fields:

```python
from typing import Annotated
from nano_agent import Desc

@dataclass
class WeatherInput:
    city: Annotated[str, Desc("City name like 'Tokyo' or 'New York'")]
    units: Annotated[str, Desc("Temperature units")] = "celsius"
```

### Type Mapping

| Python Type | JSON Schema Type |
|-------------|------------------|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |

## Built-in Tools

nano-agent provides several built-in tools. All tools support automatic output truncation (see [Output Truncation](#output-truncation) below).

### BashTool

Execute shell commands:

```python
from nano_agent import BashTool

bash = BashTool()
result = await bash.execute({"command": "ls -la"})
```

### ReadTool

Read files from the filesystem:

```python
from nano_agent import ReadTool

read = ReadTool()
result = await read.execute({"file_path": "/path/to/file.py"})
```

### WriteTool

Write content to files:

```python
from nano_agent import WriteTool

write = WriteTool()
result = await write.execute({"file_path": "/path/to/file.py", "content": "..."})
```

### EditTool

Perform string replacements in files:

```python
from nano_agent import EditTool

edit = EditTool()
result = await edit.execute({
    "file_path": "/path/to/file.py",
    "old_string": "def old",
    "new_string": "def new"
})
```

### EditConfirmTool

Confirm and apply a pending edit:

```python
from nano_agent import EditConfirmTool

confirm = EditConfirmTool()
result = await confirm.execute({"edit_id": "abc12345"})
```

### GlobTool

Find files by pattern:

```python
from nano_agent import GlobTool

glob = GlobTool()
result = await glob.execute({"pattern": "**/*.py"})
```

### SearchTool

Search file contents:

```python
from nano_agent import SearchTool

search = SearchTool()
result = await search.execute({"pattern": "def main", "path": "src/"})
```

### StatTool

Get file metadata:

```python
from nano_agent import StatTool

stat = StatTool()
result = await stat.execute({"file_path": "/path/to/file.py"})
```

### WebFetchTool

Fetch web content:

```python
from nano_agent import WebFetchTool

fetch = WebFetchTool()
result = await fetch.execute({"url": "https://example.com", "prompt": "Summarize"})
```

### PythonTool

Create and run Python scripts:

```python
from nano_agent import PythonTool

python = PythonTool()
result = await python.execute({"operation": "create", "code": "print('hello')"})
result = await python.execute({"operation": "run", "file_id": "py_abc123"})
```

### TodoWriteTool

Manage task lists:

```python
from nano_agent import TodoWriteTool

todo = TodoWriteTool()
await todo.execute({"todos": [
    {"content": "Write docs", "status": "in_progress"},
]})
```

## Using Tools with DAG

Register tools with the DAG:

```python
from nano_agent import DAG, BashTool, ReadTool

dag = (
    DAG()
    .system("You are a coding assistant.")
    .tools(BashTool(), ReadTool())
    .user("List the Python files")
)
```

## Complete Example

```python
import asyncio
from dataclasses import dataclass
from typing import Annotated
from nano_agent import DAG, ClaudeAPI, Tool, TextContent, Desc, run

@dataclass
class WeatherInput:
    city: Annotated[str, Desc("City name")]

@dataclass
class GetWeather(Tool):
    name: str = "get_weather"
    description: str = "Get current weather for a city"

    async def __call__(self, input: WeatherInput) -> TextContent:
        weather = {"Tokyo": "22C", "Paris": "18C"}.get(input.city, "Unknown")
        return TextContent(text=f"{input.city}: {weather}")

async def main():
    api = ClaudeAPI()
    dag = (
        DAG()
        .tools(GetWeather())
        .user("What's the weather in Tokyo?")
    )
    dag = await run(api, dag)
    print(dag)

asyncio.run(main())
```

## Tool Return Types

```python
# Single result
async def __call__(self, input: MyInput) -> TextContent:
    return TextContent(text="Result")

# Multiple results
async def __call__(self, input: MyInput) -> list[TextContent]:
    return [TextContent(text="Part 1"), TextContent(text="Part 2")]
```

## Error Handling

Return error information in the TextContent:

```python
async def __call__(self, input: MyInput) -> TextContent:
    try:
        result = perform_operation(input)
        return TextContent(text=result)
    except FileNotFoundError:
        return TextContent(text="Error: File not found")
```

## Output Truncation

Tools can produce large outputs. nano-agent provides automatic truncation.

### TruncationConfig

```python
from typing import ClassVar
from nano_agent.tools import TruncationConfig

@dataclass
class MyTool(Tool):
    name: str = "my_tool"
    description: str = "A tool with custom truncation"

    _truncation_config: ClassVar[TruncationConfig] = TruncationConfig(
        max_chars=10000,
        max_lines=500,
        enabled=True
    )

    async def __call__(self, input: MyInput) -> TextContent:
        return TextContent(text=large_output)
```

### Defaults

| Parameter | Default |
|-----------|---------|
| `max_chars` | 30000 |
| `max_lines` | 1000 |
| `enabled` | True |

When output exceeds limits, full output is saved to `/tmp/nano_tool_output/` and a truncated version is returned.

## Summary

| Component | Purpose |
|-----------|---------|
| `name` | Tool identifier |
| `description` | What the tool does |
| `__call__` | Execution logic (input schema inferred from type) |
| `execute()` | Execute with truncation and type conversion |
| `Desc` | Add descriptions to input fields |
| `TruncationConfig` | Configure output truncation |

## See Also

- [Custom Tools Tutorial](custom-tools-tutorial.md) - Step-by-step guide
- [Tool Execution](tool-execution.md) - Execution patterns

---

**Next:** [Tool Execution](tool-execution.md)

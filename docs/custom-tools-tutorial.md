# Custom Tools Tutorial

This tutorial walks you through building custom tools for nano-agent, from the simplest case to advanced patterns with state management.

## How Tools Work

Tools are the bridge between AI models and the real world. In nano-agent:

1. **Schema**: Tools define their input schema (JSON Schema) for the API
2. **Execution**: Tools implement `__call__` to execute when Claude requests them
3. **Separation**: The framework never executes tools automatically—you control when side effects happen

```python
@dataclass
class Tool:
    name: str                    # Tool identifier
    description: str             # What the tool does (for the model)
    # input_schema is auto-inferred from __call__ type annotation

    async def __call__(self, input: ...) -> TextContent:
        """Execute the tool with given input."""
        ...
```

## Part 1: Simplest Tool (No Input)

A tool that takes no input—perfect for getting system information:

```python
from dataclasses import dataclass
from nano_agent import Tool, TextContent
import platform

@dataclass
class SystemInfoTool(Tool):
    name: str = "system_info"
    description: str = "Get current system information"

    async def __call__(self) -> TextContent:
        info = f"OS: {platform.system()} {platform.release()}"
        return TextContent(text=info)
```

**Key points:**
- Simply omit the `input` parameter for no-input tools
- Schema is automatically generated as empty: `{"type": "object", "properties": {}, "required": []}`
- Call with `await tool.execute()` (no arguments needed)

## Part 2: Tool with Typed Input

Add input parameters using dataclasses with `Annotated` and `Desc`:

```python
from dataclasses import dataclass
from typing import Annotated
from nano_agent import Tool, TextContent, Desc

# Define input schema as a dataclass
@dataclass
class GreetInput:
    name: Annotated[str, Desc("Person's name to greet")]
    language: Annotated[str, Desc("Language code (en, es, fr, jp)")] = "en"

@dataclass
class GreeterTool(Tool):
    name: str = "greet"
    description: str = "Generate a greeting in different languages"

    async def __call__(self, input: GreetInput) -> TextContent:
        greetings = {
            "en": f"Hello, {input.name}!",
            "es": f"¡Hola, {input.name}!",
            "fr": f"Bonjour, {input.name}!",
            "jp": f"こんにちは、{input.name}さん!",
        }
        greeting = greetings.get(input.language, greetings["en"])
        return TextContent(text=greeting)
```

**Generated JSON Schema:**

```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Person's name to greet"
    },
    "language": {
      "type": "string",
      "description": "Language code (en, es, fr, jp)"
    }
  },
  "required": ["name"],
  "additionalProperties": false
}
```

**Key points:**
- Use `Annotated[type, Desc("description")]` for documented fields
- Fields with defaults become optional in the schema
- The framework auto-converts API dict to your dataclass

## Part 3: Tool with Optional Parameters

Handle optional parameters with default values:

```python
from dataclasses import dataclass
from typing import Annotated
from nano_agent import Tool, TextContent, Desc

@dataclass
class SearchInput:
    query: Annotated[str, Desc("Search query string")]
    max_results: Annotated[int, Desc("Maximum results to return")] = 10
    include_metadata: Annotated[bool, Desc("Include result metadata")] = False

@dataclass
class SearchTool(Tool):
    name: str = "search"
    description: str = "Search for documents"

    async def __call__(self, input: SearchInput) -> TextContent:
        # Access with defaults applied
        results = f"Searching: {input.query}"
        results += f"\nMax results: {input.max_results}"
        results += f"\nWith metadata: {input.include_metadata}"
        return TextContent(text=results)
```

**Type mapping:**

| Python Type | JSON Schema Type |
|-------------|------------------|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |

## Part 4: Tool with Nested Dataclass Inputs

For complex structured inputs, nest dataclasses:

```python
from dataclasses import dataclass
from typing import Annotated
from nano_agent import Tool, TextContent, Desc

@dataclass
class Address:
    street: Annotated[str, Desc("Street address")]
    city: Annotated[str, Desc("City name")]
    country: Annotated[str, Desc("Country code")] = "US"

@dataclass
class Person:
    name: Annotated[str, Desc("Full name")]
    email: Annotated[str, Desc("Email address")]
    address: Annotated[Address, Desc("Mailing address")]

@dataclass
class RegisterInput:
    person: Annotated[Person, Desc("Person to register")]
    newsletter: Annotated[bool, Desc("Subscribe to newsletter")] = True

@dataclass
class RegisterTool(Tool):
    name: str = "register"
    description: str = "Register a new user"

    async def __call__(self, input: RegisterInput) -> TextContent:
        # Nested dataclasses are auto-converted
        person = input.person
        addr = person.address
        return TextContent(
            text=f"Registered: {person.name} at {addr.city}, {addr.country}"
        )
```

The framework recursively converts nested dicts to dataclass instances.

## Part 5: Tool with List Inputs

Handle array inputs with `list[T]`:

```python
from dataclasses import dataclass, field
from typing import Annotated
from nano_agent import Tool, TextContent, Desc

@dataclass
class FileItem:
    path: Annotated[str, Desc("File path")]
    content: Annotated[str, Desc("File content")]

@dataclass
class BatchWriteInput:
    files: Annotated[list[FileItem], Desc("Files to write")]
    dry_run: Annotated[bool, Desc("Preview without writing")] = False

@dataclass
class BatchWriteTool(Tool):
    name: str = "batch_write"
    description: str = "Write multiple files at once"

    async def __call__(self, input: BatchWriteInput) -> TextContent:
        results = []
        for file in input.files:
            action = "Would write" if input.dry_run else "Wrote"
            results.append(f"{action}: {file.path} ({len(file.content)} chars)")
        return TextContent(text="\n".join(results))
```

**Generated schema for list of dataclasses:**

```json
{
  "type": "object",
  "properties": {
    "files": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "File path"},
          "content": {"type": "string", "description": "File content"}
        },
        "required": ["path", "content"]
      },
      "description": "Files to write"
    }
  }
}
```

## Part 6: Async Tool with External API

Make HTTP requests to external services:

```python
from dataclasses import dataclass
from typing import Annotated
import httpx
from nano_agent import Tool, TextContent, Desc

@dataclass
class WeatherInput:
    city: Annotated[str, Desc("City name (e.g., 'London', 'Tokyo')")]

@dataclass
class WeatherTool(Tool):
    name: str = "weather"
    description: str = "Get current weather for a city"

    async def __call__(self, input: WeatherInput) -> TextContent:
        # Using a free weather API
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"https://wttr.in/{input.city}?format=j1",
                    timeout=10.0
                )
                data = response.json()
                current = data["current_condition"][0]

                weather = (
                    f"Weather in {input.city}:\n"
                    f"  Temperature: {current['temp_C']}°C\n"
                    f"  Condition: {current['weatherDesc'][0]['value']}\n"
                    f"  Humidity: {current['humidity']}%"
                )
                return TextContent(text=weather)
            except Exception as e:
                return TextContent(text=f"Error fetching weather: {e}")
```

**Key points:**
- Use `async with` for HTTP clients
- Always handle errors gracefully
- Return error information in `TextContent`

## Part 7: Tool with Output Truncation

For tools that may produce large outputs, use truncation:

```python
from dataclasses import dataclass
from typing import Annotated, ClassVar
from nano_agent import Tool, TextContent, Desc
from nano_agent.tools import TruncationConfig

@dataclass
class LargeOutputInput:
    lines: Annotated[int, Desc("Number of lines to generate")]

@dataclass
class LargeOutputTool(Tool):
    name: str = "large_output"
    description: str = "Generate large output for testing"

    # Configure truncation limits
    _truncation_config: ClassVar[TruncationConfig] = TruncationConfig(
        max_chars=5000,    # Truncate after 5000 characters
        max_lines=100,     # Or after 100 lines
        enabled=True       # Set to False to disable truncation
    )

    async def __call__(self, input: LargeOutputInput) -> TextContent:
        lines = [f"Line {i}: " + "x" * 50 for i in range(input.lines)]
        return TextContent(text="\n".join(lines))
```

When output exceeds limits:
1. Full output is saved to a temp file
2. Truncated output is returned with a notice
3. The temp file path is included in the response

To disable truncation for a tool:

```python
_truncation_config: ClassVar[TruncationConfig] = TruncationConfig(enabled=False)
```

## Part 8: Tool with State Management

For tools that need to maintain state across calls, add instance variables:

```python
from dataclasses import dataclass, field
from typing import Annotated, ClassVar
from enum import Enum
from nano_agent import Tool, TextContent, Desc

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

@dataclass
class Task:
    id: int
    description: str
    status: TaskStatus

@dataclass
class TaskInput:
    action: Annotated[str, Desc("Action: 'add', 'complete', 'list'")]
    description: Annotated[str, Desc("Task description (for 'add')")] = ""
    task_id: Annotated[int, Desc("Task ID (for 'complete')")] = 0

@dataclass
class TaskManagerTool(Tool):
    name: str = "task_manager"
    description: str = "Manage a simple task list"

    # Instance state (not frozen, can be modified)
    _tasks: list[Task] = field(default_factory=list, repr=False)
    _next_id: int = field(default=1, repr=False)

    async def __call__(self, input: TaskInput) -> TextContent:
        action = input.action.lower()

        if action == "add":
            task = Task(
                id=self._next_id,
                description=input.description,
                status=TaskStatus.PENDING
            )
            self._tasks.append(task)
            self._next_id += 1
            return TextContent(text=f"Added task #{task.id}: {task.description}")

        elif action == "complete":
            for task in self._tasks:
                if task.id == input.task_id:
                    task.status = TaskStatus.COMPLETED
                    return TextContent(text=f"Completed task #{task.id}")
            return TextContent(text=f"Task #{input.task_id} not found")

        elif action == "list":
            if not self._tasks:
                return TextContent(text="No tasks")
            lines = []
            for task in self._tasks:
                status = "✓" if task.status == TaskStatus.COMPLETED else "○"
                lines.append(f"{status} #{task.id}: {task.description}")
            return TextContent(text="\n".join(lines))

        return TextContent(text=f"Unknown action: {action}")

    # Helper methods for external access
    def get_pending_tasks(self) -> list[Task]:
        return [t for t in self._tasks if t.status == TaskStatus.PENDING]

    def clear(self) -> None:
        self._tasks = []
        self._next_id = 1
```

**Usage:**

```python
task_mgr = TaskManagerTool()

# Tool maintains state across calls
await task_mgr.execute({"action": "add", "description": "Write docs"})
await task_mgr.execute({"action": "add", "description": "Run tests"})
await task_mgr.execute({"action": "complete", "task_id": 1})
await task_mgr.execute({"action": "list"})

# Access state externally
pending = task_mgr.get_pending_tasks()
task_mgr.clear()
```

## Part 9: Returning Multiple Results

Tools can return a list of `TextContent` for multi-part results:

```python
from dataclasses import dataclass
from typing import Annotated
from nano_agent import Tool, TextContent, Desc

@dataclass
class MultiSearchInput:
    queries: Annotated[list[str], Desc("List of search queries")]

@dataclass
class MultiSearchTool(Tool):
    name: str = "multi_search"
    description: str = "Search multiple queries at once"

    async def __call__(self, input: MultiSearchInput) -> list[TextContent]:
        results = []
        for query in input.queries:
            # Simulate search
            result = TextContent(text=f"Results for '{query}': 42 matches")
            results.append(result)
        return results
```

## Part 10: Error Handling

Return error information in `TextContent`, or use `ToolResultContent.is_error`:

```python
from dataclasses import dataclass
from typing import Annotated
from nano_agent import Tool, TextContent, Desc

@dataclass
class FileReadInput:
    path: Annotated[str, Desc("File path to read")]

@dataclass
class SafeReadTool(Tool):
    name: str = "safe_read"
    description: str = "Safely read a file with error handling"

    async def __call__(self, input: FileReadInput) -> TextContent:
        try:
            with open(input.path) as f:
                content = f.read()
            return TextContent(text=content)
        except FileNotFoundError:
            return TextContent(text=f"Error: File not found: {input.path}")
        except PermissionError:
            return TextContent(text=f"Error: Permission denied: {input.path}")
        except Exception as e:
            return TextContent(text=f"Error: {type(e).__name__}: {e}")
```

For structured error handling when building results manually:

```python
from nano_agent import ToolResultContent, TextContent

# Mark as error for the API
result = ToolResultContent(
    tool_use_id=call.id,
    content=[TextContent(text="File not found: /path/to/file")],
    is_error=True  # Tells Claude this was an error
)
```

## Complete Example: Database Tool

Putting it all together—a tool with input validation, state, async operations:

```python
import asyncio
from dataclasses import dataclass, field
from typing import Annotated, ClassVar
from nano_agent import DAG, ClaudeAPI, Tool, TextContent, ToolResultContent, Desc, run

@dataclass
class DBRecord:
    id: int
    name: str
    data: dict

@dataclass
class DBInput:
    operation: Annotated[str, Desc("Operation: 'insert', 'get', 'list', 'delete'")]
    id: Annotated[int, Desc("Record ID (for get/delete)")] = 0
    name: Annotated[str, Desc("Record name (for insert)")] = ""
    data: Annotated[dict, Desc("Record data (for insert)")] = field(default_factory=dict)

@dataclass
class InMemoryDB(Tool):
    name: str = "database"
    description: str = "Simple in-memory database for storing records"

    _records: dict[int, DBRecord] = field(default_factory=dict, repr=False)
    _next_id: int = field(default=1, repr=False)

    async def __call__(self, input: DBInput) -> TextContent:
        op = input.operation.lower()

        if op == "insert":
            if not input.name:
                return TextContent(text="Error: 'name' required for insert")
            record = DBRecord(id=self._next_id, name=input.name, data=input.data)
            self._records[record.id] = record
            self._next_id += 1
            return TextContent(text=f"Inserted record #{record.id}: {record.name}")

        elif op == "get":
            record = self._records.get(input.id)
            if not record:
                return TextContent(text=f"Error: Record #{input.id} not found")
            return TextContent(
                text=f"Record #{record.id}:\n  Name: {record.name}\n  Data: {record.data}"
            )

        elif op == "list":
            if not self._records:
                return TextContent(text="No records in database")
            lines = [f"#{r.id}: {r.name}" for r in self._records.values()]
            return TextContent(text="Records:\n" + "\n".join(lines))

        elif op == "delete":
            if input.id in self._records:
                del self._records[input.id]
                return TextContent(text=f"Deleted record #{input.id}")
            return TextContent(text=f"Error: Record #{input.id} not found")

        return TextContent(text=f"Error: Unknown operation '{op}'")

async def main():
    db = InMemoryDB()
    api = ClaudeAPI()

    dag = (
        DAG()
        .tools(db)
        .user("Create a record for 'Alice' with email alice@example.com, then list all records")
    )

    # Use run() for automatic tool execution
    dag = await run(api, dag)
    print(dag)

    # Check database state
    print(f"\nRecords in DB: {len(db._records)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

### 1. Use Descriptive Names and Descriptions

```python
# Good: Clear purpose
name: str = "file_search"
description: str = "Search for files matching a glob pattern in the codebase"

# Bad: Vague
name: str = "search"
description: str = "Search stuff"
```

### 2. Document All Parameters

```python
@dataclass
class Input:
    # Good: Clear description
    pattern: Annotated[str, Desc("Glob pattern like '*.py' or 'src/**/*.ts'")]

    # Bad: No description
    pattern: str
```

### 3. Handle Errors Gracefully

```python
async def __call__(self, input: MyInput) -> TextContent:
    try:
        result = await risky_operation(input)
        return TextContent(text=result)
    except SpecificError as e:
        return TextContent(text=f"Error: {e}")
    except Exception as e:
        return TextContent(text=f"Unexpected error: {type(e).__name__}: {e}")
```

### 4. Validate Inputs Early

```python
async def __call__(self, input: MyInput) -> TextContent:
    if not input.required_field:
        return TextContent(text="Error: 'required_field' is required")

    if input.count < 1 or input.count > 100:
        return TextContent(text="Error: 'count' must be between 1 and 100")

    # ... proceed with valid input
```

### 5. Use Async for I/O Operations

```python
# Good: Non-blocking
async def __call__(self, input: MyInput) -> TextContent:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    return TextContent(text=response.text)

# Avoid: Blocking the event loop
async def __call__(self, input: MyInput) -> TextContent:
    response = requests.get(url)  # Blocks!
    return TextContent(text=response.text)
```

## Summary

| Pattern | Use Case |
|---------|----------|
| `input: None` | No-input tools |
| `Annotated[T, Desc(...)]` | Documented parameters |
| Default values | Optional parameters |
| Nested dataclasses | Complex structured input |
| `list[T]` | Array inputs |
| `_truncation_config` | Control output size |
| Instance `field()` | Stateful tools |
| `list[TextContent]` | Multi-part results |

---

**Next:** [Configuration](configuration.md) - Auth setup and environment configuration

# nano-agent Documentation

A minimalistic functional asynchronous agent library.

## Quick Start

```python
import asyncio
from nano_agent import DAG, ClaudeAPI

async def main():
    dag = DAG().system("You are helpful.").user("Hello!")
    api = ClaudeAPI()  # Uses ANTHROPIC_API_KEY
    response = await api.send(dag)
    dag = dag.assistant(response.content)
    print(dag)

asyncio.run(main())
```

## Documentation

| Document | Description |
|----------|-------------|
| [Quickstart](quickstart.md) | Installation and first steps |
| [Custom Tools Tutorial](custom-tools-tutorial.md) | Building custom tools |
| [Configuration](configuration.md) | Authentication setup |
| [Architecture](architecture.md) | DAG/Node design |
| [Tools](tools.md) | Built-in tools reference |
| [Tool Execution](tool-execution.md) | Execution patterns |
| [DAG Builder](dag-builder.md) | Fluent builder API |
| [Data Structures](data-structures.md) | Type reference |
| [API Clients](api-clients.md) | Claude, OpenAI, Gemini |
| [Serialization](serialization.md) | Save/load graphs |

## Examples

```bash
uv run python examples/hello_world.py
uv run python examples/simple_tool.py
uv run python examples/parallel_tools.py
```

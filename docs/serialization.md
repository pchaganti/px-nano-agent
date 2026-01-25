# Graph Serialization & Visualization

nano-agent supports saving conversation graphs to JSON and visualizing them for debugging and analysis.

## Saving Graphs

### Using DAG.save()

```python
from nano_agent import DAG, ClaudeAPI, run

async def main():
    api = ClaudeAPI()
    dag = DAG().system("You are helpful.").user("Hello")
    dag = await run(api, dag)

    # Save to JSON
    dag.save("conversation.json")

    # With session ID
    dag.save("conversation.json", session_id="session_001")

    # With metadata
    dag.save(
        "conversation.json",
        session_id="session_001",
        user_name="Alice",
        task="greeting",
    )
```

### Using Node.save_graph()

For lower-level control:

```python
from nano_agent import Node

# Save from a single head node
Node.save_graph(node, "graph.json")

# Save from multiple heads (parallel branches)
Node.save_graph([head1, head2], "graph.json", session_id="my_session")
```

### JSON Format

The saved JSON has this structure:

```json
{
  "session_id": "session_001",
  "created_at": 1706123456.789,
  "metadata": {
    "user_name": "Alice",
    "task": "greeting"
  },
  "head_ids": ["abc123"],
  "nodes": {
    "node_001": {
      "id": "node_001",
      "parent_ids": [],
      "data": {
        "type": "system_prompt",
        "content": "You are helpful."
      },
      "timestamp": 1706123456.0,
      "metadata": {}
    },
    "node_002": {
      "id": "node_002",
      "parent_ids": ["node_001"],
      "data": {
        "role": "user",
        "content": "Hello"
      },
      "timestamp": 1706123456.1,
      "metadata": {}
    }
  }
}
```

## Loading Graphs

### Using DAG.load()

```python
from nano_agent import DAG

# Load returns (DAG, metadata)
dag, metadata = DAG.load("conversation.json")

print(metadata["session_id"])  # "session_001"
print(metadata["user_name"])   # "Alice"

# Continue the conversation
dag = dag.user("How are you?")
```

### Using Node.load_graph()

```python
from nano_agent import Node

# Load returns (list of head nodes, metadata)
heads, metadata = Node.load_graph("graph.json")

# Get the primary head
node = heads[0]

# Traverse ancestors
for ancestor in node.ancestors():
    print(ancestor.data)
```

## Visualization Tools

### Console Visualization

The DAG has built-in console visualization:

```python
dag = DAG().system("You are helpful.").user("Hello").assistant("Hi!")
print(dag)
```

Output:
```
  SYSTEM: You are helpful.
      |
      v
  USER: Hello
      |
      v
  ASSISTANT: Hi!
```

### Console Viewer Script

For saved graphs:

```bash
# Using convenience command
uv run nano-agent-console-viewer conversation.json

# Or directly
uv run python scripts/console_viewer.py conversation.json
```

Output:
```
Session: session_001
Created: 2024-01-24 10:30:00

  [SYSTEM] You are helpful.
      |
      v
  [USER] Hello
      |
      v
  [ASSISTANT] Hi there! How can I help?
      |
      v
  [STOP] end_turn (input: 50, output: 20)
```

### HTML Visualization

Generate interactive D3.js visualizations:

```bash
# Using convenience command
uv run nano-agent-viewer conversation.json

# Or directly
uv run python scripts/viewer.py conversation.json
```

This creates `conversation.html` with:
- Interactive node graph
- Zoom and pan
- Click nodes for details
- Color-coded node types
- Parallel execution visualization

## Visualization of Parallel Execution

When tools execute in parallel, the graph shows branching:

```
  [ASSISTANT] Let me check multiple sources...
      |
   ___+___
   |  |  |
   v  v  v
[EXEC] [EXEC] [EXEC]
Bash   Read   Glob
   |__|__|
      |
      v
  [RESULT] Combined results
      |
      v
  [ASSISTANT] Based on my findings...
```

The HTML viewer renders this as a proper DAG with fan-out/fan-in arrows.

## Use Cases

### Debugging

```python
# Save conversation when something goes wrong
try:
    dag = await run(api, dag)
except Exception as e:
    dag.save(f"debug_{timestamp}.json")
    raise
```

### Replay Analysis

```python
# Load and analyze a past conversation
dag, meta = DAG.load("conversation.json")

# Find all tool executions
for node in dag.head.ancestors():
    if isinstance(node.data, ToolExecution):
        print(f"Tool: {node.data.tool_name}")
        print(f"Result: {node.data.result[0].text[:100]}...")
```

### Token Usage Analysis

```python
# Get cumulative usage
dag, _ = DAG.load("conversation.json")
usage = dag.head.get_usage_totals()

print(f"Total input tokens: {usage['input_tokens']}")
print(f"Total output tokens: {usage['output_tokens']}")
print(f"Cache hits: {usage['cache_read_input_tokens']}")
```

### Conversation Branching

```python
# Load a conversation and create a branch
dag, _ = DAG.load("conversation.json")

# Create alternative response
branch = dag.user("Actually, let me ask differently...")

# Both dag and branch exist independently
dag.save("original.json")
branch.save("branch.json")
```

## Node Metadata

Each node can store metadata (captured automatically for API responses):

```python
# Metadata is stored per-node
node = dag.head
print(node.metadata)
# {
#   "usage": {"input_tokens": 50, "output_tokens": 20, ...},
#   "stop_reason": "end_turn",
#   "model": "claude-haiku-4-5-20251001"
# }
```

## StopReason Nodes

Conversations end with a StopReason node:

```python
from nano_agent import StopReason

# Added automatically by run()
# Or manually:
dag = dag._with_heads(
    dag._append_to_heads(
        StopReason(
            reason="end_turn",
            usage=dag.head.get_usage_totals(),
        )
    )
)
```

StopReason nodes show:
- Why the conversation ended (`end_turn`, `max_tokens`, `tool_use`, etc.)
- Cumulative token usage for the entire conversation
- Visualized as a red terminal node

## Complete Example

```python
import asyncio
from nano_agent import DAG, ClaudeAPI, BashTool, run

async def main():
    api = ClaudeAPI()

    dag = (
        DAG()
        .system("You are a helpful coding assistant.")
        .tools(BashTool())
        .user("List Python files and count them")
    )

    # Run conversation
    dag = await run(api, dag)

    # Save for analysis
    dag.save(
        "coding_session.json",
        session_id="code_001",
        task="file_listing",
    )

    # Print to console
    print(dag)

    # Later: load and analyze
    loaded_dag, meta = DAG.load("coding_session.json")
    print(f"Session: {meta['session_id']}")
    print(f"Task: {meta['task']}")

    # Count tool executions
    tool_count = sum(
        1 for n in loaded_dag.head.ancestors()
        if isinstance(n.data, ToolExecution)
    )
    print(f"Tools executed: {tool_count}")

asyncio.run(main())
```

## Summary

| Operation | Method | Output |
|-----------|--------|--------|
| Save DAG | `dag.save(path)` | JSON file |
| Load DAG | `DAG.load(path)` | (DAG, metadata) |
| Save nodes | `Node.save_graph(heads, path)` | JSON file |
| Load nodes | `Node.load_graph(path)` | (heads, metadata) |
| Console view | `print(dag)` | ASCII graph |
| Console viewer | `nano-agent-console-viewer` | Colored ASCII |
| HTML viewer | `nano-agent-viewer` | Interactive HTML |

---

**Back to:** [README](README.md) - Documentation hub

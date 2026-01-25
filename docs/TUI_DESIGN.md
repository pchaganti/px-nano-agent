# TUI Design Documentation

This document provides comprehensive documentation for the Simple TUI system in nano-agent. It covers the design philosophy, architecture, implementation details, and guidelines for future development.

**Audience:** Developers who will maintain, debug, or extend the TUI system.

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [Architecture Overview](#architecture-overview)
4. [Component Deep Dive](#component-deep-dive)
5. [Shared Patterns](#shared-patterns)
6. [Comparison Matrix](#comparison-matrix)
7. [Extending the TUI](#extending-the-tui)
8. [Design Decisions](#design-decisions)
9. [Future Considerations](#future-considerations)
10. [File Reference](#file-reference)

---

## Overview

The TUI (Terminal User Interface) system provides two distinct terminal interfaces for interacting with nano-agent. Both implementations offer a conversational interface to Claude with tool execution capabilities, but they take fundamentally different approaches to terminal rendering and user interaction.

The system includes two implementations: **NanoAgentApp** (built on the Textual framework) and **SimpleTerminalApp** (built on Rich with prompt_toolkit). This dual-implementation approach isn't redundant—each serves different use cases and preferences, offering developers and users a choice based on their specific needs.

Both TUIs share the same core functionality: they build conversation DAGs, communicate with the Claude API via `ClaudeCodeAPI`, and execute tools in an agent loop. They also share a common display formatting layer (`display.py`) to ensure consistent visual output across implementations. The key differences lie in how they manage input, rendering, and terminal interaction.

This document is intended for developers who need to understand the TUI architecture for maintenance, debugging, or extension. It assumes familiarity with Python async programming and basic terminal concepts.

---

## Design Philosophy

### Why Two Implementations?

The choice to maintain two TUI implementations reflects a deliberate design decision about trade-offs:

**Textual (app.py) - Widget-Based Power**
- Full widget system with composable UI components
- App-controlled scrollback via `RichLog` widget
- Multi-line input support with `TextArea`
- Custom history navigation with `InputHistory` dataclass
- More complex but more powerful for interactive features
- Best for: Power users who want rich widget interactions and custom keybindings

**Rich (simple_app.py) - Terminal-Native Minimalism**
- Leverages terminal's native scrollback buffer
- Single-line input via prompt_toolkit's `PromptSession`
- Built-in history navigation (comes free from prompt_toolkit)
- Lighter runtime footprint
- Terminal-native feel—behaves like a traditional CLI tool
- Best for: Quick interactions, minimal memory footprint, users who prefer terminal-native behavior

### Core Principles

**1. Separation of Display from Logic**

The formatting layer (`display.py`) is completely separate from the application logic. This means:
- Display functions return Rich objects (`Text`, `Panel`), not strings
- Both apps import and use the same formatting functions
- Styling changes propagate to both implementations automatically

**2. Async-First Design**

Both TUIs are built on async foundations:
- API calls are non-blocking (`await api.send()`)
- Tool execution is async (`await tool.execute()`)
- Even input gathering uses async patterns (executor for prompt_toolkit)

This ensures the UI remains responsive during long-running operations.

**3. Consistent User Experience**

Despite different rendering approaches, both TUIs provide:
- Same slash commands (`/quit`, `/clear`, `/help`)
- Same tool set (Bash, Read, Write, Edit, Glob, Grep, TodoWrite)
- Same system prompt (`TUI_SYSTEM_PROMPT`)
- Same visual styling for messages and tool calls

**4. Leverage Terminal Capabilities**

Rather than fighting the terminal, both implementations work with it:
- Textual uses its CSS-like styling system
- Rich uses the terminal's native scrollback
- prompt_toolkit handles terminal-specific input quirks

---

## Architecture Overview

### File Structure

```
tui/
├── __init__.py      ← Public exports (NanoAgentApp, SimpleTerminalApp)
├── __main__.py      ← Entry point (python -m tui)
├── display.py       ← Shared Rich formatting functions
├── app.py           ← Textual widget-based TUI
└── simple_app.py    ← Rich + prompt_toolkit minimal TUI
```

### Integration Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TUI Layer                                   │
│  ┌─────────────────────┐       ┌─────────────────────────────────┐ │
│  │   NanoAgentApp      │       │     SimpleTerminalApp           │ │
│  │   (Textual)         │       │     (Rich + prompt_toolkit)     │ │
│  └──────────┬──────────┘       └────────────────┬────────────────┘ │
│             │                                   │                   │
│             └──────────────┬────────────────────┘                   │
│                            │                                        │
│                   ┌────────▼────────┐                              │
│                   │   display.py    │  ← Shared formatting         │
│                   └─────────────────┘                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      nano_agent Core                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │    DAG      │ ←→ │ ClaudeCode  │ ←→ │   Tools (Bash, Read,    │ │
│  │  (Graph)    │    │    API      │    │   Edit, Glob, Grep...)  │ │
│  └─────────────┘    └──────┬──────┘    └─────────────────────────┘ │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Claude API     │
                    │  (Anthropic)    │
                    └─────────────────┘
```

### Data Flow Diagram

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  User Input  │ ─→ │  DAG.user()  │ ─→ │  api.send()  │ ─→ │  Response    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
       ┌───────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              Agent Loop                                  │
│  ┌───────────────┐    ┌────────────────┐    ┌─────────────────────────┐ │
│  │ dag.assistant │ ─→ │ Display text/  │ ─→ │ tool_calls present?     │ │
│  │   (response)  │    │ thinking       │    │                         │ │
│  └───────────────┘    └────────────────┘    └──────────┬──────────────┘ │
│                                                        │                 │
│                                   ┌────────────────────┴────────────┐    │
│                                   │ Yes                    No       │    │
│                                   ▼                        ▼        │    │
│                           ┌───────────────┐        ┌────────────┐   │    │
│                           │ execute_tool  │        │   break    │   │    │
│                           │ dag.tool_res  │        │   (done)   │   │    │
│                           └───────┬───────┘        └────────────┘   │    │
│                                   │                                 │    │
│                                   └─────── loop back ───────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### display.py - Shared Formatting Layer

The display module provides stateless formatting functions that convert data into Rich renderables. All functions are pure—they take input and return styled output without side effects.

| Function | Purpose | Return Type | Styling |
|----------|---------|-------------|---------|
| `format_user_message(text)` | Format user input | `Text` | Cyan bold "You: " prefix |
| `format_assistant_message(text)` | Format Claude's response | `RenderableType` | Green bold prefix + **Markdown rendering** (native theme) |
| `format_thinking_message(thinking)` | Format extended thinking | `Text` | Magenta italic, dim (truncated to 15 lines) |
| `format_tool_call(name, params)` | Format tool invocation | `Panel` | Yellow border, tool name as title |
| `format_tool_result(result, is_error)` | Format tool output | `Text` | Dim (red if error, truncated to 10 lines) |
| `format_system_message(text)` | Format system info | `Text` | Dim, wrapped in brackets |
| `format_error_message(text)` | Format errors | `Text` | Red bold "Error: " prefix |

**Markdown Rendering:**

The `format_assistant_message` function uses `LimitedMarkdown` (a custom subclass of Rich's `Markdown`) to render Claude's responses with simplified header styling:
- **Headers** (`#`, `##`, `###`) — rendered as **bold text with `#` prefix preserved**, left-aligned (not centered/underlined like Rich's default)
- **Emphasis** (`*italic*`, `**bold**`)
- **Code blocks** with syntax highlighting (using "native" Pygments theme)
- **Inline code** (`` `code` ``)
- **Lists** (ordered and unordered)
- **Links** (clickable in supported terminals)

**Header Style:**
| Rich Default | LimitedMarkdown (simplified) |
|--------------|------------------------------|
| Centered, large, underlined | Left-aligned, bold, with `#` prefix |
| `# Header` → **Header** (centered box) | `# Header` → **# Header** (left-aligned) |

**Code Block Style:**
| Rich Default | LimitedMarkdown (simplified) |
|--------------|------------------------------|
| `padding=1` (adds left space) | `padding=0` (no left padding) |
| Blank lines above/below | No extra blank lines |

The left padding in Rich's default causes issues when copying code - the extra space gets included, breaking Python indentation and other whitespace-sensitive code.

These simplified styles are achieved via custom `TextElement` subclasses in `display.py`:
- `SimpleHeading`: Overrides heading rendering
- `SimpleCodeBlock`: Overrides both `code_block` and `fence` elements

**Truncation Policy:**
- Thinking content: 15 lines max, then "... (N more lines)"
- Tool results: 10 lines max, then "... (N more lines)"
- Tool parameters: 60 characters per value, then "..."

**Example Usage:**
```python
from tui.display import format_user_message, format_tool_call

# Both apps use these identically
log.write(format_user_message("Hello Claude"))
log.write(format_tool_call("Bash", {"command": "ls -la"}))
```

---

### app.py - Textual Implementation

The Textual implementation provides a full widget-based UI with sophisticated layout and input handling.

#### NanoAgentApp Class

```python
class NanoAgentApp(App[None]):
    """Textual-based TUI application for nano-agent."""
```

**Key Attributes:**

| Attribute | Type | Purpose |
|-----------|------|---------|
| `api` | `ClaudeCodeAPI \| None` | API client (initialized on mount) |
| `dag` | `DAG \| None` | Conversation graph |
| `tools` | `list[Tool]` | Available tools |
| `tool_map` | `dict[str, Tool]` | Tool lookup by name |
| `history` | `InputHistory` | Input history manager |

#### Widget Composition

```python
def compose(self) -> ComposeResult:
    with Vertical():
        yield RichLog(id="messages", wrap=True, highlight=True, markup=True)
        yield LoadingIndicator(id="status")
        yield TextArea(id="input")
```

**Widget Layout:**

```
┌──────────────────────────────────────────────────┐
│  #messages (RichLog)                             │
│  ├── User message                                │
│  ├── Claude response                             │
│  ├── Tool call panel                             │
│  └── ... (scrollable)                            │
│                                          height: 1fr
├──────────────────────────────────────────────────┤
│  #status (LoadingIndicator)              height: 1
│  (hidden when not loading)              display: none
├──────────────────────────────────────────────────┤
│  #input (TextArea)                               │
│  > Type here...                          dock: bottom
│                                          height: auto
│                                          min: 3, max: 10
└──────────────────────────────────────────────────┘
```

#### InputHistory Dataclass

Custom history navigation with temp buffer pattern:

```python
@dataclass
class InputHistory:
    entries: list[str] = field(default_factory=list)
    index: int = -1          # -1 = not navigating
    temp_buffer: str = ""    # Stores current input when navigating
```

**Navigation Flow:**
1. User types "hello" → presses Ctrl+Up
2. `temp_buffer = "hello"`, `index = len(entries) - 1`
3. Previous entry displayed
4. User presses Ctrl+Down repeatedly until `index = -1`
5. `temp_buffer` ("hello") restored

#### CSS Styling

```python
CSS = """
Screen { layout: vertical; }

#messages {
    height: 1fr;              /* Flex grow to fill space */
    scrollbar-gutter: stable; /* Reserve scrollbar space */
    border: none;
    padding: 0 1;
}

#status {
    height: 1;
    padding: 0 1;
    display: none;            /* Hidden until loading */
}

#input {
    dock: bottom;             /* Always at bottom */
    margin: 0 1;
    height: auto;
    min-height: 3;            /* At least 3 lines */
    max-height: 10;           /* Cap at 10 lines */
}
"""
```

#### Keybindings

| Binding | Action | Description |
|---------|--------|-------------|
| `ctrl+c` | `quit` | Exit application |
| `ctrl+d` | `quit` | Exit application |
| `enter` | `submit` | Send message (priority binding) |
| `ctrl+enter` | `newline` | Insert newline (priority binding) |
| `ctrl+up` | `history_prev` | Previous history entry |
| `ctrl+down` | `history_next` | Next history entry |

---

### simple_app.py - Rich Implementation

The Rich implementation provides a minimal terminal app that leverages native terminal capabilities.

#### SimpleTerminalApp Dataclass

```python
@dataclass
class SimpleTerminalApp:
    console: Console = field(default_factory=Console)
    api: ClaudeCodeAPI | None = None
    dag: DAG | None = None
    tools: list[Tool] = field(default_factory=list)
    tool_map: dict[str, Tool] = field(default_factory=dict)
    session: PromptSession[str] | None = None
```

**Key Differences from Textual:**
- Uses `@dataclass` instead of inheriting from `App`
- Console is a simple Rich Console, not a widget system
- History comes from prompt_toolkit's `PromptSession`

#### Terminal Size Validation

```python
MIN_TERMINAL_WIDTH = 60
MIN_TERMINAL_HEIGHT = 10
```

**wait_for_valid_terminal_size():**
```python
async def wait_for_valid_terminal_size(self) -> None:
    """Block and display message until terminal is resized."""
    with Live(console=self.console, refresh_per_second=4) as live:
        while True:
            is_valid, width, height = self.check_terminal_size()
            if is_valid:
                break
            # Display resize message in a panel
            live.update(Panel(message, title="Resize Required"))
            await asyncio.sleep(0.25)
```

This ensures the TUI doesn't render poorly in tiny terminals.

#### Input Handling with prompt_toolkit

```python
async def get_user_input(self) -> str | None:
    if self.session is None:
        self.session = PromptSession(history=InMemoryHistory())

    # Run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None, lambda: self.session.prompt("> ")
    )
    return text.strip() if text else None
```

**Why executor?** prompt_toolkit's `prompt()` is blocking. Running it in an executor allows other async tasks to proceed.

#### Thinking Indicator with Rich Live

```python
with Live(
    Spinner("dots", text="Claude is thinking..."),
    console=self.console,
    refresh_per_second=10,
    transient=True,  # Removes spinner when done
) as live:
    response = await self.api.send(self.dag)
    live.update(Text("✓ Response received", style="green dim"))
```

The `transient=True` flag means the spinner disappears after the context exits, leaving no trace in the scrollback.

---

## Shared Patterns

### Agent Loop Pattern

Both implementations use an identical agent loop structure:

```python
# Agent loop (identical in both apps)
while True:
    # 1. Send to API
    response = await self.api.send(self.dag)

    # 2. Add to DAG
    self.dag = self.dag.assistant(response.content)

    # 3. Display thinking (if extended thinking enabled)
    thinking_blocks = response.get_thinking()
    for thinking in thinking_blocks:
        if thinking.thinking:
            display(format_thinking_message(thinking.thinking))

    # 4. Display text content
    text_content = response.get_text()
    if text_content:
        display(format_assistant_message(text_content))

    # 5. Check for tool calls
    tool_calls = response.get_tool_use()
    if not tool_calls:
        break  # No tools = conversation turn complete

    # 6. Execute each tool
    for tool_call in tool_calls:
        await self.execute_tool(tool_call)
```

### Tool Mapping

Both apps use a dictionary for O(1) tool lookup:

```python
self.tools = [BashTool(), ReadTool(), WriteTool(), ...]
self.tool_map = {tool.name: tool for tool in self.tools}

# Later, during execution:
tool = self.tool_map.get(tool_call.name)
```

### Slash Commands

Both apps support the same commands:

| Command | Aliases | Action |
|---------|---------|--------|
| `/quit` | `/exit`, `/q` | Exit application |
| `/clear` | - | Reset conversation, clear display |
| `/help` | - | Show help message |

### Error Handling

Both apps use try/except with formatted error output:

```python
try:
    result = await tool.execute(tool_call.input)
    # ... handle success
except Exception as e:
    error_result = TextContent(text=f"Tool error: {e}")
    display(format_tool_result(error_result.text, is_error=True))
    self.dag = self.dag.tool_result(
        ToolResultContent(
            tool_use_id=tool_call.id,
            content=[error_result],
            is_error=True
        )
    )
```

---

## Comparison Matrix

| Aspect | Textual (app.py) | Rich (simple_app.py) |
|--------|------------------|---------------------|
| **Framework** | Textual widgets | Rich + prompt_toolkit |
| **Base Class** | `App[None]` | `@dataclass` |
| **Scrollback** | App-controlled (`RichLog`) | Terminal-native |
| **History** | Custom `InputHistory` class | Built-in `PromptSession` |
| **Multi-line Input** | Yes (`TextArea` + Ctrl+Enter) | No (single line) |
| **Loading Indicator** | `LoadingIndicator` widget | `Rich Live` Spinner |
| **Terminal Validation** | None | 60×10 minimum |
| **CSS Styling** | Yes (CSS-in-Python) | No |
| **Keybindings** | Custom (`BINDINGS`) | prompt_toolkit defaults |
| **Entry Point** | `uv run nano-tui` | `uv run nano-tui-simple` |
| **Complexity** | Higher | Lower |
| **Dependencies** | Textual, Rich | Rich, prompt_toolkit |

---

## Extending the TUI

### Adding a New Tool

**Step 1:** Create the tool in `nano_agent/tools.py` (see [tools.md](tools.md))

**Step 2:** Add to both TUI apps:

```python
# In app.py and simple_app.py
from nano_agent.tools import MyNewTool

# In __init__ or __post_init__:
self.tools = [
    BashTool(),
    ReadTool(),
    # ... existing tools ...
    MyNewTool(),  # Add here
]
self.tool_map = {tool.name: tool for tool in self.tools}
```

**Step 3:** Update `TUI_SYSTEM_PROMPT`:

```python
TUI_SYSTEM_PROMPT = """You are a helpful terminal assistant. You have access to tools for:
- Running bash commands (Bash)
- ... existing tools ...
- Your new capability (MyNewTool)

Be concise but helpful. When using tools, explain briefly what you're doing."""
```

### Adding a New Slash Command

**In both `handle_command()` methods:**

```python
def handle_command(self, command: str) -> bool:
    cmd = command.lower().strip()

    # ... existing commands ...

    if cmd == "/mystuff":
        # Do something
        self.print_history(format_system_message("Did the thing!"))
        return True  # Continue running

    # Unknown command handling...
```

### Adding a New Display Format

**Step 1:** Add to `display.py`:

```python
def format_my_new_thing(data: MyData) -> Text:
    """Format my new thing with custom styling."""
    result = Text()
    result.append("My Thing: ", style="bold blue")
    result.append(str(data))
    return result
```

**Step 2:** Import and use in both apps:

```python
from .display import format_my_new_thing

# In the appropriate place:
self.print_history(format_my_new_thing(data))  # simple_app
# or
log.write(format_my_new_thing(data))  # app.py
```

### Creating a Third TUI Implementation

If you need a third implementation (e.g., for a specific use case):

1. **Create `tui/my_app.py`**
2. **Follow the pattern:**
   - Import from `nano_agent` and `.display`
   - Define `TUI_SYSTEM_PROMPT` (or import shared)
   - Create tool list and tool_map
   - Implement `initialize_api()`, `send_message()`, `execute_tool()`
   - Handle slash commands
3. **Add to `tui/__init__.py`:**
   ```python
   from .my_app import MyApp, my_main
   __all__ = [..., "MyApp", "my_main"]
   ```
4. **Add entry point in `pyproject.toml`:**
   ```toml
   [project.scripts]
   nano-my-tui = "tui.my_app:main"
   ```

---

## Design Decisions

### Why Dataclasses for Both Apps?

**SimpleTerminalApp:** Uses `@dataclass` because it's a simple container with default factory fields. No inheritance needed—it's just state + methods.

**NanoAgentApp:** Still uses dataclass-like patterns internally (`InputHistory`), but inherits from `App` for Textual's lifecycle management. The `App` base class provides compose/mount/action machinery that dataclasses can't replicate.

### Why Async Throughout?

1. **Non-blocking API calls:** Claude API can take seconds to respond
2. **Tool execution:** Some tools (Bash) may run for extended periods
3. **UI responsiveness:** Loading indicators can animate while waiting
4. **Future-proofing:** Streaming responses will require async

### Why Shared TUI_SYSTEM_PROMPT?

Both apps define identical `TUI_SYSTEM_PROMPT` strings. This ensures:
- Consistent behavior regardless of which TUI is used
- Same tool descriptions for Claude
- Same expected interaction patterns

**Note:** Currently duplicated in both files. A future refactor could move this to `display.py` or a new `common.py`.

### Why No Streaming (Yet)?

The current implementation waits for complete responses:

```python
response = await self.api.send(self.dag)  # Waits for full response
```

Streaming would require:
1. API client support for streaming
2. Incremental display updates
3. More complex state management

The current approach is simpler and sufficient for most use cases.

---

## Future Considerations

### Streaming Responses

To add streaming:
1. Add `send_streaming()` to ClaudeCodeAPI
2. Update agent loop to yield partial content
3. Use Rich Live (simple_app) or periodic RichLog updates (app.py)

### Configuration File

Consider adding `~/.nano-tui.yaml` for:
- Custom system prompt
- Tool selection
- Theme preferences
- Default max_tokens

### Session Persistence

Save and load conversations:
```python
# Save
Node.save_graph(self.dag.head, "session.json")

# Load
heads, metadata = Node.load_graph("session.json")
```

Would require:
- Slash command: `/save [filename]`, `/load [filename]`
- Auto-save on exit option

### Theme Support

Both Rich and Textual support theming:
- Rich: Custom `Theme` with color definitions
- Textual: CSS variables for colors

Could add `/theme dark` or `/theme light` commands.

---

## File Reference

### TUI Files

| File | Purpose | Key Exports |
|------|---------|-------------|
| `tui/__init__.py` | Public API | `NanoAgentApp`, `SimpleTerminalApp`, `main`, `simple_main` |
| `tui/__main__.py` | Module entry point | (runs `main()`) |
| `tui/display.py` | Formatting functions | `format_*` functions |
| `tui/app.py` | Textual implementation | `NanoAgentApp`, `InputHistory`, `main` |
| `tui/simple_app.py` | Rich implementation | `SimpleTerminalApp`, `main` |

### Related nano_agent Files

| File | Purpose | Used By TUI |
|------|---------|-------------|
| `nano_agent/dag.py` | DAG construction | `.user()`, `.assistant()`, `.tool_result()` |
| `nano_agent/claude_code_api.py` | API client | `.send()`, `.model` |
| `nano_agent/tools.py` | Tool definitions | `BashTool`, `ReadTool`, etc. |
| `nano_agent/data_structures.py` | Data types | `TextContent`, `ToolUseContent`, etc. |

### Entry Points (pyproject.toml)

```toml
[project.scripts]
nano-tui = "tui.app:main"
nano-tui-simple = "tui.simple_app:main"
```

---

## Summary

The TUI system provides two complementary terminal interfaces:

- **NanoAgentApp (Textual):** Full-featured widget-based UI with custom history, multi-line input, and CSS styling
- **SimpleTerminalApp (Rich):** Lightweight terminal-native feel with built-in history and spinner

Both share:
- Common display formatting (`display.py`)
- Same tools and system prompt
- Identical agent loop structure
- Same slash commands

Choose Textual for power features, Rich for simplicity. Both are production-ready and maintainable.

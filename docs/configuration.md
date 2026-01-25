# Configuration & Authentication

This guide covers authentication setup for nano-agent's API clients.

## Quick Reference

| API Client | Primary Auth | Fallback |
|------------|--------------|----------|
| `ClaudeAPI` | `ANTHROPIC_API_KEY` | Explicit `api_key` |
| `ClaudeCodeAPI` | `~/.nano-agent.json` | None |
| `OpenAIAPI` | `OPENAI_API_KEY` | Explicit `api_key` |
| `GeminiAPI` | `GEMINI_API_KEY` | Explicit `api_key` |

## ClaudeAPI

The standard client for the Anthropic API.

### Environment Variable (Recommended)

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

```python
from nano_agent import ClaudeAPI

api = ClaudeAPI()
```

### Explicit API Key

```python
api = ClaudeAPI(api_key="sk-ant-api03-...")
```

### Configuration Options

```python
api = ClaudeAPI(
    api_key="sk-ant-...",
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    temperature=1.0,
    thinking_budget=10000,
)
```

## ClaudeCodeAPI

Uses a config file captured from Claude CLI. Useful for:
- Reusing Claude CLI's OAuth tokens
- Sharing config across scripts

### Setup

1. **Capture config from Claude CLI:**

```bash
uv run nano-agent-capture-auth
```

This creates `~/.nano-agent.json`.

2. **Use the config:**

```python
from nano_agent import ClaudeCodeAPI

api = ClaudeCodeAPI()
```

### Custom Config Path

```python
api = ClaudeCodeAPI(config_path="/path/to/config.json")
```

## OpenAI API

### Environment Variable (Recommended)

```bash
export OPENAI_API_KEY="sk-..."
```

```python
from nano_agent import OpenAIAPI

api = OpenAIAPI()
```

### Configuration Options

```python
api = OpenAIAPI(
    api_key="sk-...",
    model="gpt-4o",
    max_tokens=4096,
    temperature=1.0,
)
```

## Gemini API

### Environment Variable (Recommended)

```bash
export GEMINI_API_KEY="..."
```

```python
from nano_agent import GeminiAPI

api = GeminiAPI()
```

### Configuration Options

```python
api = GeminiAPI(
    api_key="...",
    model="gemini-2.0-flash",
    max_tokens=8192,
    temperature=1.0,
)
```

## Environment Variables

| Variable | API Client | Description |
|----------|------------|-------------|
| `ANTHROPIC_API_KEY` | ClaudeAPI | Anthropic API key |
| `OPENAI_API_KEY` | OpenAIAPI | OpenAI API key |
| `GEMINI_API_KEY` | GeminiAPI | Google Gemini API key |

## Multi-Provider Example

```python
from nano_agent import ClaudeAPI, OpenAIAPI, GeminiAPI, DAG

dag = DAG().system("Be helpful.").user("Hello!")

# Pick your provider
api = ClaudeAPI()      # or OpenAIAPI() or GeminiAPI()
response = await api.send(dag)
```

---

**Next:** [Architecture](architecture.md) - Understand the DAG/Node design

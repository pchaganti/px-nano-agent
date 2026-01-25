# Performance Optimizations

## Startup Time Improvements

### Problem
The `a3` library had slow startup times (~6.4 seconds) when creating `ClaudeAPI()` instances due to the automatic configuration capture from Claude CLI.

### Root Cause Analysis
Profiling revealed two main bottlenecks in `capture_auth.py`:

1. **DNS Lookup (5.0s)**: HTTPServer binding to empty string `""` triggered slow reverse DNS lookup via `socket.getfqdn()` → `gethostbyaddr()`
2. **Inefficient Polling (1.4s)**: Long sleep intervals and subprocess cleanup timeouts

### Optimizations Applied

#### 1. Fixed DNS Lookup Issue
**Before:**
```python
server = HTTPServer(("", port), AuthCaptureHandler)
```

**After:**
```python
server = HTTPServer(("localhost", port), AuthCaptureHandler)
```

**Impact:** Eliminated 5+ second DNS lookup delay

#### 2. Optimized Polling and Cleanup
- Reduced initial server startup sleep: `0.5s` → `0.2s`
- Faster polling interval: `0.1s` → `0.05s`
- Quicker subprocess cleanup: `2s timeout` → `0.5s timeout`
- Added graceful termination before force kill

**Impact:** Reduced polling/cleanup overhead by ~50%

#### 3. Random Port Assignment
Use port `0` to let the OS assign a random available port automatically. This eliminates port conflicts when running multiple instances concurrently.

### Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **DNS Lookup** | 5.0s | ~0ms | **99.9%** |
| **Total Startup** | 6.4s | 1.2s | **82%** |
| **Import Time** | 65ms | 47ms | 28% |

### Verification

Run profiling script:
```bash
# Test API creation time
time uv run python -c "from claude_api import ClaudeAPI; ClaudeAPI()"
# Should complete in ~1.2 seconds
```

### Recommendations

For applications that create multiple `ClaudeAPI` instances:
1. **Reuse instances**: Create once, use many times
2. **Cache config**: Call `get_config()` once and pass `headers` to constructor
3. **Explicit keys**: Use `ClaudeAPI(api_key="...")` to skip auto-capture entirely

Example:
```python
# Option 1: Reuse API instance (async)
api = ClaudeAPI()
for task in tasks:
    response = await api.send(...)

# Option 2: Cache captured config
from scripts.capture_claude_code_auth import get_config
headers, body_params = get_config()  # Only capture once

api1 = ClaudeAPI(headers=headers, model=body_params['model'])
api2 = ClaudeAPI(headers=headers, model=body_params['model'])
```

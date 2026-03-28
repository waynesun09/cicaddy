# MCP Integration Guide

Cicaddy connects to MCP (Model Context Protocol) servers to extend AI agent capabilities with external tools — databases, code search, documentation lookup, metrics APIs, and more.

## Supported Protocols

All protocols are included in the base installation — no extras needed.

| Protocol | Use Case | Notes |
|----------|----------|-------|
| **HTTP/HTTPS** | Production remote servers | Primary protocol, recommended for most deployments |
| **Stdio** | Local dev, CLI tools | Popular for local MCP servers (e.g., `npx` packages) |
| **WebSocket** | Bidirectional real-time | Specialized; use only when HTTP won't work |
| **SSE** | Legacy streaming | Rarely needed for new deployments |

## Configuration

MCP servers are configured via the `MCP_SERVERS_CONFIG` environment variable — a JSON array of server objects.

### Server Config Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique server identifier |
| `protocol` | Yes | `http`, `stdio`, `websocket`, or `sse` |
| `endpoint` | For http/ws/sse | Full URL to MCP endpoint |
| `command` | For stdio | Command to execute (e.g., `npx`) |
| `args` | For stdio | Array of command arguments |
| `env` | No | Environment variables for stdio servers |
| `headers` | No | HTTP headers (auth tokens, API keys) |
| `timeout` | No | Total timeout in seconds (default: 300) |
| `idle_timeout` | No | Max seconds between progress updates (default: 60) |
| `retry_count` | No | Number of retries on failure |
| `retry_delay` | No | Initial retry delay in seconds |
| `scan_mode` | No | Security scanning mode: `disabled`, `audit`, or `enforce` |

### HTTP Server

```yaml
MCP_SERVERS_CONFIG: >-
  [{
    "name": "data-analysis",
    "protocol": "http",
    "endpoint": "https://mcp-tools.company.com/analysis",
    "headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
    "timeout": 120
  }]
```

### Stdio Server (Local)

```yaml
MCP_SERVERS_CONFIG: >-
  [{
    "name": "context7",
    "protocol": "stdio",
    "command": "npx",
    "args": ["-y", "@context7/mcp-server"],
    "env": {"CONTEXT7_API_KEY": "${CONTEXT7_KEY}"},
    "timeout": 300
  }]
```

### Multiple Servers

```yaml
MCP_SERVERS_CONFIG: >-
  [
    {"name": "context7", "protocol": "http", "endpoint": "https://mcp.context7.com/mcp", "timeout": 300, "scan_mode": "enforce"},
    {"name": "devlake", "protocol": "http", "endpoint": "https://devlake-mcp.example.com/mcp", "timeout": 600, "idle_timeout": 300, "scan_mode": "disabled"}
  ]
```

**Note**: The `scan_mode` field enables per-server security scanning. See [MCP Security Scanning](mcp-security-scanning.md) for details.

## Authentication

```yaml
# Bearer token
"headers": {"Authorization": "Bearer ${MCP_TOKEN}"}

# API key
"headers": {"X-API-Key": "${MCP_API_KEY}"}

# Custom headers
"headers": {
  "X-Auth-Token": "${CUSTOM_TOKEN}",
  "X-Project-ID": "${CI_PROJECT_ID}"
}
```

## Dual Timeout Strategy

For long-running MCP tools that report progress, cicaddy supports two timeouts:

- **`timeout`** — absolute maximum execution time (safety limit)
- **`idle_timeout`** — max time between progress updates (activity check)

```yaml
{
  "name": "metrics-server",
  "protocol": "http",
  "endpoint": "https://mcp-server.example.com/mcp",
  "timeout": 300,
  "idle_timeout": 60
}
```

**How it works:**
- Tools reporting progress every 30-60s can run well beyond `idle_timeout`
- Tools that hang with no progress fail quickly at `idle_timeout`
- `timeout` is the absolute upper bound regardless of progress

| Tool Type | timeout | idle_timeout | Rationale |
|-----------|---------|--------------|-----------|
| Quick scans | 30-60s | Not needed | Completes quickly |
| Standard analysis | 120-180s | Not needed | Completes in 2-3 min |
| Data processing | 300-600s | 60-90s | Reports progress regularly |
| Comprehensive scans | 600-900s | 90-120s | Long analysis with periodic updates |

## Tool Discovery and Execution

The agent automatically discovers available tools from each connected MCP server. The AI model selects and invokes tools based on the task prompt and context — no manual tool specification needed.

## Error Handling

The agent handles MCP errors gracefully:

- **Connection failures**: Continues with other servers
- **Tool errors**: Reported in analysis results
- **Timeouts**: Retried with exponential backoff
- **Auth failures**: Logged, server skipped

## Token-Aware MCP Execution

MCP tool results are managed by the token-aware execution engine. As token utilization increases, the engine progressively degrades tool result detail:

| Utilization | Level | Behavior |
|-------------|-------|----------|
| 0-70% | COMPREHENSIVE | Full tool results with debug info |
| 70-85% | DETAILED | Key findings, debug info removed |
| 85-95% | SUMMARY | Summarized results, raw data filtered |
| 95%+ | CRITICAL_ONLY | Only critical findings and errors |

See [Token-Aware Execution](token-aware-execution.md) for full details.

## Prompt Engineering for MCP Tools

Tool calls must happen **early** in the prompt workflow. The agent uses a "last message only" approach — tool results from early iterations won't appear in output unless referenced in the final message.

**Correct pattern** — tools first, then analysis:

```yaml
AI_TASK_PROMPT: |
  **1. Detection Phase** — Identify what needs MCP tool lookup
  **2. Tool Invocation** — Execute MCP tools to gather data
  **3. Analysis** — Analyze using tool results
  **4. Output** — Single comprehensive review
```

See the [cicaddy-gitlab prompt engineering guide](https://github.com/redhat-community-ai-tools/cicaddy-gitlab/blob/main/docs/prompt-engineering-best-practices.md) for detailed patterns and examples.

## Monitoring

```yaml
LOG_LEVEL: "DEBUG"   # Detailed MCP communication logs
JSON_LOGS: "true"    # Structured logs for analysis
```

The agent logs MCP server connections, tool discovery, execution times, token usage, degradation triggers, and errors.

## Related Documentation

- [Architecture Guide](architecture.md) — System overview and agent framework
- [Token-Aware Execution](token-aware-execution.md) — Token management and progressive degradation
- [Metadata Architecture](metadata-architecture.md) — Knowledge store and compression

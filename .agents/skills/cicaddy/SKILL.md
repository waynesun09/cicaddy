---
name: cicaddy
description: >
  Run and configure the Cicaddy pipeline AI agent CLI. Covers cicaddy run/validate/config
  commands, env file setup, DSPy task YAML authoring, MCP server configuration, sub-agent
  delegation, agent factory extension, and security scanning. Use when working with cicaddy
  CLI, writing .env files for cicaddy, creating DSPy task definitions, configuring MCP servers,
  setting up delegation, or extending the agent registry.
compatibility: Requires Python 3.11+ and uv. Dev install with uv pip install -e ".[dev,test]" or released package with uv pip install cicaddy.
metadata:
  version: "0.12.0"
  author: waynesun09
---

# Cicaddy Agent

Cicaddy is a platform-agnostic pipeline AI agent that runs LLM-powered
analysis tasks via MCP tool servers. It supports DSPy task YAML definitions,
multiple AI providers (Gemini, OpenAI, Claude, Gemini and Claude via Vertex AI), and a registry-based agent
factory that can be extended without modifying core code.

## CLI Usage

### Run the agent

```bash
# From an env file (recommended)
uv run cicaddy run --env-file .env

# With inline overrides
uv run cicaddy run --env-file .env --ai-model gemini-3-pro-preview --log-level DEBUG

# Dry-run: print resolved config without executing
uv run cicaddy run --env-file .env --dry-run

# Verbose output (sets LOG_LEVEL=DEBUG)
uv run cicaddy run --env-file .env --verbose
```

Key `run` flags:

| Flag | Description |
|------|-------------|
| `--env-file, -e FILE` | Load env vars from a `.env` file |
| `-t, --agent-type` | `task` (default), `branch_review`, `merge_request`, or custom |
| `--ai-provider` | `gemini`, `gemini-vertex`, `openai`, `claude`, `anthropic-vertex` |
| `--ai-model` | e.g. `gemini-3-pro-preview`, `gpt-4o` |
| `--mcp-config` | JSON string or path with MCP server configs |
| `--max-iters` | Maximum inference iterations |
| `--task-prompt` | Inline task prompt (alternative to AI_TASK_FILE) |
| `--dry-run` | Show config, don't run |
| `--verbose` | Debug logging |
| `--delegation-mode` | `none` (single-agent) or `auto` (sub-agent delegation) |
| `--max-sub-agents` | Max concurrent sub-agents (1-10) |

### Inspect configuration

```bash
uv run cicaddy config show --env-file .env
```

### Validate before running

```bash
uv run cicaddy validate --env-file .env
```

Performs pre-flight checks: AI provider + API key, agent type, MCP server config validity.
Exits `0` on pass (warnings allowed), `1` on error.

### Version

```bash
uv run cicaddy version
```

---

## Environment Configuration

Core env vars for a cicaddy run:

```bash
# Agent type
AGENT_TYPE=task           # task | branch_review | merge_request | <custom>
TASK_TYPE=custom          # for custom DSPy task file mode

# AI provider
AI_PROVIDER=gemini        # gemini | gemini-vertex | openai | claude | anthropic-vertex
AI_MODEL=gemini-3-pro-preview
AI_TEMPERATURE=0.0
GEMINI_API_KEY=<key>      # or OPENAI_API_KEY / ANTHROPIC_API_KEY
# For Gemini Vertex AI: GOOGLE_CLOUD_PROJECT=<project> GOOGLE_CLOUD_LOCATION=global
# For Claude Vertex AI: ANTHROPIC_VERTEX_PROJECT_ID=<project> GOOGLE_CLOUD_LOCATION=global

# DSPy task file (alternative to AI_TASK_PROMPT)
AI_TASK_FILE=examples/dora_metrics_task.yaml

# Local file tools (required for read_file in task prompts)
ENABLE_LOCAL_TOOLS=true
LOCAL_TOOLS_WORKING_DIR=/path/to/working/dir

# MCP servers (JSON array)
MCP_SERVERS_CONFIG='[{
  "name": "devlake-mcp-instance",
  "protocol": "http",
  "endpoint": "https://your-mcp-server/mcp",
  "headers": {"Authorization": "Bearer <token>"},
  "timeout": 900,
  "connection_timeout": 120,
  "idle_timeout": 450,
  "retry_count": 5
}]'

# Execution limits
MAX_INFER_ITERS=40
MAX_EXECUTION_TIME=3600
LOG_LEVEL=INFO
```

### DSPy task file inputs

Inputs declared in the task YAML are resolved from environment variables.
See `references/dspy-task-yaml.md` for the full task file specification.

```yaml
inputs:
  - name: project_name
    env_var: PROJECT_NAME   # set PROJECT_NAME=<value> in .env
    required: true
  - name: analysis_days
    env_var: ANALYSIS_DAYS
    default: "30"
```

---

## Gotchas

- `AI_TASK_FILE` paths are relative to CWD, not the env file location
- `LOCAL_TOOLS_WORKING_DIR` must be absolute when running from output directories
- `MCP_SERVERS_CONFIG` server `name` must exactly match `tools.servers` in task YAML
- Delegation mode `auto` requires at least 2 sub-agents to be useful
- `--dry-run` does NOT validate MCP server connectivity, only config parsing
- Session JSONL files can be very large (50MB+) -- never read directly
- Report output goes to CWD, not the source repo directory
- Bundled skills (`model-reference`, `cicaddy-config`) are always available, no install needed
- `validate` checks API key presence but does not test the key against the provider

---

## Reference Topics

For detailed documentation on these topics, read the corresponding reference file:

- **Sub-Agent Delegation** (v0.8.0+): Enable `DELEGATION_MODE=auto` for AI-powered triage with parallel sub-agents. See `references/delegation.md`
- **Security Scanning** (v0.6.1+): Prompt injection protection for external content. See `references/security-scanning.md`
- **Agent Factory Extension**: Register custom agents and type detectors via the registry pattern. See `references/factory-extension.md`
- **DSPy Task YAML**: Declarative task definitions with persona, tools, constraints, and output format. See `references/dspy-task-yaml.md`
- **Local Testing & Evaluation**: Use `cicaddy-runner` and `cicaddy-eval` sub-agents for local runs. See `references/local-testing.md`

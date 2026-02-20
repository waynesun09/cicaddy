---
name: cicaddy
description: >
  Use this skill when working with the Cicaddy platform-agnostic pipeline AI
  agent. Covers CLI commands, environment configuration, DSPy task files,
  MCP server setup, and extending the agent factory with custom agents.
---

# Cicaddy Agent

Cicaddy is a platform-agnostic pipeline AI agent that runs LLM-powered
analysis tasks via MCP tool servers. It supports DSPy task YAML definitions,
multiple AI providers (Gemini, OpenAI, Claude), and a registry-based agent
factory that can be extended without modifying core code.

## CLI Usage

### Run the agent

```bash
# From an env file (recommended)
uv run cicaddy run --env-file .env

# With inline overrides
uv run cicaddy run --env-file .env --ai-model gemini-2.5-pro --log-level DEBUG

# Dry-run: print resolved config without executing
uv run cicaddy run --env-file .env --dry-run

# Verbose output (sets LOG_LEVEL=DEBUG)
uv run cicaddy run --env-file .env --verbose
```

Key `run` flags:

| Flag | Description |
|------|-------------|
| `--env-file, -e FILE` | Load env vars from a `.env` file |
| `-t, --agent-type` | `cron` (default), `branch_review`, `merge_request`, or custom |
| `--ai-provider` | `gemini`, `openai`, `claude` |
| `--ai-model` | e.g. `gemini-3-pro-preview`, `gpt-4o` |
| `--mcp-config` | JSON string or path with MCP server configs |
| `--max-iters` | Maximum inference iterations |
| `--task-prompt` | Inline task prompt (alternative to AI_TASK_FILE) |
| `--dry-run` | Show config, don't run |
| `--verbose` | Debug logging |

### Inspect configuration

```bash
uv run cicaddy config show --env-file .env
```

### Validate before running

```bash
uv run cicaddy validate --env-file .env
```

Performs pre-flight checks without running the agent:

- **AI Provider** — `AI_PROVIDER` is set and the corresponding API key exists
  (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
- **Agent type** — `AGENT_TYPE` is set; warns if it will be auto-detected
- **MCP servers** — `MCP_SERVERS_CONFIG` is valid JSON, lists each server name
  and protocol (credentials are masked)

Exits `0` on pass (warnings allowed), `1` on error.

### Version

```bash
uv run cicaddy version
```

---

## Environment Configuration

Core env vars for a DSPy task run:

```bash
# Agent type
AGENT_TYPE=cron           # cron | branch_review | merge_request | <custom>
CRON_TASK_TYPE=custom     # for custom DSPy task file mode

# AI provider
AI_PROVIDER=gemini        # gemini | openai | claude
AI_MODEL=gemini-3-pro-preview
AI_TEMPERATURE=0.0
GEMINI_API_KEY=<key>      # or OPENAI_API_KEY / ANTHROPIC_API_KEY

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

Inputs declared in the task YAML are resolved from environment variables:

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

## DSPy Task YAML

A task file defines the prompt, tools, constraints, and output format
declaratively. The MCP server name in `tools.servers` must match the `name`
field in `MCP_SERVERS_CONFIG`. The `local` server is auto-provided when
`ENABLE_LOCAL_TOOLS=true`.

```yaml
name: my_analysis
type: data_analysis
version: "1.0"

persona: >
  expert analyst for {{PROJECT_NAME}}

inputs:
  - name: project_name
    env_var: PROJECT_NAME
    required: true

tools:
  servers:
    - devlake-mcp-instance   # must match MCP_SERVERS_CONFIG name
    - local                  # built-in: requires ENABLE_LOCAL_TOOLS=true
  required_tools:
    - connect_database
    - execute_query
    - read_file

constraints:
  - "NEVER use WITH clauses (CTEs)"
  - "Always use fully qualified table names: lake.table_name"

reasoning: react
output_format: html   # html | markdown | json

context: |
  Analyze {{PROJECT_NAME}} for the last {{ANALYSIS_DAYS}} days.
  ...
```

---

## Agent Factory Extension

The `AgentFactory` uses a registry pattern. Custom agents and type detectors
can be registered from any package — no need to modify cicaddy's core.

### Register a custom agent

```python
from cicaddy.agent.base import BaseAIAgent
from cicaddy.agent.factory import AgentFactory

class MyCustomAgent(BaseAIAgent):
    async def _build_prompt(self, context):
        return "Analyze the pipeline..."

    async def _process_result(self, result, analysis_result):
        # post-process the AI result
        return result

# Register under a type name
AgentFactory.register("my_custom", MyCustomAgent)
```

Activate with `AGENT_TYPE=my_custom` in the env file.

### Register a type detector

Detectors auto-select agent type from environment/settings.
Lower priority number = checked first. First non-None result wins.

```python
from cicaddy.agent.factory import AgentFactory
from cicaddy.config.settings import Settings
from typing import Optional

def detect_my_platform(settings: Settings) -> Optional[str]:
    import os
    if os.getenv("MY_PLATFORM_EVENT") == "pipeline":
        return "my_custom"
    return None

# Register with priority 10 (runs before built-in CI detector at 50)
AgentFactory.register_detector(detect_my_platform, priority=10)
```

### Entry point plugin registration (for installable packages)

Cicaddy discovers plugins automatically via `importlib.metadata.entry_points()`.
Register callables in your package's `pyproject.toml`:

```toml
[project.entry-points."cicaddy.agents"]
my_platform = "my_plugin.plugin:register_agents"

[project.entry-points."cicaddy.cli_args"]
my_platform = "my_plugin.plugin:get_cli_args"

[project.entry-points."cicaddy.env_vars"]
my_platform = "my_plugin.plugin:get_env_vars"

[project.entry-points."cicaddy.config_sections"]
my_platform = "my_plugin.plugin:config_section"

[project.entry-points."cicaddy.validators"]
my_platform = "my_plugin.plugin:validate"

[project.entry-points."cicaddy.settings_loader"]
my_platform = "my_plugin.config:load_settings"
```

Plugin callable signatures:

| Group | Signature |
|-------|-----------|
| `cicaddy.agents` | `() -> None` — calls `AgentFactory.register()` / `register_detector()` |
| `cicaddy.cli_args` | `() -> List[ArgMapping]` |
| `cicaddy.env_vars` | `() -> List[str]` |
| `cicaddy.config_sections` | `(config: Dict, mask_fn: Callable, sensitive_vars: frozenset) -> None` |
| `cicaddy.validators` | `(config: Dict) -> Tuple[List[str], List[str]]` (errors, warnings) |
| `cicaddy.settings_loader` | `() -> CoreSettings` |

Example agent registration callable:

```python
# my_plugin/plugin.py
def register_agents():
    from cicaddy.agent.factory import AgentFactory
    from my_plugin.agents import MyCustomAgent, detect_my_platform

    AgentFactory.register("my_custom", MyCustomAgent)
    AgentFactory.register_detector(detect_my_platform, priority=10)
```

After `pip install cicaddy my-plugin`, cicaddy discovers and loads the plugin
automatically — no manual imports needed.

### Built-in agent types

| Type | Class | Activated by |
|------|-------|--------------|
| `cron` | `CronAIAgent` | `AGENT_TYPE=cron` or `CRON_TASK_TYPE` env var |
| `branch_review` | `BranchReviewAgent` | `AGENT_TYPE=branch` or branch push CI |
| `merge_request` | *(platform plugin)* | `CI_MERGE_REQUEST_IID` or `AGENT_TYPE=mr` |

### Available AgentFactory methods

```python
AgentFactory.register(agent_type, agent_class)         # register agent class
AgentFactory.register_detector(detector_fn, priority)  # register type detector
AgentFactory.create_agent(settings)                    # create agent instance
AgentFactory.get_available_agent_types()               # list registered types
AgentFactory.validate_agent_requirements(type, settings)
```

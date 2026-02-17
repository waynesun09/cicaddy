# cicaddy

Platform-agnostic AI agent for running AI workflows in CI pipelines, with MCP tool integration and multi-step execution engine.

## Features

- **Multi-provider AI**: Gemini, OpenAI, Claude
- **MCP integration**: Connect to any MCP-compatible tool server
- **Multi-step execution**: Token-aware execution engine with recovery
- **YAML task definitions**: DSPy-based task configuration
- **Notifications**: Slack and email notification support
- **HTML reports**: Customizable analysis report generation
- **Extensible agents**: Registry-based agent factory for custom agents

## Installation

```bash
pip install cicaddy
```

## Quick Start

```bash
# Run with environment file
cicaddy run --env-file .env

# Run with CLI arguments
cicaddy run --ai-provider gemini --agent-type cron --log-level DEBUG

# Show configuration
cicaddy config show --env-file .env

# Validate configuration
cicaddy validate --env-file .env
```

## Configuration

Configure via environment variables or `.env` file:

```env
# AI Provider
AI_PROVIDER=gemini
AI_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your-key-here

# Agent
AGENT_TYPE=cron
CRON_TASK_TYPE=scheduled_analysis

# MCP Servers (JSON array)
MCP_SERVERS_CONFIG=[]

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/...

# DSPy Task File (takes precedence over AI_TASK_PROMPT)
AI_TASK_FILE=tasks/dora_report.yaml
```

### DSPy Task Definition (YAML)

Instead of raw prompt strings (`AI_TASK_PROMPT`), define structured tasks in YAML with typed inputs, expected outputs, MCP tool constraints, and reasoning strategy. Set `AI_TASK_FILE` to your task file path.

See [`examples/dora_metrics_task.yaml`](examples/dora_metrics_task.yaml) for a complete DORA metrics analysis task using DevLake MCP, and [`examples/templates/report_template.html`](examples/templates/report_template.html) for the HTML report template.

Key schema fields:

| Field | Description |
|-------|-------------|
| `inputs[].env_var` | Resolve value from environment variable at load time |
| `inputs[].format` | `diff` or `code` for fenced rendering in prompt |
| `tools.servers` | Restrict to specific MCP servers |
| `tools.required_tools` | Tools the AI must use during execution |
| `tools.forbidden_tools` | Tools the AI must not use |
| `reasoning` | `chain_of_thought`, `react`, or `simple` |
| `output_format` | `markdown`, `html`, or `json` |
| `context` | Supports `{{VAR}}` placeholders resolved at load time |

## Extending with Platform Plugins

`cicaddy` is designed to be extended by platform-specific packages:

```python
from cicaddy.config.settings import CoreSettings
from cicaddy.agent.factory import AgentFactory
from cicaddy.agent.base import BaseAIAgent

# Create platform-specific settings
class GitLabSettings(CoreSettings):
    gitlab_token: str = ""
    project_id: str = ""

# Register custom agent
class MergeRequestAgent(BaseAIAgent):
    ...

AgentFactory.register("merge_request", MergeRequestAgent)
```

## License

Apache-2.0

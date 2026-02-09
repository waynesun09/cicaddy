# Claude Code Rules

## Project Overview

Cicaddy is a platform-agnostic pipeline AI agent library. It provides the core agent framework, AI providers, MCP client, execution engine, and reporting — designed to be extended by platform-specific packages (e.g., GitLab, GitHub).

## Architecture

### Agent & Factory System

The agent system uses a **registry-based factory** pattern:

- `BaseAIAgent` (in `agent/base.py`) — abstract base with shared init, AI provider setup, MCP manager, execution engine, Slack notifier, and the `analyze()` template method.
- `AgentFactory` (in `agent/factory.py`) — class-level registry mapping type strings to agent classes, with pluggable type detectors.
- Built-in agents: `BranchReviewAgent`, `CronAIAgent`. Platform packages register their own (e.g., `MergeRequestAgent`).

#### Extending with a platform-specific agent

1. **Subclass `BaseAIAgent`** (or `BaseReviewAgent` for code review agents).
2. **Override `_setup_platform_integration()`** to inject platform-specific analyzers (the base implementation is a no-op).
3. **Register the agent** with `AgentFactory.register("my_type", MyAgent)`.
4. **Register a detector** with `AgentFactory.register_detector(my_detector_fn, priority=40)` — detectors receive `Settings` and return an agent type string or `None`.

```python
from cicaddy.agent.base import BaseAIAgent
from cicaddy.agent.factory import AgentFactory

class MyPlatformAgent(BaseAIAgent):
    async def _setup_platform_integration(self):
        # Set up platform-specific analyzer
        self.gitlab_analyzer = MyAnalyzer(...)

    async def _gather_context(self):
        ...

    async def _build_prompt(self, context):
        ...

AgentFactory.register("my_platform", MyPlatformAgent)
```

#### Factory detection priority

Detectors run in priority order (lower = first). Built-in CI detector runs at priority 50. Platform packages should use 30-40 for their detectors so they take precedence.

### Settings

- `CoreSettings` (pydantic-settings) — base config with AI provider, model, MCP, Slack, logging fields. Reads from env vars.
- Platform packages extend `CoreSettings` with platform-specific fields (tokens, project IDs, etc.).
- `load_core_settings()` / `load_settings()` — factory functions for instantiation.

### Key Subpackages

| Package | Purpose |
|---------|---------|
| `ai_providers/` | Provider abstraction (Gemini, Claude, OpenAI) |
| `execution/` | Token-aware multi-step executor, recovery, context compaction |
| `mcp_client/` | MCP client with SSE, HTTP, stdio, WebSocket transports |
| `tools/` | Local file tool registry with decorator-based registration |
| `dspy/` | DSPy task loading and prompt building |
| `notifications/` | Slack (webhook + rich blocks) and email notifiers |
| `reports/` | HTML report generation |
| `config/` | Settings and advanced configuration |

## Code Quality

- Run `pre-commit run --files <changed-files>` before committing
- Run `uv run pytest tests/unit/ -q` before committing
- Prefer shared/utility modules over code duplication

## Git Workflow

- Sign commits: `git commit -s`
- Only commit files modified by Claude in current session
- No "Generated with Claude Code" or "Co-Authored-By" in commits
- Ask permission before pushing

## Python

- Use `uv` for package management
- Always use virtual environments
- Dev install: `uv pip install -e ".[dev,test]"`
- Run tests: `uv run pytest tests/unit/ -q`
- Type checking: `uv run ty check`

## CLI

```
cicaddy run --env-file .env
cicaddy run --agent-type cron --ai-provider gemini
cicaddy config show --env-file .env
cicaddy validate --env-file .env
cicaddy version
```

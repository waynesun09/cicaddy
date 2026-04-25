# Core Architecture

## Agent & Factory System

The agent system uses a **registry-based factory** pattern:

- `BaseAIAgent` (in `agent/base.py`) — abstract base with shared init, AI provider setup, MCP manager, execution engine, Slack notifier, and the `analyze()` template method
- `AgentFactory` (in `agent/factory.py`) — class-level registry mapping type strings to agent classes, with pluggable type detectors
- Built-in agents: `BranchReviewAgent`, `TaskAgent`. Platform packages register their own (e.g., `MergeRequestAgent`)

**Factory detection priority**: Detectors run in priority order (lower = first). Built-in CI detector runs at priority 50. Platform packages should use 30-40.

See `references/factory-extension.md` for full extension guide.

## Settings

- `CoreSettings` (pydantic-settings) — base config with AI provider, model, MCP, Slack, logging fields. Reads from env vars
- Platform packages extend `CoreSettings` with platform-specific fields (tokens, project IDs, etc.)
- `load_core_settings()` / `load_settings()` — factory functions for instantiation

## Key Subpackages

| Package | Purpose |
|---------|---------|
| `ai_providers/` | Provider abstraction (Gemini, Claude, OpenAI — with Vertex AI support for Gemini and Claude) |
| `execution/` | Token-aware multi-step executor, recovery, context compaction |
| `mcp_client/` | MCP client with SSE, HTTP, stdio, WebSocket transports + security scanning |
| `tools/` | Local file tool registry with decorator-based registration + scanning |
| `security/` | Provenance detection and security utilities |
| `dspy/` | DSPy task loading and prompt building |
| `notifications/` | Slack (webhook + rich blocks) and email notifiers |
| `reports/` | HTML report generation |
| `config/` | Settings and advanced configuration |
| `delegation/` | AI-powered sub-agent triage, orchestration, summarization, and registry |
| `knowledge.py` | Bundled knowledge context (model reference, config guidance) |
| `rules.py` | Agent rules auto-loading with external content scanning |
| `skills.py` | Skills discovery with supply chain protection |

## Agent Rules & Skills (v0.6.0+)

Cicaddy automatically loads:
- **Rule files**: `AGENT.md`/`AGENTS.md` (generic), `CLAUDE.md`, `GEMINI.md`, `COPILOT.md` (provider-specific)
- **Skills**: From `.agents/skills/` (cross-tool), `.claude/skills/`, `.gemini/skills/`, `.github/skills/` (provider-specific)

Rules and skills are injected into agent prompts during initialization. See `src/cicaddy/rules.py` and `src/cicaddy/skills.py`.

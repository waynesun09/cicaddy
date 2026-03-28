# Cicaddy Development Guidelines

## Project Overview

Cicaddy is a platform-agnostic pipeline AI agent library. It provides the core agent framework, AI providers, MCP client, execution engine, and reporting — designed to be extended by platform-specific packages (e.g., GitLab, GitHub).

**Key capabilities (v0.6.0+):**
- Auto-loads agent rules from `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `COPILOT.md` in workspace
- Auto-discovers skills from `.agents/skills/`, `.claude/skills/`, `.gemini/skills/`, `.github/skills/`
- Provider-specific skills and rules take precedence over cross-tool defaults

## Architecture

### Agent & Factory System

The agent system uses a **registry-based factory** pattern:

- `BaseAIAgent` (in `agent/base.py`) — abstract base with shared init, AI provider setup, MCP manager, execution engine, Slack notifier, and the `analyze()` template method
- `AgentFactory` (in `agent/factory.py`) — class-level registry mapping type strings to agent classes, with pluggable type detectors
- Built-in agents: `BranchReviewAgent`, `TaskAgent`. Platform packages register their own (e.g., `MergeRequestAgent`)

#### Extending with a platform-specific agent

1. **Subclass `BaseAIAgent`** (or `BaseReviewAgent` for code review agents)
2. **Override `_setup_platform_integration()`** to inject platform-specific analyzers (the base implementation is a no-op)
3. **Register the agent** with `AgentFactory.register("my_type", MyAgent)`
4. **Register a detector** with `AgentFactory.register_detector(my_detector_fn, priority=40)` — detectors receive `Settings` and return an agent type string or `None`

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

**Factory detection priority**: Detectors run in priority order (lower = first). Built-in CI detector runs at priority 50. Platform packages should use 30-40 for their detectors so they take precedence.

### Settings

- `CoreSettings` (pydantic-settings) — base config with AI provider, model, MCP, Slack, logging fields. Reads from env vars
- Platform packages extend `CoreSettings` with platform-specific fields (tokens, project IDs, etc.)
- `load_core_settings()` / `load_settings()` — factory functions for instantiation

### Key Subpackages

| Package | Purpose |
|---------|---------|
| `ai_providers/` | Provider abstraction (Gemini, Claude, OpenAI) |
| `execution/` | Token-aware multi-step executor, recovery, context compaction |
| `mcp_client/` | MCP client with SSE, HTTP, stdio, WebSocket transports + security scanning |
| `tools/` | Local file tool registry with decorator-based registration + scanning |
| `security/` | Provenance detection and security utilities |
| `dspy/` | DSPy task loading and prompt building |
| `notifications/` | Slack (webhook + rich blocks) and email notifiers |
| `reports/` | HTML report generation |
| `config/` | Settings and advanced configuration |
| `rules.py` | Agent rules auto-loading with external content scanning |
| `skills.py` | Skills discovery with supply chain protection |

### Agent Rules & Skills (v0.6.0+)

Cicaddy automatically loads:
- **Rule files**: `AGENT.md`/`AGENTS.md` (generic), `CLAUDE.md`, `GEMINI.md`, `COPILOT.md` (provider-specific)
- **Skills**: From `.agents/skills/` (cross-tool), `.claude/skills/`, `.gemini/skills/`, `.github/skills/` (provider-specific)

Rules and skills are injected into agent prompts during initialization. See `src/cicaddy/rules.py` and `src/cicaddy/skills.py`.

### Security Scanning (v0.6.1+)

Cicaddy provides comprehensive prompt injection protection across all content sources:

**What gets scanned:**
- **MCP tool responses** — External API responses (enforce mode by default)
- **Local file tools** — File reads that could access external content (audit mode by default)
- **External rules** — Rule files from git submodules or untracked sources (audit mode)
- **External skills** — Skills from dependencies or global directories (enforce mode)

**Provenance-based policies:**
- **Local/git-tracked content** — Trusted via code review, skips scanning
- **Submodule content** — Detected via `.gitmodules`, scanned as external
- **Untracked content** — Detected via `git ls-files`, scanned as external
- **Global skills** — Always treated as external (supply chain risk)

**Configuration:**
```bash
# Local file tools
LOCAL_TOOLS_SCAN_MODE=audit                    # disabled|audit|enforce
LOCAL_TOOLS_BLOCKING_THRESHOLD=0.3             # 0.0-1.0

# Rules (AGENT.md, CLAUDE.md from submodules)
RULES_SCAN_MODE=audit
RULES_BLOCKING_THRESHOLD=0.3

# Skills (.agents/skills/ from dependencies)
SKILLS_SCAN_MODE=enforce                       # Stricter for supply chain
SKILLS_BLOCKING_THRESHOLD=0.2
```

**Scan modes:**
- `disabled` — No scanning
- `audit` — Log warnings, don't block
- `enforce` — Block content above threshold

See `docs/mcp-security-scanning.md` and `docs/SCANNING-REVIEW.md` for details.

## Code Quality

- Run `pre-commit run --files <changed-files>` before committing
- Run `uv run pytest tests/unit/ -q` before committing (must pass 935+ tests)
- Prefer shared/utility modules over code duplication
- Follow type hints, Google-style docstrings, async where appropriate

## Git Workflow

- **Sign commits**: `git commit -s` (DCO sign-off required)
- Only commit files modified in current session
- **No "Generated with Claude Code"** or **"Co-Authored-By"** in commits, PR descriptions, or MR descriptions
- Use `-` (hyphens) not `/` (slashes) in branch/worktree names
- Ask permission before pushing to remote

## Python

- Use `uv` for package management
- Always use virtual environments
- Dev install: `uv pip install -e ".[dev,test]"`
- Run tests: `uv run pytest tests/unit/ -q`
- Type checking: `uv run ty check`
- Format: `pre-commit run ruff-format --files <changed-files>`

## CLI Usage

```bash
# Run agent with environment file
cicaddy run --env-file .env

# Run with CLI arguments
cicaddy run --agent-type task --ai-provider gemini

# Show configuration
cicaddy config show --env-file .env

# Validate configuration
cicaddy validate --env-file .env

# Show version
cicaddy version

# Extract code review graph context (requires cicaddy[graph])
cicaddy graph-context --base-ref origin/main --update -o graph_context.json
```

## Testing

### MCP Integration

When working with MCP servers:
- Test with both stdio and HTTP transports
- Verify prompt injection scanner modes: `disabled`, `audit`, `enforce`
- Check MCP_SERVERS_CONFIG parsing in DSPy task files
- Test graceful degradation when MCP servers are unavailable

### Security Scanning

When testing security features:
- Test local tools scanning with external files (should be scanned in audit mode)
- Test git submodule rule files (should be detected as external and scanned)
- Test external skills from `.agents/skills/` (should be scanned in enforce mode)
- Verify provenance detection: git-tracked files should skip scanning
- Test threshold behavior: content below threshold should not be blocked
- Run tests: `uv run pytest tests/unit/test_*scanning*.py tests/unit/test_provenance.py -v`

## Release Process

1. Bump version in `pyproject.toml`
2. Update `AGENTS.md` if architecture changes
3. Run full test suite: `uv run pytest tests/unit/ -q`
4. Create release with `gh release create v<version>`
5. PyPI publish is automated via `.github/workflows/python-publish.yml`
6. Downstream packages (cicaddy-gitlab, cicaddy-action) auto-pick latest via `>=` constraints

# Cicaddy Development Guidelines

## Project Overview

Cicaddy is a platform-agnostic pipeline AI agent library. It provides the core agent framework, AI providers, MCP client, execution engine, and reporting — designed to be extended by platform-specific packages (e.g., GitLab, GitHub).

**Key capabilities:**
- Auto-loads agent rules from `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `COPILOT.md` in workspace (v0.6.0+)
- Auto-discovers skills from `.agents/skills/`, `.claude/skills/`, `.gemini/skills/`, `.github/skills/` (v0.6.0+)
- Provider-specific skills and rules take precedence over cross-tool defaults
- Claude via Vertex AI (`anthropic-vertex` provider) — uses Google Cloud ADC, no API key needed (v0.7.0+)
- Bundled knowledge and skills shipped with the package for model guidance and config reference (v0.8.0+)
- AI-powered sub-agent delegation with parallel execution, triage, sibling awareness, and context-aware review triage (v0.8.0+)
- AI-powered review summarization with structured findings and inline comment support (v0.10.0+)

## Architecture

Cicaddy uses a registry-based factory pattern for agents, pydantic-settings for configuration, and async execution with MCP tool servers.

For detailed architecture documentation, see:
- **Core architecture**: `.agents/skills/cicaddy/references/architecture.md`
- **Factory extension guide**: `.agents/skills/cicaddy/references/factory-extension.md`
- **Sub-agent delegation**: `.agents/skills/cicaddy/references/delegation.md` and `docs/sub-agent-delegation.md`
- **Security scanning**: `.agents/skills/cicaddy/references/security-scanning.md` and `docs/mcp-security-scanning.md`
- **DSPy task format**: `.agents/skills/cicaddy/references/dspy-task-yaml.md`
- **Local testing**: `.agents/skills/cicaddy/references/local-testing.md`

## Local Testing & Evaluation

Two sub-agents in `.agents/agents/` support local cicaddy testing without flooding the context window:

- **cicaddy-runner** — Executes cicaddy runs locally, manages env files and output directories. Returns compact summary tables only.
- **cicaddy-eval** — Analyzes cicaddy JSON reports and session files via `python3 -c` extraction scripts. Produces structured metrics and comparisons.

Run outputs are stored in `_cicaddy_runs/` (gitignored). **Never read cicaddy output files directly** — always delegate to the sub-agents.

**Maintenance**: When changing the agent output format (e.g., fields in `analysis_result` like `summarized`, `findings`, or `sub_agent_details`), update the JSON Report Structure and extraction scripts in `.agents/agents/cicaddy-eval.md` to match.

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

### Delegation

When testing delegation features:
- Run delegation unit tests: `uv run pytest tests/unit/test_delegation_*.py -v`
- Test triage prompt construction and JSON response parsing
- Test registry loading: built-in agents, YAML overrides, JSON config merges
- Test sub-agent tool filtering (blocked/allowed tools)
- Test orchestrator parallel execution and partial failure handling
- Verify `DELEGATION_MODE=none` preserves existing single-agent behavior

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

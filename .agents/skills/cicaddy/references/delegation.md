# Sub-Agent Delegation (v0.8.0+)

Cicaddy supports AI-powered sub-agent delegation where an AI triage step
selects specialized sub-agents, runs them in parallel with sibling awareness
(each agent knows what others cover to avoid duplication), and aggregates results.

## Enable delegation

```bash
# In .env file
DELEGATION_MODE=auto          # none (default, single-agent) | auto (delegation)
MAX_SUB_AGENTS=3              # Max concurrent sub-agents (1-10)
SUB_AGENT_MAX_ITERS=5         # Max iterations per sub-agent (1-15)
TRIAGE_PROMPT=""              # Optional custom triage instructions
DELEGATION_AGENTS_DIR=.agents/delegation  # Custom agent YAML directory
DELEGATION_AGENTS=""          # JSON inline custom agent definitions
```

## Run with delegation

```bash
# Delegated code review (set DELEGATION_MODE=auto in .env)
uv run cicaddy run --env-file .env

# Or override via CLI flags
uv run cicaddy run --env-file .env --delegation-mode auto --max-sub-agents 2
```

When using `AI_TASK_FILE` with delegation, the task definition (persona, constraints,
tool restrictions, output format) is loaded and provided to the triage agent as context.
The task's `forbidden_tools` are cascaded to all sub-agents.

## Built-in sub-agents

**Review**: `security-reviewer`, `architecture-reviewer`, `api-reviewer`,
`database-reviewer`, `ui-reviewer`, `devops-reviewer`, `performance-reviewer`,
`general-reviewer`

**Task**: `data-analyst`, `report-writer`, `general-task`

## Custom sub-agents (YAML)

Place YAML files in `.agents/delegation/{agent_type}/`:

```yaml
# .agents/delegation/review/compliance-reviewer.yaml
name: compliance-reviewer
agent_type: review
persona: compliance engineer specializing in regulatory requirements
description: Reviews changes for regulatory and compliance impact
categories: [security, configuration]
constraints:
  - Focus on regulatory compliance (SOC2, GDPR, HIPAA)
  - Flag any PII handling changes
  - Check audit logging requirements
output_sections:
  - Compliance Impact
  - Regulatory Risks
  - Required Controls
priority: 15
# Optional: restrict tools (strict whitelist if set)
# allowed_tools: ["read_file", "list_directory"]
# blocked_tools: []
```

## How it works

1. **Triage** -- AI analyzes the context and selects sub-agents from the registry
2. **Parallel execution** -- Sub-agents run with focused prompts, filtered tools, workspace context (bundled skills, agent rules, repo skills), sibling awareness, and divided token budgets
3. **Aggregation** -- Results merged into unified output with per-agent sections

Sub-agents share parent's MCP connections and tool registry (no new server processes). They also inherit the parent's workspace context: bundled skills, per-repo agent rules (`AGENT.md`/`CLAUDE.md`/`GEMINI.md`), and per-repo skills (`.agents/skills/`). Side-effect tools (post comments, merge PRs) are blocked by default via plugin entry points.

## Review summarization (v0.10.0+)

When 2+ sub-agents produce reviews, the **SummarizationAgent** condenses output into a ~300-500 word summary with structured `Finding` objects (file, line, severity, message). Line numbers are resolved via a two-step process: deterministic diff snippet matching, then AI fallback.

```bash
DELEGATION_SUMMARIZE=true                         # Enable (default: true)
DELEGATION_SUMMARIZATION_PROMPT=""                # Optional custom instructions
```

See `docs/sub-agent-delegation.md` for full details on findings, line resolution, and fallback behavior.

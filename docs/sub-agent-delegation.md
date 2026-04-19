# Sub-Agent Delegation

Cicaddy supports AI-powered sub-agent delegation: instead of running a single agent, an AI triage step analyzes the context, selects specialized sub-agents, runs them in parallel with sibling awareness (each agent knows what others cover to avoid duplication), and aggregates results.

## How It Works

1. **Triage** — An AI call analyzes the context (diff, task description, etc.) and selects which sub-agents to activate from the registry
2. **Parallel Execution** — Selected sub-agents run concurrently with focused prompts, filtered tools, sibling awareness, and divided token budgets
3. **Aggregation** — Results are merged into a unified output with per-agent sections

Each sub-agent receives a `SiblingInfo` list describing the other agents in the same batch. When running solo (common for small changes), the agent is told it is the sole reviewer and should provide comprehensive coverage. When running alongside specialists, it knows their categories and avoids duplicating their work.

Sub-agents share the parent's MCP connections and tool registry — no new server processes are created. They also inherit the parent's workspace context: bundled skills, per-repo agent rules (`AGENT.md`/`CLAUDE.md`/`GEMINI.md`), and per-repo skills (`.agents/skills/`). This ensures sub-agents have the same knowledge and guidelines as the parent agent. Side-effect tools (posting comments, merging PRs) are blocked by default via plugin entry points.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DELEGATION_MODE` | `none` | `none` (single-agent) or `auto` (AI-powered delegation) |
| `MAX_SUB_AGENTS` | `3` | Maximum concurrent sub-agents (1-10) |
| `SUB_AGENT_MAX_ITERS` | `5` | Max inference iterations per sub-agent (1-15) |
| `DELEGATION_AGENTS` | `""` | JSON config for custom sub-agent definitions |
| `DELEGATION_AGENTS_DIR` | `.agents/delegation` | Directory for user-defined sub-agent YAML files |
| `TRIAGE_PROMPT` | `""` | Optional custom instructions for the triage AI |
| `DELEGATION_SUMMARIZE` | `true` | Enable AI-powered review summarization for multi-agent results (v0.10.0+) |
| `DELEGATION_SUMMARIZATION_PROMPT` | `""` | Optional custom instructions for the summarization AI (v0.10.0+) |

### CLI flags

```bash
cicaddy run --env-file .env --delegation-mode auto --max-sub-agents 2
```

These override the corresponding env vars. Other delegation settings (`SUB_AGENT_MAX_ITERS`, `TRIAGE_PROMPT`, etc.) are env-var only.

## Built-in Sub-Agents

### Review agents

Activated when the parent agent type is a review type (e.g., `merge_request`, `branch_review`).

| Agent | Focus Areas |
|-------|-------------|
| `security-reviewer` | Auth, crypto, secrets, injection, access control |
| `architecture-reviewer` | Design patterns, module boundaries, interfaces |
| `api-reviewer` | Endpoints, schemas, versioning, backward compat |
| `database-reviewer` | Queries, migrations, schema changes, indexes |
| `ui-reviewer` | Frontend components, accessibility, UX |
| `devops-reviewer` | CI/CD pipelines, Docker, deployment |
| `performance-reviewer` | Algorithms, caching, concurrency, resource usage |
| `general-reviewer` | Code correctness, clarity, naming, error handling, test coverage |

### Task agents

Activated when `AGENT_TYPE=task`.

| Agent | Focus Areas |
|-------|-------------|
| `data-analyst` | Data processing, statistics, pattern recognition |
| `report-writer` | Report generation, formatting, documentation |
| `general-task` | General-purpose catch-all |

## Custom Sub-Agents

### YAML files

Place YAML files in `.agents/delegation/{agent_type}/` to define custom sub-agents:

```
.agents/delegation/
├── review/                    # Review-specific agents
│   └── compliance-reviewer.yaml
├── task/                      # Task-specific agents
│   └── data-validator.yaml
└── shared-analyst.yaml        # agent_type: * (available to all)
```

YAML format:

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
# Optional tool filtering
allowed_tools: ["read_file", "list_directory"]  # strict whitelist (if set)
blocked_tools: []                                # additional blocks
```

### JSON inline

Define agents inline via the `DELEGATION_AGENTS` env var:

```bash
DELEGATION_AGENTS='[{"name": "compliance-reviewer", "agent_type": "review", "persona": "compliance engineer", "description": "Reviews compliance impact", "categories": ["security"]}]'
```

### Merge precedence

1. Built-in agents (lowest priority)
2. User YAML files from `DELEGATION_AGENTS_DIR`
3. `DELEGATION_AGENTS` JSON overrides (highest priority)

User-defined agents with the same name as a built-in agent will replace it.

## Tool Filtering

Sub-agents receive a filtered subset of the parent's tools:

1. **Base blocked**: `delegate_task` (prevents recursive delegation)
2. **Plugin blocked**: platform plugins register side-effect tools via `cicaddy.delegation_blocked_tools` entry point
3. **Per-agent**: `SubAgentSpec.allowed_tools` (strict whitelist) and `blocked_tools` (additional blocks)

### Plugin entry point

Plugins register their blocked tools in `pyproject.toml`:

```toml
[project.entry-points."cicaddy.delegation_blocked_tools"]
my_platform = "my_plugin.plugin:get_delegation_blocked_tools"
```

```python
def get_delegation_blocked_tools() -> set[str]:
    return {"create_comment", "merge_pr", "update_issue"}
```

## Review Summarization & Inline Comments (v0.10.0+)

When multiple sub-agents review code, the combined output can exceed 2000 words. The **SummarizationAgent** condenses multi-agent reviews into a concise ~300-500 word summary with structured findings for inline comment placement.

### How It Works

1. **Sub-agents quote code** — Instead of guessing line numbers, reviewers quote `existing_code` snippets from the diff
2. **Deterministic resolution** — `find_line_in_diff()` matches snippets to diff line numbers using exact, normalized, and fuzzy matching (93% cutoff, single-line only)
3. **AI fallback** — Unresolved snippets trigger a lightweight AI call with annotated diff context
4. **Structured output** — Findings include file path, line range, severity, category, and message

### Configuration

```bash
DELEGATION_SUMMARIZE=true                         # Enable multi-agent summarization (default)
DELEGATION_SUMMARIZATION_PROMPT=""                # Optional custom instructions
```

### Result Fields

The delegation result includes:

- `summarized: bool` — True if any summarization was performed (AI or fallback concatenation)
- `ai_summarized: bool` — True only if AI summarization succeeded (distinguishes from fallback)
- `findings: list[Finding]` — Structured findings with line numbers for inline comments

Each `Finding` includes:

```python
@dataclass
class Finding:
    file: str                    # File path relative to repo root
    line: int | None             # Best-effort line number (null for file-level)
    severity: str                # critical | major | minor | nit
    message: str                 # Human-readable finding description
    suggestion: str | None       # Concrete fix or null if none
    agent_source: str            # Which sub-agent identified this
    existing_code: str | None    # Code snippet quoted by reviewer
    start_line: int | None       # Start line in diff (null for file-level)
    end_line: int | None         # End line in diff (null for single-line)
```

### Single-Agent Bypass

When delegation selects only one sub-agent (common for small changes), summarization is skipped to avoid AI overhead. The single agent's output is used directly with `summarized=False`.

### Fallback Behavior

If AI summarization fails (JSON parse error, empty response, timeout), the orchestrator falls back to deterministic concatenation with `ai_summarized=False` but `summarized=True` to indicate the result was processed.

## DSPy Task Files + Delegation

When using `AI_TASK_FILE` with `DELEGATION_MODE=auto`, the task definition (persona, constraints, tool restrictions, output format) is loaded and provided to the triage agent as context. This enables task-aware delegation — the triage model understands the task's intent and can make better sub-agent assignments.

The task's `forbidden_tools` are automatically cascaded to all sub-agents' `blocked_tools`, ensuring sub-agents respect the task's tool restrictions.

```bash
# DSPy task with delegation
AI_TASK_FILE=examples/dora_metrics_task.yaml
DELEGATION_MODE=auto
# Triage sees: task name, persona, constraints, required_tools, output_format
# → selects data-analyst for SQL queries + report-writer for HTML output
```

## Examples

See [`examples/delegation/`](../examples/delegation/) for:
- `delegation_review.env` — example .env for delegated code review
- `review/compliance-reviewer.yaml` — custom compliance reviewer
- `review/dependency-reviewer.yaml` — custom dependency reviewer

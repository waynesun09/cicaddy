# Local Testing & Evaluation

Two sub-agents in `.agents/agents/` handle local cicaddy runs and output evaluation:

- **cicaddy-runner** -- Runs cicaddy locally with env file management, batch support
  across models/delegation modes, and output directory isolation. Returns only
  compact run summary tables.
- **cicaddy-eval** -- Evaluates cicaddy output (JSON reports, session files). Produces
  structured metric summaries and side-by-side comparisons. Never loads full report
  content into context.

## Workflow

1. Spawn `cicaddy-runner` with run configurations -> returns output paths + status
2. Spawn `cicaddy-eval` with output paths -> returns compact comparison table
3. Relay summary to user

## Forcing Multi-Agent Delegation

When `DELEGATION_MODE=auto`, the triage AI decides how many sub-agents to
dispatch based on perceived complexity. Small or focused PRs often get triaged
as "low" complexity with only 1 agent, which means:

- The **summarizer never runs** (requires 2+ successful agents)
- No structured **findings** are extracted
- **Line resolution** and **hunk validation** are skipped
- The output is prose-only with no inline comment placement

To force multi-agent execution and exercise the full findings pipeline:

```bash
# In the .env file:
DELEGATION_MODE=auto
MAX_SUB_AGENTS=3
TRIAGE_PROMPT=Always assign at least 2 sub-agents for thorough coverage.
DELEGATION_SUMMARIZE=true
LOG_LEVEL=DEBUG          # optional: see line resolution + hunk validation stats
```

The key is `TRIAGE_PROMPT` — it instructs the triage AI to select 2+ agents
regardless of perceived complexity. `MAX_SUB_AGENTS` sets the upper bound.

**Pipeline chain with 2+ agents:**

```
Sub-agents (parallel) → Summarizer (extracts Finding objects)
  → resolve_findings()           — deterministic line resolution
  → _ai_resolve_lines()          — AI fallback for unresolved
  → validate_findings_in_hunks() — clamp/clear lines outside diff hunks
  → Findings with valid line numbers → platform plugins (inline comments)
```

**Log messages to verify the pipeline ran:**

```
Line resolution: N resolved, N unresolved out of N findings
Hunk validation: N valid, N clamped, N cleared out of N findings
```

If these messages are absent, the summarizer didn't produce findings (check
that 2+ agents succeeded).

## Context Protection Rules

- **NEVER** read cicaddy output files (JSON/HTML/session JSONL) directly
- Always delegate to `cicaddy-runner` or `cicaddy-eval` sub-agents
- cicaddy runs MUST redirect stdout/stderr (never pipe to terminal)
- Output summaries are capped at 2000-3000 characters
- Run outputs are stored in `_cicaddy_runs/` (gitignored)

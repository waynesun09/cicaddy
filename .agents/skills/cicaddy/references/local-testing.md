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

## Context Protection Rules

- **NEVER** read cicaddy output files (JSON/HTML/session JSONL) directly
- Always delegate to `cicaddy-runner` or `cicaddy-eval` sub-agents
- cicaddy runs MUST redirect stdout/stderr (never pipe to terminal)
- Output summaries are capped at 2000-3000 characters
- Run outputs are stored in `_cicaddy_runs/` (gitignored)

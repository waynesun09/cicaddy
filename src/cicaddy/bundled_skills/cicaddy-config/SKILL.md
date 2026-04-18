---
name: cicaddy-config
description: Cicaddy configuration variables, task definitions, MCP server setup, and agent rules/skills reference
---

## Task Configuration

### Inline Prompt
Set `AI_TASK_PROMPT` with a custom review/analysis prompt:
```yaml
variables:
  AI_TASK_PROMPT: |
    Review this code for security issues and performance bottlenecks.
    Focus on OWASP top 10 vulnerabilities.
```

### DSPy YAML Task File
Set `AI_TASK_FILE` to a structured task definition (takes precedence over `AI_TASK_PROMPT`):
```yaml
# task.yaml
name: security-review
type: code_review
description: Security-focused code review
reasoning_strategy: chain_of_thought

inputs:
  - name: diff
    description: Code diff to review

outputs:
  - name: review
    format: markdown
    description: Security review findings

constraints:
  - Focus on OWASP top 10
  - Cite specific line numbers
```

## MCP Server Configuration

```yaml
# JSON/YAML list in MCP_SERVERS_CONFIG
variables:
  MCP_SERVERS_CONFIG: |
    [
      {
        "name": "Context7",
        "protocol": "http",
        "endpoint": "https://mcp.context7.com/mcp",
        "headers": {"CONTEXT7_API_KEY": "$CONTEXT7_API_KEY"},
        "timeout": 300,
        "idle_timeout": 60,
        "scan_mode": "enforce"
      }
    ]
```

### MCP Server Fields
- `name`: Display name
- `protocol`: `sse`, `http`, `stdio`, or `websocket`
- `endpoint`: URL for remote servers
- `command`/`args`: For stdio servers
- `tools`: Optional list to restrict available tools
- `timeout`: Request timeout (default: 30s)
- `scan_mode`: `disabled` (default for MCP), `audit`, or `enforce`

## Agent Rules & Skills

### Rules Auto-Loading
Cicaddy auto-loads rule files from the workspace root:
- `AGENT.md` / `AGENTS.md` — Generic rules (always loaded)
- `GEMINI.md` — Loaded when `AI_PROVIDER=gemini`
- `CLAUDE.md` — Loaded when `AI_PROVIDER=claude` or `anthropic-vertex`
- `COPILOT.md` — Loaded when `AI_PROVIDER=openai`

### Skills Discovery
Skills are discovered from (highest to lowest precedence):
1. Provider-specific: `.claude/skills/`, `.gemini/skills/`
2. Cross-tool: `.agents/skills/`
3. Global: `~/.agents/skills/`
4. Bundled: shipped with cicaddy package

Each skill directory must contain a `SKILL.md` with YAML frontmatter:
```yaml
---
name: skill-name
description: What this skill provides
---
Skill instructions here...
```

## Execution Tuning

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `MAX_INFER_ITERS` | `15` | 1+ | Max AI planning iterations |
| `MAX_EXECUTION_TIME` | `600` | 60-7200 | Total time limit (seconds) |
| `CONTEXT_SAFETY_FACTOR` | `0.85` | 0.5-0.97 | Token budget utilization |
| `MAX_TOKENS_RECOVERY_LIMIT` | `3` | 1-10 | Token overflow recovery attempts |
| `AI_TEMPERATURE` | `0.0` | 0.0-2.0 | Generation temperature |

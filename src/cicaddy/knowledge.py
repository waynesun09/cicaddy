"""Bundled knowledge constants for cicaddy agents.

Provides reference knowledge that is always available to cicaddy during task
execution, regardless of what the user's workspace contains. This is the
package-level knowledge tier — lowest precedence, overridden by workspace
rules (AGENT.md) and provider-specific rules (GEMINI.md, CLAUDE.md).

Pattern inspired by Claude Code (prompts.ts), Gemini CLI (snippets.ts),
and Hermes (prompt_builder.py) — all embed core knowledge as source constants.
"""

from __future__ import annotations

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)

# Bundled context injected before workspace rules in every prompt.
# Keep concise — this consumes tokens on every request.
CICADDY_CONTEXT = """\
<cicaddy_reference source="bundled">

## AI Model Reference

### Gemini (Google)
- **gemini-3.1-pro-preview** — Latest flagship, best for complex analysis and code review
- **gemini-3-flash-preview** — Fast and capable, good default for most tasks
- **gemini-3.1-flash-lite-preview** — Lightweight, lowest cost
- The `-preview` suffix is acceptable for Vertex AI enterprise use
- Model names may include date suffixes (e.g., `-YYYYMMDD`) when pinning versions
- Set via `AI_MODEL` env var; default is `gemini-3-flash-preview`

### Claude (Anthropic)
- **claude-opus-4-6** — Most capable, best for complex reasoning
- **claude-sonnet-4-6** — Balanced performance and cost, good default
- **claude-haiku-4-5** — Fast and lightweight
- Acceptable name formats: `claude-sonnet-4-6`, `claude-sonnet-4-6@default`,
  `claude-sonnet-4-6@latest`, or with date suffix `claude-sonnet-4-6-20250514`
- For Vertex AI enterprise, the model name format depends on admin configuration
- Set via `AI_MODEL` env var; default for claude/anthropic-vertex is `claude-sonnet-4-6`

### OpenAI
- **gpt-4o** — Default model for OpenAI provider
- Set via `AI_MODEL` env var

## Cicaddy Configuration Quick Reference

### Core Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `AI_PROVIDER` | `gemini` | Provider: gemini, openai, claude, anthropic, anthropic-vertex |
| `AI_MODEL` | per-provider | Model name (see above) |
| `AI_TEMPERATURE` | `0.0` | Generation temperature (0.0-2.0) |
| `AI_TASK_PROMPT` | — | Inline task prompt (overrides default) |
| `AI_TASK_FILE` | — | Path to DSPy YAML task file (takes precedence over AI_TASK_PROMPT) |
| `AI_RESPONSE_FORMAT` | `markdown` | Output format: markdown, html, json |
| `AGENT_TASKS` | `code_review` | Comma-separated task list |
| `MAX_INFER_ITERS` | `15` | Maximum AI planning iterations |
| `MAX_EXECUTION_TIME` | `600` | Max execution time in seconds (60-7200) |

### MCP & Tools
| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVERS_CONFIG` | `[]` | JSON/YAML list of MCP server configs |
| `ENABLE_LOCAL_TOOLS` | `false` | Enable built-in file tools (glob, read) |
| `LOCAL_TOOLS_WORKING_DIR` | — | Working directory for local file tools |
| `LOCAL_TOOLS_SCAN_MODE` | `audit` | Scan mode for local tool results |

### Agent Rules & Skills
| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_RULES_ENABLED` | `true` | Enable auto-loading of AGENT.md, CLAUDE.md, etc. |
| `AGENT_RULES_WORKSPACE` | — | Override workspace path for rules discovery |
| `RULES_SCAN_MODE` | `audit` | Scan mode for external rule files |
| `SKILLS_SCAN_MODE` | `enforce` | Scan mode for external skill files |

### Security Scanning Modes
- **disabled** — No scanning (default for MCP server results)
- **audit** — Log warnings but allow content (default for rules, local tools)
- **enforce** — Block content exceeding risk threshold (default for skills)

### Anthropic Vertex AI
| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_VERTEX_PROJECT_ID` | — | GCP project ID (required) |
| `CLOUD_ML_REGION` | `us-east5` | Vertex AI region |

Uses Application Default Credentials — no API key needed.
Set `GOOGLE_APPLICATION_CREDENTIALS` or use workload identity.

</cicaddy_reference>
"""


def get_bundled_context() -> str:
    """Return the bundled cicaddy knowledge context.

    Returns:
        Bundled context string wrapped in XML tags, or empty string
        if bundled knowledge is disabled (future extensibility).
    """
    return CICADDY_CONTEXT

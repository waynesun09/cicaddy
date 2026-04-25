"""Bundled knowledge constants for cicaddy agents.

Provides reference knowledge that is always available to cicaddy during task
execution, regardless of what the user's workspace contains. This is the
package-level knowledge tier — lowest precedence, overridden by workspace
rules (AGENTS.md/AGENT.md) and provider-specific rules (GEMINI.md, CLAUDE.md).

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

## AI Model Quick Reference

Default models per provider (set via `AI_MODEL` env var):
- **gemini / gemini-vertex**: `gemini-3-flash-preview` — tiers: pro, flash, flash-lite (pattern: `gemini-{ver}-{tier}-preview`)
- **claude / anthropic-vertex**: `claude-sonnet-4-6` — tiers: opus, sonnet, haiku (pattern: `claude-{tier}-{major}-{minor}`)
- **openai**: `gpt-5.4`

Gemini `-preview` suffix is acceptable for Vertex AI enterprise. Claude names accept `@default`, `@latest`, or date suffixes.

## Key Configuration

- `AI_PROVIDER`: gemini, gemini-vertex, openai, claude, anthropic, anthropic-vertex (default: gemini)
- `AI_TASK_FILE`: DSPy YAML task definition (takes precedence over `AI_TASK_PROMPT`)
- `AGENT_TASKS`: Comma-separated task list (default: code_review)
- `AGENTS.md` / `AGENT.md`: Generic rules loaded for all providers. Provider-specific: `GEMINI.md`, `CLAUDE.md`, `COPILOT.md`
- Scan modes: disabled (MCP default), audit (rules/tools default), enforce (skills default)
- Gemini Vertex: set `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` (default: global), uses Application Default Credentials
- Gemini auto-fallback: `AI_PROVIDER=gemini` without `GEMINI_API_KEY` auto-switches to Vertex AI when `GOOGLE_CLOUD_PROJECT` is set
- Anthropic Vertex: set `ANTHROPIC_VERTEX_PROJECT_ID` + `GOOGLE_CLOUD_LOCATION` (default: global), uses Application Default Credentials

For detailed configuration, DSPy task format, and MCP server setup, see the bundled `cicaddy-config` and `model-reference` skills.

</cicaddy_reference>
"""


def get_bundled_context() -> str:
    """Return the bundled cicaddy knowledge context.

    Returns:
        Bundled context string wrapped in XML tags, or empty string
        if bundled knowledge is disabled (future extensibility).
    """
    return CICADDY_CONTEXT

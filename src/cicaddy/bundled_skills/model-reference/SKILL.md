---
name: model-reference
description: Current AI model names, capabilities, and naming conventions for Gemini, Claude, and OpenAI providers
---

## Gemini Models (Google)

| Model | Use Case | Notes |
|-------|----------|-------|
| `gemini-3.1-pro-preview` | Complex analysis, deep code review | Latest flagship |
| `gemini-3-flash-preview` | General tasks, default for most use cases | Fast, capable |
| `gemini-3.1-flash-lite-preview` | Simple tasks, cost-sensitive | Lightweight |

- The `-preview` suffix is acceptable for Vertex AI enterprise use
- `gemini-3.1-pro-preview` requires the `global` Vertex endpoint; other Gemini preview models support regional endpoints
- Date suffixes (`-YYYYMMDD`) are used to pin specific versions
- All Gemini models support function calling (MCP tool use)

## Claude Models (Anthropic)

| Model | Use Case | Notes |
|-------|----------|-------|
| `claude-opus-4-*` | Complex reasoning, architecture review | Most capable |
| `claude-sonnet-4-*` | Balanced code review, default | Good cost/performance |
| `claude-haiku-4-*` | Fast triage, simple checks | Lightweight |

- Model names follow `claude-{tier}-{major}-{minor}` pattern (e.g., `claude-sonnet-4-6`)
- Acceptable suffixes: `@default`, `@latest`, or date `-YYYYMMDD` (e.g., `claude-sonnet-4-6@default`)
- For Vertex AI enterprise: model name format depends on the organization's admin configuration
- New versions (e.g., opus-4-7) become available as Anthropic releases them
- All Claude models support tool use

## OpenAI Models

| Model | Use Case | Notes |
|-------|----------|-------|
| `gpt-5.4` | Default for OpenAI provider | 1M context, strong coding |
| `gpt-5.4-mini` | Cost-effective alternative | 1M context |
| `gpt-5.4-nano` | Lightweight, high-volume tasks | 1M context |

Override via `AI_MODEL` to use other models (e.g., `o3`, `gpt-4o`).

## Setting the Model

```bash
# Environment variable
AI_MODEL=gemini-3.1-pro-preview

# Or in .gitlab-ci.yml
variables:
  AI_MODEL: "gemini-3.1-pro-preview"
```

The default model per provider is defined in cicaddy's factory:
- gemini: `gemini-3-flash-preview`
- claude/anthropic/anthropic-vertex: `claude-sonnet-4-6`
- openai: `gpt-5.4`

---
name: model-reference
description: Current AI model names, capabilities, and naming conventions for Gemini, Claude, and OpenAI providers
---

## Gemini Models (Google)

| Tier | Use Case | Notes |
|------|----------|-------|
| `gemini-*-pro-preview` | Complex analysis, deep code review | Flagship |
| `gemini-*-flash-preview` | General tasks, default for most use cases | Fast, capable |
| `gemini-*-flash-lite-preview` | Simple tasks, cost-sensitive | Lightweight |

- Model names follow `gemini-{version}-{tier}-preview` pattern (e.g., `gemini-3-flash-preview`, `gemini-3.1-pro-preview`)
- The `-preview` suffix is acceptable for Vertex AI enterprise use
- Pro and Flash tiers typically require the `global` Vertex endpoint; Flash-Lite also supports regional endpoints
- Date suffixes (`-YYYYMMDD`) are used to pin specific versions
- All Gemini models support function calling (MCP tool use)

## Claude Models (Anthropic)

| Tier | Use Case | Notes |
|------|----------|-------|
| `claude-opus-*` | Complex reasoning, architecture review | Most capable |
| `claude-sonnet-*` | Balanced code review, default | Good cost/performance |
| `claude-haiku-*` | Fast triage, simple checks | Lightweight |

- Model names follow `claude-{tier}-{major}-{minor}` pattern (e.g., `claude-sonnet-4-6`)
- Acceptable suffixes: `@default`, `@latest`, or date `-YYYYMMDD` (e.g., `claude-sonnet-4-6@default`)
- For Vertex AI enterprise: model name format depends on the organization's admin configuration
- All Claude models support tool use

## OpenAI Models

| Tier | Use Case | Notes |
|------|----------|-------|
| `gpt-*` | Default for OpenAI provider | Flagship, strong coding |
| `gpt-*-mini` | Cost-effective alternative | Balanced |
| `gpt-*-nano` | Lightweight, high-volume tasks | Lowest cost |

- Model names follow `gpt-{version}` pattern (e.g., `gpt-5.4`, `gpt-5.4-mini`)
- Override via `AI_MODEL` to use other models (e.g., `o3`)

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

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
- Date suffixes (`-YYYYMMDD`) are used to pin specific versions
- All Gemini models support function calling (MCP tool use)

## Claude Models (Anthropic)

| Model | Use Case | Notes |
|-------|----------|-------|
| `claude-opus-4-6` | Complex reasoning, architecture review | Most capable |
| `claude-sonnet-4-6` | Balanced code review, default | Good cost/performance |
| `claude-haiku-4-5` | Fast triage, simple checks | Lightweight |

- Acceptable formats: `claude-sonnet-4-6`, with `@default`, `@latest`, or date suffix `-20250514`
- For Vertex AI enterprise: model name format depends on the organization's admin configuration
- All Claude models support tool use

## OpenAI Models

| Model | Use Case | Notes |
|-------|----------|-------|
| `gpt-4o` | Default for OpenAI provider | Multimodal |

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
- openai: `gpt-4o`

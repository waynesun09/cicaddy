# Metadata Architecture: Dual Storage and Compression

This document covers the dual-storage pattern (Knowledge Store + Conversation Context), tool metadata structure, and multi-layer compression coordination.

## Problem

When the agent completed 10 iterations using MCP tools, reports and Slack notifications contained minimal information despite successfully executing many tool calls. The data was being lost during conversation compaction.

## Solution: Dual Storage

Two parallel data streams with different purposes:

| | Conversation Context | Knowledge Store |
|---|---|---|
| **Purpose** | Keep AI under token limits | Preserve complete tool data |
| **Compression** | Layers 1-3 applied | NEVER compressed |
| **Used by** | AI model for planning | Reports, Slack, audit |
| **Contains** | Compressed reasoning + tool summaries | Full original tool results |

```
                    +→ Compressed for conversation → AI planning
Tool Result --------+
                    +→ Full data in Knowledge Store → Reports + Slack
                                                       (ACCURATE)
```

## Knowledge Store

**Location**: `execution/knowledge_store.py`

`AccumulatedKnowledge` stores every MCP tool result with triple indexing:
- `tool_results` — execution order (list)
- `results_by_server` — grouped by MCP server
- `results_by_tool` — grouped by tool name

Each entry records: `iteration`, `server`, `tool`, `arguments`, `result` (full), `execution_time`, `result_size_bytes`, `timestamp`, `inference_id`, `unique_ref`.

### Integration Points

1. **Executor** (`token_aware_executor.py`): Stores full result in Knowledge Store BEFORE any compression
2. **Turn** (`turn.py`): Carries `accumulated_knowledge` through execution pipeline
3. **Agent** (`base.py`): Returns `accumulated_knowledge` alongside `ai_analysis`
4. **Reports** (`html_formatter.py`): Renders complete tool data from Knowledge Store
5. **Notifications** (`rich_slack.py`): Generates summaries from original tool results

## Tool Metadata

Tool interactions use structured metadata in `ProviderResponse.tool_calls` (`ai_providers/base.py`):

- **Requests**: `msg.tool_calls` — array of `{id, type, function: {name, arguments}}`
- **Responses**: `msg.tool_call_id` — links response to its request

This enables tool pair preservation during compression — the compactor keeps request/response pairs together to prevent AI confusion.

## Compression Coordination

Three independent layers operate at different points:

| Layer | Trigger | Target | Method |
|-------|---------|--------|--------|
| 1. Pre-execution | >70% utilization AND prompt >1000 tokens | Last message | `compress_prompt_before_send()` |
| 2. Real-time | Result exceeds `max_tokens_per_tool_result` | Individual tool result | `compress_tool_result_realtime()` |
| 3. Iterative | iteration >3, conversation >50% budget, or projected overflow | Entire conversation | `compact_iteration_context()` |

**Tool pair preservation**: `_identify_tool_pairs()` scans messages for matching `tool_calls[].id` ↔ `tool_call_id` pairs. These pairs are passed to the compactor, which preserves them intact during compression.

**Quality metrics**: Each compression records `compression_ratio` and `information_preserved` in `ExecutionState`. Typical results: 40-60% size reduction, >90% information retained.

## Related Documentation

- [Architecture Guide](architecture.md) — System overview
- [Token-Aware Execution](token-aware-execution.md) — Resource management and degradation
- [Prompt Engineering Best Practices](https://github.com/redhat-community-ai-tools/cicaddy-gitlab/blob/main/docs/prompt-engineering-best-practices.md) — Last Message Only pattern

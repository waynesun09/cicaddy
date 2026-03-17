# Token-Aware Execution System

The execution engine implements LlamaStack-inspired dual-limit patterns for intelligent resource management with progressive degradation.

## Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `TokenAwareExecutor` | `execution/token_aware_executor.py` | Orchestrates AI inference with resource monitoring |
| `ProgressiveAnalyzer` | `execution/progressive_analyzer.py` | Adapts analysis quality based on utilization |
| `GenericResultFormatter` | `execution/result_formatter.py` | Structured BEGIN/END content markers |
| `ContextCompactor` | `execution/context_compactor.py` | AI-powered conversation compression |
| `RecoveryManager` | `execution/recovery.py` | Handles premature completion and limit recovery |

## ExecutionLimits

```python
from cicaddy.execution.token_aware_executor import ExecutionLimits

limits = ExecutionLimits(
    max_infer_iters=15,            # Maximum AI planning iterations
    max_tokens_total=1048576,      # Total token budget (dynamically set from model capacity)
    max_tokens_per_iteration=8000, # Per-iteration token limit
    max_tokens_per_tool_result=4000, # Per-tool result limit (triggers Layer 2 compression)
    max_tools_per_iteration=5,     # Tools per iteration
    max_total_tools=50,            # Total tool executions
    max_execution_time=600,        # 10 min total
    max_tool_timeout=300,          # 5 min per tool call
    max_tool_idle_timeout=60,      # 1 min between progress updates
    warning_threshold=0.8,         # Warn at 80% utilization
    degradation_threshold=0.9,     # Degrade at 90%
    recovery_enabled=True,
    max_recovery_attempts=3,
)
```

At runtime, `BaseAIAgent` dynamically sets `max_tokens_total` from the model's actual input capacity (e.g., Gemini: 1,048,576, Claude 4: 1,048,576, Claude 3.5: 200,000).

## Safety Valves

Five independent limits, each with its own `StopReason`:

| Limit | Fields | StopReason |
|-------|--------|------------|
| Iterations | `max_infer_iters` | `max_iterations` |
| Tokens | `max_tokens_total`, `max_tokens_per_iteration` | `out_of_tokens` |
| Tools | `max_tools_per_iteration`, `max_total_tools` | `max_tools` |
| Time | `max_execution_time`, `max_tool_timeout` | `timeout` |
| Size | `max_result_size_bytes` | `max_result_size` |

Token estimation uses 4 chars = 1 token (conservative).

## Progressive Degradation

Analysis quality automatically adapts as resources are consumed:

| Utilization | Level | Result Size | Raw Data | Debug |
|-------------|-------|-------------|----------|-------|
| 0-70% | COMPREHENSIVE | 4x tool limit | Yes | Yes |
| 70-85% | DETAILED | 2x tool limit | Yes | No |
| 85-95% | SUMMARY | 1x tool limit | No | No |
| 95%+ | CRITICAL_ONLY | 500 tokens, 1 tool/iter | No | No |

DETAILED and SUMMARY levels keep the configured `max_tools_per_iteration`. Only CRITICAL_ONLY reduces to 1 tool per iteration.

## Recovery System

Detects and recovers from:

| Error Type | Detection | Recovery |
|------------|-----------|----------|
| `AI_PREMATURE_COMPLETION` | Continuation phrases ("Let me...", "I'll...") or ultra-short responses | Guided retry with context |
| `MAX_TOKENS_EXCEEDED` | Token budget exhausted | Compact context, extend iterations |
| `MAX_ITERATIONS_EXCEEDED` | Iteration limit hit | Buffer iterations (up to 3) |
| `TOOL_EXECUTION_ERROR` | Tool call failures | Retry with error context |
| `INVALID_TOOL_CALL` | Malformed tool calls | Correction prompt |

**Final synthesis**: When all recovery attempts fail, triggers a dedicated AI call with `tools=None`, providing all Knowledge Store data and forcing a final deliverable.

Max 3 recovery attempts per error pattern prevents infinite loops.

## Configuration Profiles

```python
# Quick analysis (under 2 min)
ExecutionLimits(max_infer_iters=5, max_tokens_total=25000, max_execution_time=120)

# Comprehensive (up to 15 min)
ExecutionLimits(max_infer_iters=15, max_tokens_total=200000, max_execution_time=900)

# Deep analysis (up to 30 min)
ExecutionLimits(max_infer_iters=25, max_tokens_total=500000, max_execution_time=1800)
```

Environment variables for override: `MAX_INFER_ITERS`, `MAX_TOKENS_TOTAL`, `MAX_EXECUTION_TIME`.

## Execution Summary

After execution, `executor.get_execution_summary()` returns:

```python
{
    "stop_reason": "end_of_turn",
    "iterations": {"completed": 5, "max_allowed": 15, "utilization": 0.33},
    "tokens": {"total_used": 45000, "max_allowed": 1048576, "utilization": 0.04},
    "tools": {"total_executed": 8, "max_allowed": 50, "utilization": 0.16},
    "degradation_active": False,
    "compression": {
        "total_compressions": 3,
        "avg_compression_ratio": 2.3,
        "avg_info_preserved": 0.87
    },
    "recovery": {"activations": 0, "success_rate": 1.0}
}
```

## Related Documentation

- [Architecture Guide](architecture.md) — System overview
- [Metadata Architecture](metadata-architecture.md) — Knowledge Store and compression
- [MCP Integration](mcp-integration.md) — MCP server configuration

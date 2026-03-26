# Shared Task Notes — PR #20 (MCP Prompt Injection Scanner)

## Current State

PR #20 (`feature/mcp-prompt-injection-scanner`) has CodeRabbit review feedback. A worktree exists at `../cicaddy-scanner-wt/` with **unstaged changes** that address all major review issues.

## What's Been Done (unstaged in worktree)

The following CodeRabbit review issues are fixed in unstaged changes:

- **Critical**: Raw findings no longer echoed in logs or `[BLOCKED]` message (`client.py`)
- **Minor**: `scan_mode` validation added — raises `ValueError` for invalid modes (`client.py`)
- **Minor**: `else` changed to `elif self.scan_mode == "audit"` — no silent pass-through (`client.py`)
- **Major**: `CompositeScanner` now passes `llm_guard` config (threshold, use_onnx) (`client.py`)
- **Major**: `EnterpriseClient` forwards `scan_config` to `MCPClientManager` (`enterprise_client.py`)
- **Minor**: `asyncio.get_running_loop()` replaces deprecated `get_event_loop()` (`llm_guard_scanner.py`)
- **Minor**: Docs note that `scan_mode` requires global enablement (`mcp-integration.md`)

## What's Next

1. **Commit and push** the unstaged changes in `../cicaddy-scanner-wt/` to `feature/mcp-prompt-injection-scanner`
2. **Re-request CodeRabbit review** (`@coderabbitai review`)
3. **Composite scanner fast path** — CodeRabbit flagged that non-consensus mode skips ML scanner when heuristic says clean. This is by design (performance optimization), but consider clarifying the docstring or adding a config option to force all scanners to run.
4. **Clean up session files** — Many `session_*.jsonl` and `task_*.{html,json}` files in worktree root should be `.gitignore`d or deleted

## Verification

- All 843 unit tests pass
- All pre-commit hooks pass (ruff, ruff-format, ty check, bandit, detect-secrets)

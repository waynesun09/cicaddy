# Shared Task Notes — PR 20 Review Fixes

## What was done

Applied fixes for 7 CodeRabbit review comments on PR #20 (MCP Prompt Injection Scanner).
Changes are in the existing worktree at `../cicaddy-scanner-wt/` on branch `feature/mcp-prompt-injection-scanner`.

### Fixes applied (in `cicaddy-scanner-wt`):

1. **scan_mode validation** (`client.py`): Added ValueError if scan_mode not in `(disabled, audit, enforce)`
2. **Explicit audit branch** (`client.py`): Changed `else` to `elif self.scan_mode == "audit"` to prevent silent fallthrough
3. **CRITICAL: No raw findings in logs/blocked payload** (`client.py`): Log only count + risk_score, not attacker-controlled text
4. **CompositeScanner llm_guard config** (`client.py`): `_create_scanner_for_server` now passes llm_guard threshold/use_onnx to composite
5. **EnterpriseClient scan_config** (`enterprise_client.py`): Forward `scan_config` from AdvancedMCPConfig to MCPClientManager
6. **Deprecated asyncio API** (`llm_guard_scanner.py`): `get_event_loop()` → `get_running_loop()`, `sanitized` → `_sanitized`
7. **docs/mcp-integration.md**: Clarified scan_mode requires global scanning to be enabled

## Status

- All 843 unit tests pass
- All pre-commit checks pass (ruff lint, ruff format, ty check, bandit, gitleaks)
- Changes NOT committed yet — waiting for review

## Next steps

- Commit and push fixes to `feature/mcp-prompt-injection-scanner` branch (from the `cicaddy-scanner-wt` worktree)
- Some outdated CodeRabbit comments (composite fast-path logic, LLMGuardScanner fail-open) were already addressed in prior commits — verify those are outdated in the PR thread
- Consider adding a test for the new scan_mode validation (ValueError on invalid mode)
- Address docstring coverage (79.44% vs 80% threshold per CodeRabbit pre-merge check)
- Once review comments are resolved, merge PR #20

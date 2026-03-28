# Prompt Injection Defense Review Summary

**Date:** 2026-03-28
**Worktree:** `cicaddy-prompt-defense-wt`
**Branch:** `prompt-defense-balanced-check`

## Your Concerns Addressed

### ✅ 1. Defense Should Be General Tool Level, Not MCP-Only

**Current State (v0.6.0):**
- Scanning is implemented **only in `mcp_client/client.py`**
- Local file tools (`read_file`, `glob_files`) bypass scanning completely
- They return content directly without any security checks

**Solution:**
- Phase 1: Create `ToolScanner` interface that works with **any tool type**
- Integrate at `ToolRegistry` level (affects MCP and local tools equally)
- Configuration: `LOCAL_TOOLS_SCAN_MODE` for local tools, keeps per-server config for MCP

**Result:** Scanning applies uniformly across all tool types, not just MCP.

---

### ✅ 2. Apply Scanning to Initial Loading of Rules and Skills

**Current State:**
- `rules.py:load_agent_rules()` - loads AGENT.md/CLAUDE.md **without scanning**
- `skills.py:discover_skills()` - loads .agents/skills/*/SKILL.md **without scanning**
- Both can load external/untrusted content (submodules, dependencies)

**Solution:**
- Phase 2: Add provenance detection to distinguish local vs. external content
- Scan external rule files in **audit mode** (log warnings)
- Scan external skills in **enforce mode** (block on high risk)
- Skip scanning for local project files (trust code review process)

**Provenance Detection:**
```python
def _is_external_source(path: Path) -> bool:
    """Detect if file is from submodule or outside version control."""
    # Check .gitmodules for submodule paths
    # Check git ls-files for tracked status
    return is_submodule or not_in_version_control
```

**Result:** Initial content loading is protected against supply chain attacks (ToxicSkills, rule injection).

---

### ✅ 3. Apply Scanning to Initial Prompt in Task

**Current State:**
- Task prompts are built from:
  1. System prompt (hardcoded in source)
  2. Task YAML file (`AI_TASK_FILE`)
  3. Agent rules (loaded above)
  4. Skills (loaded above)
  5. Context data (project info, diffs)

**Analysis:**
- **System prompt:** No scanning needed (developer-authored code)
- **Task YAML:** Operator-controlled, version-controlled → trust code review
- **Agent rules:** ✅ Covered in Phase 2 (scan external rules)
- **Skills:** ✅ Covered in Phase 2 (scan external skills)
- **Context data:** Comes from trusted sources (git, GitLab API) → no scanning needed

**Conclusion:** The "initial prompt" is protected by scanning rules/skills that compose it. We don't need to scan the final assembled prompt (that would be over-scanning).

---

## Balanced Approach: What Gets Scanned

### ✅ Scan These (Trust Boundaries)

| Content | Scan Mode | Why |
|---------|-----------|-----|
| **MCP tool responses** | enforce | External APIs, documented attack vector (ContextCrush) |
| **Local file tools responses** | audit | Could read external files, moderate risk |
| **External skills** (from registries) | enforce | Supply chain risk (ToxicSkills campaign) |
| **Submodule rule files** | audit | External repos, moderate organizational trust |
| **Fetched web content** (future) | enforce | Indirect injection vector |

### ❌ Don't Scan These (Trusted Sources)

| Content | Why Not | Alternative Control |
|---------|---------|-------------------|
| **System prompts** | Developer-authored code | Code review, version control |
| **Task YAML config** | Operator-controlled, checked in | PR review, branch protection |
| **Local AGENT.md** (project root) | Team-authored, reviewed | Version control, git provenance |
| **User input** (interactive) | User is the principal | Permission system for actions |
| **Internal APIs** (DevLake, GitLab) | Authenticated infrastructure | Network controls, API auth |

---

## Key Innovation: Blocking Threshold

**Problem:** Current implementation blocks on **any** detection (`risk_score > 0.0`).

**Result:** False positives in legitimate documentation:
- Node.js docs contain `process.env` examples
- Admin guides use `sudo` commands
- API docs have base64-encoded tokens

**Solution (Phase 3):**

```python
# Separate detection from blocking
detection_threshold = 0.0    # Log any finding
blocking_threshold  = 0.3    # Block only when cumulative risk > 0.3

if scan_result.risk_score >= blocking_threshold:
    if scan_mode == "enforce":
        # Block content
    else:
        # Audit: log but pass through
```

**Example:**
- Single `process.env` match → risk_score = 0.3 → logged but **not blocked**
- `process.env` + `send to webhook` → risk_score = 0.6 → **blocked**

---

## Research Findings (Key Takeaways)

From `Research/prompt-injection-defense-agent-systems.md`:

### 🔴 Real Attack Vectors

1. **ContextCrush (Feb 2026):** Context7 MCP server served attacker instructions to all users. Credential exfiltration in minutes.

2. **ToxicSkills (Snyk, Feb 2026):** 36% of ClawHub skills contained prompt injection. 76 confirmed malicious payloads.

3. **Rule File Injection:** AGENTS.md auto-loaded with high trust in every AI IDE. 30+ CVEs disclosed (IDEsaster campaign).

### 🟢 Defense-in-Depth Consensus

**All major players agree:** Anthropic, OpenAI, Google DeepMind, Meta, OWASP, NIST
- No single defense works
- Adaptive attacks bypass 100% of published defenses
- Must layer multiple defenses
- Cicaddy's composite scanner (heuristic + ML) is architecturally sound

### 🟡 The Agents Rule of Two (Meta AI)

An agent is vulnerable when it has all three:
1. **Untrusted input** (web, external docs)
2. **Sensitive data** (credentials, private repos)
3. **External actions** (send email, create issues)

CI/CD agents inherently have all three → must mitigate with scanning, least-privilege, and human approval.

---

## Implementation Roadmap

### ✅ Phase 1: General Tool-Level Scanning (COMPLETE)
**Priority:** P0
**Status:** ✅ **Implemented**
**Impact:** Closes MCP-only gap

**Completed:**
- ✅ Create `tools/scanner.py` with `ToolScanner` interface
- ✅ Add scanning to `ToolRegistry.call_tool()`
- ✅ Add `LOCAL_TOOLS_SCAN_MODE` and `LOCAL_TOOLS_BLOCKING_THRESHOLD` env vars
- ✅ Tests for local file tool scanning (17 tests passing)
- ✅ Separate detection and blocking thresholds

**Files Added:**
- `src/cicaddy/tools/scanner.py` - ToolScanner wrapper
- `tests/unit/test_tool_scanner.py` - 10 tests
- `tests/unit/test_tool_registry_scanning.py` - 7 tests

**Commit:** `259d921`

### ✅ Phase 2: Rules and Skills Scanning (COMPLETE)
**Priority:** P1
**Status:** ✅ **Implemented**
**Impact:** Blocks supply chain attacks

**Completed:**
- ✅ Add provenance detection (`is_external_source()`, `get_provenance_label()`)
- ✅ Scan external rule files in `rules.py`
- ✅ Scan external skills in `skills.py`
- ✅ Add `RULES_SCAN_MODE`, `RULES_BLOCKING_THRESHOLD` env vars
- ✅ Add `SKILLS_SCAN_MODE`, `SKILLS_BLOCKING_THRESHOLD` env vars
- ✅ Tests for malicious rule/skill detection (32 tests passing)
- ✅ Git-based provenance (submodules, tracking status)

**Files Added:**
- `src/cicaddy/security/__init__.py` - Security package
- `src/cicaddy/security/provenance.py` - Provenance detection
- `tests/unit/test_provenance.py` - 9 tests
- `tests/unit/test_rules_scanning.py` - 8 tests
- `tests/unit/test_skills_scanning.py` - 15 tests

**Commit:** `113de6a`

### Phase 3: MCP Blocking Threshold (Optional)
**Priority:** P2
**Effort:** 1-2 days
**Impact:** Reduces false positives for MCP tools

**Tasks:**
- Apply blocking threshold separation to MCP scanning (currently blocks on any detection)
- Unify threshold logic across all scanners
- Add `MCP_SCAN_BLOCKING_THRESHOLD` env var
- Tests for MCP threshold behavior

**Note:** Local tools, rules, and skills already have blocking threshold separation (Phase 1 & 2). This phase would extend the same pattern to MCP tools for consistency.

### Phase 4: Source-Aware Policies (Optional)
**Priority:** P3
**Effort:** 2-3 days
**Impact:** Fine-grained control per source

**Tasks:**
- Add `SourceScanPolicy` model
- Per-source threshold overrides
- Policy lookup logic

**Example:**
```yaml
source_policies:
  context7:
    blocking_threshold: 0.2  # Strict for external docs
  internal-devlake:
    scan_mode: disabled      # Trust internal APIs
```

---

## Configuration Examples

### Development (Audit Everything)
```bash
MCP_SCAN_ENABLED=true
MCP_SCAN_MODE=audit                    # Log, don't block
LOCAL_TOOLS_SCAN_MODE=audit
RULES_SCAN_MODE=audit
SKILLS_SCAN_MODE=audit
```

### Production (Enforce External, Audit Internal)
```bash
MCP_SCAN_ENABLED=true
MCP_SCAN_MODE=enforce                  # Block malicious MCP responses
MCP_SCAN_BLOCKING_THRESHOLD=0.3        # Only block high-risk

LOCAL_TOOLS_SCAN_MODE=audit            # Local tools are safer
RULES_SCAN_MODE=audit                  # Submodule rules logged
SKILLS_SCAN_MODE=enforce               # External skills blocked
SKILLS_BLOCKING_THRESHOLD=0.2          # Strict for supply chain
```

---

## Next Steps

1. **Review this plan** and the detailed plan in `docs/prompt-injection-balanced-defense-plan.md`

2. **Decide on phases:**
   - Phase 1 (general tool scanning) - must-have
   - Phase 2 (rules/skills scanning) - recommended
   - Phase 3 (blocking threshold) - quality-of-life
   - Phase 4 (source policies) - nice-to-have

3. **Implementation:**
   - Create issues/tickets for each phase
   - Start with Phase 1 (smallest, highest impact)
   - Each phase is independently testable

4. **Testing strategy:**
   - Unit tests with malicious payloads
   - Integration tests with real attack samples
   - False positive tests with legitimate docs (Node.js, React, DevOps)

5. **Documentation:**
   - Update `docs/mcp-security-scanning.md`
   - Add configuration examples
   - Document threshold tuning

---

## Questions for Discussion

1. **Provenance detection:** Use `git submodule` command or parse `.gitmodules`?
   - `.gitmodules` is faster (no subprocess), but command is more reliable

2. **Default scan mode for skills:** `audit` or `enforce`?
   - Research recommends `enforce` for external skills (ToxicSkills precedent)
   - But `audit` reduces friction for teams getting started

3. **Allowlist implementation:** Per-server patterns or global config?
   - Proposal: YAML config file (`scan_allowlists.yaml`) in workspace root

4. **Scan result caching:** Should we cache identical content?
   - Reduces redundant scans for repeated tool calls
   - Adds complexity (cache invalidation, TTL)

5. **Task YAML scanning:** Should operator-controlled task files be scanned?
   - Current plan: No (trust code review)
   - Alternative: Audit scan if loaded from URL or external path

---

## Files in This Worktree

```
cicaddy-prompt-defense-wt/
├── SCANNING-REVIEW.md                              # This file (summary)
├── docs/
│   └── prompt-injection-balanced-defense-plan.md   # Detailed implementation plan
├── Research/
│   └── prompt-injection-defense-agent-systems.md   # Comprehensive research (by claude-researcher agent)
└── src/cicaddy/
    └── (current source code)
```

---

## Conclusion

Your concerns are valid and well-founded. The current implementation has three gaps:

1. ❌ **MCP-only scanning** (not general tool level)
2. ❌ **No scanning of initial rules/skills loading**
3. ⚠️ **No distinction between detection and blocking** (causes false positives)

This plan addresses all three with a balanced approach:

- ✅ Scan at trust boundaries (external content)
- ✅ Don't over-scan trusted sources (team code)
- ✅ Separate detection thresholds from blocking thresholds
- ✅ Provide clear configuration options

**Ready to proceed with Phase 1?** Let me know if you want to:
- Review the detailed plan first
- Adjust priorities
- Start implementation
- Discuss any of the open questions

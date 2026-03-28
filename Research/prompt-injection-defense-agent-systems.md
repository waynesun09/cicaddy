# Prompt Injection Defense in AI Agent Systems

*Date: 2026-03-28*

## Overview

Prompt injection remains the #1 vulnerability in the OWASP Top 10 for LLM Applications (2025), and it is intensifying as AI agents gain tool-calling, file system access, and network capabilities. This document synthesizes academic research, industry best practices, and real-world attack patterns to inform a balanced scanning strategy for cicaddy's agent framework.

The fundamental challenge: LLMs process instructions and data in the same channel with no hardware-enforced separation. No single defense eliminates prompt injection. The consensus across Anthropic, OpenAI, Google DeepMind, Meta, OWASP, and NIST is that **defense-in-depth** is the only viable strategy.

Key statistics (as of early 2026):
- Attack success rates reach **84% in agentic systems** without defenses
- Only **34.7% of organizations** have deployed dedicated prompt injection defenses
- Anthropic's layered defenses reduced Claude browser agent attack success from 23.6% to **~1%**
- Critical CVEs: Microsoft Copilot (CVSS 9.3), GitHub Copilot (CVSS 9.6), Cursor IDE (CVSS 9.8)
- A meta-analysis of 78 studies (2021-2026) found attack success **exceeds 85%** against state-of-the-art defenses when adaptive attacks are used

---

## 1. Defense-in-Depth Strategies

### 1.1 The Agents Rule of Two (Meta AI, Oct 2025)

The most actionable framework for agent security design. An agent must not satisfy more than two of three properties simultaneously:

```
    [A] Untrusted Input          [B] Sensitive Data          [C] External Actions
    (web, user content,          (credentials, private       (send email, execute
     external docs)               repos, .env files)          code, write files)

              A + B = OK if C is restricted (human approval)
              A + C = OK if B is restricted (sandboxed, no secrets)
              B + C = OK if A is restricted (trusted input only)
              A + B + C = VULNERABLE (the "lethal trifecta")
```

This maps directly to cicaddy's architecture:
- **CI/CD agents** process untrusted code [A], have repo access [B], and can post comments/create issues [C] -- all three properties
- **Mitigation**: scanning tool responses [A], least-privilege tokens [B], human-in-the-loop for destructive actions [C]

### 1.2 OWASP Defense Layers

The OWASP LLM Prompt Injection Prevention Cheat Sheet recommends a five-stage pipeline:

| Stage | Function | Latency Impact |
|-------|----------|----------------|
| 1. Input validation | Pattern matching, length limits, encoding detection | <1ms |
| 2. HITL routing | Flag high-risk requests for human review | Variable |
| 3. Sanitization | Clean and structure prompts with clear data/instruction separation | <1ms |
| 4. Generation + validation | LLM call with output validation | N/A |
| 5. Output filtering | Prevent leakage of system prompts, credentials | <1ms |

### 1.3 Anthropic's Approach

Anthropic combines three layers:
1. **Training-time robustness**: Reinforcement learning that rewards correct identification of injection in simulated content
2. **Content classification**: Real-time classifiers scanning untrusted content entering the context window
3. **Human red teaming**: Continuous probing by security researchers, plus external arena challenges

Result: Claude Opus 4.5 achieves ~1% attack success rate (down from 23.6% without defenses). Anthropic publicly acknowledges that "no browser agent is immune to prompt injection."

### 1.4 The PALADIN Framework

Academic research proposes five protective layers:
1. Input sanitization and normalization
2. Provenance tracking (where did this content come from?)
3. Anomaly detection in agent reasoning
4. Context isolation between trust domains
5. Output verification before action execution

Results: 94% detection accuracy, 70% reduction in trust leakage, 96% task accuracy retention.

---

## 2. Where to Apply Scanning

### 2.1 Trust Hierarchy

Not all inputs carry equal risk. The key design principle is: **scan at trust boundaries, not everywhere**.

```
TRUST LEVEL         SOURCE                          SCAN APPROACH
-----------         ------                          -------------
Highest trust       System prompt (hardcoded)        No scanning needed
                    |
High trust          Operator instructions            Validate at deployment
                    (task YAML, CI config)
                    |
Medium trust        Workspace rule files             Scan on load (audit mode)
                    (AGENT.md, CLAUDE.md)
                    |
Medium-low trust    Skill definitions from           Scan on load (enforce mode)
                    workspace (.agents/skills/)
                    |
Low trust           User inputs                      Scan before processing
                    |
Lowest trust        Tool responses                   Scan before context injection
                    (MCP servers, web fetches,        (enforce mode)
                    external APIs)
```

### 2.2 Current Implementation (cicaddy v0.6.0)

Cicaddy currently scans **tool responses** from MCP servers. This covers the lowest-trust tier. The scanner architecture is sound:

- **HeuristicScanner**: 30+ regex patterns, 6-pass normalization, <1ms latency
- **LLMGuardScanner**: DeBERTa v3 ML classifier, ~50-200ms latency
- **CompositeScanner**: Heuristic first, ML only for suspicious content (adaptive latency)
- Three modes: disabled, audit, enforce (per-server granularity)

### 2.3 Recommended Scanning Expansion

Based on the research, here is where scanning should and should not be applied:

#### SHOULD scan (high value, real attack vectors):

| Source | Why | Scanner | Mode | Rationale |
|--------|-----|---------|------|-----------|
| **MCP tool responses** | Lowest trust, active attack vector (ContextCrush) | Heuristic or composite | enforce | Already implemented. Primary defense point. |
| **Skill definitions** (.agents/skills/) | Supply chain attack vector (ToxicSkills, ClawHavoc) | Heuristic | enforce on external, audit on local | Snyk found 36% of ClawHub skills contain prompt injection. Skills execute with full agent permissions. |
| **Fetched web content** | Indirect injection via web pages | Heuristic | audit or enforce | Palo Alto Unit 42 documented web-based indirect injection in the wild. |
| **External rule files** (from dependencies, submodules) | Rule injection attack vector | Heuristic | audit | AGENTS.md in cloned repos or submodules can contain persistent injection. |

#### SHOULD NOT scan (low value, high false-positive risk):

| Source | Why Not | Alternative Control |
|--------|---------|-------------------|
| **System prompts** (hardcoded) | Written by developers, under version control | Code review, PR process |
| **Task YAML definitions** | Operator-controlled, checked into repo | Code review, PR process |
| **Local AGENT.md/CLAUDE.md** (project root) | Written by the team, version-controlled | Code review; flag if modified outside normal git flow |
| **User prompts** (interactive mode) | The user IS the principal; scanning their input creates friction | Permission system for destructive actions |
| **Internal API responses** (DevLake, GitLab) | Authenticated, controlled infrastructure | Network-level controls, API auth |

#### GRAY AREA (context-dependent):

| Source | When to Scan | When to Skip |
|--------|-------------|-------------|
| **AGENT.md in submodules** | When submodule is from untrusted/external source | When submodule is from your own org |
| **PR/MR descriptions** | When processing external contributor PRs | When processing internal team PRs |
| **Git commit messages** | When summarizing external repos | When processing internal repos |

### 2.4 Decision Framework

```
Is the content authored by the operator/developer?
  YES --> Skip scanning (trust code review process)
  NO  --> Is it from a controlled internal system?
            YES --> Skip or audit mode (network controls sufficient)
            NO  --> Is it from an external/public source?
                      YES --> Enforce mode scanning
                      NO  --> Audit mode (log but don't block)
```

---

## 3. False Positive Mitigation

False positives are the primary barrier to adoption. If scanning blocks legitimate content, teams disable it entirely -- which is worse than no scanning at all.

### 3.1 Known False Positive Triggers

| Pattern | Legitimate Context | Mitigation |
|---------|-------------------|------------|
| `process.env.` | Node.js documentation, tutorials | Context-aware: only flag in non-code contexts, or require co-occurrence with exfil patterns |
| `ignore previous` | Changelog entries, documentation updates | Require verb + instruction noun pattern, not just substring |
| `sudo` | Linux administration docs, system setup guides | Only flag when combined with other escalation patterns |
| `base64` strings | API examples, JWT tokens, image data | Only flag when decoded content contains injection keywords |
| `act as` | Role-playing documentation, test scenarios | Require "act as" + authority role (admin, root, system) |
| `rm -rf` | Build scripts, cleanup documentation | Only flag `rm -rf /` (root), not `rm -rf ./build/` |

### 3.2 Techniques

**3.2.1 Contextual Scoring (Already Implemented)**
Cicaddy's cumulative risk scoring is a good foundation. Single low-severity matches alone should not block. Enhancement: introduce a **blocking threshold** separate from detection threshold.

```
detection_threshold = 0.0   (any match is logged)
blocking_threshold  = 0.3   (only block when cumulative score exceeds this)
```

**3.2.2 Co-occurrence Requirements**
Require multiple pattern categories to co-occur before blocking. A single `process.env` reference in docs is likely legitimate; `process.env` + `send to webhook` is almost certainly malicious.

**3.2.3 Source-Aware Thresholds**
Different blocking thresholds for different sources:
- External documentation servers (Context7): threshold 0.2 (strict)
- Internal APIs: threshold 0.5 (lenient) or disabled
- Skill definitions: threshold 0.3

**3.2.4 Allowlisting**
Maintain per-server allowlists for known false-positive patterns. For example, a Node.js documentation server will legitimately contain `process.env` patterns.

```yaml
mcp_scan_config:
  allowlists:
    context7:
      - "process\\.env"    # Common in Node.js docs
    internal-docs:
      - "sudo"             # Common in admin guides
```

**3.2.5 InjecGuard MOF Strategy**
Academic research (InjecGuard) proposes "Mitigating Over-defense for Free" (MOF), a training strategy that reduces bias on trigger words. This reduced false positives by 30.8% on the NotInject benchmark while maintaining detection accuracy.

**3.2.6 Consensus Mode (Already Implemented)**
The CompositeScanner's consensus mode (both heuristic AND ML must agree) significantly reduces false positives at the cost of slightly higher latency for suspicious content. This is the recommended approach for production deployments where false positives are costly.

---

## 4. Performance Considerations

### 4.1 Latency Budget

In a CI/CD pipeline context, agent tasks run for minutes to hours. Scanner latency is negligible compared to LLM inference time. The real performance concern is:
- **Model loading time** for ML-based scanners (one-time, ~2-5s)
- **Memory footprint** of transformer models (~500MB for DeBERTa)

### 4.2 When to Use Each Approach

| Approach | Latency | Memory | Best For |
|----------|---------|--------|----------|
| **Heuristic only** | <1ms per scan | ~0 | CI/CD pipelines, resource-constrained environments, high-throughput scanning |
| **ML classifier** (DeBERTa) | 50-200ms per scan | ~500MB | High-security environments, when false positives are costly |
| **Composite** (heuristic-first) | <1ms clean, 50-200ms suspicious | ~500MB | Production default -- best balance of speed and accuracy |
| **LLM-based deep scan** | 300-500ms+ | N/A (API call) | Highest-stakes scenarios, human-in-the-loop review |

### 4.3 Scanning Frequency Optimization

Not every tool call needs scanning. Optimization strategies:

1. **Cache scan results** for identical content (with TTL)
2. **Skip scanning** for tools that return structured data (SQL results, metrics) vs. free-text (documentation, web content)
3. **Batch scanning** when processing multiple tool responses in sequence
4. **Progressive analysis**: scan only the first N characters for initial triage, full scan only if suspicious

### 4.4 PromptGuard Benchmark (Nature, Jan 2026)

The PromptGuard hybrid framework (regex + MiniBERT) achieved:
- 67% reduction in injection success rate
- F1-score of 0.91 in detection
- **Latency increase below 8%** over unscanned baseline
- Regex handled 62% of detections; MiniBERT caught the remaining semantic cases
- False negatives reduced by 22% compared to regex alone

---

## 5. Specific Attack Vectors in Agent Systems

### 5.1 Malicious Rule Files in Workspace

**Attack**: Attacker places poisoned AGENT.md/CLAUDE.md/.cursorrules in a repository, submodule, or PR. The agent loads it automatically on every interaction.

**Real-world examples**:
- IDEsaster vulnerabilities (30+ CVEs across every major AI IDE, disclosed 2025)
- Datadog incident: attacker opened GitHub issues with injection payloads targeting `claude-code-action`
- Lasso Security: documented indirect prompt injection in Claude Code via CLAUDE.md files

**Why it is dangerous**: Rule files are loaded automatically, with high trust, on every session. Unlike a web page the agent might visit once, a malicious rule file persists.

**Defenses**:
1. **Audit mode scanning** of rule files from external sources (submodules, dependencies)
2. **Git provenance checking**: flag rule files modified outside normal PR workflow
3. **Diff-based scanning**: scan only the diff of rule files on PR, not the full file
4. **Human review**: require approval for changes to agent configuration files

**Cicaddy relevance**: cicaddy v0.6.0 auto-loads AGENT.md/CLAUDE.md/GEMINI.md and .agents/skills/ files. The auto-loading follows agentskills.io standard, which is good for usability but creates an attack surface. Recommended: scan external/submodule rule files in audit mode.

### 5.2 Compromised Skill Definitions

**Attack**: Malicious skills distributed via registries (ClawHub) or included in project dependencies.

**Real-world examples**:
- **ToxicSkills (Snyk, Feb 2026)**: 36% of ClawHub skills contained prompt injection. 76 confirmed malicious payloads for credential theft, backdoor installation, and data exfiltration.
- **ClawHavoc campaign**: 341 malicious skills (12% of registry) delivered Atomic Stealer (AMOS) macOS infostealer. Campaign window: Jan 27-29, 2026.
- Barrier to publishing: one SKILL.md file + 1-week-old GitHub account. No code signing, no review.

**Why it is dangerous**: Skills execute with the full permissions of the AI agent. Unlike packages in npm/PyPI, there is no sandboxing, no permission model, and no supply chain verification.

**Defenses**:
1. **Enforce mode scanning** on all externally-sourced skill definitions
2. **Provenance verification**: only load skills from trusted sources (pinned commits, signed manifests)
3. **Permission boundaries**: skills should declare required capabilities; agent should enforce least-privilege
4. **Registry scanning**: tools like Snyk's mcp-scan for automated skill auditing

**Cicaddy relevance**: cicaddy loads skills from `.agents/skills/` in the workspace. If the workspace is a cloned external repo or contains submodules with skills, those skills are untrusted. Recommended: scan skill files, especially from external sources, in enforce mode.

### 5.3 Tool Response Injection (Current Coverage)

**Attack**: MCP server or external API returns content containing injection payloads mixed with legitimate data.

**Real-world examples**:
- **ContextCrush (Feb 2026)**: Context7's "Custom Rules" feature served attacker-controlled instructions verbatim to all users querying a library. Researchers demonstrated credential exfiltration in minutes.
- **PromptPwnd**: Prompt injection via CI/CD pipelines when AI agents process untrusted GitHub Actions/GitLab CI inputs.

**Why it is dangerous**: Tool responses enter the LLM context window with implicit trust. The LLM cannot distinguish between legitimate documentation and injected instructions.

**Defenses** (already implemented in cicaddy):
1. Heuristic scanning with 30+ patterns and 6-pass normalization
2. ML-based classification (DeBERTa) for semantic detection
3. Composite scanning with early-exit optimization
4. Per-server scan mode configuration (disabled/audit/enforce)

### 5.4 Task File Manipulation

**Attack**: Attacker modifies task YAML files to alter agent behavior -- changing prompts, adding malicious tool configurations, or modifying scan settings.

**Defense approach**: Task files are operator-controlled configuration, not untrusted input. The primary defense is the software development lifecycle:
1. Version control (git)
2. Code review (PR process)
3. Branch protection rules
4. CI/CD pipeline validation

**Scanning task files would be over-scanning** -- it treats your own configuration as untrusted, which undermines the trust model. Instead, protect the integrity of the configuration through access controls.

### 5.5 Emerging Vectors

**Multi-modal injection**: Hiding instructions in images that accompany benign text. As agents gain vision capabilities, this becomes relevant.

**RAG poisoning**: Injecting malicious documents into knowledge bases. Research shows 5 carefully crafted documents can manipulate AI responses 90% of the time.

**Tool description rug-pulls**: MCP server changes tool descriptions mid-session to inject new instructions. Pipelock addresses this by fingerprinting tool descriptions on first contact.

**DNS rebinding for SSRF**: Hostname resolves to public IP on first lookup, private IP on second. Relevant when agents can fetch URLs.

---

## 6. Balanced Defense Framework for Cicaddy

### 6.1 Recommended Architecture

```
                    TRUST BOUNDARY MAP
                    ==================

 +------------------------------------------------------------------+
 |  OPERATOR ZONE (highest trust - no scanning)                     |
 |                                                                  |
 |  System prompts, task YAML, CI/CD config, local AGENT.md         |
 |  Defense: code review, version control, branch protection        |
 +------------------------------------------------------------------+
                              |
                    [auto-loaded at startup]
                              |
 +------------------------------------------------------------------+
 |  WORKSPACE ZONE (medium trust - audit scanning)                  |
 |                                                                  |
 |  Submodule AGENT.md, external rule files, local skills           |
 |  Defense: heuristic scan in audit mode, git provenance check     |
 +------------------------------------------------------------------+
                              |
                    [loaded during execution]
                              |
 +------------------------------------------------------------------+
 |  EXTERNAL ZONE (low trust - enforce scanning)                    |
 |                                                                  |
 |  MCP tool responses, fetched web content, external skills,       |
 |  documentation servers, user-generated content in PRs            |
 |  Defense: composite scanning in enforce mode, egress filtering   |
 +------------------------------------------------------------------+
```

### 6.2 Implementation Priority

| Priority | What | Effort | Impact |
|----------|------|--------|--------|
| **P0 (done)** | MCP tool response scanning | Implemented | Blocks ContextCrush, tool injection |
| **P1** | External skill scanning | Medium | Blocks ToxicSkills, ClawHavoc-class attacks |
| **P2** | Submodule/dependency rule file scanning | Low | Blocks rule injection via supply chain |
| **P3** | Fetched web content scanning | Low | Blocks indirect injection from web |
| **P4** | Tool description fingerprinting | Medium | Blocks rug-pull attacks |
| **P5** | Egress filtering / SSRF protection | High | Blocks data exfiltration |

### 6.3 What NOT to Scan

Explicitly listing what should NOT be scanned is as important as what should be:

1. **System prompts**: These are your own code. Scanning them would be like running your own source through an antivirus.
2. **Task YAML files**: Operator configuration under version control.
3. **Local AGENT.md/CLAUDE.md**: Written by your team, reviewed via PRs.
4. **User prompts in interactive mode**: The user is the principal. Restricting their input breaks the trust model.
5. **Internal API responses**: DevLake, internal GitLab -- these are behind auth and network controls.
6. **Structured data responses**: SQL query results, JSON metrics -- these contain data, not instructions.

### 6.4 Configuration Defaults

Recommended defaults for cicaddy:

```yaml
mcp_scan_config:
  enabled: true
  scanner: "heuristic"           # Default scanner (no ML dependency)
  default_mode: "audit"          # Log warnings, don't block by default
  blocking_threshold: 0.3        # Only block when risk exceeds this

  # Per-source overrides
  source_policies:
    external_docs:               # Context7, rtfmbro, etc.
      mode: "enforce"
      scanner: "composite"       # ML confirmation for external docs
      blocking_threshold: 0.2

    external_skills:             # Skills from registries or external repos
      mode: "enforce"
      scanner: "heuristic"
      blocking_threshold: 0.2

    submodule_rules:             # AGENT.md from submodules
      mode: "audit"
      scanner: "heuristic"
      blocking_threshold: 0.3

    internal_apis:               # DevLake, internal GitLab
      mode: "disabled"

    user_input:                  # Interactive user prompts
      mode: "disabled"
```

---

## 7. Key Research Papers and Frameworks

### Academic Research

| Paper/Framework | Key Finding | Relevance |
|----------------|-------------|-----------|
| **Agents Rule of Two** (Meta AI, Oct 2025) | Agent must not have untrusted input + sensitive data + external actions simultaneously | Core design principle for agent security |
| **The Attacker Moves Second** (Anthropic+OpenAI+DeepMind, Oct 2025) | 12 published defenses largely bypassed by adaptive attacks; static tests are useless for evaluation | Don't rely on any single defense; continuous red-teaming required |
| **PALADIN** (2025) | Five-layer defense-in-depth framework; 94% detection, 96% task accuracy retention | Validates multi-layer approach |
| **PromptGuard** (Nature, Jan 2026) | Regex + MiniBERT hybrid: 91% F1, <8% latency increase | Validates cicaddy's composite scanner approach |
| **InjecGuard MOF** (2025) | "Mitigating Over-defense for Free" training reduces false positives by 30.8% | Approach for reducing false positives in ML classifiers |
| **PromptScreen** (Dec 2025) | TF-IDF + Linear SVM: 93.4% accuracy, 96.5% specificity, negligible overhead | Lightweight classifier option between heuristic and transformer |
| **SmoothLLM** (2024) | Random perturbation + aggregation reduces attack success to <1% | Defense technique for high-stakes scenarios |
| **Comprehensive Review** (MDPI, Jan 2025) | Meta-analysis of 78 studies; adaptive attack success >85% against all defenses | No defense is perfect; defense-in-depth is mandatory |

### Industry Tools and Frameworks

| Tool | Approach | Status |
|------|----------|--------|
| **Pipelock** | Agent firewall: 11-layer scanner pipeline, capability separation, bidirectional MCP scanning | Open source, actively maintained |
| **Lakera Guard** | Cloud-based prompt injection detection API | Commercial |
| **Snyk mcp-scan** | Scanner for MCP servers and SKILL.md files | Open source |
| **promptfoo** | Red teaming and prompt injection testing framework | Open source |
| **LLM Guard** (ProtectAI) | DeBERTa-based classifier for prompt injection detection | Open source, used in cicaddy |
| **Microsoft Prompt Shields** | Integrated with Defender for Cloud; "spotlighting" for input isolation | Commercial (Azure) |
| **Microsoft FIDES** | Information flow control for deterministic prevention of indirect injection | Research prototype |

---

## 8. Actionable Recommendations

### Immediate (Next Release)

1. **Add blocking threshold** to HeuristicScanner: separate detection (log) from blocking (enforce) thresholds
2. **Skill scanning**: extend scanner to cover `.agents/skills/` file loading
3. **Allowlisting**: add per-server pattern allowlists to reduce false positives on known patterns

### Short-Term (Next Quarter)

4. **Submodule rule file scanning**: audit-mode scan of AGENT.md files from submodules and external dependencies
5. **Co-occurrence scoring**: require multiple pattern categories to trigger blocking (reduces false positives)
6. **Source-aware thresholds**: different blocking thresholds based on content provenance

### Medium-Term

7. **Tool description fingerprinting**: detect mid-session changes to MCP tool descriptions
8. **Egress filtering**: restrict agent network access to approved domains
9. **Scan result caching**: cache results for identical content to reduce redundant scanning
10. **Red team testing**: integrate promptfoo or similar for automated adversarial testing of scanning pipeline

### Design Principles (Ongoing)

- **Scan at trust boundaries, not everywhere**: over-scanning trusted sources erodes usability and teaches teams to disable scanning
- **Audit before enforce**: always run in audit mode first to measure false positive rates before switching to enforce
- **Fail open for low-risk, fail closed for high-risk**: audit mode for workspace content, enforce mode for external content
- **Make scanning invisible for clean content**: <1ms latency for legitimate content via heuristic-first composite scanning
- **Never scan the user's own words**: the user is the principal, not the threat

---

## Sources

- [OWASP LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
- [OWASP LLM Prompt Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html)
- [Anthropic: Prompt Injection Defenses](https://www.anthropic.com/research/prompt-injection-defenses)
- [Meta AI: Agents Rule of Two](https://www.mbgsec.com/weblog/2025-11-01-agents-rule-of-two-a-practical-approach-to-ai-agent-security/)
- [Simon Willison: New Prompt Injection Papers](https://simonwillison.net/2025/Nov/2/new-prompt-injection-papers/)
- [ContextCrush Vulnerability (Noma Security)](https://noma.security/blog/contextcrush-context7-the-mcp-server-vulnerability/)
- [Snyk ToxicSkills Study](https://snyk.io/blog/toxicskills-malicious-ai-agent-skills-clawhub/)
- [Snyk: From SKILL.md to Shell Access](https://snyk.io/articles/skill-md-shell-access/)
- [AGENTS.md Is an Attack Surface](https://blog.themenonlab.com/blog/agents-md-security-risk-ai-agents/)
- [Lasso Security: Indirect Prompt Injection in Claude Code](https://www.lasso.security/blog/the-hidden-backdoor-in-claude-coding-assistant)
- [Pipelock Agent Firewall](https://github.com/luckyPipewrench/pipelock)
- [PromptPwnd: GitHub Actions AI Agent Attacks](https://www.aikido.dev/blog/promptpwnd-github-actions-ai-agents)
- [PromptGuard Framework (Nature, Jan 2026)](https://www.nature.com/articles/s41598-025-31086-y)
- [PromptScreen: Efficient Jailbreak Mitigation](https://arxiv.org/html/2512.19011)
- [DMPI-PMHFE: Pre-trained Model + Heuristic Detection](https://link.springer.com/chapter/10.1007/978-981-95-3072-4_6)
- [Comprehensive Review: Prompt Injection in LLMs and Agent Systems (MDPI)](https://www.mdpi.com/2078-2489/17/1/54)
- [Microsoft: Protecting Against Indirect Injection in MCP](https://developer.microsoft.com/blog/protecting-against-indirect-injection-attacks-mcp)
- [Palo Alto Unit 42: Web-Based Indirect Prompt Injection](https://unit42.paloaltonetworks.com/ai-agent-prompt-injection/)
- [Datadog: Stopping HackerBot-Claw](https://www.datadoghq.com/blog/engineering/stopping-hackerbot-claw-with-bewaire/)
- [Penligent: AI Agents Hacking in 2026](https://www.penligent.ai/hackinglabs/ai-agents-hacking-in-2026-defending-the-new-execution-boundary/)
- [Prompt Injection Defenses Catalog (tldrsec)](https://github.com/tldrsec/prompt-injection-defenses)
- [cicaddy MCP Security Scanning Documentation](../docs/mcp-security-scanning.md)

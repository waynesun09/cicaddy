# Security Scanning (v0.6.1+)

Cicaddy includes prompt injection protection for all external content.

## Scan configuration

```bash
# MCP tool responses (per-server in MCP_SERVERS_CONFIG)
# - scan_mode: disabled|audit|enforce

# Local file tools
LOCAL_TOOLS_SCAN_MODE=audit
LOCAL_TOOLS_BLOCKING_THRESHOLD=0.3

# Agent rules from submodules
RULES_SCAN_MODE=audit
RULES_BLOCKING_THRESHOLD=0.3

# Skills from dependencies
SKILLS_SCAN_MODE=enforce
SKILLS_BLOCKING_THRESHOLD=0.2
```

## How it works

- **Local/git-tracked content**: Trusted (skips scanning)
- **External content** (submodules, untracked, global skills): Scanned
- **Audit mode**: Logs warnings, doesn't block
- **Enforce mode**: Blocks content above threshold
- **Threshold separation**: Detection (0.0) vs blocking (0.2-0.3)

See `docs/mcp-security-scanning.md` for details.

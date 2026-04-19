# DSPy Task YAML

A task file defines the prompt, tools, constraints, and output format
declaratively. The MCP server name in `tools.servers` must match the `name`
field in `MCP_SERVERS_CONFIG`. The `local` server is auto-provided when
`ENABLE_LOCAL_TOOLS=true`.

```yaml
name: my_analysis
type: data_analysis
version: "1.0"

persona: >
  expert analyst for {{PROJECT_NAME}}

inputs:
  - name: project_name
    env_var: PROJECT_NAME
    required: true

tools:
  servers:
    - devlake-mcp-instance   # must match MCP_SERVERS_CONFIG name
    - local                  # built-in: requires ENABLE_LOCAL_TOOLS=true
  required_tools:
    - connect_database
    - execute_query
    - read_file

constraints:
  - "NEVER use WITH clauses (CTEs)"
  - "Always use fully qualified table names: lake.table_name"

reasoning: react
output_format: html   # html | markdown | json

context: |
  Analyze {{PROJECT_NAME}} for the last {{ANALYSIS_DAYS}} days.
  ...
```

## Task file inputs

Inputs declared in the task YAML are resolved from environment variables:

```yaml
inputs:
  - name: project_name
    env_var: PROJECT_NAME   # set PROJECT_NAME=<value> in .env
    required: true
  - name: analysis_days
    env_var: ANALYSIS_DAYS
    default: "30"
```

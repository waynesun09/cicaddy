# Agent Factory Extension

The `AgentFactory` uses a registry pattern. Custom agents and type detectors
can be registered from any package -- no need to modify cicaddy's core.

## Register a custom agent

```python
from cicaddy.agent.base import BaseAIAgent
from cicaddy.agent.factory import AgentFactory

class MyCustomAgent(BaseAIAgent):
    async def _build_prompt(self, context):
        return "Analyze the pipeline..."

    async def _process_result(self, result, analysis_result):
        # post-process the AI result
        return result

# Register under a type name
AgentFactory.register("my_custom", MyCustomAgent)
```

Activate with `AGENT_TYPE=my_custom` in the env file.

## Register a type detector

Detectors auto-select agent type from environment/settings.
Lower priority number = checked first. First non-None result wins.

```python
from cicaddy.agent.factory import AgentFactory
from cicaddy.config.settings import Settings
from typing import Optional

def detect_my_platform(settings: Settings) -> Optional[str]:
    import os
    if os.getenv("MY_PLATFORM_EVENT") == "pipeline":
        return "my_custom"
    return None

# Register with priority 10 (runs before built-in CI detector at 50)
AgentFactory.register_detector(detect_my_platform, priority=10)
```

## Entry point plugin registration (for installable packages)

Cicaddy discovers plugins automatically via `importlib.metadata.entry_points()`.
Register callables in your package's `pyproject.toml`:

```toml
[project.entry-points."cicaddy.agents"]
my_platform = "my_plugin.plugin:register_agents"

[project.entry-points."cicaddy.cli_args"]
my_platform = "my_plugin.plugin:get_cli_args"

[project.entry-points."cicaddy.env_vars"]
my_platform = "my_plugin.plugin:get_env_vars"

[project.entry-points."cicaddy.config_sections"]
my_platform = "my_plugin.plugin:config_section"

[project.entry-points."cicaddy.validators"]
my_platform = "my_plugin.plugin:validate"

[project.entry-points."cicaddy.settings_loader"]
my_platform = "my_plugin.config:load_settings"
```

Plugin callable signatures:

| Group | Signature |
|-------|-----------|
| `cicaddy.agents` | `() -> None` -- calls `AgentFactory.register()` / `register_detector()` |
| `cicaddy.cli_args` | `() -> List[ArgMapping]` |
| `cicaddy.env_vars` | `() -> List[str]` |
| `cicaddy.config_sections` | `(config: Dict, mask_fn: Callable, sensitive_vars: frozenset) -> None` |
| `cicaddy.validators` | `(config: Dict) -> Tuple[List[str], List[str]]` (errors, warnings) |
| `cicaddy.settings_loader` | `() -> CoreSettings` |

Example agent registration callable:

```python
# my_plugin/plugin.py
def register_agents():
    from cicaddy.agent.factory import AgentFactory
    from my_plugin.agents import MyCustomAgent, detect_my_platform

    AgentFactory.register("my_custom", MyCustomAgent)
    AgentFactory.register_detector(detect_my_platform, priority=10)
```

After `pip install cicaddy my-plugin`, cicaddy discovers and loads the plugin
automatically -- no manual imports needed.

## Built-in agent types

| Type | Class | Activated by |
|------|-------|--------------|
| `task` | `TaskAgent` | `AGENT_TYPE=task` or `TASK_TYPE` env var |
| `branch_review` | `BranchReviewAgent` | `AGENT_TYPE=branch` or branch push CI |
| `merge_request` | *(platform plugin)* | `CI_MERGE_REQUEST_IID` or `AGENT_TYPE=mr` |

## Available AgentFactory methods

```python
AgentFactory.register(agent_type, agent_class)         # register agent class
AgentFactory.register_detector(detector_fn, priority)  # register type detector
AgentFactory.create_agent(settings)                    # create agent instance
AgentFactory.get_available_agent_types()               # list registered types
AgentFactory.validate_agent_requirements(type, settings)
```

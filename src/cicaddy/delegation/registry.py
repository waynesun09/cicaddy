"""Sub-agent specification model and registry.

Provides SubAgentSpec dataclass and SubAgentRegistry for loading built-in
and user-defined sub-agent definitions, organized by agent type.

Built-in agents are shipped as YAML files in the ``builtin/`` directory
alongside this module, included in the release package via
``package-data`` in ``pyproject.toml``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)

# Directory containing built-in agent YAML files, shipped with the package.
_BUILTIN_DIR = Path(__file__).parent / "builtin"


@dataclass
class SubAgentSpec:
    """Specification for a delegation sub-agent."""

    name: str
    persona: str
    description: str
    categories: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    output_sections: List[str] = field(default_factory=list)
    priority: int = 0
    allowed_tools: Optional[List[str]] = None
    blocked_tools: List[str] = field(default_factory=list)
    agent_type: str = "*"  # review|task|cron|* (which parent types can use this)
    source_file: Optional[str] = None


class SubAgentRegistry:
    """Registry for loading and managing sub-agent specifications.

    Loads built-in agents for the requested agent type, then merges
    user-defined agents from YAML files and JSON config overrides.
    """

    def load_registry(
        self,
        agent_type: str,
        user_config: str = "",
        agents_dir: str = ".agents/delegation",
    ) -> Dict[str, SubAgentSpec]:
        """Load sub-agent specs for a given agent type.

        Merge order (later overrides earlier):
        1. Built-in agents for the agent_type
        2. User YAML files from agents_dir/{agent_type}/ and agents_dir/ (type=*)
        3. DELEGATION_AGENTS JSON env var overrides

        Args:
            agent_type: Parent agent type (review, task, cron).
            user_config: JSON string with custom agent definitions.
            agents_dir: Base directory for user-defined YAML files.

        Returns:
            Dict mapping agent name to SubAgentSpec.
        """
        registry: Dict[str, SubAgentSpec] = {}

        # 1. Load built-in agents from bundled YAML files
        built_in = self._load_builtin_agents(agent_type)
        registry.update(built_in)
        logger.info(f"Loaded {len(built_in)} built-in agents for type '{agent_type}'")

        # 2. Load user YAML files
        user_agents = self._load_yaml_agents(agent_type, agents_dir)
        for name, spec in user_agents.items():
            registry[name] = spec
        if user_agents:
            logger.info(
                f"Loaded {len(user_agents)} user-defined agents from {agents_dir}"
            )

        # 3. Parse JSON config overrides
        if user_config and user_config.strip():
            json_agents = self._parse_json_config(user_config, agent_type)
            for name, spec in json_agents.items():
                registry[name] = spec
            if json_agents:
                logger.info(
                    f"Applied {len(json_agents)} agent overrides from JSON config"
                )

        logger.info(
            f"Registry loaded: {len(registry)} total agents for type '{agent_type}'"
        )
        return registry

    def _load_builtin_agents(self, agent_type: str) -> Dict[str, SubAgentSpec]:
        """Load built-in agents from bundled YAML files for the given type."""
        type_dir = _BUILTIN_DIR / agent_type
        if not type_dir.is_dir():
            return {}
        return self._scan_yaml_dir(type_dir)

    def _load_yaml_agents(
        self, agent_type: str, agents_dir: str
    ) -> Dict[str, SubAgentSpec]:
        """Load user-defined agents from YAML files."""
        agents: Dict[str, SubAgentSpec] = {}
        base_path = Path(agents_dir)

        # Scan type-specific directory: .agents/delegation/{agent_type}/*.yaml
        type_dir = base_path / agent_type
        if type_dir.is_dir():
            for name, spec in self._scan_yaml_dir(type_dir).items():
                if spec.agent_type in ("*", agent_type):
                    agents[name] = spec
                else:
                    logger.warning(
                        f"Skipping agent '{name}' from {type_dir}: "
                        f"agent_type '{spec.agent_type}' does not match '{agent_type}'"
                    )

        # Scan root directory for wildcard agents: .agents/delegation/*.yaml
        if base_path.is_dir():
            for yaml_file in sorted(base_path.glob("*.yaml")):
                spec = self._parse_yaml_file(yaml_file)
                if spec and spec.agent_type in ("*", agent_type):
                    agents[spec.name] = spec

        return agents

    @staticmethod
    def _as_list(value: Any, field_name: str, source: str = "") -> List[str]:
        """Coerce a value to a list of strings, warning on type mismatch."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            logger.warning(
                f"Field '{field_name}' should be a list, got string"
                f"{f' in {source}' if source else ''}: wrapping in list"
            )
            return [value]
        logger.warning(
            f"Field '{field_name}' has unexpected type {type(value).__name__}"
            f"{f' in {source}' if source else ''}: wrapping in list"
        )
        return [str(value)]

    def _scan_yaml_dir(self, directory: Path) -> Dict[str, SubAgentSpec]:
        """Scan a directory for YAML agent spec files."""
        agents: Dict[str, SubAgentSpec] = {}
        for yaml_file in sorted(directory.glob("*.yaml")):
            spec = self._parse_yaml_file(yaml_file)
            if spec:
                agents[spec.name] = spec
        return agents

    def _parse_yaml_file(self, yaml_file: Path) -> Optional[SubAgentSpec]:
        """Parse a single YAML file into a SubAgentSpec."""
        try:
            content = yaml_file.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            if not isinstance(data, dict) or "name" not in data:
                logger.warning(f"Invalid agent YAML (missing 'name'): {yaml_file}")
                return None

            src = str(yaml_file)
            allowed = data.get("allowed_tools")
            return SubAgentSpec(
                name=data["name"],
                persona=data.get("persona", ""),
                description=data.get("description", ""),
                categories=self._as_list(data.get("categories", []), "categories", src),
                constraints=self._as_list(
                    data.get("constraints", []), "constraints", src
                ),
                output_sections=self._as_list(
                    data.get("output_sections", []), "output_sections", src
                ),
                priority=data.get("priority", 50),
                allowed_tools=(
                    self._as_list(allowed, "allowed_tools", src)
                    if allowed is not None
                    else None
                ),
                blocked_tools=self._as_list(
                    data.get("blocked_tools", []), "blocked_tools", src
                ),
                agent_type=data.get("agent_type", "*"),
                source_file=src,
            )
        except (yaml.YAMLError, OSError) as e:
            logger.warning(f"Failed to parse agent YAML {yaml_file}: {e}")
            return None

    def _parse_json_config(
        self, config_json: str, agent_type: str
    ) -> Dict[str, SubAgentSpec]:
        """Parse JSON config string into agent specs."""
        agents: Dict[str, SubAgentSpec] = {}
        try:
            data = json.loads(config_json)
            if not isinstance(data, list):
                data = [data]

            for entry in data:
                if not isinstance(entry, dict) or "name" not in entry:
                    continue
                entry_type = entry.get("agent_type", "*")
                if entry_type not in ("*", agent_type):
                    continue

                src = f"JSON config ({entry['name']})"
                allowed = entry.get("allowed_tools")
                spec = SubAgentSpec(
                    name=entry["name"],
                    persona=entry.get("persona", ""),
                    description=entry.get("description", ""),
                    categories=self._as_list(
                        entry.get("categories", []), "categories", src
                    ),
                    constraints=self._as_list(
                        entry.get("constraints", []), "constraints", src
                    ),
                    output_sections=self._as_list(
                        entry.get("output_sections", []), "output_sections", src
                    ),
                    priority=entry.get("priority", 50),
                    allowed_tools=(
                        self._as_list(allowed, "allowed_tools", src)
                        if allowed is not None
                        else None
                    ),
                    blocked_tools=self._as_list(
                        entry.get("blocked_tools", []), "blocked_tools", src
                    ),
                    agent_type=entry_type,
                )
                agents[spec.name] = spec

        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse DELEGATION_AGENTS JSON: {e}")

        return agents

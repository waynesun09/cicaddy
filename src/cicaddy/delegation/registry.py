"""Sub-agent specification model and registry.

Provides SubAgentSpec dataclass and SubAgentRegistry for loading built-in
and user-defined sub-agent definitions, organized by agent type.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


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


# Built-in review sub-agents
_BUILT_IN_REVIEW_AGENTS: Dict[str, SubAgentSpec] = {
    "security-reviewer": SubAgentSpec(
        name="security-reviewer",
        persona="senior application security engineer with OWASP Top 10 expertise",
        description="Reviews security-sensitive changes including auth, crypto, secrets, and access control",
        categories=["security", "api"],
        constraints=[
            "Focus exclusively on security vulnerabilities and risks",
            "Rate findings: Critical/High/Medium/Low",
            "Include specific remediation steps for each finding",
        ],
        output_sections=[
            "Security Vulnerabilities",
            "Risk Assessment",
            "Remediation Steps",
        ],
        priority=10,
        agent_type="review",
    ),
    "architecture-reviewer": SubAgentSpec(
        name="architecture-reviewer",
        persona="senior software architect focused on design patterns and system boundaries",
        description="Reviews architectural changes including design patterns, abstractions, module boundaries, and interfaces",
        categories=["architecture"],
        constraints=[
            "Focus on structural design decisions and their implications",
            "Evaluate separation of concerns and coupling",
            "Consider backward compatibility and migration paths",
        ],
        output_sections=[
            "Design Analysis",
            "Architectural Concerns",
            "Recommendations",
        ],
        priority=20,
        agent_type="review",
    ),
    "api-reviewer": SubAgentSpec(
        name="api-reviewer",
        persona="API design specialist with expertise in REST, GraphQL, and backward compatibility",
        description="Reviews API changes including endpoints, schemas, versioning, and backward compatibility",
        categories=["api"],
        constraints=[
            "Focus on API contract changes and their impact",
            "Check for breaking changes in request/response schemas",
            "Evaluate versioning and deprecation strategy",
        ],
        output_sections=["API Changes", "Breaking Changes", "Compatibility Assessment"],
        priority=30,
        agent_type="review",
    ),
    "database-reviewer": SubAgentSpec(
        name="database-reviewer",
        persona="database engineer specializing in query optimization, migrations, and data integrity",
        description="Reviews database changes including queries, migrations, ORM models, schema changes, and indexes",
        categories=["database"],
        constraints=[
            "Focus on data integrity, migration safety, and query performance",
            "Check for missing indexes on new queries",
            "Evaluate migration rollback strategy",
        ],
        output_sections=["Schema Changes", "Query Analysis", "Migration Safety"],
        priority=25,
        agent_type="review",
    ),
    "ui-reviewer": SubAgentSpec(
        name="ui-reviewer",
        persona="frontend engineer with expertise in UI/UX, accessibility, and component design",
        description="Reviews frontend changes including components, templates, CSS/styling, accessibility, and UX",
        categories=["ui"],
        constraints=[
            "Focus on user experience and accessibility (WCAG)",
            "Check for responsive design and cross-browser compatibility",
            "Evaluate component reusability and state management",
        ],
        output_sections=["UI/UX Analysis", "Accessibility", "Component Review"],
        priority=35,
        agent_type="review",
    ),
    "devops-reviewer": SubAgentSpec(
        name="devops-reviewer",
        persona="DevOps/SRE engineer with CI/CD pipeline and infrastructure expertise",
        description="Reviews CI/CD, Docker, deployment, and infrastructure changes",
        categories=["cicd", "configuration"],
        constraints=[
            "Focus on pipeline reliability and deployment safety",
            "Check for secrets exposure in CI configuration",
            "Evaluate infrastructure cost and scaling implications",
        ],
        output_sections=[
            "Pipeline Analysis",
            "Deployment Safety",
            "Infrastructure Impact",
        ],
        priority=40,
        agent_type="review",
    ),
    "performance-reviewer": SubAgentSpec(
        name="performance-reviewer",
        persona="performance engineer specializing in algorithms, caching, concurrency, and resource usage",
        description="Reviews performance-sensitive changes including algorithms, caching, concurrency, and resource usage",
        categories=["performance"],
        constraints=[
            "Focus on algorithmic complexity and resource efficiency",
            "Identify potential bottlenecks and memory leaks",
            "Suggest concrete performance improvements with expected impact",
        ],
        output_sections=[
            "Performance Analysis",
            "Bottlenecks",
            "Optimization Opportunities",
        ],
        priority=45,
        agent_type="review",
    ),
    "general-reviewer": SubAgentSpec(
        name="general-reviewer",
        persona="senior software engineer with broad expertise in code quality and best practices",
        description="General code review covering anything not handled by specialized reviewers",
        categories=["architecture", "tests", "documentation", "error_handling"],
        constraints=[
            "Provide balanced, actionable feedback",
            "Focus on code quality, readability, and maintainability",
            "Avoid duplicating findings from specialized reviewers",
        ],
        output_sections=["Code Quality", "Test Coverage", "General Feedback"],
        priority=100,
        agent_type="review",
    ),
}

# Built-in task sub-agents
_BUILT_IN_TASK_AGENTS: Dict[str, SubAgentSpec] = {
    "data-analyst": SubAgentSpec(
        name="data-analyst",
        persona="data analyst specializing in data processing, statistics, and pattern recognition",
        description="Analyzes data patterns, trends, and anomalies",
        categories=["data_processing", "performance"],
        constraints=[
            "Present findings with supporting data",
            "Quantify observations where possible",
        ],
        output_sections=["Data Analysis", "Key Findings", "Recommendations"],
        priority=10,
        agent_type="task",
    ),
    "report-writer": SubAgentSpec(
        name="report-writer",
        persona="technical writer focused on clear, structured reports and documentation",
        description="Generates formatted reports, summaries, and documentation",
        categories=["documentation"],
        constraints=[
            "Use clear, concise language",
            "Structure output for readability",
            "Include actionable conclusions",
        ],
        output_sections=["Summary", "Details", "Conclusions"],
        priority=20,
        agent_type="task",
    ),
    "general-task": SubAgentSpec(
        name="general-task",
        persona="versatile analyst capable of handling diverse tasks",
        description="General-purpose task handler for anything not covered by specialized agents",
        categories=["data_processing", "documentation", "configuration"],
        constraints=[
            "Provide thorough, well-structured output",
            "Adapt approach to the specific task requirements",
        ],
        output_sections=["Analysis", "Results", "Recommendations"],
        priority=100,
        agent_type="task",
    ),
}

# All built-in agents indexed by type
_ALL_BUILT_IN: Dict[str, Dict[str, SubAgentSpec]] = {
    "review": _BUILT_IN_REVIEW_AGENTS,
    "task": _BUILT_IN_TASK_AGENTS,
}


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

        # 1. Load built-in agents
        built_in = _ALL_BUILT_IN.get(agent_type, {})
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

    def _load_yaml_agents(
        self, agent_type: str, agents_dir: str
    ) -> Dict[str, SubAgentSpec]:
        """Load user-defined agents from YAML files."""
        agents: Dict[str, SubAgentSpec] = {}
        base_path = Path(agents_dir)

        # Scan type-specific directory: .agents/delegation/{agent_type}/*.yaml
        type_dir = base_path / agent_type
        if type_dir.is_dir():
            agents.update(self._scan_yaml_dir(type_dir))

        # Scan root directory for wildcard agents: .agents/delegation/*.yaml
        if base_path.is_dir():
            for yaml_file in sorted(base_path.glob("*.yaml")):
                spec = self._parse_yaml_file(yaml_file)
                if spec and spec.agent_type in ("*", agent_type):
                    agents[spec.name] = spec

        return agents

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

            return SubAgentSpec(
                name=data["name"],
                persona=data.get("persona", ""),
                description=data.get("description", ""),
                categories=data.get("categories", []),
                constraints=data.get("constraints", []),
                output_sections=data.get("output_sections", []),
                priority=data.get("priority", 50),
                allowed_tools=data.get("allowed_tools"),
                blocked_tools=data.get("blocked_tools", []),
                agent_type=data.get("agent_type", "*"),
                source_file=str(yaml_file),
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

                spec = SubAgentSpec(
                    name=entry["name"],
                    persona=entry.get("persona", ""),
                    description=entry.get("description", ""),
                    categories=entry.get("categories", []),
                    constraints=entry.get("constraints", []),
                    output_sections=entry.get("output_sections", []),
                    priority=entry.get("priority", 50),
                    allowed_tools=entry.get("allowed_tools"),
                    blocked_tools=entry.get("blocked_tools", []),
                    agent_type=entry_type,
                )
                agents[spec.name] = spec

        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse DELEGATION_AGENTS JSON: {e}")

        return agents

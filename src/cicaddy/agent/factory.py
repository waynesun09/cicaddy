"""Agent factory for dynamic instantiation of AI agents."""

import os
from typing import Callable, Dict, List, Optional, Tuple, Type

from cicaddy.config.settings import MCPServerConfig, Settings
from cicaddy.utils.logger import get_logger

from .base import BaseAIAgent

logger = get_logger(__name__)

# Type alias for detector functions: (settings) -> Optional[agent_type_str]
TypeDetector = Callable[[Settings], Optional[str]]


class AgentFactory:
    """Registry-based factory for creating AI agents based on context and configuration.

    Agents and type detectors are registered via class methods, allowing
    platform-specific packages to extend the factory without modifying this module.
    """

    # Registry: agent_type_name -> agent class
    _registry: Dict[str, Type[BaseAIAgent]] = {}

    # Detectors: list of (detector_fn, priority) sorted by priority (lower = higher priority)
    _detectors: List[Tuple[TypeDetector, int]] = []

    @classmethod
    def register(cls, agent_type: str, agent_class: Type[BaseAIAgent]) -> None:
        """Register an agent class for a given type name.

        Args:
            agent_type: String identifier (e.g., "merge_request", "cron")
            agent_class: BaseAIAgent subclass to instantiate
        """
        cls._registry[agent_type] = agent_class
        logger.debug(f"Registered agent type '{agent_type}': {agent_class.__name__}")

    @classmethod
    def register_detector(cls, detector: TypeDetector, priority: int = 100) -> None:
        """Register a type detector function.

        Detectors are called in priority order (lowest first) during
        auto-detection. The first detector to return a non-None value wins.

        Args:
            detector: Callable that takes Settings and returns an agent type
                      string, or None if it cannot determine the type.
            priority: Lower values are checked first. Default: 100.
        """
        cls._detectors.append((detector, priority))
        cls._detectors.sort(key=lambda x: x[1])
        logger.debug(f"Registered type detector with priority {priority}")

    @staticmethod
    def create_agent(settings: Optional[Settings] = None) -> BaseAIAgent:
        """
        Create appropriate agent based on environment and configuration.

        Uses the registry to look up agent classes. Falls back to
        _determine_agent_type for auto-detection.

        Args:
            settings: Optional settings object, will load from environment if None

        Returns:
            BaseAIAgent instance

        Raises:
            ValueError: If unable to determine agent type or invalid configuration
        """
        if not settings:
            from cicaddy.config.settings import load_settings

            settings = load_settings()

        agent_type = AgentFactory._determine_agent_type(settings)

        logger.info(f"Creating {agent_type} agent")

        agent_class = AgentFactory._registry.get(agent_type)
        if agent_class:
            return agent_class(settings)

        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Registered types: {list(AgentFactory._registry.keys())}"
        )

    @staticmethod
    def _determine_agent_type(settings: Settings) -> str:
        """
        Determine which type of agent to create based on environment and settings.

        Checks registered detectors first (in priority order), then falls back
        to the built-in detection logic.

        Args:
            settings: Settings object to analyze

        Returns:
            String agent type: "merge_request", "branch_review", or "cron"
        """
        # 1. Check explicit agent type override
        explicit_type = os.getenv("AGENT_TYPE")
        if explicit_type:
            if explicit_type.lower() in ["mr", "merge_request"]:
                logger.info(
                    f"Using explicit agent type: merge_request (from AGENT_TYPE={explicit_type})"
                )
                return "merge_request"
            elif explicit_type.lower() in ["branch", "branch_review"]:
                logger.info(
                    f"Using explicit agent type: branch_review (from AGENT_TYPE={explicit_type})"
                )
                return "branch_review"
            elif explicit_type.lower() in ["cron", "scheduled"]:
                logger.info(
                    f"Using explicit agent type: cron (from AGENT_TYPE={explicit_type})"
                )
                return "cron"
            else:
                # Check if explicit type is a registered custom type
                if explicit_type.lower() in AgentFactory._registry:
                    logger.info(
                        f"Using explicit registered agent type: {explicit_type.lower()}"
                    )
                    return explicit_type.lower()
                logger.warning(
                    f"Unknown AGENT_TYPE value: {explicit_type}, falling back to auto-detection"
                )

        # 2. Run registered detectors in priority order
        for detector, priority in AgentFactory._detectors:
            try:
                result = detector(settings)
                if result:
                    logger.info(
                        f"Type detector (priority={priority}) returned: {result}"
                    )
                    return result
            except Exception as e:
                logger.warning(f"Type detector (priority={priority}) failed: {e}")

        # 3. Default to cron for general/scheduled analysis
        logger.info(
            "No specific context detected, defaulting to cron agent for general analysis"
        )
        return "cron"

    @staticmethod
    def get_available_agent_types() -> list[str]:
        """
        Get list of available agent types.

        Returns:
            List of registered agent type strings
        """
        return list(AgentFactory._registry.keys())

    @staticmethod
    def validate_agent_requirements(agent_type: str, settings: Settings) -> bool:
        """
        Validate that the agent type can be created with the given settings.

        Args:
            agent_type: Type of agent to validate
            settings: Settings to validate against

        Returns:
            True if agent can be created, False otherwise
        """
        if agent_type == "merge_request":
            # MR agent requires merge request IID
            mr_iid = getattr(settings, "merge_request_iid", None)
            if not mr_iid and not os.getenv("CI_MERGE_REQUEST_IID"):
                logger.error(
                    "Merge request agent requires merge_request_iid or CI_MERGE_REQUEST_IID"
                )
                return False

        elif agent_type == "branch_review":
            # Branch review agent needs basic AI provider
            if not settings.ai_provider:
                logger.warning("No AI provider configured, using default")

        elif agent_type == "cron":
            # Cron agent is more flexible, but check for basic requirements
            if not settings.ai_provider:
                logger.warning("No AI provider configured, using default")

        elif agent_type not in AgentFactory._registry:
            logger.error(f"Unknown agent type: {agent_type}")
            return False

        return True

    @staticmethod
    def get_mcp_servers_for_context(
        agent_type: str,
        cron_scope: Optional[str] = None,
        cron_task_type: Optional[str] = None,
    ) -> List[MCPServerConfig]:
        """
        Load MCP servers from environment configuration.

        This function simply loads whatever MCP servers the user has configured
        via environment variables, trusting their configuration choices.

        Args:
            agent_type: Type of agent being created (for logging only)
            cron_scope: Scope for cron agents (for logging only)
            cron_task_type: Task type for cron agents (for logging only)

        Returns:
            List of MCP server configurations from environment
        """
        servers = []

        # Load servers from MCP_SERVERS_CONFIG environment variable
        mcp_config_json = os.getenv("MCP_SERVERS_CONFIG")
        if mcp_config_json:
            try:
                import json

                server_configs = json.loads(mcp_config_json)

                for config_dict in server_configs:
                    try:
                        server = MCPServerConfig(**config_dict)
                        servers.append(server)
                        logger.debug(
                            f"Loaded MCP server: {server.name} ({server.protocol})"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to parse MCP server config {config_dict}: {e}"
                        )
                        continue

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse MCP_SERVERS_CONFIG JSON: {e}")
            except Exception as e:
                logger.error(f"Error loading MCP servers from environment: {e}")

        logger.info(
            f"Loaded {len(servers)} MCP servers from environment for {agent_type} agent"
        )
        return servers


def _detect_ci_agent_type(settings: Settings) -> Optional[str]:
    """CI environment agent type detector.

    Checks common CI environment variables to determine the appropriate agent type.

    Args:
        settings: Settings object to analyze

    Returns:
        Agent type string or None if not determinable
    """
    # Check for merge request context
    ci_mr_iid = os.getenv("CI_MERGE_REQUEST_IID")
    if ci_mr_iid:
        logger.info(f"Detected merge request context: CI_MERGE_REQUEST_IID={ci_mr_iid}")
        return "merge_request"

    # Check for cron task configuration
    cron_task_type = os.getenv("CRON_TASK_TYPE")
    if cron_task_type:
        logger.info(f"Detected cron context: CRON_TASK_TYPE={cron_task_type}")
        return "cron"

    # Check CI pipeline source
    pipeline_source = os.getenv("CI_PIPELINE_SOURCE")
    if pipeline_source:
        if pipeline_source == "merge_request_event":
            logger.info(
                f"Detected merge request context: CI_PIPELINE_SOURCE={pipeline_source}"
            )
            return "merge_request"
        elif pipeline_source == "schedule":
            logger.info(f"Detected cron context: CI_PIPELINE_SOURCE={pipeline_source}")
            return "cron"
        elif pipeline_source == "push":
            # Check if this is a branch push (not to default branch)
            ci_commit_branch = os.getenv("CI_COMMIT_BRANCH")
            ci_default_branch = os.getenv("CI_DEFAULT_BRANCH", "main")
            if ci_commit_branch and ci_commit_branch != ci_default_branch:
                logger.info(
                    f"Detected branch push context: CI_PIPELINE_SOURCE={pipeline_source}, branch={ci_commit_branch}"
                )
                return "branch_review"
            else:
                logger.info(
                    "Push to default branch detected, checking other indicators"
                )
        else:
            logger.info(
                f"Unknown pipeline source: {pipeline_source}, checking other indicators"
            )

    # Check settings for merge request IID
    mr_iid = getattr(settings, "merge_request_iid", None)
    if mr_iid:
        logger.info(f"Found merge request IID in settings: {mr_iid}")
        return "merge_request"

    return None


# --- Register built-in agents and detectors ---
from .branch_agent import BranchReviewAgent  # noqa: E402
from .cron_agent import CronAIAgent  # noqa: E402
from .mr_agent import MergeRequestAgent  # noqa: E402

AgentFactory.register("merge_request", MergeRequestAgent)
AgentFactory.register("branch_review", BranchReviewAgent)
AgentFactory.register("cron", CronAIAgent)

# Register CI environment detector with default priority
AgentFactory.register_detector(_detect_ci_agent_type, priority=50)

# Discover and register plugin agents (idempotent)
from cicaddy.plugin import discover_plugins  # noqa: E402

discover_plugins()


def create_agent_from_environment() -> BaseAIAgent:
    """
    Convenience function to create agent from environment variables.

    This is the main entry point for agent creation in most scenarios.

    Returns:
        BaseAIAgent instance based on environment detection

    Raises:
        ValueError: If unable to create agent or validation fails
    """
    from cicaddy.config.settings import load_settings

    settings = load_settings()
    factory = AgentFactory()

    # Determine agent type
    agent_type = factory._determine_agent_type(settings)

    # Validate requirements
    if not factory.validate_agent_requirements(agent_type, settings):
        raise ValueError(f"Cannot create {agent_type} agent: requirements not met")

    # Create and return agent
    return factory.create_agent(settings)

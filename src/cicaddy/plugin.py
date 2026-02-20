"""Plugin discovery for cicaddy using importlib.metadata entry_points.

Entry point groups:
    cicaddy.agents          - Register agents + detectors
    cicaddy.cli_args        - Additional CLI argument mappings
    cicaddy.env_vars        - Additional environment variable names
    cicaddy.config_sections - Config show display sections
    cicaddy.validators      - Validation checks
    cicaddy.settings_loader - Settings factory override
"""

import logging
from importlib.metadata import entry_points
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

_plugins_discovered = False


def discover_plugins() -> None:
    """Load and invoke all ``cicaddy.agents`` entry points.

    This is idempotent — calling it more than once is a no-op.
    Each entry point should be a callable that registers agents and/or
    detectors with :class:`cicaddy.agent.factory.AgentFactory`.
    """
    global _plugins_discovered
    if _plugins_discovered:
        return
    _plugins_discovered = True

    eps = entry_points()
    agent_eps = eps.select(group="cicaddy.agents") if hasattr(eps, "select") else eps.get("cicaddy.agents", [])
    for ep in agent_eps:
        try:
            register_fn = ep.load()
            register_fn()
            logger.debug("Loaded agent plugin: %s", ep.name)
        except Exception as e:
            logger.warning("Failed to load agent plugin '%s': %s", ep.name, e)


def get_plugin_cli_args() -> List:
    """Return additional CLI argument mappings from all ``cicaddy.cli_args`` plugins.

    Each entry point should be a callable returning a list of
    :class:`cicaddy.cli.arg_mapping.ArgMapping` instances.
    """
    from cicaddy.cli.arg_mapping import ArgMapping  # noqa: F811 — deferred import

    result: List[ArgMapping] = []
    eps = entry_points()
    cli_eps = eps.select(group="cicaddy.cli_args") if hasattr(eps, "select") else eps.get("cicaddy.cli_args", [])
    for ep in cli_eps:
        try:
            args_fn = ep.load()
            result.extend(args_fn())
            logger.debug("Loaded CLI args plugin: %s", ep.name)
        except Exception as e:
            logger.warning("Failed to load CLI args plugin '%s': %s", ep.name, e)
    return result


def get_plugin_env_vars() -> List[str]:
    """Return additional environment variable names from ``cicaddy.env_vars`` plugins.

    Each entry point should be a callable returning a list of env var name strings.
    """
    result: List[str] = []
    eps = entry_points()
    env_eps = eps.select(group="cicaddy.env_vars") if hasattr(eps, "select") else eps.get("cicaddy.env_vars", [])
    for ep in env_eps:
        try:
            vars_fn = ep.load()
            result.extend(vars_fn())
            logger.debug("Loaded env vars plugin: %s", ep.name)
        except Exception as e:
            logger.warning("Failed to load env vars plugin '%s': %s", ep.name, e)
    return result


def get_plugin_config_sections() -> List[Callable]:
    """Return config section display callables from ``cicaddy.config_sections`` plugins.

    Each entry point should be a callable with signature:
        (config: Dict, mask_fn: Callable, sensitive_vars: frozenset) -> None
    """
    result: List[Callable] = []
    eps = entry_points()
    cfg_eps = eps.select(group="cicaddy.config_sections") if hasattr(eps, "select") else eps.get("cicaddy.config_sections", [])
    for ep in cfg_eps:
        try:
            result.append(ep.load())
            logger.debug("Loaded config section plugin: %s", ep.name)
        except Exception as e:
            logger.warning("Failed to load config section plugin '%s': %s", ep.name, e)
    return result


def get_plugin_validators() -> List[Callable]:
    """Return validation callables from ``cicaddy.validators`` plugins.

    Each entry point should be a callable with signature:
        (config: Dict) -> Tuple[List[str], List[str]]  # (errors, warnings)
    """
    result: List[Callable] = []
    eps = entry_points()
    val_eps = eps.select(group="cicaddy.validators") if hasattr(eps, "select") else eps.get("cicaddy.validators", [])
    for ep in val_eps:
        try:
            result.append(ep.load())
            logger.debug("Loaded validator plugin: %s", ep.name)
        except Exception as e:
            logger.warning("Failed to load validator plugin '%s': %s", ep.name, e)
    return result


def get_settings_loader() -> Optional[Callable]:
    """Return a settings loader callable from ``cicaddy.settings_loader`` plugin, if any.

    Only the first registered entry point is used. The callable should
    return a :class:`cicaddy.config.settings.CoreSettings` (or subclass) instance.
    """
    eps = entry_points()
    loader_eps = eps.select(group="cicaddy.settings_loader") if hasattr(eps, "select") else eps.get("cicaddy.settings_loader", [])
    for ep in loader_eps:
        try:
            loader = ep.load()
            logger.debug("Loaded settings loader plugin: %s", ep.name)
            return loader
        except Exception as e:
            logger.warning("Failed to load settings loader plugin '%s': %s", ep.name, e)
    return None

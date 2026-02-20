"""CLI command implementations."""

import asyncio
import sys
from argparse import Namespace
from typing import Any, Dict

from cicaddy.cli.arg_mapping import SENSITIVE_ENV_VARS
from cicaddy.cli.env_loader import (
    apply_cli_args_to_env,
    get_effective_config,
    load_env_file,
    mask_sensitive_value,
)


def get_version() -> str:
    """Get the package version."""
    try:
        from importlib.metadata import version

        return version("cicaddy")
    except Exception:
        # Fallback to reading pyproject.toml
        try:
            import tomllib
            from pathlib import Path

            pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"
            if pyproject.exists():
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                return data.get("project", {}).get("version", "unknown")
        except Exception:  # nosec B110 - intentional fallback to "unknown"
            pass
    return "unknown"


def cmd_version(args: Namespace) -> int:
    """Handle the 'version' command."""
    print(f"cicaddy version {get_version()}")
    return 0


def cmd_config_show(args: Namespace) -> int:
    """Handle the 'config show' command."""
    # Load env file if specified
    if args.env_file:
        try:
            load_env_file(args.env_file)
            print(f"Loaded environment from: {args.env_file}\n")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Get effective configuration
    config = get_effective_config()

    print("Current Configuration:")
    print("=" * 50)

    # Group by category
    print("\n[Agent Settings]")
    agent_vars = ["AGENT_TYPE", "AI_PROVIDER", "AI_MODEL", "MAX_INFER_ITERS"]
    for var in agent_vars:
        value = config.get(var)
        if var in SENSITIVE_ENV_VARS:
            print(f"  {var}: {mask_sensitive_value(value)}")
        else:
            print(f"  {var}: {value or '(not set)'}")

    print("\n[API Keys]")
    api_keys = ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    for var in api_keys:
        value = config.get(var)
        print(f"  {var}: {mask_sensitive_value(value)}")

    print("\n[Logging]")
    print(f"  LOG_LEVEL: {config.get('LOG_LEVEL') or '(not set)'}")

    print("\n[MCP]")
    mcp_config = config.get("MCP_SERVERS_CONFIG")
    if mcp_config:
        # Truncate long config strings
        if len(mcp_config) > 80:
            print(f"  MCP_SERVERS_CONFIG: {mcp_config[:77]}...")
        else:
            print(f"  MCP_SERVERS_CONFIG: {mcp_config}")
    else:
        print("  MCP_SERVERS_CONFIG: (not set)")

    print("\nNote: Sensitive values (API keys, tokens) are masked with ****.")

    return 0


def cmd_validate(args: Namespace) -> int:
    """Handle the 'validate' command - pre-flight configuration checks."""
    # Load env file if specified
    if args.env_file:
        try:
            load_env_file(args.env_file)
            print(f"Loaded environment from: {args.env_file}\n")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    print("Configuration Validation")
    print("=" * 50)

    errors: list[str] = []
    warnings: list[str] = []

    config = get_effective_config()

    # Check AI Provider configuration
    print("\n[AI Provider]")
    ai_provider = config.get("AI_PROVIDER")
    if not ai_provider:
        errors.append("AI_PROVIDER is not set")
        print("  AI_PROVIDER: (not set) ✗")
    else:
        print(f"  AI_PROVIDER: {ai_provider} ✓")

        # Check for corresponding API key
        api_key_map = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure": "AZURE_OPENAI_KEY",
            "ollama": None,  # Ollama doesn't require API key
        }
        required_key = api_key_map.get(ai_provider.lower())
        if required_key:
            if config.get(required_key):
                print(
                    f"  {required_key}: {mask_sensitive_value(config.get(required_key))} ✓"
                )
            else:
                errors.append(f"{required_key} is required for {ai_provider} provider")
                print(f"  {required_key}: (not set) ✗")
        elif ai_provider.lower() == "ollama":
            print("  API Key: (not required for Ollama) ✓")

    # Check Agent Type configuration
    print("\n[Agent Configuration]")
    agent_type = config.get("AGENT_TYPE")
    if agent_type:
        print(f"  AGENT_TYPE: {agent_type} ✓")
    else:
        warnings.append("AGENT_TYPE not set (will be auto-detected)")
        print("  AGENT_TYPE: (will be auto-detected) ~")

    # Check MCP configuration
    print("\n[MCP Servers]")
    mcp_config = config.get("MCP_SERVERS_CONFIG")
    if mcp_config:
        try:
            import json

            servers = json.loads(mcp_config)
            if isinstance(servers, list):
                print(f"  Configured servers: {len(servers)} ✓")
                for server in servers:
                    name = server.get("name", "unnamed")
                    protocol = server.get("protocol", "unknown")
                    # Show config summary without exposing secrets
                    extras = []
                    if server.get("headers"):
                        extras.append("headers")
                    if server.get("env"):
                        extras.append("env")
                    if server.get("tools"):
                        extras.append(f"{len(server['tools'])} tools")
                    extra_info = f" [{', '.join(extras)}]" if extras else ""
                    print(f"    - {name} ({protocol}){extra_info}")
            else:
                warnings.append("MCP_SERVERS_CONFIG is not a valid JSON array")
                print("  MCP_SERVERS_CONFIG: invalid format ~")
        except json.JSONDecodeError:
            # Don't include error details as they may contain secrets from config
            errors.append("MCP_SERVERS_CONFIG is not valid JSON")
            print("  MCP_SERVERS_CONFIG: invalid JSON ✗")
    else:
        print("  MCP_SERVERS_CONFIG: (not configured) ~")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"\n✗ Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        if warnings:
            print(f"\n~ {len(warnings)} warning(s):")
            for warn in warnings:
                print(f"  - {warn}")
        return 1
    elif warnings:
        print(f"\n✓ Validation PASSED with {len(warnings)} warning(s):")
        for warn in warnings:
            print(f"  - {warn}")
        return 0
    else:
        print("\n✓ Validation PASSED - configuration is valid")
        return 0


def cmd_run(args: Namespace) -> int:
    """Handle the 'run' command."""
    # Load env file if specified (lowest priority)
    if args.env_file:
        try:
            loaded = load_env_file(args.env_file)
            print(
                f"Read {len(loaded)} variables from {args.env_file} (existing env vars preserved)"
            )
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Apply CLI arguments to environment (highest priority)
    args_dict = vars(args)
    applied = apply_cli_args_to_env(args_dict)
    if applied:
        print(f"Applied {len(applied)} CLI arguments to environment")

    # Handle dry-run mode
    if args.dry_run:
        print("\n[DRY RUN] Would run agent with configuration:")
        return cmd_config_show(args)

    # Run the agent
    return _run_agent()


def _run_agent() -> int:
    """Run the AI agent."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        from cicaddy.config.settings import load_settings
        from cicaddy.utils.logger import get_logger, setup_logging

        # Load settings (will read from environment)
        settings = load_settings()

        # Setup structured logging
        setup_logging(
            level=settings.log_level,
            json_logs=settings.json_logs,
        )

        logger = get_logger(__name__)
        logger.info("Starting Cicaddy agent via CLI")

        # Run async main
        return asyncio.run(_run_agent_async(settings, logger))

    except ImportError as e:
        print(f"Error: Failed to import agent modules: {e}", file=sys.stderr)
        print(
            "Make sure you're running from the project root or have installed the package.",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def _run_agent_async(settings: Any, logger: Any) -> int:
    """Async agent execution."""
    from cicaddy.agent.factory import create_agent_from_environment
    from cicaddy.ai_providers.base import ProviderError, TemporaryServiceError

    try:
        logger.info("Configuration loaded successfully")

        # Create agent using factory
        agent = create_agent_from_environment()
        await agent.initialize()

        # Determine which method to call based on agent type
        if hasattr(agent, "run_scheduled_analysis"):
            logger.info("Running scheduled analysis")
            results = await agent.run_scheduled_analysis()
            logger.info(
                "Scheduled analysis completed",
                task_type=results.get("task_type", "unknown"),
                execution_time=results.get("execution_time", 0),
            )
        elif hasattr(agent, "process_merge_request"):
            logger.info("Running merge request analysis")
            results = await agent.process_merge_request()
            logger.info("MR analysis completed")
        else:
            logger.info("Running base agent analysis")
            results = await agent.analyze()
            logger.info(
                "Analysis completed",
                execution_time=results.get("execution_time", 0),
            )

        # Cleanup
        try:
            await agent.cleanup()
        except Exception as cleanup_error:
            logger.debug(f"Expected cleanup error: {cleanup_error}")

        logger.info("Cicaddy agent completed successfully")
        return 0

    except ProviderError as e:
        error_msg = str(e) if str(e) else f"{type(e).__name__}: (no message)"

        if isinstance(e, TemporaryServiceError):
            exit_code = 2
            logger.error(
                "AI Provider service temporarily unavailable",
                error=error_msg,
                retry_count=getattr(e, "retry_count", 0),
                exc_info=True,
            )
        else:
            exit_code = 3
            logger.error(
                "AI Provider configuration error",
                error=error_msg,
                exception_type=type(e).__name__,
                exc_info=True,
            )

        # Try to send error notification
        await _send_error_notification(settings, e, error_msg, exit_code)
        return exit_code

    except Exception as e:
        error_msg = str(e) if str(e) else f"{type(e).__name__}: (no message)"
        logger.error(
            "Cicaddy agent failed",
            error=error_msg,
            exception_type=type(e).__name__,
            exc_info=True,
        )

        # Try to send error notification
        await _send_error_notification(settings, e, error_msg, 1)
        return 1


async def _send_error_notification(
    settings: Any, error: Exception, error_msg: str, exit_code: int
) -> None:
    """Send error notification via Slack if configured."""
    try:
        slack_webhook_urls = settings.get_slack_webhook_urls()
        if slack_webhook_urls:
            from cicaddy.ai_providers.base import TemporaryServiceError
            from cicaddy.notifications.slack import SlackNotifier

            notifier = SlackNotifier(slack_webhook_urls, ssl_verify=settings.ssl_verify)

            context: Dict[str, Any] = {
                "project_name": getattr(settings, "project_name", "")
                or "Unknown Project",
                "project_id": getattr(settings, "project_id", None),
                "error_type": "ai_provider_failure"
                if exit_code in [2, 3]
                else "general_failure",
                "exit_code": exit_code,
            }

            if isinstance(error, TemporaryServiceError):
                context["retry_count"] = getattr(error, "retry_count", 0)

            await notifier.send_error_notification(
                f"AI Agent Failure: {error_msg}",
                context=context,
            )
    except Exception:  # nosec B110
        pass  # Silently ignore notification failures

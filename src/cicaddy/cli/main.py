#!/usr/bin/env python3
"""Main CLI entry point for Cicaddy."""

import argparse
import sys
from typing import Any, Dict, List, Optional

from cicaddy.cli.arg_mapping import get_run_arg_mappings
from cicaddy.cli.commands import (
    cmd_config_show,
    cmd_run,
    cmd_validate,
    cmd_version,
    get_version,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="cicaddy",
        description="Cicaddy - Platform-agnostic pipeline AI agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cicaddy run --env-file .env
  cicaddy run --agent-type task --ai-provider gemini --log-level DEBUG
  cicaddy config show --env-file .env
  cicaddy version

For more information, visit:
  https://github.com/waynesun09/cicaddy
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {get_version()}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
        metavar="COMMAND",
    )

    # 'run' subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Run the AI agent",
        description="Run the Cicaddy agent with specified configuration",
    )
    _add_run_arguments(run_parser)

    # 'config' subcommand with 'show' subsubcommand
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management",
        description="View and manage agent configuration",
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
        title="config commands",
        metavar="SUBCOMMAND",
    )

    config_show_parser = config_subparsers.add_parser(
        "show",
        help="Show current configuration",
        description="Display the effective agent configuration",
    )
    config_show_parser.add_argument(
        "--env-file",
        "-e",
        metavar="FILE",
        help="Load environment from a .env file",
    )

    # 'version' subcommand
    subparsers.add_parser(
        "version",
        help="Show version information",
        description="Display the cicaddy version",
    )

    # 'validate' subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration before running",
        description="Perform pre-flight checks on agent configuration",
    )
    validate_parser.add_argument(
        "--env-file",
        "-e",
        metavar="FILE",
        help="Load environment from a .env file",
    )

    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'run' subcommand."""
    from cicaddy.agent.factory import AgentFactory
    from cicaddy.plugin import discover_plugins

    # Ensure plugins are loaded so agent types are registered
    discover_plugins()

    # Environment file option
    parser.add_argument(
        "--env-file",
        "-e",
        metavar="FILE",
        help="Load environment from a .env file (lowest priority, CLI args override)",
    )

    # Build dynamic agent-type choices from registry + standard aliases
    registered = AgentFactory.get_available_agent_types()
    agent_type_choices = sorted(set(registered) | {"mr", "task", "cron", "branch"})

    # Add all mapped arguments (base + plugin)
    for mapping in get_run_arg_mappings():
        kwargs: Dict[str, Any] = {
            "help": mapping.help_text or f"Set {mapping.env_var}",
            "dest": mapping.cli_arg.lstrip("-").replace("-", "_"),
        }

        if mapping.env_var == "AGENT_TYPE":
            # Use dynamically built choices for --agent-type
            kwargs["choices"] = agent_type_choices
            kwargs["metavar"] = "AGENT_TYPE"
        elif mapping.choices:
            kwargs["choices"] = mapping.choices
            kwargs["metavar"] = mapping.cli_arg.lstrip("-").upper().replace("-", "_")

        if mapping.arg_type is int:
            kwargs["type"] = int

        if mapping.default is not None:
            kwargs["default"] = None  # Don't set default, let env/settings handle it

        args = [mapping.cli_arg]
        if mapping.short_arg:
            args.insert(0, mapping.short_arg)

        parser.add_argument(*args, **kwargs)

    # Verbose flag
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (sets LOG_LEVEL=DEBUG)",
    )

    # Dry-run flag
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running the agent",
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # No command specified - show help
    if args.command is None:
        parser.print_help()
        return 0

    # Route to appropriate command handler
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "config":
        if args.config_command == "show":
            return cmd_config_show(args)
        else:
            # No config subcommand - show config help
            parser.parse_args(["config", "--help"])
            return 0
    elif args.command == "version":
        return cmd_version(args)
    elif args.command == "validate":
        return cmd_validate(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())

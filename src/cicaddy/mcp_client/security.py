"""Security validation for MCP client operations."""

import re
from typing import Dict, List, Optional

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


class SecurityValidator:
    """Validates MCP server configurations and commands for security."""

    # List of sensitive env vars that should not be overridden (from Goose patterns)
    DISALLOWED_ENV_VARS = [
        # 🔧 Binary path manipulation
        "PATH",  # Controls executable lookup paths — critical for command hijacking
        "PATHEXT",  # Windows: Determines recognized executable extensions (e.g., .exe, .bat)
        "SystemRoot",  # Windows: Can affect system DLL resolution (e.g., `kernel32.dll`)
        "windir",  # Windows: Alternative to SystemRoot (used in legacy apps)
        # 🧬 Dynamic linker hijacking (Linux/macOS)
        "LD_LIBRARY_PATH",  # Alters shared library resolution
        "LD_PRELOAD",  # Forces preloading of shared libraries — common attack vector
        "LD_AUDIT",  # Loads a monitoring library that can intercept execution
        "LD_DEBUG",  # Enables verbose linker logging (information disclosure risk)
        "LD_BIND_NOW",  # Forces immediate symbol resolution, affecting ASLR
        "LD_ASSUME_KERNEL",  # Tricks linker into thinking it's running on an older kernel
        # 🍎 macOS dynamic linker variables
        "DYLD_LIBRARY_PATH",  # Same as LD_LIBRARY_PATH but for macOS
        "DYLD_INSERT_LIBRARIES",  # macOS equivalent of LD_PRELOAD
        "DYLD_FRAMEWORK_PATH",  # Overrides framework lookup paths
        # 🐍 Python / Node / Ruby / Java / Golang hijacking
        "PYTHONPATH",  # Overrides Python module resolution
        "PYTHONHOME",  # Overrides Python root directory
        "NODE_OPTIONS",  # Injects options/scripts into every Node.js process
        "RUBYOPT",  # Injects Ruby execution flags
        "GEM_PATH",  # Alters where RubyGems looks for installed packages
        "GEM_HOME",  # Changes RubyGems default install location
        "CLASSPATH",  # Java: Controls where classes are loaded from — critical for RCE attacks
        "GO111MODULE",  # Go: Forces use of module proxy or disables it
        "GOROOT",  # Go: Changes root installation directory (could lead to execution hijacking)
        # 🖥️ Process & DLL hijacking
        "TEMP",
        "TMP",  # Redirects temporary file storage (useful for injection attacks)
        "LOCALAPPDATA",  # Controls application data paths (can be abused for persistence)
        "USERPROFILE",  # User directory (can affect profile-based execution paths)
        "HOMEDRIVE",
        "HOMEPATH",  # Changes where the user's home directory is located
    ]

    # Dangerous command patterns that should be blocked
    DANGEROUS_COMMAND_PATTERNS = [
        r";\s*rm\s+-rf",  # Command injection with rm -rf
        r"&&\s*rm\s+-rf",  # Command chaining with rm -rf
        r"\|\s*sh",  # Piping to shell
        r"\|\s*bash",  # Piping to bash
        r"\$\(",  # Command substitution
        r"`[^`]*`",  # Backtick command substitution
        r">\s*/dev/",  # Writing to device files
        r"<\s*/dev/",  # Reading from device files
        r"curl\s+.*\|\s*sh",  # Download and execute pattern
        r"wget\s+.*\|\s*sh",  # Download and execute pattern
        r"rm\s+-rf\s+/",  # Direct rm -rf / commands
        r"os\.system",  # Python os.system calls
        r"subprocess\.",  # Python subprocess calls in args
        r"child_process",  # Node.js child_process
        r"exec\s*\(",  # Exec function calls
    ]

    # Dangerous command executables that should be blocked
    DANGEROUS_COMMANDS = [
        "sh",
        "bash",
        "zsh",
        "fish",  # Direct shell access
        "eval",
        "exec",  # Code execution
        "nc",
        "netcat",
        "telnet",  # Network tools
        "wget",
        "curl",  # Download tools when used with shell
        "dd",  # Disk manipulation
    ]

    @classmethod
    def validate_env_vars(cls, env_vars: Optional[Dict[str, str]]) -> Dict[str, str]:
        """
        Validate environment variables, removing dangerous ones.

        Args:
            env_vars: Dictionary of environment variables to validate

        Returns:
            Dict of validated environment variables with dangerous ones removed
        """
        if not env_vars:
            return {}

        validated = {}

        for key, value in env_vars.items():
            if cls._is_env_var_disallowed(key):
                logger.warning(f"Skipping disallowed environment variable: {key}")
                continue

            validated[key] = value

        return validated

    @classmethod
    def validate_command(cls, command: str, args: Optional[List[str]] = None) -> bool:
        """
        Validate command and arguments for dangerous patterns.

        Args:
            command: The command to execute
            args: List of command arguments

        Returns:
            True if command is safe, False if dangerous
        """
        # Check if command itself is dangerous
        command_basename = command.split("/")[-1]  # Get just the command name
        if command_basename in cls.DANGEROUS_COMMANDS:
            logger.error(f"Dangerous command blocked: {command_basename}")
            return False

        # Check the full command string including arguments
        full_command = command
        if args:
            full_command += " " + " ".join(args)

        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_COMMAND_PATTERNS:
            if re.search(pattern, full_command, re.IGNORECASE):
                logger.error(f"Dangerous command pattern detected: {pattern}")
                logger.error(f"Command: {full_command}")
                return False

        # Additional checks for absolute paths outside safe directories
        if command.startswith("/") and not cls._is_safe_absolute_path(command):
            logger.warning(f"Command uses potentially unsafe absolute path: {command}")

        return True

    @classmethod
    def sanitize_working_directory(cls, working_dir: Optional[str]) -> Optional[str]:
        """
        Sanitize working directory path.

        Args:
            working_dir: Working directory path to sanitize

        Returns:
            Sanitized working directory path or None if invalid
        """
        if not working_dir:
            return None

        # Remove dangerous path components
        if ".." in working_dir:
            logger.warning(f"Working directory contains path traversal: {working_dir}")
            return None

        # Ensure it's an absolute path
        if not working_dir.startswith("/"):
            logger.warning(f"Working directory must be absolute path: {working_dir}")
            return None

        return working_dir

    @classmethod
    def _is_env_var_disallowed(cls, key: str) -> bool:
        """Check if environment variable key is disallowed."""
        return any(
            disallowed.lower() == key.lower() for disallowed in cls.DISALLOWED_ENV_VARS
        )

    @classmethod
    def _is_safe_absolute_path(cls, path: str) -> bool:
        """Check if absolute path is in a safe directory."""
        safe_prefixes = [
            "/usr/bin/",
            "/usr/local/bin/",
            "/bin/",
            "/opt/",
            "/home/",
        ]

        return any(path.startswith(prefix) for prefix in safe_prefixes)

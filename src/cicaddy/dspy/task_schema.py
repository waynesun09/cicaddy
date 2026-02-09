"""Pydantic models for YAML task configuration.

These models define the schema for declarative task definitions that replace
monolithic AI_TASK_PROMPT strings.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TaskInput(BaseModel):
    """Input parameter for a task.

    Inputs can be sourced from environment variables or provided with defaults.

    Two mechanisms exist for injecting environment values:

    * **``{{VAR}}`` placeholders** in any YAML string value (``context``,
      ``description``, ``default``, etc.).  These are resolved at load time
      by ``TaskLoader`` *after* YAML parsing and are best for inline text
      embedding.

    * **``env_var`` on this model** — resolved at load time via
      ``TaskLoader._resolve_inputs()`` with support for defaults and
      ``required`` validation.  Use this for structured inputs that need
      explicit tracking and error handling.
    """

    name: str = Field(..., description="Parameter name used in prompt templates")
    env_var: Optional[str] = Field(
        None, description="Environment variable to read value from"
    )
    default: Optional[str] = Field(None, description="Default value if env_var not set")
    description: Optional[str] = Field(
        None, description="Human-readable description of this input"
    )
    required: bool = Field(
        True, description="Whether this input must have a value (from env or default)"
    )
    format: Optional[Literal["diff", "code"]] = Field(
        None,
        description=(
            "Content format hint for rendering large values in the prompt. "
            "'diff' uses ~~~diff fencing, 'code' uses plain ~~~ fencing."
        ),
    )


class TaskOutput(BaseModel):
    """Output section expected from the task.

    Outputs define the structure of what the AI should produce.
    """

    name: str = Field(..., description="Section name (e.g., 'executive_summary')")
    description: Optional[str] = Field(
        None, description="Description of what this section should contain"
    )
    required: bool = Field(True, description="Whether this section is required")
    format: Optional[str] = Field(
        None, description="Expected format (e.g., 'table', 'list', 'paragraph')"
    )


class ToolConfig(BaseModel):
    """MCP tool configuration for a task.

    Specifies which MCP servers and tools are available for the task.
    """

    servers: List[str] = Field(
        default_factory=list, description="List of MCP server names to use"
    )
    required_tools: List[str] = Field(
        default_factory=list, description="Tools that must be used during execution"
    )
    forbidden_tools: List[str] = Field(
        default_factory=list, description="Tools that should not be used"
    )


class TaskDefinition(BaseModel):
    """Complete task definition loaded from YAML.

    This is the main schema for declarative task configuration.
    """

    # Core identification
    name: str = Field(..., description="Unique task identifier")
    description: str = Field(..., description="What this task does")
    type: Literal["data_analysis", "monitoring", "code_review", "custom"] = Field(
        "custom", description="Task category for specialized handling"
    )
    persona: Optional[str] = Field(
        None,
        description=(
            "Explicit AI persona/role description. "
            "Overrides the default ROLE_MAPPING derived from 'type'."
        ),
    )

    # Input/Output specification
    inputs: List[TaskInput] = Field(
        default_factory=list, description="Input parameters from env/context"
    )
    outputs: List[TaskOutput] = Field(
        default_factory=list, description="Required output sections"
    )

    # Tool configuration
    tools: ToolConfig = Field(
        default_factory=ToolConfig, description="MCP server configuration"
    )

    # Execution constraints
    constraints: List[str] = Field(
        default_factory=list, description="Rules the AI must follow"
    )

    # Reasoning strategy
    reasoning: Literal["chain_of_thought", "react", "simple"] = Field(
        "chain_of_thought", description="Reasoning approach to use"
    )

    # Output configuration
    output_format: Literal["markdown", "html", "json"] = Field(
        "markdown", description="Expected output format"
    )

    # Optional metadata
    version: str = Field("1.0", description="Task definition version")
    author: Optional[str] = Field(None, description="Task author")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")

    # Additional prompt context
    context: Optional[str] = Field(
        None, description="Additional context to include in the prompt"
    )
    examples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Example input/output pairs for few-shot learning",
    )

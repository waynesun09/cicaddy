"""Prompt builder that converts TaskDefinition to structured prompts.

Generates AI prompts from declarative YAML task definitions.
"""

import logging
from typing import Any, Dict, List, Optional

from cicaddy.dspy.task_schema import TaskDefinition

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds structured prompts from TaskDefinition objects."""

    # Legacy input names treated as diff content when TaskInput.format is None.
    _LEGACY_DIFF_NAMES = frozenset(("diff_content", "code_changes", "diff"))

    # Task type to role mapping
    ROLE_MAPPING = {
        "data_analysis": "expert data analyst and SQL specialist",
        "monitoring": "infrastructure monitoring specialist",
        "code_review": "senior software engineer and code reviewer",
        "custom": "AI assistant specialized in the requested task",
    }

    # Reasoning strategy templates
    REASONING_TEMPLATES = {
        "chain_of_thought": (
            "Think through this step-by-step:\n"
            "1. Understand the requirements\n"
            "2. Plan your approach\n"
            "3. Execute the analysis\n"
            "4. Synthesize findings\n"
            "5. Provide recommendations"
        ),
        "react": (
            "Use the ReAct pattern:\n"
            "- Thought: Reason about what to do next\n"
            "- Action: Take an action using available tools\n"
            "- Observation: Observe the result\n"
            "- Repeat until task is complete"
        ),
        "simple": "Complete the task directly and efficiently.",
    }

    def __init__(
        self,
        task: TaskDefinition,
        resolved_inputs: Optional[Dict[str, Optional[str]]] = None,
    ):
        """Initialize the prompt builder.

        Args:
            task: TaskDefinition to build prompts from.
            resolved_inputs: Pre-resolved input values from TaskLoader.
                             If not provided, falls back to each input's default.
        """
        self.task = task
        self._base_resolved: Dict[str, Optional[str]] = resolved_inputs or {
            inp.name: inp.default for inp in task.inputs
        }

    def _resolve_inputs_with_context(
        self, context: Dict[str, Any]
    ) -> Dict[str, Optional[str]]:
        """Build a local dictionary of resolved input values overlaid with context.

        Context keys matching input names will override the loader-resolved
        values.  The TaskDefinition object is never mutated.

        Args:
            context: Runtime context dictionary.

        Returns:
            Dictionary mapping input names to their resolved string values.
        """
        resolved = dict(self._base_resolved)

        for inp in self.task.inputs:
            if inp.name in context and context[inp.name] is not None:
                value = context[inp.name]
                resolved[inp.name] = str(value) if not isinstance(value, str) else value
                logger.debug(f"Populated input {inp.name} from context")

        return resolved

    def build(
        self,
        context: Optional[Dict[str, Any]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build a complete prompt from the task definition.

        Args:
            context: Additional runtime context (e.g., project info, timestamp).
                     Context values matching input names will populate those inputs.
            mcp_tools: Available MCP tools with their schemas

        Returns:
            Complete prompt string for the AI model
        """
        context = context or {}
        mcp_tools = mcp_tools or []

        # Build a local resolved-inputs dictionary (no mutation)
        resolved_inputs = self._resolve_inputs_with_context(context)

        sections = [
            self._build_role_section(),
            self._build_objective_section(),
            self._build_inputs_section(resolved_inputs),
            self._build_outputs_section(),
            self._build_tools_section(mcp_tools),
            self._build_constraints_section(),
            self._build_context_section(context),
            self._build_examples_section(),
            self._build_reasoning_section(),
            self._build_format_section(),
        ]

        # Filter out empty sections and join
        prompt = "\n\n".join(section for section in sections if section)

        logger.debug(
            f"Built prompt for task {self.task.name}",
            extra={
                "prompt_length": len(prompt),
                "sections": len([s for s in sections if s]),
            },
        )

        return prompt

    def _build_role_section(self) -> str:
        """Build the role/persona section."""
        role = self.task.persona or self.ROLE_MAPPING.get(
            self.task.type, self.ROLE_MAPPING["custom"]
        )
        return f"You are an {role}.\n\nTask: {self.task.name}\nDescription: {self.task.description}"

    def _build_objective_section(self) -> str:
        """Build the objective section from task description."""
        if not self.task.context:
            return ""
        return f"## Background\n\n{self.task.context}"

    def _build_inputs_section(self, resolved_inputs: Dict[str, Optional[str]]) -> str:
        """Build the inputs section.

        Args:
            resolved_inputs: Dictionary mapping input names to resolved values.
        """
        if not self.task.inputs:
            return ""

        lines = ["## Inputs\n"]
        large_content_inputs: list[tuple[str, str, Optional[str]]] = []

        for inp in self.task.inputs:
            value = resolved_inputs.get(inp.name) or "(not provided)"
            desc = f" - {inp.description}" if inp.description else ""
            required = " (required)" if inp.required else " (optional)"

            resolved_value = resolved_inputs.get(inp.name)

            # Determine effective format: explicit field takes precedence,
            # then fall back to legacy name matching for backward compat.
            effective_format = inp.format
            if effective_format is None and inp.name in self._LEGACY_DIFF_NAMES:
                effective_format = "diff"

            # Handle large/formatted content inputs separately
            if effective_format in ("diff", "code") and resolved_value:
                large_content_inputs.append(
                    (inp.name, resolved_value, effective_format)
                )
                label = "Code Changes" if effective_format == "diff" else inp.name
                lines.append(f"- **{inp.name}**: (see {label} below){required}{desc}")
            elif len(str(value)) > 200:
                large_content_inputs.append((inp.name, value, effective_format))
                lines.append(f"- **{inp.name}**: (see below){required}{desc}")
            else:
                lines.append(f"- **{inp.name}**: `{value}`{required}{desc}")

        # Add large content sections
        for name, content, fmt in large_content_inputs:
            if fmt == "diff":
                lines.append(f"\n### Code Changes\n\n~~~diff\n{content}\n~~~")
            elif fmt == "code":
                lines.append(f"\n### {name}\n\n~~~\n{content}\n~~~")
            else:
                lines.append(f"\n### {name}\n\n{content}")

        return "\n".join(lines)

    def _build_outputs_section(self) -> str:
        """Build the expected outputs section."""
        if not self.task.outputs:
            return ""

        lines = ["## Expected Output Sections\n"]
        for out in self.task.outputs:
            required = " (required)" if out.required else " (optional)"
            format_hint = f" [{out.format}]" if out.format else ""
            desc = f": {out.description}" if out.description else ""
            lines.append(f"- **{out.name}**{required}{format_hint}{desc}")

        return "\n".join(lines)

    def _build_tools_section(self, mcp_tools: List[Dict[str, Any]]) -> str:
        """Build the available tools section with schemas."""
        if not mcp_tools:
            return "## Available Tools\n\nNo MCP tools available."

        # Filter tools by server if specified
        if self.task.tools.servers:
            filtered_tools = [
                t for t in mcp_tools if t.get("server") in self.task.tools.servers
            ]
        else:
            filtered_tools = mcp_tools

        # Filter out forbidden tools
        if self.task.tools.forbidden_tools:
            filtered_tools = [
                t
                for t in filtered_tools
                if t.get("name") not in self.task.tools.forbidden_tools
            ]

        if not filtered_tools:
            return "## Available Tools\n\nNo matching MCP tools available."

        lines = ["## Available Tools\n"]
        for tool in filtered_tools:
            tool_name = tool.get("name", "unknown")
            tool_desc = tool.get("description", "No description")
            server = tool.get("server", "")
            server_note = f" (from {server})" if server else ""

            lines.append(f"### {tool_name}{server_note}")
            lines.append(f"{tool_desc}\n")

            # Add schema information
            if "inputSchema" in tool and tool["inputSchema"]:
                schema = tool["inputSchema"]
                if "properties" in schema:
                    required_params = schema.get("required", [])
                    lines.append("**Parameters:**")
                    for param_name, param_info in schema["properties"].items():
                        param_type = param_info.get("type", "unknown")
                        param_desc = param_info.get("description", "")
                        is_required = param_name in required_params
                        req_text = " (required)" if is_required else " (optional)"
                        lines.append(
                            f"- `{param_name}` ({param_type}){req_text}: {param_desc}"
                        )
                    lines.append("")

        # Note required tools
        if self.task.tools.required_tools:
            lines.append(
                f"\n**Required tools (must use):** {', '.join(self.task.tools.required_tools)}"
            )

        return "\n".join(lines)

    def _build_constraints_section(self) -> str:
        """Build the constraints/rules section."""
        if not self.task.constraints:
            return ""

        lines = ["## Constraints\n"]
        for i, constraint in enumerate(self.task.constraints, 1):
            lines.append(f"{i}. {constraint}")

        return "\n".join(lines)

    def _build_context_section(self, context: Dict[str, Any]) -> str:
        """Build the runtime context section."""
        if not context:
            return ""

        lines = ["## Execution Context\n"]

        # Add project info if available
        project = context.get("project", {})
        if project:
            lines.append(f"- **Project**: {project.get('name', 'Unknown')}")
            if project.get("id"):
                lines.append(f"- **Project ID**: {project['id']}")

        # Add timestamp
        if context.get("timestamp"):
            lines.append(f"- **Timestamp**: {context['timestamp']}")

        # Add scope and task type
        if context.get("scope"):
            lines.append(f"- **Scope**: {context['scope']}")
        if context.get("task_type"):
            lines.append(f"- **Task Type**: {context['task_type']}")

        # Add analysis type if present
        if context.get("analysis_type"):
            lines.append(f"- **Analysis Type**: {context['analysis_type']}")

        return "\n".join(lines)

    def _build_examples_section(self) -> str:
        """Build the examples section for few-shot learning."""
        if not self.task.examples:
            return ""

        lines = ["## Examples\n"]
        for i, example in enumerate(self.task.examples, 1):
            lines.append(f"### Example {i}")
            if "input" in example:
                lines.append(f"**Input:** {example['input']}")
            if "output" in example:
                lines.append(f"**Output:** {example['output']}")
            lines.append("")

        return "\n".join(lines)

    def _build_reasoning_section(self) -> str:
        """Build the reasoning strategy section."""
        template = self.REASONING_TEMPLATES.get(
            self.task.reasoning, self.REASONING_TEMPLATES["simple"]
        )
        return f"## Approach\n\n{template}"

    def _build_format_section(self) -> str:
        """Build the output format section."""
        format_instructions = {
            "markdown": "Format your response in Markdown.",
            "html": "Format your response in HTML. Return only the HTML content without doctype or html/head/body tags.",
            "json": "Format your response as valid JSON.",
        }

        instruction = format_instructions.get(
            self.task.output_format, format_instructions["markdown"]
        )

        return f"## Output Format\n\n{instruction}"

    def get_input_values(self) -> Dict[str, str]:
        """Get resolved input values as a dictionary.

        Returns:
            Dictionary mapping input names to their string values.
        """
        return {name: value or "" for name, value in self._base_resolved.items()}

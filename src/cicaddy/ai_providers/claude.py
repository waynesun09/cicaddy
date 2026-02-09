"""Anthropic Claude provider adapter following Llama Stack patterns."""

import json
import os
from typing import Any, Dict, List, Optional

try:
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from cicaddy.utils.logger import get_logger
from cicaddy.utils.tool_converter import convert_mcp_tools_to_claude

from .base import BaseProvider, ProviderMessage, ProviderResponse, StopReason

logger = get_logger(__name__)


class ClaudeProvider(BaseProvider):
    """Anthropic Claude provider adapter."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client: Optional[Any] = None

        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not available")

    async def initialize(self) -> None:
        """Initialize Anthropic connection."""
        api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = AsyncAnthropic(api_key=api_key)

        logger.info(f"Initialized Claude provider with model: {self.model_id}")

    async def chat_completion(
        self,
        messages: List[ProviderMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ProviderResponse:
        """Generate chat completion using Claude."""
        if not self.client:
            raise RuntimeError("Provider not initialized")

        # Convert messages to Claude format
        claude_messages: List[Dict[str, str]] = []
        system_message: Optional[str] = None

        for msg in messages:
            if msg.role == "system":
                # Claude expects system message separately
                system_message = msg.content
            else:
                claude_messages.append({"role": msg.role, "content": msg.content})

        # Build prompt for token validation
        prompt_parts: List[str] = []
        if system_message:
            prompt_parts.append(f"System: {system_message}")
        for claude_msg in claude_messages:
            prompt_parts.append(f"{claude_msg['role']}: {claude_msg['content']}")
        prompt = "\n".join(prompt_parts)

        # Validate and truncate if necessary
        validated_prompt, was_truncated = self._validate_token_limits(prompt, "claude")
        if was_truncated:
            # Rebuild messages from truncated prompt
            claude_messages, system_message = self._rebuild_claude_messages_from_prompt(
                validated_prompt
            )
            logger.info("Prompt was truncated due to Claude token limits")

        try:
            # Calculate dynamic max_tokens based on prompt length
            prompt_tokens = len(validated_prompt) // 4  # Rough estimate
            max_tokens = self._get_dynamic_max_tokens("claude", prompt_tokens)

            # Prepare request parameters
            request_params = {
                "model": self.model_id,
                "messages": claude_messages,
                "max_tokens": max_tokens,
                "temperature": float(self.config.get("temperature", 0.0)),
            }

            # Add system message if present
            if system_message:
                request_params["system"] = system_message

            # Add tools if provided
            if tools:
                claude_tools = convert_mcp_tools_to_claude(tools)
                if claude_tools:
                    request_params["tools"] = claude_tools
                    logger.debug(f"Converted {len(claude_tools)} tools for Claude API")

            # Generate response
            response = await self.client.messages.create(**request_params)

            # Extract content
            content = ""
            tool_calls = []

            for content_block in response.content:
                if content_block.type == "text":
                    content += content_block.text
                elif content_block.type == "tool_use":
                    tool_calls.append(
                        {
                            "id": content_block.id,
                            "function": {
                                "name": content_block.name,
                                "arguments": json.dumps(content_block.input),
                            },
                        }
                    )

            # Convert Claude's stop reason to our standard format
            stop_reason = self._convert_stop_reason(response.stop_reason)

            # Handle max_tokens responses
            if response.stop_reason == "max_tokens":
                content = self._handle_max_tokens_response(content, tool_calls, prompt)
                logger.warning("Claude response was truncated due to max tokens")

            return ProviderResponse(
                content=content,
                model=response.model,
                stop_reason=stop_reason,
                usage=self._extract_usage(response),
                tool_calls=tool_calls,
            )

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise

    def _rebuild_claude_messages_from_prompt(
        self, prompt: str
    ) -> tuple[List[Dict[str, str]], Optional[str]]:
        """Rebuild Claude messages from truncated prompt."""
        messages = []
        system_message = None
        current_role = None
        current_content = []

        for line in prompt.split("\n"):
            if (
                line.startswith("system:")
                or line.startswith("user:")
                or line.startswith("assistant:")
            ):
                # Save previous message
                if current_role and current_content:
                    if current_role == "system":
                        system_message = "\n".join(current_content).strip()
                    else:
                        messages.append(
                            {
                                "role": current_role,
                                "content": "\n".join(current_content).strip(),
                            }
                        )

                # Start new message
                parts = line.split(":", 1)
                current_role = parts[0].strip().lower()
                current_content = [parts[1].strip()] if len(parts) > 1 else []
            elif current_role:
                current_content.append(line)

        # Add final message
        if current_role and current_content:
            if current_role == "system":
                system_message = "\n".join(current_content).strip()
            else:
                messages.append(
                    {
                        "role": current_role,
                        "content": "\n".join(current_content).strip(),
                    }
                )

        return messages, system_message

    def _extract_usage(self, response) -> Dict[str, Any]:
        """Extract usage statistics from Claude response."""
        if hasattr(response, "usage") and response.usage:
            return {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens
                + response.usage.output_tokens,
            }
        return {}

    def _convert_stop_reason(self, claude_stop_reason: str) -> StopReason:
        """Convert Claude stop reason to standard StopReason enum."""
        stop_reason_mapping = {
            "end_turn": StopReason.end_of_turn,
            "max_tokens": StopReason.out_of_tokens,
            "stop_sequence": StopReason.end_of_turn,
            "tool_use": StopReason.end_of_message,
        }
        return stop_reason_mapping.get(claude_stop_reason, StopReason.end_of_turn)

    async def shutdown(self) -> None:
        """Cleanup Claude resources."""
        # Anthropic client handles cleanup automatically
        self.client = None
        logger.info("Claude provider shutdown completed")

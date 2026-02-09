"""OpenAI provider adapter following Llama Stack patterns."""

import os
from typing import Any, Dict, List, Optional

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from cicaddy.utils.logger import get_logger
from cicaddy.utils.tool_converter import convert_mcp_tools_to_openai

from .base import BaseProvider, ProviderMessage, ProviderResponse

logger = get_logger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI provider adapter."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client: Optional[Any] = None

        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not available")

    async def initialize(self) -> None:
        """Initialize OpenAI connection."""
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = AsyncOpenAI(api_key=api_key, base_url=self.config.get("base_url"))

        logger.info(f"Initialized OpenAI provider with model: {self.model_id}")

    async def chat_completion(
        self,
        messages: List[ProviderMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ProviderResponse:
        """Generate chat completion using OpenAI."""
        if not self.client:
            raise RuntimeError("Provider not initialized")

        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # Build prompt for token validation
        prompt = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in openai_messages]
        )

        # Validate and truncate if necessary
        validated_prompt, was_truncated = self._validate_token_limits(prompt, "openai")
        if was_truncated:
            # Rebuild messages from truncated prompt
            openai_messages = self._rebuild_messages_from_prompt(validated_prompt)
            logger.info("Prompt was truncated due to OpenAI token limits")

        try:
            # Calculate dynamic max_tokens
            prompt_tokens = len(validated_prompt) // 4  # Rough estimate
            max_tokens = self._get_dynamic_max_tokens("openai", prompt_tokens)

            # Prepare request parameters
            request_params = {
                "model": self.model_id,
                "messages": openai_messages,
                "temperature": float(self.config.get("temperature", 0.0)),
                "max_tokens": max_tokens,
            }

            # Add tools if provided
            if tools:
                request_params["tools"] = convert_mcp_tools_to_openai(tools)
                request_params["tool_choice"] = "auto"

            # Generate response
            response = await self.client.chat.completions.create(**request_params)

            # Safely extract content using base provider's robust method
            choice = response.choices[0]
            content = self._safe_extract_content(choice.message)

            # Extract finish reason and convert to stop reason
            finish_reason = getattr(choice, "finish_reason", None)
            stop_reason = self._convert_finish_reason_to_stop_reason(finish_reason)

            # Handle length/max_tokens responses
            if finish_reason == "length":
                content = self._handle_max_tokens_response(content, [], prompt)
                logger.warning("OpenAI response was truncated due to max tokens")

            # Extract tool calls if present
            tool_calls = []
            if choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]

            return ProviderResponse(
                content=content,
                model=response.model,
                stop_reason=stop_reason,
                usage=self._extract_usage(response),
                tool_calls=tool_calls,
            )

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _rebuild_messages_from_prompt(self, prompt: str) -> List[Dict[str, str]]:
        """Rebuild OpenAI messages from truncated prompt."""
        messages = []
        current_role = None
        current_content = []

        for line in prompt.split("\n"):
            if (
                line.startswith("user:")
                or line.startswith("assistant:")
                or line.startswith("system:")
            ):
                # Save previous message
                if current_role and current_content:
                    messages.append(
                        {
                            "role": current_role,
                            "content": "\n".join(current_content).strip(),
                        }
                    )

                # Start new message
                parts = line.split(":", 1)
                current_role = parts[0].strip()
                current_content = [parts[1].strip()] if len(parts) > 1 else []
            elif current_role:
                current_content.append(line)

        # Add final message
        if current_role and current_content:
            messages.append(
                {"role": current_role, "content": "\n".join(current_content).strip()}
            )

        return messages if messages else [{"role": "user", "content": prompt}]

    def _extract_usage(self, response) -> Dict[str, Any]:
        """Extract usage statistics from OpenAI response."""
        if hasattr(response, "usage") and response.usage:
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return {}

    async def shutdown(self) -> None:
        """Cleanup OpenAI resources."""
        # OpenAI client handles cleanup automatically
        self.client = None
        logger.info("OpenAI provider shutdown completed")

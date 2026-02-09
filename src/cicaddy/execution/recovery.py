"""Recovery mechanism for early break issues in AI agent execution.

Implements "Secondary AI Call" pattern inspired by Unifai's LLM-driven recovery:
1. Detect early stop conditions (premature completion, tool errors, etc.)
2. Build context from the event store
3. Make a secondary AI call with a focused recovery prompt
4. Inject the recovery response into conversation to continue execution

Based on real-world analysis of 12 jobs across 4 pipelines showing:
- AI_PREMATURE_COMPLETION: 25% (most common recoverable error)
- TIMEOUT/CONTINUED: 42% (often legitimate, may not need recovery)
- TOOL_ERROR: 8% (recoverable with alternative approaches)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from cicaddy.ai_providers.base import BaseProvider, ProviderMessage
from cicaddy.execution.error_classifier import ClassifiedError, ErrorType
from cicaddy.execution.event_log import EventLog
from cicaddy.execution.steps import InferenceStep, StepType
from cicaddy.utils.logger import get_logger

# TYPE_CHECKING import to avoid circular dependency
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cicaddy.execution.knowledge_store import AccumulatedKnowledge

logger = get_logger(__name__)


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""

    should_continue: bool
    reason: str
    recovery_message: Optional[str] = None
    strategy_used: Optional[str] = None
    require_synthesis: bool = False  # If True, trigger dedicated synthesis inference
    recovery_tool_calls: Optional[List[Dict[str, Any]]] = (
        None  # Tool calls from recovery AI
    )


# Scenario-specific recovery prompts for each error type
# These prompts are designed to provide focused guidance for recovery
RECOVERY_PROMPTS: Dict[ErrorType, str] = {
    ErrorType.INVALID_TOOL_CALL: """⚠️ TOOL CALL ERROR - RECOVERY NEEDED

The previous tool call failed because it used unsupported parameters.

**Failed Tool Call:**
- Tool: {tool_name} (server: {tool_server})
- Arguments: {arguments}
- Error: {error_message}

**What Happened:**
The {tool_server} MCP server does not support the operation you attempted.
Common issues:
- Using update/write operations on read-only tools
- Passing parameters the tool doesn't accept
- Incorrect parameter types or formats

**Execution Context:**
{event_history_summary}

**Original Task:**
{original_prompt}

**Your Task:**
1. Analyze what went wrong
2. Determine an alternative approach to accomplish the original goal
3. Use only READ operations on {tool_server} tools if available
4. If no alternative exists, explain what data you CAN provide

**Important:** Focus on completing the original task using valid tool calls only.
""",
    ErrorType.AI_INFERENCE_FAILURE: """⚠️ AI INFERENCE ERROR - RECOVERY NEEDED

The AI model returned an empty or invalid response (possibly due to token budget exhaustion).

**Error Context:**
- Iteration: {iteration}
- Error: {error_message}

**Execution Progress So Far:**
{event_history_summary}

**Original Task:**
{original_prompt}

**Your Task:**
1. Review what has been accomplished so far
2. Determine the next logical step to continue the analysis
3. Provide clear, specific guidance on what the main AI should do next

**IMPORTANT:**
- Do NOT make tool calls yourself - your role is to guide the main AI
- Provide TEXT GUIDANCE ONLY explaining what tools to call and with what parameters
- The main AI will execute the tool calls based on your guidance
- Be concrete about: which tools to call (by exact name), what arguments to pass, expected next steps

**Expected Response Format:**
Provide a clear message that will be shown to the main AI to guide it. Example:

"To continue this task, you need to:
1. Call [tool_name] with parameters: {{param1: value1}}
2. Analyze the results
3. Complete the remaining analysis

Please proceed with these tool calls now."
""",
    ErrorType.TOOL_EXECUTION_ERROR: """⚠️ TOOL EXECUTION ERROR - RECOVERY NEEDED

A tool execution failed during processing.

**Failed Tool:**
- Tool: {tool_name} (server: {tool_server})
- Arguments: {arguments}
- Error: {error_message}

**Execution Context:**
{event_history_summary}

**Original Task:**
{original_prompt}

**Recovery Options:**
1. Try a different tool that provides similar data
2. Proceed with partial data already collected
3. Acknowledge the limitation and provide best-effort analysis

**Your Task:**
Determine the best recovery strategy and continue the analysis.
""",
    ErrorType.REPEATED_FAILURE: """⚠️ REPEATED FAILURE DETECTED - STRATEGY CHANGE NEEDED

The same error has occurred {retry_count} times.

**Repeated Error:**
- Type: {original_error_type}
- Error: {error_message}

**Failed Approaches:**
{failed_approaches}

**Original Task:**
{original_prompt}

**Your Task:**
1. The previous approaches have failed repeatedly
2. You MUST try a completely different strategy
3. Consider alternative data sources or simplified analysis
4. If no alternatives exist, provide best-effort results with limitations noted
""",
    # MOST COMMON CASE - AI says "I will continue" but makes no tool calls
    # OR gives ultra-short response and stops silently
    ErrorType.AI_PREMATURE_COMPLETION: """⚠️ INCOMPLETE EXECUTION - CONTINUATION REQUIRED

The execution stopped prematurely at iteration {iteration} with an incomplete response.

**Your Last Response:**
"{last_response_preview}"

**Problem: No Tool Calls Were Made!**

**What Has Been Accomplished So Far:**
{event_history_summary}

**Original Task Requirements:**
{original_prompt}

**Why This Is Incomplete:**
- The task is NOT complete - you've provided minimal output
- The original task requires comprehensive work with multiple tool calls
- You stopped after minimal exploration without completing the required work
- No tool calls were made to gather the data or perform the necessary operations

**Critical Instructions:**
1. Review the original task requirements carefully
2. The task likely requires:
   - Multiple tool calls to gather information or perform operations
   - Analysis of data, patterns, and results
   - Comprehensive output addressing all task requirements
3. The main AI needs clear, specific guidance on what to do next
4. Your role is to provide that guidance, NOT to execute the operations yourself

**Available Tools:**
Tools are available to:
- Gather required data or information
- Perform necessary operations
- Process and analyze results
- Complete all aspects of the original task

**IMPORTANT:** Provide clear, specific guidance on what tool calls the main AI should make next.
Be concrete about:
- Which tools to call (by exact name)
- What arguments to pass
- What data to analyze
- Expected next steps after getting results

Do NOT make tool calls yourself - your role is to guide the main AI to make the correct tool calls
in the next iteration.

**Expected Response Format:**
Provide a clear message that will be shown to the main AI to guide it. Example:

"To complete this task, you need to:
1. Call [tool_name] with parameters: {{param1: value1, param2: value2}}
2. Call [another_tool] with parameters: {{param: value}}
3. Analyze the results from those tool calls
4. Provide comprehensive output addressing the original requirements

Please proceed with these tool calls now."
""",
    # Token budget exhausted - fresh context recovery (allows continued work)
    ErrorType.MAX_TOKENS_EXCEEDED: """# ⚠️ TOKEN LIMIT REACHED - FRESH CONTEXT RECOVERY

## Context
You reached the token limit. You now have a **fresh context** with accumulated tool results.

## Original Task
{original_prompt}

## Accumulated Tool Results ({tool_count} tools executed)
Previous work is preserved - review before making new calls:

{tool_calls_summary}

## Your Options
1. **If critical data is missing:** Make targeted tool calls for specific missing data
2. **If you have enough data:** Produce your response directly

## ⛔ CRITICAL CONSTRAINTS
- **STAY ON TASK** - Continue the SAME analysis from the original task, do NOT pivot to different analysis
- Do NOT repeat tool calls that already succeeded above
- Do NOT start the task from scratch or switch to a different report/analysis type
- Do NOT make broad exploratory calls unrelated to the original task

## ✅ Required Behavior
- Review accumulated results first
- Only call tools for genuinely missing critical data **for the original task**
- If data is empty/missing, produce a report stating "No data available for the specified criteria"
- **PRESERVE OUTPUT FORMAT** - If the original task specifies HTML/charts/specific format, produce that exact format
- Work efficiently - a final synthesis step will follow if needed

---
**Continue your work with the context above. Stay on the original task.**
""",
    # Iteration limit reached - fresh context recovery with remaining token budget
    ErrorType.MAX_ITERATIONS_EXCEEDED: """# ⚠️ ITERATION LIMIT REACHED - FRESH CONTEXT RECOVERY

## Context
You reached the iteration limit ({iterations_completed} iterations). You have {remaining_tokens:,} tokens
remaining and a **fresh context** with accumulated tool results.

## Original Task
{original_prompt}

## Accumulated Tool Results ({tool_count} tools executed)
Previous work is preserved - review before making new calls:

{tool_calls_summary}

## Your Options
1. **If critical data is missing:** Make targeted tool calls (limited budget remaining)
2. **If you have enough data:** Produce your response directly

## ⛔ CRITICAL CONSTRAINTS
- **STAY ON TASK** - Continue the SAME analysis from the original task, do NOT pivot to different analysis
- Do NOT repeat tool calls that already succeeded above
- Do NOT start the task from scratch or switch to a different report/analysis type
- Do NOT call tools unrelated to the original task (e.g., if asked for coverage report, don't call PR retest tools)
- If original task requires specific output format (HTML, charts, etc.), you MUST produce that format

## ✅ Required Behavior
- Review accumulated results first
- Only call tools for genuinely missing critical data **for the original task**
- If data is empty/missing for the original task, produce a report with that finding (not a different analysis)
- **PRESERVE OUTPUT FORMAT** - If the original task specifies HTML with Chart.js/specific format, produce that exact format
- Work efficiently - a final synthesis step will follow if needed

---
**Continue your work with the context above. Stay on the original task and preserve the required output format.**
""",
}

# Dedicated synthesis prompt - used for final inference after recovery exhaustion
# This prompt focuses ONLY on synthesizing accumulated data into a final deliverable
SYNTHESIS_PROMPT = """# 📊 FINAL SYNTHESIS REQUIRED

You have exhausted the available {limit_type} after extensive data collection.
This is your **FINAL inference** - you MUST produce the complete deliverable now.

## Original Task
{original_prompt}

## Your Previous Work (Last 3 Responses)
Below is your recent work from the last 3 messages. Continue and complete this work:

{last_responses_preview}

## Accumulated Data ({tool_count} tool calls executed)
Below are ALL the results from your previous tool calls. This data represents
significant work and should be sufficient to address the task.

{tool_calls_summary}

## Your Mission
**SYNTHESIZE** all the above into a comprehensive final response that:
1. **CONTINUES** from your previous work shown above
2. Addresses every requirement in the original task
3. Uses ALL relevant data from the tool results
4. Structures the response clearly and professionally
5. Notes any limitations if critical data is missing
6. **USES THE EXACT OUTPUT FORMAT specified in the original task** (HTML, charts, etc.)

## ⛔ CRITICAL CONSTRAINTS
- **NO tool calls allowed** - tools are disabled for this synthesis
- This is your FINAL response - make it complete
- Do NOT request additional data or suggest further analysis
- Do NOT apologize for limitations - work with what you have
- **STAY ON TASK** - produce the deliverable requested in the original task
- **PRESERVE OUTPUT FORMAT** - If original task requires HTML/Chart.js/specific format, produce that exact format

## ✅ Expected Output
Produce a complete, well-structured deliverable that:
- Fully addresses the original task
- Continues from your previous work
- Uses the accumulated data above
- Matches the exact output format specified in the original task
- If data is empty/missing, include that finding in the report

---
**Begin your final synthesis now. Stay on task and preserve the required output format.**
"""

# Max size for message content in recovery context (~3000 tokens)
# Used to truncate large/problematic responses while preserving context
MAX_RECOVERY_MESSAGE_SIZE = 12000


class RecoveryManager:
    """
    Manages recovery from execution errors using secondary AI calls.

    Implements the "LLM-driven recovery" pattern:
    1. Classify the error type
    2. Build context from the event log
    3. Generate scenario-specific recovery prompt
    4. Make secondary AI call for recovery
    5. Return recovery instructions to continue execution
    """

    def __init__(
        self,
        ai_provider: BaseProvider,
        event_log: Optional[EventLog] = None,
        max_recovery_attempts: int = 3,
    ):
        """
        Initialize the recovery manager.

        Args:
            ai_provider: AI provider for making recovery calls
            event_log: Event log for extracting recovery context
            max_recovery_attempts: Maximum recovery attempts per error pattern
        """
        self.ai_provider = ai_provider
        self.event_log = event_log
        self.max_recovery_attempts = max_recovery_attempts

        # Track recovery attempts by error pattern to prevent infinite loops
        self._recovery_attempts: Dict[str, int] = {}

        # Track failed approaches for REPEATED_FAILURE handling
        self._failed_approaches: List[str] = []

    def _safe_truncate_for_recovery(
        self, content: str, max_size: int = MAX_RECOVERY_MESSAGE_SIZE
    ) -> str:
        """
        Truncate message content for inclusion in recovery context.

        Args:
            content: Message content to truncate
            max_size: Maximum size in characters (default: 12000 chars ~3000 tokens)

        Returns:
            Original content if within limit, otherwise truncated with marker
        """
        if not content or len(content) <= max_size:
            return content or ""
        return content[:max_size] + "\n... [truncated for recovery context]"

    def _format_last_messages_for_prompt(
        self,
        conversation_messages: Optional[List[ProviderMessage]],
    ) -> str:
        """
        Format last 3 messages as a string for inclusion in synthesis prompt.

        Reuses _safe_truncate_for_recovery for consistent truncation behavior
        with auto-recovery (12000 chars per message).

        Args:
            conversation_messages: Full conversation history

        Returns:
            Formatted string with last 3 messages for prompt inclusion
        """
        if not conversation_messages or len(conversation_messages) < 2:
            return "No previous responses available."

        # Get last 3 messages (same pattern as _prepare_context_messages_for_recovery)
        last_messages = conversation_messages[-3:]

        lines = []
        for i, msg in enumerate(last_messages, 1):
            role = msg.role.upper()
            # Reuse existing truncation method
            content = self._safe_truncate_for_recovery(msg.content)

            lines.append(f"### Message {i} ({role})")
            lines.append(content)
            lines.append("")  # blank line between messages

        return "\n".join(lines)

    def _prepare_context_messages_for_recovery(
        self,
        conversation_messages: List[ProviderMessage],
        recovery_message: ProviderMessage,
    ) -> List[ProviderMessage]:
        """
        Prepare context messages for recovery AI call.

        Unified pattern used by all recovery methods:
        - First message (original task)
        - Last 3 messages (recent context)
        - Recovery prompt

        Each message is truncated at MAX_RECOVERY_MESSAGE_SIZE to handle
        large/problematic responses while preserving context.

        Args:
            conversation_messages: Full conversation history
            recovery_message: The recovery prompt message

        Returns:
            List of ProviderMessages ready for AI call
        """
        context_messages = []

        if conversation_messages:
            # First message (original task) - truncated for safety
            first_msg = conversation_messages[0]
            context_messages.append(
                ProviderMessage(
                    role=first_msg.role,
                    content=self._safe_truncate_for_recovery(first_msg.content),
                )
            )

            # Last 3 messages (truncated) - recent context
            for msg in conversation_messages[-3:]:
                context_messages.append(
                    ProviderMessage(
                        role=msg.role,
                        content=self._safe_truncate_for_recovery(msg.content),
                    )
                )

        # Add recovery prompt at end
        context_messages.append(recovery_message)

        return context_messages

    async def attempt_recovery(
        self,
        error: ClassifiedError,
        original_prompt: str,
        conversation_messages: List[ProviderMessage],
        available_tools: Optional[List[Dict[str, Any]]] = None,
        recovery_steps_collector: Optional[List[InferenceStep]] = None,
        knowledge_store: Optional["AccumulatedKnowledge"] = None,
    ) -> RecoveryResult:
        """
        Attempt to recover from an execution error.

        Args:
            error: Classified error with context
            original_prompt: Original task prompt
            conversation_messages: Current conversation history
            available_tools: Available MCP tools (for recovery AI context)
            recovery_steps_collector: Optional list to collect recovery InferenceSteps
            knowledge_store: For MAX_TOKENS_EXCEEDED, provides tool call history

        Returns:
            RecoveryResult with recovery instructions or failure reason
        """
        # Special handling for MAX_TOKENS_EXCEEDED - use fresh context recovery
        if error.error_type == ErrorType.MAX_TOKENS_EXCEEDED:
            return await self._recover_from_token_limit(
                error,
                original_prompt,
                available_tools,
                recovery_steps_collector,
                knowledge_store,
                conversation_messages,
            )

        # Special handling for MAX_ITERATIONS_EXCEEDED - use fresh context recovery
        if error.error_type == ErrorType.MAX_ITERATIONS_EXCEEDED:
            return await self._recover_from_iteration_limit(
                error,
                original_prompt,
                available_tools,
                recovery_steps_collector,
                knowledge_store,
                conversation_messages,
            )

        error_key = error.get_error_key()

        # Check if we've exceeded retry limit for this error pattern
        current_attempts = self._recovery_attempts.get(error_key, 0)
        if current_attempts >= self.max_recovery_attempts:
            logger.warning(
                f"Max recovery attempts ({self.max_recovery_attempts}) exceeded for error pattern: {error_key}"
            )
            return RecoveryResult(
                should_continue=False,
                reason="max_retries_exceeded",
                strategy_used="abort",
            )

        # Increment attempt counter
        self._recovery_attempts[error_key] = current_attempts + 1
        attempt_num = self._recovery_attempts[error_key]

        logger.info(
            f"Attempting recovery for {error.error_type.value} "
            f"(attempt {attempt_num}/{self.max_recovery_attempts})"
        )

        # Check if this is now a repeated failure
        if attempt_num > 1:
            # Track the failed approach
            self._failed_approaches.append(
                f"Attempt {attempt_num - 1}: {error.error_type.value} - {error.message[:100]}"
            )

            # If we've failed multiple times with the same pattern, escalate to REPEATED_FAILURE
            if attempt_num >= 2:
                error = ClassifiedError(
                    error_type=ErrorType.REPEATED_FAILURE,
                    message=error.message,
                    tool_name=error.tool_name,
                    tool_server=error.tool_server,
                    arguments=error.arguments,
                    iteration=error.iteration,
                    retry_count=attempt_num,
                    last_response_preview=error.last_response_preview,
                )

        # Log recovery activation
        logger.info(
            f"🔄 Recovery mechanism activated for {error.error_type.value} "
            f"at iteration {error.iteration}"
        )

        # Build recovery context from event log
        context = self._build_recovery_context(error)

        # Get scenario-specific recovery prompt
        recovery_prompt = self._format_recovery_prompt(error, context, original_prompt)

        # Make secondary AI call for recovery
        logger.info(
            f"Making secondary AI call for recovery (error: {error.error_type.value})"
        )

        # For error types that should produce TEXT guidance only, don't pass tools
        # to prevent recovery AI from making tool calls instead of providing guidance
        text_only_error_types = {
            ErrorType.AI_INFERENCE_FAILURE,
            ErrorType.AI_PREMATURE_COMPLETION,
            ErrorType.REPEATED_FAILURE,
        }
        recovery_tools = (
            None if error.error_type in text_only_error_types else available_tools
        )

        try:
            recovery_response = await self._call_recovery_ai(
                recovery_prompt,
                conversation_messages,
                recovery_tools,
                recovery_steps_collector,
            )

            if recovery_response:
                logger.info(
                    f"Recovery successful for {error.error_type.value}, "
                    f"response length: {len(recovery_response)}"
                )
                return RecoveryResult(
                    should_continue=True,
                    reason="recovery_successful",
                    recovery_message=recovery_response,
                    strategy_used=error.error_type.value,
                )
            else:
                logger.warning("Recovery AI call returned empty response")
                return RecoveryResult(
                    should_continue=False,
                    reason="recovery_response_empty",
                    strategy_used=error.error_type.value,
                )

        except Exception as e:
            logger.error(f"Recovery AI call failed: {e}")
            return RecoveryResult(
                should_continue=False,
                reason=f"recovery_call_failed: {e}",
                strategy_used=error.error_type.value,
            )

    def _build_recovery_context(self, error: ClassifiedError) -> Dict[str, Any]:
        """
        Build recovery context from event log.

        Extracts relevant history including:
        - Successful tool calls and their results
        - Failed tool calls with errors
        - AI inference history

        Args:
            error: Classified error with context

        Returns:
            Dictionary with structured recovery context
        """
        if not self.event_log:
            return {
                "successful_tools": [],
                "failed_tools": [],
                "ai_responses": [],
                "total_events": 0,
            }

        # Get recent events from event log
        events = self.event_log.get_recent_events(count=20)

        successful_tools = []
        failed_tools = []
        ai_responses = []

        for event in events:
            event_type = event.get("event_type", "")

            if event_type == "tool_execution":
                tool_info = {
                    "tool": event.get("tool", "unknown"),
                    "server": event.get("server", "unknown"),
                    "arguments": event.get("arguments", {}),
                    "iteration": event.get("iteration", 0),
                }

                result = event.get("result", "")
                result_str = str(result).lower()

                # Check if this was a failed tool call
                if any(
                    keyword in result_str
                    for keyword in ["error", "failed", "exception", "timeout"]
                ):
                    tool_info["error"] = result[:500] if len(result) > 500 else result
                    failed_tools.append(tool_info)
                else:
                    # Success - include preview of result
                    result_preview = result[:200] if len(result) > 200 else result
                    tool_info["result_preview"] = result_preview
                    successful_tools.append(tool_info)

            elif event_type == "ai_inference":
                ai_responses.append(
                    {
                        "iteration": event.get("iteration", 0),
                        "response_preview": event.get("response", "")[:300],
                        "had_tool_calls": event.get("tool_calls_count", 0) > 0,
                    }
                )

        return {
            "successful_tools": successful_tools,
            "failed_tools": failed_tools,
            "ai_responses": ai_responses,
            "total_events": len(events),
        }

    def _format_event_history_summary(self, context: Dict[str, Any]) -> str:
        """
        Format recovery context as human-readable summary for prompt.

        Args:
            context: Recovery context from _build_recovery_context

        Returns:
            Formatted string for inclusion in recovery prompt
        """
        lines = []

        successful_tools = context.get("successful_tools", [])
        failed_tools = context.get("failed_tools", [])
        ai_responses = context.get("ai_responses", [])

        if successful_tools:
            lines.append("✅ Successful tool calls:")
            for tool in successful_tools[-5:]:  # Last 5 successful calls
                lines.append(f"  - {tool['tool']} ({tool['server']})")
                if tool.get("result_preview"):
                    preview = tool["result_preview"][:100]
                    lines.append(f"    Result: {preview}...")

        if failed_tools:
            lines.append("❌ Failed tool calls:")
            for tool in failed_tools[-3:]:  # Last 3 failed calls
                lines.append(
                    f"  - {tool['tool']}: {tool.get('error', 'Unknown error')[:100]}"
                )

        if ai_responses:
            # Count iterations with tool calls vs without
            with_tools = sum(1 for r in ai_responses if r.get("had_tool_calls"))
            without_tools = len(ai_responses) - with_tools
            lines.append(
                f"📝 AI iterations: {len(ai_responses)} total "
                f"({with_tools} with tool calls, {without_tools} without)"
            )

        if not lines:
            return "No execution history available."

        return "\n".join(lines)

    def _format_recovery_prompt(
        self,
        error: ClassifiedError,
        context: Dict[str, Any],
        original_prompt: str,
    ) -> str:
        """
        Generate scenario-specific recovery prompt.

        Args:
            error: Classified error with context
            context: Recovery context from event log
            original_prompt: Original task prompt

        Returns:
            Formatted recovery prompt string
        """
        # Get template for this error type
        template = RECOVERY_PROMPTS.get(error.error_type)
        if not template:
            # Fallback to generic prompt
            template = RECOVERY_PROMPTS[ErrorType.AI_INFERENCE_FAILURE]

        # Format event history summary
        event_history_summary = self._format_event_history_summary(context)

        # Format failed approaches for REPEATED_FAILURE
        failed_approaches = "\n".join(
            f"  - {approach}" for approach in self._failed_approaches[-3:]
        )
        if not failed_approaches:
            failed_approaches = "  - No previous approaches recorded"

        # Note: original_prompt is NOT truncated here - context is managed by
        # _prepare_context_messages_for_recovery() which handles message truncation

        # Truncate last response preview if too long
        last_response_preview = error.last_response_preview or ""
        if len(last_response_preview) > 500:
            last_response_preview = last_response_preview[:500] + "..."

        # Format the template with all available context
        try:
            formatted_prompt = template.format(
                tool_name=error.tool_name or "unknown",
                tool_server=error.tool_server or "unknown",
                arguments=str(error.arguments)[:300] if error.arguments else "{}",
                error_message=error.message[:500] if error.message else "Unknown error",
                iteration=error.iteration,
                retry_count=error.retry_count,
                event_history_summary=event_history_summary,
                original_prompt=original_prompt,
                last_response_preview=last_response_preview,
                original_error_type=error.error_type.value,
                failed_approaches=failed_approaches,
            )
        except KeyError as e:
            # If template has missing placeholders, use simpler format
            logger.warning(f"Recovery prompt template missing key: {e}")
            formatted_prompt = f"""⚠️ RECOVERY NEEDED

Error: {error.message}
Iteration: {error.iteration}

Context:
{event_history_summary}

Original Task:
{original_prompt}

Please continue the analysis by making the necessary tool calls.
"""

        return formatted_prompt

    async def _call_recovery_ai(
        self,
        recovery_prompt: str,
        conversation_messages: List[ProviderMessage],
        available_tools: Optional[List[Dict[str, Any]]] = None,
        recovery_steps_collector: Optional[List[InferenceStep]] = None,
    ) -> Optional[str]:
        """
        Make secondary AI call for recovery.

        Uses enhanced context including original task and available tools.

        Args:
            recovery_prompt: Formatted recovery prompt
            conversation_messages: Current conversation history
            available_tools: Available MCP tools (for recovery AI context)
            recovery_steps_collector: Optional list to collect recovery InferenceSteps

        Returns:
            Recovery response text or None if failed
        """
        try:
            # Build recovery message
            recovery_message = ProviderMessage(
                role="user",
                content=recovery_prompt,
            )

            # Use unified context pattern: first + last 3 messages + recovery prompt
            # Each message is truncated at MAX_RECOVERY_MESSAGE_SIZE for safety
            context_messages = self._prepare_context_messages_for_recovery(
                conversation_messages, recovery_message
            )

            # Provide tools to recovery AI for context-aware guidance
            response = await self.ai_provider.chat_completion(
                messages=context_messages,
                tools=available_tools,
            )

            # Create usage stats for logging and step tracking
            usage_stats = None
            if response:
                usage_stats = {
                    "total_tokens": getattr(response, "total_tokens", 0),
                    "prompt_tokens": getattr(response, "prompt_tokens", 0),
                    "completion_tokens": getattr(response, "completion_tokens", 0),
                }

            # Log recovery AI call to event log for inference_calls tracking
            # IMPORTANT: Use "ai_inference" event type so it's counted in HTML report
            if self.event_log and response:
                self.event_log.append_event(
                    "ai_inference",
                    {
                        "recovery_call": True,
                        "model_used": getattr(response, "model", None),
                        "usage_stats": usage_stats,
                        "input_tokens": usage_stats.get("prompt_tokens", 0)
                        if usage_stats
                        else 0,
                        "output_tokens": usage_stats.get("completion_tokens", 0)
                        if usage_stats
                        else 0,
                    },
                )

            # Create InferenceStep for recovery AI call and collect it
            # This ensures recovery calls are counted in the HTML report's inference count
            # The step will be added to the turn by the caller after turn creation
            if recovery_steps_collector is not None and response:
                # Use timestamp-based step_id since we don't have turn_id yet
                step_id = (
                    f"recovery_inference_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                )
                recovery_step = InferenceStep(
                    turn_id="",  # Will be set when added to turn
                    step_id=step_id,
                    step_type=StepType.inference,
                    started_at=datetime.now(),
                    model_response=response.content,
                    model_used=getattr(response, "model", None),
                    usage_stats=usage_stats,
                    completed_at=datetime.now(),
                )
                recovery_steps_collector.append(recovery_step)
                logger.debug(
                    f"Collected recovery InferenceStep {step_id}, "
                    f"total recovery steps: {len(recovery_steps_collector)}"
                )

            # Validate recovery response - should be text guidance, not tool calls
            if response and response.tool_calls:
                logger.warning(
                    f"Recovery AI made {len(response.tool_calls)} tool calls despite "
                    f"being instructed to provide text guidance only. This may indicate "
                    f"the recovery prompt needs refinement."
                )

            if response and response.content:
                return response.content

            return None

        except Exception as e:
            logger.error(f"Recovery AI call failed: {e}")
            return None

    async def _recover_from_token_limit(
        self,
        error: ClassifiedError,
        original_prompt: str,
        available_tools: Optional[List[Dict[str, Any]]],
        recovery_steps_collector: Optional[List[InferenceStep]],
        knowledge_store: Optional["AccumulatedKnowledge"],
        conversation_messages: Optional[List[ProviderMessage]] = None,
    ) -> RecoveryResult:
        """
        Handle MAX_TOKENS_EXCEEDED with fresh context recovery.

        This method:
        1. Uses unified context pattern (first + last 3 messages, truncated)
        2. Includes preprocessed tool calls from knowledge store
        3. Provides original prompt and available tools
        4. Guides AI to continue from accumulated knowledge
        """
        if not knowledge_store:
            logger.warning("MAX_TOKENS_EXCEEDED recovery requires knowledge_store")
            return RecoveryResult(
                should_continue=False,
                reason="no_knowledge_store",
                strategy_used="max_tokens_exceeded",
            )

        # Get minimal tool calls summary
        tool_calls_summary = knowledge_store.get_tool_calls_summary_for_prompt()

        # Format recovery prompt with full original prompt (no truncation)
        template = RECOVERY_PROMPTS[ErrorType.MAX_TOKENS_EXCEEDED]
        recovery_prompt = template.format(
            original_prompt=original_prompt,
            tool_calls_summary=tool_calls_summary,
            tool_count=knowledge_store.total_tools_executed,
        )

        logger.info(
            f"MAX_TOKENS_EXCEEDED recovery: fresh context with "
            f"{knowledge_store.total_tools_executed} tool results"
        )

        try:
            # Build recovery message
            recovery_message = ProviderMessage(
                role="user",
                content=recovery_prompt,
            )

            # Use unified context pattern: first + last 3 messages + recovery prompt
            # Each message is truncated at MAX_RECOVERY_MESSAGE_SIZE for safety
            context_messages = self._prepare_context_messages_for_recovery(
                conversation_messages or [], recovery_message
            )

            # Make recovery AI call with tools (so it can continue execution)
            response = await self.ai_provider.chat_completion(
                messages=context_messages,
                tools=available_tools,
            )

            # Track in event log
            usage_stats = None
            if self.event_log and response:
                usage_stats = {
                    "total_tokens": getattr(response, "total_tokens", 0),
                    "prompt_tokens": getattr(response, "prompt_tokens", 0),
                    "completion_tokens": getattr(response, "completion_tokens", 0),
                }
                # IMPORTANT: Use "ai_inference" event type so it's counted in HTML report
                self.event_log.append_event(
                    "ai_inference",
                    {
                        "recovery_call": True,
                        "recovery_type": "max_tokens_exceeded_fresh_context",
                        "model_used": getattr(response, "model", None),
                        "usage_stats": usage_stats,
                        "tool_results_included": knowledge_store.total_tools_executed,
                        "input_tokens": usage_stats.get("prompt_tokens", 0)
                        if usage_stats
                        else 0,
                        "output_tokens": usage_stats.get("completion_tokens", 0)
                        if usage_stats
                        else 0,
                    },
                )

            # Create InferenceStep for tracking
            if recovery_steps_collector is not None and response:
                step_id = (
                    f"recovery_max_tokens_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                )
                recovery_step = InferenceStep(
                    turn_id="",
                    step_id=step_id,
                    step_type=StepType.inference,
                    started_at=datetime.now(),
                    model_response=response.content,
                    model_used=getattr(response, "model", None),
                    usage_stats=usage_stats,
                    completed_at=datetime.now(),
                )
                recovery_steps_collector.append(recovery_step)
                logger.debug(
                    f"Collected recovery InferenceStep {step_id} for MAX_TOKENS_EXCEEDED"
                )

            if response and (response.content or response.tool_calls):
                # CRITICAL: Pass recovery_prompt as fresh context, not response.content
                # When AI makes tool calls, response.content is empty, causing context loss.
                # The recovery_prompt contains original task + knowledge store summary.
                # Also capture tool_calls so executor can execute them.
                tool_calls = None
                if response.tool_calls:
                    # Tool calls are already dicts in OpenAI format:
                    # {"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
                    tool_calls = [
                        {
                            "name": tc.get("function", {}).get(
                                "name", tc.get("name", "")
                            ),
                            "arguments": tc.get("function", {}).get(
                                "arguments", tc.get("arguments", "{}")
                            ),
                        }
                        for tc in response.tool_calls
                    ]
                return RecoveryResult(
                    should_continue=True,
                    reason="fresh_context_recovery_successful",
                    recovery_message=recovery_prompt,  # Pass task context, not empty content
                    strategy_used="max_tokens_exceeded_fresh_context",
                    recovery_tool_calls=tool_calls,
                )
            else:
                return RecoveryResult(
                    should_continue=False,
                    reason="recovery_response_empty",
                    strategy_used="max_tokens_exceeded_fresh_context",
                )

        except Exception as e:
            logger.error(f"MAX_TOKENS_EXCEEDED recovery failed: {e}")
            return RecoveryResult(
                should_continue=False,
                reason=f"recovery_call_failed: {e}",
                strategy_used="max_tokens_exceeded_fresh_context",
            )

    async def _recover_from_iteration_limit(
        self,
        error: ClassifiedError,
        original_prompt: str,
        available_tools: Optional[List[Dict[str, Any]]],
        recovery_steps_collector: Optional[List[InferenceStep]],
        knowledge_store: Optional["AccumulatedKnowledge"],
        conversation_messages: Optional[List[ProviderMessage]] = None,
        iterations_completed: int = 0,
        remaining_tokens: int = 0,
    ) -> RecoveryResult:
        """
        Handle MAX_ITERATIONS_EXCEEDED with fresh context recovery.

        This method:
        1. Uses unified context pattern (first + last 3 messages, truncated)
        2. Includes preprocessed tool calls from knowledge store
        3. Provides original prompt and available tools
        4. Guides AI to continue from accumulated knowledge
        5. Is more conservative than token recovery (synthesis focus)
        """
        if not knowledge_store:
            logger.warning("MAX_ITERATIONS_EXCEEDED recovery requires knowledge_store")
            return RecoveryResult(
                should_continue=False,
                reason="no_knowledge_store",
                strategy_used="max_iterations_exceeded",
            )

        # Get minimal tool calls summary
        tool_calls_summary = knowledge_store.get_tool_calls_summary_for_prompt()

        # Format recovery prompt with full original prompt (no truncation)
        template = RECOVERY_PROMPTS[ErrorType.MAX_ITERATIONS_EXCEEDED]
        recovery_prompt = template.format(
            original_prompt=original_prompt,
            tool_calls_summary=tool_calls_summary,
            iterations_completed=iterations_completed or error.iteration,
            remaining_tokens=remaining_tokens,
            tool_count=knowledge_store.total_tools_executed,
        )

        logger.info(
            f"MAX_ITERATIONS_EXCEEDED recovery: fresh context with "
            f"{knowledge_store.total_tools_executed} tool results, "
            f"{remaining_tokens:,} tokens remaining"
        )

        try:
            # Build recovery message
            recovery_message = ProviderMessage(
                role="user",
                content=recovery_prompt,
            )

            # Use unified context pattern: first + last 3 messages + recovery prompt
            # Each message is truncated at MAX_RECOVERY_MESSAGE_SIZE for safety
            context_messages = self._prepare_context_messages_for_recovery(
                conversation_messages or [], recovery_message
            )

            # Make recovery AI call with tools (so it can continue execution)
            response = await self.ai_provider.chat_completion(
                messages=context_messages,
                tools=available_tools,
            )

            # Track in event log
            usage_stats = None
            if self.event_log and response:
                usage_stats = {
                    "total_tokens": getattr(response, "total_tokens", 0),
                    "prompt_tokens": getattr(response, "prompt_tokens", 0),
                    "completion_tokens": getattr(response, "completion_tokens", 0),
                }
                # IMPORTANT: Use "ai_inference" event type so it's counted in HTML report
                self.event_log.append_event(
                    "ai_inference",
                    {
                        "recovery_call": True,
                        "recovery_type": "max_iterations_exceeded_fresh_context",
                        "model_used": getattr(response, "model", None),
                        "usage_stats": usage_stats,
                        "tool_results_included": knowledge_store.total_tools_executed,
                        "iterations_completed": iterations_completed or error.iteration,
                        "remaining_tokens": remaining_tokens,
                        "input_tokens": usage_stats.get("prompt_tokens", 0)
                        if usage_stats
                        else 0,
                        "output_tokens": usage_stats.get("completion_tokens", 0)
                        if usage_stats
                        else 0,
                    },
                )

            # Create InferenceStep for tracking
            if recovery_steps_collector is not None and response:
                step_id = (
                    f"recovery_max_iters_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                )
                recovery_step = InferenceStep(
                    turn_id="",
                    step_id=step_id,
                    step_type=StepType.inference,
                    started_at=datetime.now(),
                    model_response=response.content,
                    model_used=getattr(response, "model", None),
                    usage_stats=usage_stats,
                    completed_at=datetime.now(),
                )
                recovery_steps_collector.append(recovery_step)
                logger.debug(
                    f"Collected recovery InferenceStep {step_id} for MAX_ITERATIONS_EXCEEDED"
                )

            if response and (response.content or response.tool_calls):
                # CRITICAL: Pass recovery_prompt as fresh context, not response.content
                # When AI makes tool calls, response.content is empty, causing context loss.
                # The recovery_prompt contains original task + knowledge store summary.
                # Also capture tool_calls so executor can execute them.
                tool_calls = None
                if response.tool_calls:
                    # Tool calls are already dicts in OpenAI format:
                    # {"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
                    tool_calls = [
                        {
                            "name": tc.get("function", {}).get(
                                "name", tc.get("name", "")
                            ),
                            "arguments": tc.get("function", {}).get(
                                "arguments", tc.get("arguments", "{}")
                            ),
                        }
                        for tc in response.tool_calls
                    ]
                return RecoveryResult(
                    should_continue=True,
                    reason="fresh_context_recovery_successful",
                    recovery_message=recovery_prompt,  # Pass task context, not empty content
                    strategy_used="max_iterations_exceeded_fresh_context",
                    recovery_tool_calls=tool_calls,
                )
            else:
                return RecoveryResult(
                    should_continue=False,
                    reason="recovery_response_empty",
                    strategy_used="max_iterations_exceeded_fresh_context",
                )

        except Exception as e:
            logger.error(f"MAX_ITERATIONS_EXCEEDED recovery failed: {e}")
            return RecoveryResult(
                should_continue=False,
                reason=f"recovery_call_failed: {e}",
                strategy_used="max_iterations_exceeded_fresh_context",
            )

    def reset_recovery_state(self) -> None:
        """Reset recovery state for a new execution session."""
        self._recovery_attempts.clear()
        self._failed_approaches.clear()
        logger.debug("Recovery state reset for new session")

    def get_recovery_stats(self) -> Dict[str, Any]:
        """
        Get statistics about recovery attempts.

        Returns:
            Dictionary with recovery statistics
        """
        total_attempts = sum(self._recovery_attempts.values())
        unique_patterns = len(self._recovery_attempts)

        return {
            "total_attempts": total_attempts,
            "unique_error_patterns": unique_patterns,
            "attempts_by_pattern": dict(self._recovery_attempts),
            "failed_approaches_count": len(self._failed_approaches),
        }

    async def perform_final_synthesis(
        self,
        original_prompt: str,
        knowledge_store: "AccumulatedKnowledge",
        limit_type: str,
        recovery_steps_collector: Optional[List[InferenceStep]] = None,
        conversation_messages: Optional[List[ProviderMessage]] = None,
    ) -> Optional[str]:
        """
        Perform dedicated synthesis inference to produce final deliverable.

        This method is called when recovery attempts are exhausted and the AI
        needs to synthesize accumulated data into a final response. Unlike
        recovery prompts, this uses tools=None to prevent further exploration.

        Args:
            original_prompt: Original task prompt
            knowledge_store: Accumulated tool results from all inferences
            limit_type: Type of limit reached ("token budget" or "iteration limit")
            recovery_steps_collector: Optional list to collect InferenceSteps
            conversation_messages: Conversation history for context (truncated)

        Returns:
            Synthesized response text or None if failed
        """
        if not knowledge_store or knowledge_store.total_tools_executed == 0:
            logger.warning("No tool results to synthesize")
            return None

        # Get tool calls summary
        tool_calls_summary = knowledge_store.get_tool_calls_summary_for_prompt()

        # Extract last 3 responses (truncated) from conversation
        # Reuses _safe_truncate_for_recovery internally for consistent behavior
        last_responses_preview = self._format_last_messages_for_prompt(
            conversation_messages
        )

        # Format synthesis prompt with full context
        synthesis_prompt = SYNTHESIS_PROMPT.format(
            limit_type=limit_type,
            original_prompt=original_prompt,
            last_responses_preview=last_responses_preview,
            tool_count=knowledge_store.total_tools_executed,
            tool_calls_summary=tool_calls_summary,
        )

        logger.info(
            f"Performing final synthesis with {knowledge_store.total_tools_executed} "
            f"tool results (limit: {limit_type})"
        )

        try:
            # Create synthesis message
            synthesis_message = ProviderMessage(
                role="user",
                content=synthesis_prompt,
            )

            # Use unified context pattern: first + last 3 messages + synthesis prompt
            # Each message is truncated at MAX_RECOVERY_MESSAGE_SIZE for safety
            context_messages = self._prepare_context_messages_for_recovery(
                conversation_messages or [], synthesis_message
            )

            # Make synthesis AI call WITHOUT tools to prevent further exploration
            response = await self.ai_provider.chat_completion(
                messages=context_messages,
                tools=None,  # Critical: No tools = pure synthesis
            )

            # Track in event log
            usage_stats = None
            if self.event_log and response:
                usage_stats = {
                    "total_tokens": getattr(response, "total_tokens", 0),
                    "prompt_tokens": getattr(response, "prompt_tokens", 0),
                    "completion_tokens": getattr(response, "completion_tokens", 0),
                }
                # IMPORTANT: Use "ai_inference" event type so it's counted in HTML report
                self.event_log.append_event(
                    "ai_inference",
                    {
                        "synthesis_call": True,
                        "limit_type": limit_type,
                        "model_used": getattr(response, "model", None),
                        "usage_stats": usage_stats,
                        "tool_results_synthesized": knowledge_store.total_tools_executed,
                        "input_tokens": usage_stats.get("prompt_tokens", 0)
                        if usage_stats
                        else 0,
                        "output_tokens": usage_stats.get("completion_tokens", 0)
                        if usage_stats
                        else 0,
                    },
                )

            # Create InferenceStep for tracking
            if recovery_steps_collector is not None and response:
                step_id = f"synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                synthesis_step = InferenceStep(
                    turn_id="",
                    step_id=step_id,
                    step_type=StepType.inference,
                    started_at=datetime.now(),
                    model_response=response.content,
                    model_used=getattr(response, "model", None),
                    usage_stats=usage_stats,
                    completed_at=datetime.now(),
                )
                recovery_steps_collector.append(synthesis_step)
                logger.debug(f"Collected synthesis InferenceStep {step_id}")

            if response and response.content:
                logger.info(f"Final synthesis completed: {len(response.content)} chars")
                return response.content
            else:
                logger.warning("Synthesis AI call returned empty response")
                return None

        except Exception as e:
            logger.error(f"Final synthesis failed: {e}")
            return None

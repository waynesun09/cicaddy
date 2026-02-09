"""Multi-step execution engine for AI agent planning and execution."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from cicaddy.ai_providers.base import BaseProvider, ProviderMessage
from cicaddy.mcp_client.client import OfficialMCPClientManager
from cicaddy.tools import ToolRegistry
from cicaddy.utils.logger import get_logger

from .progressive_analyzer import ProgressiveAnalyzer
from .result_formatter import GenericResultFormatter
from .steps import InferenceStep, StepType, ToolExecutionStep
from .token_aware_executor import ExecutionLimits, TokenAwareExecutor
from .turn import Turn

logger = get_logger(__name__)


class ExecutionEngine:
    """
    Multi-step execution engine with LlamaStack-inspired token-aware patterns.

    Features:
    - Multi-level safety valves (iterations, tokens, results, time)
    - Progressive degradation with graceful fallback
    - Generic result formatting for any MCP server
    - Intelligent resource management
    - LlamaStack dual-limit approach (max_infer_iters + out_of_tokens)
    """

    def __init__(
        self,
        ai_provider: BaseProvider,
        mcp_manager: Optional[OfficialMCPClientManager] = None,
        local_tool_registry: Optional[ToolRegistry] = None,
        session_id: Optional[str] = None,
        execution_limits: Optional[ExecutionLimits] = None,
        context_safety_factor: float = 0.7,  # NEW: Configurable via CONTEXT_SAFETY_FACTOR env var
    ):
        self.ai_provider = ai_provider
        self.mcp_manager = mcp_manager
        self.local_tool_registry = local_tool_registry
        self.session_id = session_id or str(uuid.uuid4())
        self.turns: List[Turn] = []

        # Token-aware execution components
        self.execution_limits = execution_limits or ExecutionLimits()
        self.token_aware_executor = TokenAwareExecutor(
            ai_provider=ai_provider,
            mcp_manager=mcp_manager,
            local_tool_registry=local_tool_registry,
            limits=self.execution_limits,
            session_id=self.session_id,
            context_safety_factor=context_safety_factor,  # NEW: Pass from settings
        )
        self.progressive_analyzer = ProgressiveAnalyzer(
            ai_provider, self.execution_limits
        )
        # Use default formatter limits so integration tests see truncation at ~2KB
        self.result_formatter = GenericResultFormatter()

        logger.info("ExecutionEngine initialized with token-aware execution")

    async def execute_turn(
        self,
        messages: List[ProviderMessage],
        available_tools: Optional[List[Dict[str, Any]]] = None,
        max_infer_iters: Optional[int] = None,
    ) -> Turn:
        """
        Execute a complete turn with iterative planning and tool execution.

        Uses token-aware execution with LlamaStack patterns for intelligent resource management.

        Args:
            messages: Initial conversation messages
            available_tools: MCP tools available for execution
            max_infer_iters: Maximum AI planning iterations

        Returns:
            Turn object with execution results and comprehensive metadata

        Features:
            - Multi-level safety valves (iterations, tokens, results, time)
            - Progressive degradation when approaching limits
            - Generic MCP tool support (any server type)
            - Structured result formatting with BEGIN/END markers
            - Intelligent truncation and prioritization
        """
        logger.info(f"Starting token-aware execution for session {self.session_id}")

        # Update limits only if caller provided an override
        if (
            isinstance(max_infer_iters, int)
            and max_infer_iters > 0
            and max_infer_iters != self.execution_limits.max_infer_iters
        ):
            self.execution_limits.max_infer_iters = max_infer_iters
            logger.debug(
                f"Updated max_infer_iters to {self.execution_limits.max_infer_iters} for backward compatibility"
            )

        # Execute with token-aware executor
        execution_result = await self.token_aware_executor.execute_with_limits(
            messages=messages, available_tools=available_tools
        )

        # Create Turn object from token-aware execution results
        turn_id = str(uuid.uuid4())
        turn = Turn(
            turn_id=turn_id, session_id=self.session_id, input_messages=messages
        )

        # Convert tool results to execution steps
        tool_results = execution_result.get("tool_results", [])
        if tool_results:
            # Apply progressive analysis to tool results
            analysis_level = self.progressive_analyzer.determine_analysis_level(
                self.token_aware_executor.state
            )

            processed_results, analysis_summary = (
                self.progressive_analyzer.apply_degradation_strategy(
                    tool_results,
                    analysis_level,
                    str(messages[0].content if messages else ""),
                )
            )

            # Create tool execution steps
            for i, result in enumerate(processed_results):
                step = ToolExecutionStep(
                    turn_id=turn_id,
                    step_id=f"tool_{i}",
                    step_type=StepType.tool_execution,
                    started_at=datetime.now(),
                    tool_name=result.get("tool_name", "unknown"),
                    tool_server=result.get("tool_server", ""),
                    tool_arguments=result.get("arguments", {}),
                    tool_response=result.get("result", ""),
                    completed_at=datetime.now(),
                )
                turn.add_step(step)

        # Create final inference step with the complete response and ACCURATE token usage
        final_response = execution_result.get("final_response", "Analysis completed")

        # Extract accurate token usage data from EventLog
        # EventLog tracks exact input_tokens and output_tokens per AI inference iteration
        # We aggregate ALL ai_inference events across all iterations into one InferenceStep
        usage_stats = None
        model_used = None

        if self.token_aware_executor.event_log:
            try:
                # Get all events from EventLog to aggregate token data
                event_count = self.token_aware_executor.event_log.get_event_count()
                all_events = self.token_aware_executor.event_log.get_recent_events(
                    count=event_count
                )

                # Filter for ai_inference events and aggregate token data
                total_input_tokens = 0
                total_output_tokens = 0
                inference_count = 0

                for event in all_events:
                    if event.get("event_type") == "ai_inference":
                        total_input_tokens += event.get("input_tokens", 0)
                        total_output_tokens += event.get("output_tokens", 0)
                        inference_count += 1

                # Only create usage_stats if we have actual AI inference data
                if inference_count > 0:
                    usage_stats = {
                        "total_tokens": total_input_tokens + total_output_tokens,
                        "prompt_tokens": total_input_tokens,
                        "completion_tokens": total_output_tokens,
                    }
                    logger.debug(
                        f"Aggregated token usage from {inference_count} AI inferences: "
                        f"{usage_stats['total_tokens']} total ({total_input_tokens} input + {total_output_tokens} output)"
                    )

            except Exception as e:
                logger.warning(f"Failed to extract token usage from EventLog: {e}")
                # usage_stats remains None, HTML report will show no token data

        # Get model information from AI provider settings
        model_used = None
        if self.token_aware_executor.ai_provider and hasattr(
            self.token_aware_executor.ai_provider, "settings"
        ):
            try:
                model_used = self.token_aware_executor.ai_provider.settings.ai_model
            except AttributeError:
                logger.warning(
                    "AI provider settings exist but 'ai_model' attribute is missing"
                )
                model_used = None

        inference_step = InferenceStep(
            turn_id=turn_id,
            step_id="final_inference",
            step_type=StepType.inference,
            started_at=datetime.now(),
            model_response=final_response,
            model_used=model_used,
            usage_stats=usage_stats,
            completed_at=datetime.now(),
        )
        turn.add_step(inference_step)

        # Add recovery InferenceSteps collected during execution
        # These are created during recovery AI calls to ensure they're counted in inference metrics
        if self.token_aware_executor.recovery_steps:
            for recovery_step in self.token_aware_executor.recovery_steps:
                # Update turn_id now that we have it
                recovery_step.turn_id = turn_id
                turn.add_step(recovery_step)
            logger.debug(
                f"Added {len(self.token_aware_executor.recovery_steps)} recovery InferenceSteps to turn"
            )

        # Mark turn as completed with execution summary
        execution_summary = execution_result.get("execution_summary", {})
        enhanced_response = self._enhance_response_with_summary(
            final_response, execution_summary
        )
        # If an error stop reason occurred, reflect that in the output message
        if (
            execution_summary.get("stop_reason") == "error"
            and "failed" not in enhanced_response.lower()
        ):
            enhanced_response = (
                enhanced_response
                + "\n[EXECUTION ERROR: AI inference failed or returned no content]"
            )
        turn.execution_summary = execution_summary
        turn.mark_completed(enhanced_response)

        # Attach accumulated knowledge from token-aware executor
        # This ensures complete tool results are preserved for reports and notifications
        turn.accumulated_knowledge = self.token_aware_executor.knowledge_store

        self.turns.append(turn)
        logger.info(
            f"Token-aware turn {turn_id} completed with {len(turn.steps)} steps "
            f"and {turn.accumulated_knowledge.total_tools_executed if turn.accumulated_knowledge else 0} tools in knowledge store"
        )

        return turn

    def _enhance_response_with_summary(
        self, response: str, execution_summary: Dict[str, Any]
    ) -> str:
        """Enhance response with execution summary if degradation occurred."""
        if not execution_summary.get("degradation_active", False):
            return response

        # Add execution metadata for transparency
        summary_parts = [response]

        stop_reason = execution_summary.get("stop_reason")
        if stop_reason in ["max_iterations", "out_of_tokens"]:
            iterations = execution_summary.get("iterations", {})
            tokens = execution_summary.get("tokens", {})

            summary_parts.append("\n📊 Execution Summary:")
            summary_parts.append(
                f"• Iterations: {iterations.get('completed', 0)}/{iterations.get('max_allowed', 0)} "
                f"({iterations.get('utilization', 0):.1%} utilization)"
            )
            summary_parts.append(
                f"• Tokens: {tokens.get('total_used', 0)}/{tokens.get('max_allowed', 0)} "
                f"({tokens.get('utilization', 0):.1%} utilization)"
            )
            summary_parts.append(f"• Stop Reason: {stop_reason}")
            summary_parts.append(
                "ℹ️ Analysis optimized due to resource constraints. "
                "Increase limits for more comprehensive results."
            )

        return "\n".join(summary_parts)

    def get_turn_history(self) -> List[Dict[str, Any]]:
        """Get execution history for all turns."""
        return [turn.to_dict() for turn in self.turns]

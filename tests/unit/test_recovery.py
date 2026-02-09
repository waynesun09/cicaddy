"""Unit tests for Early Break Recovery mechanism."""

from unittest.mock import AsyncMock, Mock

import pytest

from cicaddy.ai_providers.base import ProviderMessage, ProviderResponse
from cicaddy.execution.error_classifier import (
    ClassifiedError,
    ErrorType,
    classify_tool_error,
    has_continuation_indicator,
)
from cicaddy.execution.recovery import (
    MAX_RECOVERY_MESSAGE_SIZE,
    RECOVERY_PROMPTS,
    SYNTHESIS_PROMPT,
    RecoveryManager,
    RecoveryResult,
)


class TestErrorClassification:
    """Test error classification system."""

    def test_error_types_defined(self):
        """Test all expected error types are defined."""
        assert ErrorType.AI_PREMATURE_COMPLETION is not None
        assert ErrorType.INVALID_TOOL_CALL is not None
        assert ErrorType.AI_INFERENCE_FAILURE is not None
        assert ErrorType.TOOL_EXECUTION_ERROR is not None
        assert ErrorType.REPEATED_FAILURE is not None
        assert ErrorType.MAX_TOKENS_EXCEEDED is not None
        assert ErrorType.MAX_ITERATIONS_EXCEEDED is not None

    def test_classified_error_get_error_key(self):
        """Test error key generation for different error types."""
        # Tool execution error with tool name
        error = ClassifiedError(
            error_type=ErrorType.TOOL_EXECUTION_ERROR,
            message="Connection refused",
            tool_name="execute_query",
            tool_server="konflux-devlake",
            iteration=5,
        )
        assert "tool_execution_error" in error.get_error_key()
        assert "execute_query" in error.get_error_key()

        # AI premature completion
        error = ClassifiedError(
            error_type=ErrorType.AI_PREMATURE_COMPLETION,
            message="AI indicated continuation but made no tool calls",
            iteration=4,
        )
        assert "ai_premature_completion" in error.get_error_key()
        assert "iteration_4" in error.get_error_key()

    def test_has_continuation_indicator_positive(self):
        """Test detection of continuation indicators in AI responses."""
        # Real-world examples from pipeline analysis with intent markers
        assert has_continuation_indicator("I will now move into the analysis phase")
        assert has_continuation_indicator("I will start by calculating the summary")
        assert has_continuation_indicator("My next step is to begin filling out")
        assert has_continuation_indicator("I will now inspect the data")
        assert has_continuation_indicator("I need to proceed to the next phase")
        assert has_continuation_indicator("I will execute the query")

    def test_has_continuation_indicator_negative(self):
        """Test that non-continuation phrases are not detected."""
        assert not has_continuation_indicator("Here is the final report")
        assert not has_continuation_indicator("Analysis complete")
        assert not has_continuation_indicator("The data shows the following trends")
        assert not has_continuation_indicator("")
        assert not has_continuation_indicator(None)

    def test_has_continuation_indicator_case_insensitive(self):
        """Test that indicator detection is case insensitive."""
        assert has_continuation_indicator("I WILL NOW move into analysis")
        assert has_continuation_indicator("I Will Now Move Into")
        # "my next step" was removed to prevent false positives - replaced with specific test
        assert has_continuation_indicator("MY NEXT STEP IS TO BEGIN analysis")

    def test_classify_tool_error_invalid_tool_call(self):
        """Test classification of invalid tool call errors."""
        assert (
            classify_tool_error("Operation not supported")
            == ErrorType.INVALID_TOOL_CALL
        )
        assert classify_tool_error("Invalid parameter") == ErrorType.INVALID_TOOL_CALL
        assert (
            classify_tool_error("Write operations not allowed")
            == ErrorType.INVALID_TOOL_CALL
        )
        assert (
            classify_tool_error("Update operation failed")
            == ErrorType.INVALID_TOOL_CALL
        )

    def test_classify_tool_error_timeout(self):
        """Test classification of timeout errors."""
        assert (
            classify_tool_error("Connection timeout") == ErrorType.TOOL_EXECUTION_ERROR
        )
        assert (
            classify_tool_error("Deadline exceeded") == ErrorType.TOOL_EXECUTION_ERROR
        )
        assert (
            classify_tool_error("Connection refused") == ErrorType.TOOL_EXECUTION_ERROR
        )

    def test_classify_tool_error_default(self):
        """Test default classification for unknown errors."""
        assert (
            classify_tool_error("Unknown error occurred")
            == ErrorType.TOOL_EXECUTION_ERROR
        )
        assert (
            classify_tool_error("Something went wrong")
            == ErrorType.TOOL_EXECUTION_ERROR
        )


class TestRecoveryPrompts:
    """Test recovery prompts are correctly defined."""

    def test_all_error_types_have_prompts(self):
        """Test that all error types have corresponding recovery prompts."""
        for error_type in ErrorType:
            if error_type != ErrorType.REPEATED_FAILURE:
                # REPEATED_FAILURE can fall back to AI_INFERENCE_FAILURE prompt
                # All others should have their own prompt
                assert (
                    error_type in RECOVERY_PROMPTS
                    or error_type == ErrorType.REPEATED_FAILURE
                )

    def test_ai_premature_completion_prompt_content(self):
        """Test AI_PREMATURE_COMPLETION prompt contains key instructions."""
        prompt = RECOVERY_PROMPTS[ErrorType.AI_PREMATURE_COMPLETION]
        assert "INCOMPLETE EXECUTION" in prompt
        assert "tool calls" in prompt.lower()
        assert "IMPORTANT" in prompt or "Critical" in prompt

    def test_ai_inference_failure_prompt_no_tool_calls_instruction(self):
        """Test AI_INFERENCE_FAILURE prompt instructs not to make tool calls."""
        prompt = RECOVERY_PROMPTS[ErrorType.AI_INFERENCE_FAILURE]
        assert "Do NOT make tool calls" in prompt
        assert "TEXT GUIDANCE ONLY" in prompt

    def test_prompts_have_placeholders(self):
        """Test that prompts contain expected placeholders."""
        # AI_PREMATURE_COMPLETION prompt should have these placeholders
        prompt = RECOVERY_PROMPTS[ErrorType.AI_PREMATURE_COMPLETION]
        assert "{iteration}" in prompt
        assert "{last_response_preview}" in prompt
        assert "{original_prompt}" in prompt
        assert "{event_history_summary}" in prompt

    def test_max_tokens_exceeded_prompt_content(self):
        """Test MAX_TOKENS_EXCEEDED prompt contains key instructions."""
        prompt = RECOVERY_PROMPTS[ErrorType.MAX_TOKENS_EXCEEDED]
        assert "TOKEN LIMIT" in prompt or "token limit" in prompt.lower()
        assert "{original_prompt}" in prompt
        assert "{tool_calls_summary}" in prompt
        assert "{tool_count}" in prompt  # Now includes tool count for emphasis
        # Recovery prompt mentions synthesis step will follow (synthesis is now separate)
        assert "synthesis step will follow" in prompt.lower()

    def test_max_iterations_exceeded_prompt_content(self):
        """Test MAX_ITERATIONS_EXCEEDED prompt contains key instructions."""
        prompt = RECOVERY_PROMPTS[ErrorType.MAX_ITERATIONS_EXCEEDED]
        assert "ITERATION LIMIT" in prompt or "iteration limit" in prompt.lower()
        assert "{original_prompt}" in prompt
        assert "{tool_calls_summary}" in prompt
        assert "{iterations_completed}" in prompt
        assert "{tool_count}" in prompt  # Now includes tool count for emphasis
        # remaining_tokens uses format specifier {:,} for comma-separated numbers
        assert "remaining_tokens" in prompt
        # Recovery prompt mentions synthesis step will follow (synthesis is now separate)
        assert "synthesis step will follow" in prompt.lower()

    def test_max_iterations_exceeded_prompt_has_correct_placeholders(self):
        """Test MAX_ITERATIONS_EXCEEDED prompt has all required placeholders."""
        prompt = RECOVERY_PROMPTS[ErrorType.MAX_ITERATIONS_EXCEEDED]
        # All required placeholders for iteration limit recovery
        assert "{original_prompt}" in prompt
        assert "{tool_calls_summary}" in prompt
        assert "{iterations_completed}" in prompt
        # remaining_tokens uses format specifier {:,} for comma-separated numbers
        assert "remaining_tokens" in prompt

    def test_synthesis_prompt_content(self):
        """Test SYNTHESIS_PROMPT contains key instructions."""
        assert "FINAL SYNTHESIS" in SYNTHESIS_PROMPT
        assert "NO tool calls allowed" in SYNTHESIS_PROMPT
        assert "{original_prompt}" in SYNTHESIS_PROMPT
        assert "{tool_calls_summary}" in SYNTHESIS_PROMPT
        assert "{tool_count}" in SYNTHESIS_PROMPT
        assert "{limit_type}" in SYNTHESIS_PROMPT

    def test_synthesis_prompt_separate_from_recovery(self):
        """Test SYNTHESIS_PROMPT is separate from recovery prompts."""
        # Synthesis prompt should focus on producing final output
        assert "SYNTHESIZE" in SYNTHESIS_PROMPT
        # Should explicitly disable tools
        assert "tools are disabled" in SYNTHESIS_PROMPT.lower()
        # Should not be a recovery prompt (doesn't tell AI to make tool calls)
        assert "make tool calls" not in SYNTHESIS_PROMPT.lower()


class TestRecoveryManager:
    """Test RecoveryManager functionality."""

    @pytest.fixture
    def mock_ai_provider(self):
        """Create a mock AI provider."""
        provider = AsyncMock()
        provider.chat_completion = AsyncMock(
            return_value=ProviderResponse(
                content="I will now make the necessary tool calls to continue.",
                model="test-model",
                tool_calls=None,
            )
        )
        return provider

    @pytest.fixture
    def mock_event_log(self):
        """Create a mock event log."""
        event_log = Mock()
        event_log.get_recent_events = Mock(
            return_value=[
                {
                    "event_type": "tool_execution",
                    "tool": "execute_query",
                    "server": "konflux-devlake",
                    "arguments": {"query": "SELECT * FROM incidents"},
                    "result": "10 rows returned",
                    "iteration": 1,
                },
                {
                    "event_type": "ai_inference",
                    "iteration": 2,
                    "response": "I will now analyze the data",
                    "tool_calls_count": 0,
                },
            ]
        )
        return event_log

    @pytest.fixture
    def recovery_manager(self, mock_ai_provider, mock_event_log):
        """Create a RecoveryManager instance."""
        return RecoveryManager(
            ai_provider=mock_ai_provider,
            event_log=mock_event_log,
            max_recovery_attempts=3,
        )

    @pytest.mark.asyncio
    async def test_attempt_recovery_success(self, recovery_manager):
        """Test successful recovery attempt."""
        error = ClassifiedError(
            error_type=ErrorType.AI_PREMATURE_COMPLETION,
            message="AI indicated continuation but made no tool calls",
            iteration=4,
            last_response_preview="I will now analyze the data...",
        )

        result = await recovery_manager.attempt_recovery(
            error=error,
            original_prompt="Analyze pipeline data",
            conversation_messages=[
                ProviderMessage(role="user", content="Analyze pipeline data")
            ],
        )

        assert result is not None
        assert result.should_continue is True
        assert result.recovery_message is not None
        assert result.strategy_used == "ai_premature_completion"

    @pytest.mark.asyncio
    async def test_attempt_recovery_max_retries_exceeded(self, recovery_manager):
        """Test recovery fails when max retries exceeded."""
        error = ClassifiedError(
            error_type=ErrorType.AI_PREMATURE_COMPLETION,
            message="AI indicated continuation but made no tool calls",
            iteration=4,
        )

        # Exhaust retry limit
        for _ in range(3):
            await recovery_manager.attempt_recovery(
                error=error,
                original_prompt="Analyze data",
                conversation_messages=[
                    ProviderMessage(role="user", content="Analyze data")
                ],
            )

        # Next attempt should fail
        result = await recovery_manager.attempt_recovery(
            error=error,
            original_prompt="Analyze data",
            conversation_messages=[
                ProviderMessage(role="user", content="Analyze data")
            ],
        )

        assert result.should_continue is False
        assert result.reason == "max_retries_exceeded"

    @pytest.mark.asyncio
    async def test_attempt_recovery_escalates_to_repeated_failure(
        self, recovery_manager
    ):
        """Test that repeated failures escalate to REPEATED_FAILURE error type."""
        error = ClassifiedError(
            error_type=ErrorType.TOOL_EXECUTION_ERROR,
            message="Connection timeout",
            tool_name="execute_query",
            tool_server="konflux-devlake",
            iteration=4,
        )

        # First attempt
        await recovery_manager.attempt_recovery(
            error=error,
            original_prompt="Analyze data",
            conversation_messages=[
                ProviderMessage(role="user", content="Analyze data")
            ],
        )

        # Second attempt should escalate
        result = await recovery_manager.attempt_recovery(
            error=error,
            original_prompt="Analyze data",
            conversation_messages=[
                ProviderMessage(role="user", content="Analyze data")
            ],
        )

        # Should still succeed but with escalated error type
        assert result.should_continue is True
        assert result.strategy_used == "repeated_failure"

    def test_build_recovery_context_with_events(self, recovery_manager):
        """Test recovery context building from event log."""
        error = ClassifiedError(
            error_type=ErrorType.AI_PREMATURE_COMPLETION,
            message="Test",
            iteration=3,
        )

        context = recovery_manager._build_recovery_context(error)

        assert "successful_tools" in context
        assert "failed_tools" in context
        assert "ai_responses" in context
        assert len(context["successful_tools"]) == 1
        assert context["successful_tools"][0]["tool"] == "execute_query"

    def test_build_recovery_context_without_event_log(self, mock_ai_provider):
        """Test recovery context building when event log is None."""
        manager = RecoveryManager(ai_provider=mock_ai_provider, event_log=None)

        error = ClassifiedError(
            error_type=ErrorType.AI_PREMATURE_COMPLETION,
            message="Test",
            iteration=3,
        )

        context = manager._build_recovery_context(error)

        assert context["successful_tools"] == []
        assert context["failed_tools"] == []
        assert context["ai_responses"] == []

    def test_format_event_history_summary(self, recovery_manager):
        """Test formatting of event history summary."""
        context = {
            "successful_tools": [
                {
                    "tool": "execute_query",
                    "server": "devlake",
                    "result_preview": "10 rows",
                }
            ],
            "failed_tools": [{"tool": "update_record", "error": "Not supported"}],
            "ai_responses": [
                {"iteration": 1, "had_tool_calls": True},
                {"iteration": 2, "had_tool_calls": False},
            ],
        }

        summary = recovery_manager._format_event_history_summary(context)

        assert "execute_query" in summary
        assert "update_record" in summary
        assert "AI iterations" in summary

    def test_reset_recovery_state(self, recovery_manager):
        """Test recovery state reset."""
        # Add some state
        recovery_manager._recovery_attempts["test_key"] = 2
        recovery_manager._failed_approaches.append("Test approach")

        recovery_manager.reset_recovery_state()

        assert len(recovery_manager._recovery_attempts) == 0
        assert len(recovery_manager._failed_approaches) == 0

    def test_get_recovery_stats(self, recovery_manager):
        """Test recovery statistics retrieval."""
        recovery_manager._recovery_attempts["error_1"] = 2
        recovery_manager._recovery_attempts["error_2"] = 1
        recovery_manager._failed_approaches.append("Approach 1")

        stats = recovery_manager.get_recovery_stats()

        assert stats["total_attempts"] == 3
        assert stats["unique_error_patterns"] == 2
        assert stats["failed_approaches_count"] == 1

    @pytest.mark.asyncio
    async def test_recover_from_iteration_limit_success(self, recovery_manager):
        """Test successful recovery from MAX_ITERATIONS_EXCEEDED."""
        # Create mock knowledge store
        mock_knowledge_store = Mock()
        mock_knowledge_store.get_tool_calls_summary_for_prompt = Mock(
            return_value="Tool 1: execute_query - returned 10 rows\nTool 2: get_incidents - returned 5 incidents"
        )
        mock_knowledge_store.total_tools_executed = 2

        error = ClassifiedError(
            error_type=ErrorType.MAX_ITERATIONS_EXCEEDED,
            message="Iteration limit reached: 40/40 iterations",
            iteration=40,
        )

        result = await recovery_manager.attempt_recovery(
            error=error,
            original_prompt="Analyze pipeline data",
            conversation_messages=[],  # Fresh context mode
            available_tools=None,
            knowledge_store=mock_knowledge_store,
        )

        assert result is not None
        assert result.should_continue is True
        assert result.strategy_used == "max_iterations_exceeded_fresh_context"

    @pytest.mark.asyncio
    async def test_recover_from_iteration_limit_no_knowledge_store(
        self, recovery_manager
    ):
        """Test recovery fails without knowledge store for MAX_ITERATIONS_EXCEEDED."""
        error = ClassifiedError(
            error_type=ErrorType.MAX_ITERATIONS_EXCEEDED,
            message="Iteration limit reached: 40/40 iterations",
            iteration=40,
        )

        result = await recovery_manager.attempt_recovery(
            error=error,
            original_prompt="Analyze pipeline data",
            conversation_messages=[],
            available_tools=None,
            knowledge_store=None,  # No knowledge store
        )

        assert result is not None
        assert result.should_continue is False
        assert result.reason == "no_knowledge_store"
        assert result.strategy_used == "max_iterations_exceeded"

    @pytest.mark.asyncio
    async def test_recover_from_max_tokens_exceeded_success(self, recovery_manager):
        """Test successful recovery from MAX_TOKENS_EXCEEDED."""
        # Create mock knowledge store
        mock_knowledge_store = Mock()
        mock_knowledge_store.get_tool_calls_summary_for_prompt = Mock(
            return_value="Tool 1: execute_query - returned 10 rows"
        )
        mock_knowledge_store.total_tools_executed = 1

        error = ClassifiedError(
            error_type=ErrorType.MAX_TOKENS_EXCEEDED,
            message="Token budget exhausted: 100000/100000 tokens",
            iteration=10,
        )

        result = await recovery_manager.attempt_recovery(
            error=error,
            original_prompt="Analyze pipeline data",
            conversation_messages=[],  # Fresh context mode
            available_tools=None,
            knowledge_store=mock_knowledge_store,
        )

        assert result is not None
        assert result.should_continue is True
        assert result.strategy_used == "max_tokens_exceeded_fresh_context"

    @pytest.mark.asyncio
    async def test_text_only_error_types_dont_pass_tools(
        self, mock_ai_provider, mock_event_log
    ):
        """Test that text-only error types don't pass tools to recovery AI."""
        manager = RecoveryManager(
            ai_provider=mock_ai_provider,
            event_log=mock_event_log,
            max_recovery_attempts=3,
        )

        # Test AI_INFERENCE_FAILURE - should NOT pass tools
        error = ClassifiedError(
            error_type=ErrorType.AI_INFERENCE_FAILURE,
            message="Token budget exhausted",
            iteration=27,
        )

        available_tools = [{"name": "test_tool", "description": "A test tool"}]

        await manager.attempt_recovery(
            error=error,
            original_prompt="Analyze data",
            conversation_messages=[
                ProviderMessage(role="user", content="Analyze data")
            ],
            available_tools=available_tools,
        )

        # Verify chat_completion was called with tools=None
        call_kwargs = mock_ai_provider.chat_completion.call_args.kwargs
        assert call_kwargs.get("tools") is None

    @pytest.mark.asyncio
    async def test_tool_error_types_pass_tools(self, mock_ai_provider, mock_event_log):
        """Test that tool error types DO pass tools to recovery AI."""
        manager = RecoveryManager(
            ai_provider=mock_ai_provider,
            event_log=mock_event_log,
            max_recovery_attempts=3,
        )

        # Test INVALID_TOOL_CALL - should pass tools
        error = ClassifiedError(
            error_type=ErrorType.INVALID_TOOL_CALL,
            message="Operation not supported",
            tool_name="update_record",
            tool_server="devlake",
            iteration=5,
        )

        available_tools = [{"name": "test_tool", "description": "A test tool"}]

        await manager.attempt_recovery(
            error=error,
            original_prompt="Update data",
            conversation_messages=[ProviderMessage(role="user", content="Update data")],
            available_tools=available_tools,
        )

        # Verify chat_completion was called with tools
        call_kwargs = mock_ai_provider.chat_completion.call_args.kwargs
        assert call_kwargs.get("tools") == available_tools


class TestRecoveryResult:
    """Test RecoveryResult dataclass."""

    def test_recovery_result_creation(self):
        """Test RecoveryResult creation with all fields."""
        result = RecoveryResult(
            should_continue=True,
            reason="recovery_successful",
            recovery_message="Continue with tool calls",
            strategy_used="ai_premature_completion",
        )

        assert result.should_continue is True
        assert result.reason == "recovery_successful"
        assert result.recovery_message == "Continue with tool calls"
        assert result.strategy_used == "ai_premature_completion"

    def test_recovery_result_defaults(self):
        """Test RecoveryResult with default optional fields."""
        result = RecoveryResult(
            should_continue=False,
            reason="max_retries_exceeded",
        )

        assert result.should_continue is False
        assert result.reason == "max_retries_exceeded"
        assert result.recovery_message is None
        assert result.strategy_used is None


class TestContinuationIndicators:
    """Test continuation indicator patterns from real pipeline analysis."""

    def test_build_service_patterns(self):
        """Test patterns from job 43268795 (build_service)."""
        # Real response that caused early break
        response = "I will now move into the analysis phase. I will start by calculating the summary metrics."
        assert has_continuation_indicator(response)

    def test_summary_analysis_patterns(self):
        """Test patterns from job 43268799 (summary_analysis)."""
        # Real response that caused early break
        response = "I will now move into the final phase: Summary Report Generation. My next step is to begin filling out the 8 sections."
        assert has_continuation_indicator(response)

    def test_integration_service_patterns(self):
        """Test patterns from integration_service events."""
        assert has_continuation_indicator("I will construct the query")
        assert has_continuation_indicator("I will execute the analysis")
        assert has_continuation_indicator("I will proceed to data collection")
        assert has_continuation_indicator("I will move to query the database")
        assert has_continuation_indicator("I will continue with the report")

    def test_early_break_pattern_job_43067065(self):
        """Test pattern from job 43067065 that broke at iteration 3."""
        response = "I will now inspect the pull request data"
        assert has_continuation_indicator(response)

    def test_intent_markers_detection(self):
        """Test that all intent markers are correctly detected."""
        # Test all intent markers defined in the simplified approach
        test_cases = [
            "I will analyze the data",
            "I'll execute the query",
            "Let me check the results",
            "My next step is to verify",
            "I need to process this",
            "First, I will gather information",
            "To overcome this limitation",
            "I will pivot to a different approach",
            "I will attempt to retrieve the data",
        ]
        for test_text in test_cases:
            assert has_continuation_indicator(test_text), (
                f"Intent marker not detected in: '{test_text}'"
            )


class TestFinalSynthesis:
    """Test the dedicated synthesis inference mechanism."""

    @pytest.fixture
    def mock_ai_provider(self):
        """Create a mock AI provider for synthesis tests."""
        provider = AsyncMock()
        provider.chat_completion = AsyncMock(
            return_value=ProviderResponse(
                content="Here is the comprehensive final report based on all collected data...",
                model="test-model",
                tool_calls=None,
            )
        )
        return provider

    @pytest.fixture
    def mock_knowledge_store(self):
        """Create a mock knowledge store with tool results."""
        knowledge_store = Mock()
        knowledge_store.get_tool_calls_summary_for_prompt = Mock(
            return_value="Tool 1: execute_query - returned 10 rows\nTool 2: get_incidents - returned 5 incidents"
        )
        knowledge_store.total_tools_executed = 5
        return knowledge_store

    @pytest.fixture
    def recovery_manager(self, mock_ai_provider):
        """Create a RecoveryManager instance for synthesis tests."""
        return RecoveryManager(
            ai_provider=mock_ai_provider,
            event_log=None,
            max_recovery_attempts=3,
        )

    @pytest.mark.asyncio
    async def test_perform_final_synthesis_success(
        self, recovery_manager, mock_knowledge_store
    ):
        """Test successful final synthesis after recovery exhaustion."""
        result = await recovery_manager.perform_final_synthesis(
            original_prompt="Analyze pipeline data and generate report",
            knowledge_store=mock_knowledge_store,
            limit_type="token budget",
        )

        assert result is not None
        assert "comprehensive final report" in result
        # Verify knowledge store was used
        mock_knowledge_store.get_tool_calls_summary_for_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_final_synthesis_no_knowledge_store(self, recovery_manager):
        """Test synthesis fails gracefully without knowledge store."""
        result = await recovery_manager.perform_final_synthesis(
            original_prompt="Analyze data",
            knowledge_store=None,
            limit_type="iteration limit",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_perform_final_synthesis_empty_knowledge_store(
        self, recovery_manager
    ):
        """Test synthesis fails gracefully with empty knowledge store."""
        empty_store = Mock()
        empty_store.total_tools_executed = 0

        result = await recovery_manager.perform_final_synthesis(
            original_prompt="Analyze data",
            knowledge_store=empty_store,
            limit_type="token budget",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_perform_final_synthesis_no_tools_passed(
        self, mock_ai_provider, mock_knowledge_store
    ):
        """Test that synthesis AI call is made with tools=None."""
        manager = RecoveryManager(
            ai_provider=mock_ai_provider,
            event_log=None,
            max_recovery_attempts=3,
        )

        await manager.perform_final_synthesis(
            original_prompt="Analyze data",
            knowledge_store=mock_knowledge_store,
            limit_type="token budget",
        )

        # Verify chat_completion was called with tools=None
        call_kwargs = mock_ai_provider.chat_completion.call_args.kwargs
        assert call_kwargs.get("tools") is None

    @pytest.mark.asyncio
    async def test_perform_final_synthesis_collects_inference_step(
        self, mock_ai_provider, mock_knowledge_store
    ):
        """Test that synthesis creates an InferenceStep for tracking."""
        manager = RecoveryManager(
            ai_provider=mock_ai_provider,
            event_log=None,
            max_recovery_attempts=3,
        )

        recovery_steps = []
        await manager.perform_final_synthesis(
            original_prompt="Analyze data",
            knowledge_store=mock_knowledge_store,
            limit_type="iteration limit",
            recovery_steps_collector=recovery_steps,
        )

        # Verify an InferenceStep was collected
        assert len(recovery_steps) == 1
        assert "synthesis" in recovery_steps[0].step_id

    def test_recovery_result_require_synthesis_field(self):
        """Test RecoveryResult has require_synthesis field."""
        result = RecoveryResult(
            should_continue=True,
            reason="recovery_successful",
            require_synthesis=True,
        )

        assert result.require_synthesis is True

        # Test default value
        result_default = RecoveryResult(
            should_continue=False,
            reason="test",
        )
        assert result_default.require_synthesis is False


class TestRecoveryContextTruncation:
    """Test recovery context truncation helpers."""

    def test_max_recovery_message_size_constant(self):
        """Test MAX_RECOVERY_MESSAGE_SIZE constant is defined correctly."""
        # ~3000 tokens = ~12000 characters
        assert MAX_RECOVERY_MESSAGE_SIZE == 12000

    def test_safe_truncate_for_recovery_short_content(self):
        """Test truncation passes through short content unchanged."""
        mock_provider = AsyncMock()
        manager = RecoveryManager(ai_provider=mock_provider)

        short_content = "This is a short message."
        result = manager._safe_truncate_for_recovery(short_content)

        assert result == short_content
        assert "[truncated" not in result

    def test_safe_truncate_for_recovery_long_content(self):
        """Test truncation handles long content correctly."""
        mock_provider = AsyncMock()
        manager = RecoveryManager(ai_provider=mock_provider)

        # Create content longer than limit
        long_content = "x" * 15000  # Longer than 12000 chars
        result = manager._safe_truncate_for_recovery(long_content)

        assert len(result) < len(long_content)
        assert result.startswith("x" * 100)  # Content preserved
        assert "[truncated for recovery context]" in result

    def test_safe_truncate_for_recovery_empty_content(self):
        """Test truncation handles empty/None content."""
        mock_provider = AsyncMock()
        manager = RecoveryManager(ai_provider=mock_provider)

        assert manager._safe_truncate_for_recovery("") == ""
        assert manager._safe_truncate_for_recovery(None) == ""

    def test_safe_truncate_for_recovery_exact_limit(self):
        """Test truncation at exact limit boundary."""
        mock_provider = AsyncMock()
        manager = RecoveryManager(ai_provider=mock_provider)

        # Content exactly at limit
        exact_content = "x" * MAX_RECOVERY_MESSAGE_SIZE
        result = manager._safe_truncate_for_recovery(exact_content)

        assert result == exact_content
        assert "[truncated" not in result

    def test_prepare_context_messages_empty_conversation(self):
        """Test context preparation with empty conversation."""
        mock_provider = AsyncMock()
        manager = RecoveryManager(ai_provider=mock_provider)

        recovery_message = ProviderMessage(role="user", content="Recovery prompt")
        result = manager._prepare_context_messages_for_recovery([], recovery_message)

        assert len(result) == 1
        assert result[0] == recovery_message

    def test_prepare_context_messages_single_message(self):
        """Test context preparation with single message conversation."""
        mock_provider = AsyncMock()
        manager = RecoveryManager(ai_provider=mock_provider)

        conversation = [ProviderMessage(role="user", content="Original task")]
        recovery_message = ProviderMessage(role="user", content="Recovery prompt")

        result = manager._prepare_context_messages_for_recovery(
            conversation, recovery_message
        )

        # First message + last 3 (which is just the first) + recovery
        # But first message is same as last, so: first + recovery = 2
        assert len(result) == 3  # first + last1 (same) + recovery
        assert result[-1] == recovery_message

    def test_prepare_context_messages_full_conversation(self):
        """Test context preparation with full conversation history."""
        mock_provider = AsyncMock()
        manager = RecoveryManager(ai_provider=mock_provider)

        conversation = [
            ProviderMessage(role="user", content="Original task"),
            ProviderMessage(role="assistant", content="First response"),
            ProviderMessage(role="user", content="Second message"),
            ProviderMessage(role="assistant", content="Second response"),
            ProviderMessage(role="user", content="Third message"),
            ProviderMessage(role="assistant", content="Third response"),
        ]
        recovery_message = ProviderMessage(role="user", content="Recovery prompt")

        result = manager._prepare_context_messages_for_recovery(
            conversation, recovery_message
        )

        # First message + last 3 messages + recovery = 5
        assert len(result) == 5
        assert result[0].content == "Original task"  # First message
        assert result[-1] == recovery_message  # Recovery at end

    def test_prepare_context_messages_truncates_large_content(self):
        """Test that context preparation truncates large messages."""
        mock_provider = AsyncMock()
        manager = RecoveryManager(ai_provider=mock_provider)

        large_content = "x" * 15000  # Larger than limit
        conversation = [
            ProviderMessage(role="user", content="Original task"),
            ProviderMessage(role="assistant", content=large_content),
        ]
        recovery_message = ProviderMessage(role="user", content="Recovery prompt")

        result = manager._prepare_context_messages_for_recovery(
            conversation, recovery_message
        )

        # Check the large content was truncated
        for msg in result[:-1]:  # Exclude recovery message
            if len(msg.content) > 100:  # Only check large ones
                assert "[truncated for recovery context]" in msg.content
                assert len(msg.content) < 15000

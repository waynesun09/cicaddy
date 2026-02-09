"""Unit tests for TokenAwareExecutor and related components."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from cicaddy.ai_providers.base import BaseProvider, ProviderMessage
from cicaddy.execution.token_aware_executor import (
    ExecutionLimits,
    ExecutionState,
    StopReason,
    TokenAwareExecutor,
)


class TestExecutionLimits:
    """Test ExecutionLimits configuration and validation."""

    def test_default_limits(self):
        """Test default ExecutionLimits values."""
        limits = ExecutionLimits()

        assert limits.max_infer_iters == 10
        assert (
            limits.max_tokens_total == 128000
        )  # Conservative fallback for most modern LLMs
        assert (
            limits.max_tokens_per_iteration == 8000
        )  # Conservative fallback, dynamically configured in agents
        assert (
            limits.max_tokens_per_tool_result == 4000
        )  # Conservative fallback, dynamically configured in agents
        assert limits.max_tools_per_iteration == 5
        assert limits.max_total_tools == 50
        assert limits.max_result_size_bytes == 1024 * 1024
        assert limits.max_total_result_size_bytes == 10 * 1024 * 1024
        assert limits.max_execution_time == 600
        assert limits.max_tool_timeout == 300  # 5 minutes absolute maximum
        assert limits.max_tool_idle_timeout == 60  # 1 minute idle timeout
        assert limits.warning_threshold == 0.8
        assert limits.degradation_threshold == 0.9

    def test_custom_limits(self):
        """Test custom ExecutionLimits configuration."""
        limits = ExecutionLimits(
            max_infer_iters=5,
            max_tokens_total=50000,
            max_tokens_per_iteration=4000,
            max_tools_per_iteration=3,
            max_execution_time=300,
        )

        assert limits.max_infer_iters == 5
        assert limits.max_tokens_total == 50000
        assert limits.max_tokens_per_iteration == 4000
        assert limits.max_tools_per_iteration == 3
        assert limits.max_execution_time == 300


class TestExecutionState:
    """Test ExecutionState tracking and management."""

    def test_initial_state(self):
        """Test initial ExecutionState values."""
        state = ExecutionState()

        assert state.current_iteration == 0
        assert state.total_tokens_used == 0
        assert state.current_iteration_tokens == 0
        assert state.total_tools_executed == 0
        assert state.current_iteration_tools == 0
        assert state.total_result_size_bytes == 0
        assert state.warnings_issued == []
        assert state.degradation_active is False
        assert state.stop_reason is None

    def test_reset_iteration(self):
        """Test iteration counter reset."""
        state = ExecutionState()
        state.current_iteration_tokens = 100
        state.current_iteration_tools = 2

        state.reset_iteration()

        assert state.current_iteration_tokens == 0
        assert state.current_iteration_tools == 0

    def test_add_tokens(self):
        """Test token tracking."""
        state = ExecutionState()

        state.add_tokens(100)
        assert state.total_tokens_used == 100
        assert state.current_iteration_tokens == 100

        state.add_tokens(50)
        assert state.total_tokens_used == 150
        assert state.current_iteration_tokens == 150

    def test_add_tool_execution(self):
        """Test tool execution tracking."""
        state = ExecutionState()

        state.add_tool_execution(1024)
        assert state.total_tools_executed == 1
        assert state.current_iteration_tools == 1
        assert state.total_result_size_bytes == 1024

        state.add_tool_execution(512)
        assert state.total_tools_executed == 2
        assert state.current_iteration_tools == 2
        assert state.total_result_size_bytes == 1536

    def test_time_tracking(self):
        """Test elapsed time calculation."""
        state = ExecutionState()
        start_time = state.start_time

        # Mock time passage
        with patch("time.time", return_value=start_time + 10):
            elapsed = state.get_elapsed_time()
            assert elapsed == 10

    def test_iteration_time_tracking(self):
        """Test iteration time calculation."""
        state = ExecutionState()

        # Mock time passage since last iteration
        original_time = state.last_iteration_time
        with patch("time.time", return_value=original_time + 5):
            iteration_time = state.get_iteration_time()
            assert iteration_time == 5

    def test_reset_for_fresh_context(self):
        """Test reset_for_fresh_context resets per-inference counters while preserving cumulative metrics.

        This is critical for MAX_TOKENS_EXCEEDED and MAX_ITERATIONS_EXCEEDED
        recovery to work correctly:
        - Per-inference tokens (total_tokens_used) reset for limit checking
        - Cumulative tokens (cumulative_tokens_all_inferences) preserved for metrics/billing
        - Per-inference tools (total_tools_executed) reset for limit checking
        - Cumulative tools (cumulative_tools_all_inferences) preserved for metrics
        """
        state = ExecutionState()

        # Simulate significant token and tool usage in first inference
        state.total_tokens_used = 100000  # High token usage
        state.current_iteration_tokens = 5000
        state.current_iteration_tools = 3
        state.total_tools_executed = 25
        state.total_result_size_bytes = 50000

        # Verify initial cumulative counters are 0
        assert state.cumulative_tokens_all_inferences == 0
        assert state.cumulative_tools_all_inferences == 0
        assert state.get_total_tokens_all_inferences() == 100000
        assert state.get_total_tools_all_inferences() == 25

        # Reset for fresh context recovery
        state.reset_for_fresh_context()

        # Per-inference token counters should be reset (for limit checking)
        assert state.total_tokens_used == 0
        assert state.current_iteration_tokens == 0
        assert state.current_iteration_tools == 0

        # Per-inference tool counter should be reset (for limit checking)
        assert state.total_tools_executed == 0

        # Cumulative tokens should be preserved (for metrics/billing)
        assert state.cumulative_tokens_all_inferences == 100000
        assert state.get_total_tokens_all_inferences() == 100000  # 100000 + 0

        # Cumulative tools should be preserved (for metrics)
        assert state.cumulative_tools_all_inferences == 25
        assert state.get_total_tools_all_inferences() == 25  # 25 + 0

        # Result size bytes should be preserved (total accumulated data)
        assert state.total_result_size_bytes == 50000

        # Simulate second inference adding more tokens and tools
        state.add_tokens(50000)
        state.add_tool_execution(1000)
        state.add_tool_execution(2000)
        assert state.total_tokens_used == 50000
        assert state.total_tools_executed == 2
        assert state.get_total_tokens_all_inferences() == 150000  # 100000 + 50000
        assert state.get_total_tools_all_inferences() == 27  # 25 + 2

        # Reset for another fresh context recovery
        state.reset_for_fresh_context()

        # Cumulative should now include both inferences
        assert state.cumulative_tokens_all_inferences == 150000
        assert state.cumulative_tools_all_inferences == 27
        assert state.total_tokens_used == 0
        assert state.total_tools_executed == 0
        assert state.get_total_tokens_all_inferences() == 150000
        assert state.get_total_tools_all_inferences() == 27


class TestStopReason:
    """Test StopReason enum values."""

    def test_stop_reason_values(self):
        """Test all StopReason enum values."""
        assert StopReason.end_of_turn.value == "end_of_turn"
        assert StopReason.end_of_message.value == "end_of_message"
        assert StopReason.out_of_tokens.value == "out_of_tokens"
        assert StopReason.max_iterations.value == "max_iterations"
        assert StopReason.max_tools.value == "max_tools"
        assert StopReason.max_result_size.value == "max_result_size"
        assert StopReason.timeout.value == "timeout"
        assert StopReason.error.value == "error"


class TestTokenAwareExecutor:
    """Test TokenAwareExecutor functionality."""

    @pytest.fixture
    def mock_ai_provider(self):
        """Mock AI provider for testing."""
        provider = Mock(spec=BaseProvider)
        provider.chat_completion = AsyncMock()
        return provider

    @pytest.fixture
    def mock_mcp_manager(self):
        """Mock MCP manager for testing."""
        manager = AsyncMock()
        manager.call_tool = AsyncMock()
        return manager

    @pytest.fixture
    def executor(self, mock_ai_provider, mock_mcp_manager):
        """Create TokenAwareExecutor for testing."""
        limits = ExecutionLimits(
            max_infer_iters=3,
            max_tokens_total=1000,
            max_tokens_per_iteration=400,
            max_tools_per_iteration=2,
            max_execution_time=60,
        )
        return TokenAwareExecutor(
            ai_provider=mock_ai_provider,
            mcp_manager=mock_mcp_manager,
            limits=limits,
            session_id="test-session",
        )

    def test_initialization(self, executor):
        """Test TokenAwareExecutor initialization."""
        assert executor.session_id == "test-session"
        assert executor.limits.max_infer_iters == 3
        assert executor.state.current_iteration == 0
        assert executor.state.total_tokens_used == 0

    def test_check_limits_iterations(self, executor):
        """Test iteration limit checking."""
        # Set current iteration to limit
        executor.state.current_iteration = 3

        stop_reason = executor.check_limits()
        assert stop_reason == StopReason.max_iterations

    def test_check_limits_total_tokens(self, executor):
        """Test total token limit checking."""
        executor.state.total_tokens_used = 1000

        stop_reason = executor.check_limits()
        assert stop_reason == StopReason.out_of_tokens

    def test_check_limits_iteration_tokens(self, executor):
        """Test iteration token limit checking."""
        executor.state.current_iteration_tokens = 400

        stop_reason = executor.check_limits()
        assert stop_reason == StopReason.out_of_tokens

    def test_check_limits_tools_per_iteration(self, executor):
        """Test tools per iteration limit checking."""
        executor.state.current_iteration_tools = 2

        stop_reason = executor.check_limits()
        assert stop_reason == StopReason.max_tools

    def test_check_limits_total_tools(self, executor):
        """Test total tools limit checking."""
        executor.state.total_tools_executed = 50

        stop_reason = executor.check_limits()
        assert stop_reason == StopReason.max_tools

    def test_check_limits_result_size(self, executor):
        """Test result size limit checking."""
        executor.state.total_result_size_bytes = 10 * 1024 * 1024

        stop_reason = executor.check_limits()
        assert stop_reason == StopReason.max_result_size

    def test_check_limits_time(self, executor):
        """Test time limit checking."""
        # Mock elapsed time exceeding limit
        with patch.object(executor.state, "get_elapsed_time", return_value=61):
            stop_reason = executor.check_limits()
            assert stop_reason == StopReason.timeout

    def test_check_limits_no_violation(self, executor):
        """Test no limit violations."""
        stop_reason = executor.check_limits()
        assert stop_reason is None

    def test_check_warnings_iteration(self, executor):
        """Test iteration warning detection."""
        executor.state.current_iteration = 2  # 2/3 = 66% < 80%
        warnings = executor.check_warnings()
        assert len(warnings) == 0

        executor.state.current_iteration = 3  # 3/3 = 100% >= 80%
        warnings = executor.check_warnings()
        # May include both iteration warning and degradation warning
        assert len(warnings) >= 1
        warning_text = " ".join(warnings)
        assert "Approaching iteration limit" in warning_text

    def test_check_warnings_tokens(self, executor):
        """Test token warning detection."""
        executor.state.total_tokens_used = 800  # 800/1000 = 80%
        warnings = executor.check_warnings()
        assert len(warnings) == 1
        assert "Approaching token limit" in warnings[0]

    def test_check_warnings_degradation_activation(self, executor):
        """Test progressive degradation activation via tool limit.

        Note: With recovery_enabled=True (default), degradation is skipped for
        iteration/token limits. Tool limit still triggers degradation since
        it's not recoverable.
        """
        # Set tool utilization to 90% (degradation threshold)
        # executor fixture has max_total_tools=50 by default
        executor.state.total_tools_executed = 45  # 45/50 = 90% >= 90%

        warnings = executor.check_warnings()
        assert executor.state.degradation_active is True
        assert any("Activating progressive degradation" in w for w in warnings)

    def test_estimate_tokens(self, executor):
        """Test token estimation."""
        text = "This is a test string with words"  # 33 characters
        estimated = executor.estimate_tokens(text)
        assert estimated == max(1, 33 // 4)  # 8 tokens

    def test_should_truncate_results(self, executor):
        """Test result truncation decision."""
        # Initially should not truncate
        assert executor.should_truncate_results() is False

        # Should truncate when degradation is active
        executor.state.degradation_active = True
        assert executor.should_truncate_results() is True

        # Should truncate when token utilization is high (>90%)
        executor.state.degradation_active = False
        executor.state.total_tokens_used = 910  # 91% of 1000, > 90% threshold
        assert executor.should_truncate_results() is True

    @pytest.mark.asyncio
    async def test_execute_ai_inference_success(self, executor, mock_ai_provider):
        """Test successful AI inference execution."""
        # Mock successful response
        mock_response = Mock()
        mock_response.content = "Test AI response"
        mock_response.tool_calls = None  # Required for EventLog logging
        mock_ai_provider.chat_completion.return_value = mock_response

        messages = [ProviderMessage(content="Test prompt", role="user")]

        response = await executor._execute_ai_inference(messages)

        assert response == mock_response
        assert response.content == "Test AI response"
        assert executor.state.total_tokens_used > 0  # Tokens should be tracked
        mock_ai_provider.chat_completion.assert_called_once_with(messages, tools=None)

    @pytest.mark.asyncio
    async def test_execute_ai_inference_tool_calls_only(
        self, executor, mock_ai_provider
    ):
        """Test AI inference with tool calls but no text content."""
        # Mock response with tool calls but empty content
        mock_response = Mock()
        mock_response.content = ""  # Empty content
        mock_response.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"param": "value"}'},
            }
        ]
        mock_ai_provider.chat_completion.return_value = mock_response

        messages = [ProviderMessage(content="Test prompt", role="user")]

        response = await executor._execute_ai_inference(messages)

        # Should not be considered empty since it has tool calls
        assert response == mock_response
        assert response.content == ""
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        # Should track input tokens but output tokens should be 0 for empty content
        assert executor.state.total_tokens_used > 0  # Input tokens counted
        mock_ai_provider.chat_completion.assert_called_once_with(messages, tools=None)

    @pytest.mark.asyncio
    async def test_execute_ai_inference_timeout(self, executor, mock_ai_provider):
        """Test AI inference timeout handling."""
        # Mock timeout
        mock_ai_provider.chat_completion.side_effect = AsyncMock(
            side_effect=Exception("Timeout")
        )

        messages = [ProviderMessage(content="Test prompt", role="user")]

        response = await executor._execute_ai_inference(messages)

        assert response is None

    @pytest.mark.asyncio
    async def test_execute_ai_inference_insufficient_tokens(
        self, executor, mock_ai_provider
    ):
        """Test AI inference with insufficient token budget."""
        # Set tokens near limit
        executor.state.total_tokens_used = 999  # Very close to 1000 limit

        messages = [
            ProviderMessage(
                content="Very long test prompt that will exceed token budget",
                role="user",
            )
        ]

        response = await executor._execute_ai_inference(messages)

        assert response is None  # Should return None due to insufficient budget

    @pytest.mark.asyncio
    async def test_execute_single_tool_success(self, executor, mock_mcp_manager):
        """Test successful single tool execution."""
        tool_call = {
            "tool_name": "test_tool",
            "tool_server": "test_server",
            "arguments": {"param": "value"},
        }

        # Mock successful tool response
        mock_mcp_manager.call_tool.return_value = "Tool result"

        result = await executor._execute_single_tool(tool_call)

        assert result["tool_name"] == "test_tool"
        assert result["tool_server"] == "test_server"
        assert result["result"] == "Tool result"
        assert result["status"] == "success"
        assert result["execution_time"] > 0
        assert result["result_size_bytes"] > 0

        # Verify state tracking
        assert executor.state.total_tools_executed == 1
        assert executor.state.current_iteration_tools == 1

    @pytest.mark.asyncio
    async def test_execute_single_tool_failure(self, executor, mock_mcp_manager):
        """Test single tool execution failure."""
        tool_call = {
            "tool_name": "failing_tool",
            "tool_server": "test_server",
            "arguments": {},
        }

        # Mock tool failure
        mock_mcp_manager.call_tool.side_effect = Exception("Tool execution failed")

        result = await executor._execute_single_tool(tool_call)

        assert result["tool_name"] == "failing_tool"
        assert result["status"] == "error"
        assert "Tool execution failed" in result["result"]
        assert result["result_size_bytes"] == 0

        # Should still track the attempt
        assert executor.state.total_tools_executed == 1

    @pytest.mark.asyncio
    async def test_execute_single_tool_timeout(self, executor, mock_mcp_manager):
        """Test single tool execution timeout."""
        tool_call = {
            "tool_name": "slow_tool",
            "tool_server": "test_server",
            "arguments": {},
        }

        # Mock tool timeout
        import asyncio

        mock_mcp_manager.call_tool.side_effect = asyncio.TimeoutError()

        result = await executor._execute_single_tool(tool_call)

        assert result["status"] == "error"
        assert "exceeded timeout" in result["result"]

    @pytest.mark.asyncio
    async def test_execute_single_tool_large_result_truncation(
        self, executor, mock_mcp_manager
    ):
        """Test large tool result intelligent compression (Phase 2)."""
        tool_call = {
            "tool_name": "large_result_tool",
            "tool_server": "test_server",
            "arguments": {},
        }

        # Mock large result that exceeds token limit
        large_result = "x" * (2 * 1024 * 1024)  # 2MB, exceeds 1MB limit
        mock_mcp_manager.call_tool.return_value = large_result

        result = await executor._execute_single_tool(tool_call)

        assert result["status"] == "success"
        # Phase 2: Intelligent compression produces compressed marker
        assert (
            "[Preserved" in result["result"]
            or "[Truncated" in result["result"]
            or "[TRUNCATED:" in result["result"]
        )
        assert len(result["result"]) < len(large_result)
        # Verify compression flag is set
        assert (
            result.get("compressed", False) is True
            or "[Truncated" in result["result"]
            or "[TRUNCATED:" in result["result"]
        )

    def test_format_tool_results_for_conversation_full(self, executor):
        """Test full tool results formatting."""
        tool_results = [
            {"tool_name": "tool1", "result": "Result 1", "status": "success"},
            {"tool_name": "tool2", "result": "Error occurred", "status": "error"},
        ]

        formatted = executor._format_tool_results_for_conversation(tool_results)

        assert "Tool execution results:" in formatted
        assert "Tool tool1: Result 1" in formatted
        assert "Tool tool2 failed: Error occurred" in formatted

    def test_format_tool_results_for_conversation_summarized(self, executor):
        """Test summarized tool results formatting."""
        # Activate degradation mode
        executor.state.degradation_active = True

        tool_results = [
            {"tool_name": "tool1", "result": "Result 1", "status": "success"},
            {"tool_name": "tool2", "result": "Result 2", "status": "success"},
        ]

        formatted = executor._format_tool_results_for_conversation(tool_results)

        assert "Tool execution summary:" in formatted
        assert "Successfully executed 2 tools" in formatted
        assert "[SUMMARIZED:" in formatted

    @pytest.mark.asyncio
    async def test_generate_final_summary_simple(self, executor):
        """Test final summary generation without degradation."""
        messages = [
            ProviderMessage(content="User prompt", role="user"),
            ProviderMessage(
                content="AI final response with substantive content", role="assistant"
            ),
        ]
        tool_results = []

        summary = await executor._generate_final_summary(messages, tool_results)

        assert summary == "AI final response with substantive content"

    @pytest.mark.asyncio
    async def test_generate_final_summary_with_degradation(self, executor):
        """Test final summary generation with degradation active."""
        messages = [
            ProviderMessage(content="User prompt", role="user"),
            ProviderMessage(
                content="AI analysis with substantive content for testing",
                role="assistant",
            ),
        ]
        tool_results = [
            {"status": "success"},
            {"status": "success"},
            {"status": "error"},
        ]

        # Set degradation conditions
        executor.state.degradation_active = True
        executor.state.stop_reason = StopReason.max_iterations
        executor.state.current_iteration = 3

        summary = await executor._generate_final_summary(messages, tool_results)

        assert "AI analysis with substantive content for testing" in summary
        assert "[EXECUTION SUMMARY:" in summary
        assert "Completed 3 iterations" in summary
        assert "Tools executed: 2/3 successful" in summary

    def test_get_execution_summary(self, executor):
        """Test execution summary generation."""
        # Set up some execution state
        executor.state.current_iteration = 2
        executor.state.total_tokens_used = 500
        executor.state.total_tools_executed = 3
        executor.state.degradation_active = True
        executor.state.stop_reason = StopReason.out_of_tokens

        summary = executor.get_execution_summary()

        assert summary["session_id"] == "test-session"
        assert summary["stop_reason"] == "out_of_tokens"
        assert summary["iterations"]["completed"] == 2
        assert summary["iterations"]["max_allowed"] == 3
        assert summary["iterations"]["utilization"] == 2 / 3
        assert summary["tokens"]["total_used"] == 500
        assert summary["tokens"]["max_allowed"] == 1000
        assert summary["tokens"]["utilization"] == 0.5
        assert summary["tools"]["total_executed"] == 3
        assert summary["degradation_active"] is True

    @pytest.mark.asyncio
    async def test_execute_with_limits_complete_workflow(
        self, executor, mock_ai_provider, mock_mcp_manager
    ):
        """Test complete execution workflow with limits."""
        # Mock AI responses
        mock_response_1 = Mock(content="I need to call a tool")
        mock_response_1.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"param": "value"}'},
            }
        ]
        mock_response_2 = Mock(content="Final analysis based on tool results")
        mock_response_2.tool_calls = []
        mock_responses = [mock_response_1, mock_response_2]
        mock_ai_provider.chat_completion.side_effect = mock_responses

        # Mock tool parsing to return empty (no tools to execute)
        with patch.object(executor, "_parse_and_execute_tools") as mock_parse_tools:
            mock_parse_tools.side_effect = [
                [
                    {
                        "tool_name": "test_tool",
                        "result": "tool result",
                        "status": "success",
                    }
                ],
                [],  # No more tools on second iteration
            ]

            messages = [ProviderMessage(content="Analyze this", role="user")]
            available_tools = [{"name": "test_tool", "server": "test_server"}]

            result = await executor.execute_with_limits(messages, available_tools)

            assert "final_response" in result
            assert "tool_results" in result
            assert "execution_summary" in result
            assert "conversation_messages" in result

            # Should have completed normally
            assert executor.state.stop_reason == StopReason.end_of_turn

            # Should have executed some iterations
            assert executor.state.current_iteration > 0

    @pytest.mark.asyncio
    async def test_execute_with_limits_max_iterations(self, executor, mock_ai_provider):
        """Test execution stopping at max iterations."""
        # Mock AI to always request tools (infinite loop scenario)
        mock_response = Mock(content="I need more tools")
        mock_response.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"param": "value"}'},
            }
        ]
        mock_ai_provider.chat_completion.return_value = mock_response

        # Mock tool parsing to always return tools
        with patch.object(executor, "_parse_and_execute_tools") as mock_parse_tools:
            mock_parse_tools.return_value = [
                {"tool_name": "test_tool", "result": "result", "status": "success"}
            ]

            messages = [ProviderMessage(content="Analyze this", role="user")]
            available_tools = [{"name": "test_tool", "server": "test_server"}]

            await executor.execute_with_limits(messages, available_tools)

            # Should stop due to max iterations
            assert executor.state.stop_reason == StopReason.max_iterations
            assert executor.state.current_iteration == 3  # Hit the limit

    @pytest.mark.asyncio
    async def test_execute_with_limits_error_handling(self, executor, mock_ai_provider):
        """Test execution error handling."""
        # Mock AI provider to raise exception
        mock_ai_provider.chat_completion.side_effect = Exception("AI provider error")

        messages = [ProviderMessage(content="Analyze this", role="user")]

        result = await executor.execute_with_limits(messages)

        assert executor.state.stop_reason == StopReason.error
        # When AI inference fails immediately, it defaults to "Analysis completed"
        # because no assistant messages were added to the conversation
        assert result["final_response"] == "Analysis completed"

    # ===== Phase 2: Integration Tests for Compression Triggers =====

    @pytest.mark.asyncio
    async def test_execute_with_limits_triggers_conversation_compaction_after_iteration_3(
        self, mock_ai_provider
    ):
        """Test that conversation compaction is triggered after iteration 3."""
        # Create executor with smaller iteration limit to trigger compaction faster
        limits = ExecutionLimits(max_infer_iters=5, max_tokens_total=10000)
        executor = TokenAwareExecutor(
            ai_provider=mock_ai_provider, mcp_manager=None, limits=limits
        )

        # Mock compactor's compact_iteration_context
        mock_compactor = Mock()
        mock_compactor.compact_iteration_context = AsyncMock(
            return_value=(
                [ProviderMessage(content="Compacted", role="user")],
                Mock(compression_ratio=2.0, information_preserved=0.85),
            )
        )
        executor.compactor = mock_compactor

        # Mock AI provider to return responses that trigger iterations
        response_iteration = [0]  # Track which iteration we're on

        async def create_response(*args, **kwargs):
            response_iteration[0] += 1
            if response_iteration[0] <= 3:
                # First 3 iterations: return tool calls to continue
                response = Mock(content="Working on it")
                response.tool_calls = [
                    {
                        "id": f"call_{response_iteration[0]}",
                        "type": "function",
                        "function": {"name": "tool", "arguments": "{}"},
                    }
                ]
                return response
            else:
                # After 3rd iteration: return final response
                response = Mock(content="Done")
                response.tool_calls = None
                return response

        mock_ai_provider.chat_completion = create_response

        # Mock tool parsing to return success
        with patch.object(executor, "_parse_and_execute_tools") as mock_parse_tools:
            mock_parse_tools.return_value = [
                {"tool_name": "tool", "result": "result", "status": "success"}
            ]

            messages = [ProviderMessage(content="Initial task", role="user")]
            await executor.execute_with_limits(messages)

            # compact_iteration_context should be called for iteration 4 and beyond
            assert mock_compactor.compact_iteration_context.called
            # Should be called at least once (for iteration 4)
            assert mock_compactor.compact_iteration_context.call_count >= 1

    @pytest.mark.asyncio
    async def test_execute_ai_inference_compresses_prompt_when_high_utilization(
        self, mock_ai_provider
    ):
        """Test that prompt is compressed before AI inference if token utilization is high."""
        limits = ExecutionLimits(max_tokens_total=10000, max_tokens_per_iteration=5000)
        executor = TokenAwareExecutor(
            ai_provider=mock_ai_provider, mcp_manager=None, limits=limits
        )

        # Mock compactor's compress_prompt_before_send
        mock_compactor = Mock()
        mock_compactor.compress_prompt_before_send.return_value = (
            "compressed_prompt",
            Mock(compression_ratio=2.0, information_preserved=0.9),
        )
        executor.compactor = mock_compactor

        # Set up executor state to trigger compression (>70% utilization)
        executor.state.total_tokens_used = 7500  # 75% of 10000

        # Create a long message that exceeds 1000 tokens (estimate_tokens uses len//4)
        long_message_content = (
            "x " * 2500
        )  # 5000 characters / 4 = 1250 tokens (exceeds 1000 threshold)
        messages = [
            ProviderMessage(content="System instruction", role="system"),
            ProviderMessage(content=long_message_content, role="user"),
        ]

        # Mock AI response
        mock_response = Mock(content="Response")
        mock_response.tool_calls = None
        mock_ai_provider.chat_completion = AsyncMock(return_value=mock_response)

        await executor._execute_ai_inference(messages, [])

        # Assert that compress_prompt_before_send was called
        assert mock_compactor.compress_prompt_before_send.called
        # Verify it was called with the long message content and token budget
        call_args = mock_compactor.compress_prompt_before_send.call_args
        assert long_message_content in call_args[0][0]
        assert call_args.kwargs["token_budget"] == limits.max_tokens_per_iteration

    @pytest.mark.asyncio
    async def test_execute_single_tool_compresses_large_result(self, mock_ai_provider):
        """Test that large tool results are compressed intelligently."""
        limits = ExecutionLimits(max_tokens_per_tool_result=100)
        mock_mcp_manager = Mock()
        executor = TokenAwareExecutor(
            ai_provider=mock_ai_provider, mcp_manager=mock_mcp_manager, limits=limits
        )

        # Mock compactor's compress_tool_result_realtime
        mock_compactor = Mock()
        compressed_result = {
            "tool_name": "large_tool",
            "tool_server": "test_server",
            "arguments": {},
            "result": "compressed_result",
            "status": "success",
            "compressed": True,
        }
        mock_compactor.compress_tool_result_realtime.return_value = (
            compressed_result,
            Mock(compression_ratio=10.0, information_preserved=0.80),
        )
        executor.compactor = mock_compactor

        # Mock large tool result (will exceed token limit)
        large_result = "x " * 500  # Approximately 1000 tokens
        mock_mcp_manager.call_tool = AsyncMock(return_value=large_result)

        tool_call = {
            "tool_name": "large_tool",
            "tool_server": "test_server",
            "arguments": {},
        }

        result = await executor._execute_single_tool(tool_call)

        # Assert that compress_tool_result_realtime was called
        assert mock_compactor.compress_tool_result_realtime.called
        # Verify compression was applied
        assert result["compressed"] is True
        assert result["result"] == "compressed_result"

    # ===== Phase 3: Unit Tests for Multi-Factor Compression Triggers =====

    def test_calculate_conversation_tokens(self, executor):
        """Test conversation token calculation."""
        messages = [
            ProviderMessage(
                content="Short message", role="user"
            ),  # ~14 chars / 4 = 3 tokens
            ProviderMessage(
                content="A" * 100, role="assistant"
            ),  # 100 chars / 4 = 25 tokens
            ProviderMessage(
                content="B" * 200, role="user"
            ),  # 200 chars / 4 = 50 tokens
        ]

        total_tokens = executor._calculate_conversation_tokens(messages)

        # Should sum up all message tokens
        assert total_tokens == 3 + 25 + 50  # 78 tokens

    def test_should_compact_conversation_trigger_1_iteration(self, executor):
        """Test Trigger 1: Iteration > 3."""
        executor.state.current_iteration = 4
        messages = [ProviderMessage(content="Short", role="user")]

        should_compact = executor._should_compact_conversation(messages)

        assert should_compact is True

    def test_should_compact_conversation_trigger_2_conversation_size(self, executor):
        """Test Trigger 2: Conversation size exceeds 50% of iteration budget."""
        executor.state.current_iteration = 2  # Early iteration
        # max_tokens_per_iteration = 400, so 50% threshold = 200 tokens
        # Create message that's 250 tokens (250 * 4 = 1000 chars)
        large_message = "x" * 1000
        messages = [ProviderMessage(content=large_message, role="user")]

        should_compact = executor._should_compact_conversation(messages)

        assert should_compact is True

    def test_should_compact_conversation_trigger_3_projected_usage(self, executor):
        """Test Trigger 3: Token utilization > 60% AND projected > 80%."""
        # Set up state for high utilization
        executor.state.current_iteration = 5
        executor.state.total_tokens_used = 650  # 650/1000 = 65% utilization

        # Projected usage: (650 + 650/5) / 1000 = 780/1000 = 78%... not quite 80%
        # Let's make it higher: 700 tokens used over 5 iterations = 140 avg per iteration
        executor.state.total_tokens_used = 700  # 70% utilization
        # Projected: (700 + 140) / 1000 = 84% > 80% ✓

        messages = [ProviderMessage(content="Message", role="user")]

        should_compact = executor._should_compact_conversation(messages)

        assert should_compact is True

    def test_should_compact_conversation_trigger_4_large_tool_result(self, executor):
        """Test Trigger 4: Single tool result > 30% of iteration budget."""
        executor.state.current_iteration = 1  # Very early iteration
        # max_tokens_per_iteration = 400, so 30% threshold = 120 tokens
        # Tool result of 150 tokens (exceeds 30% threshold of 120)

        messages = [ProviderMessage(content="Small", role="user")]

        should_compact = executor._should_compact_conversation(
            messages, last_tool_result_tokens=150
        )

        assert should_compact is True

    def test_should_compact_conversation_no_triggers(self, executor):
        """Test that no triggers fire when all conditions are safe."""
        executor.state.current_iteration = 2
        executor.state.total_tokens_used = 100  # Low utilization
        messages = [ProviderMessage(content="Small message", role="user")]

        should_compact = executor._should_compact_conversation(
            messages, last_tool_result_tokens=10
        )

        assert should_compact is False

    def test_get_compaction_trigger_reason_iteration(self, executor):
        """Test trigger reason identification for iteration trigger."""
        executor.state.current_iteration = 5
        messages = [ProviderMessage(content="Test", role="user")]

        reason = executor._get_compaction_trigger_reason(messages)

        assert reason == "iteration_5"

    def test_get_compaction_trigger_reason_conversation_size(self, executor):
        """Test trigger reason identification for conversation size."""
        executor.state.current_iteration = 2
        large_message = "x" * 1000  # 250 tokens
        messages = [ProviderMessage(content=large_message, role="user")]

        reason = executor._get_compaction_trigger_reason(messages)

        assert "conversation_size_" in reason
        assert "tokens" in reason

    def test_get_compaction_trigger_reason_large_tool_result(self, executor):
        """Test trigger reason identification for large tool result."""
        executor.state.current_iteration = 1
        messages = [ProviderMessage(content="Small", role="user")]

        reason = executor._get_compaction_trigger_reason(
            messages, last_tool_result_tokens=150
        )

        assert "large_tool_result_" in reason
        assert "tokens" in reason

    def test_execution_state_record_compression(self):
        """Test compression metrics recording in ExecutionState."""
        state = ExecutionState()

        # Record first compression
        state.record_compression(2.5, 0.85, "iteration_4")

        assert state.compression_count == 1
        assert state.total_compression_ratio == 2.5
        assert state.total_information_preserved == 0.85
        assert state.compression_triggers == ["iteration_4"]

        # Record second compression
        state.record_compression(3.0, 0.90, "conversation_size_500tokens")

        assert state.compression_count == 2
        assert state.total_compression_ratio == 5.5  # 2.5 + 3.0
        assert state.total_information_preserved == 1.75  # 0.85 + 0.90
        assert state.compression_triggers == [
            "iteration_4",
            "conversation_size_500tokens",
        ]

    def test_get_execution_summary_includes_compression_metrics(self, executor):
        """Test that execution summary includes compression metrics."""
        # Record some compressions
        executor.state.record_compression(2.0, 0.80, "iteration_4")
        executor.state.record_compression(3.0, 0.90, "large_tool_result_300tokens")

        summary = executor.get_execution_summary()

        assert "compression" in summary
        assert summary["compression"]["count"] == 2
        # Use approximate equality for float comparisons
        assert (
            abs(summary["compression"]["avg_compression_ratio"] - 2.5) < 0.01
        )  # (2.0 + 3.0) / 2
        assert (
            abs(summary["compression"]["avg_information_preserved"] - 0.85) < 0.01
        )  # (0.80 + 0.90) / 2
        # Check triggers list
        assert len(summary["compression"]["triggers"]) == 2
        assert summary["compression"]["triggers"][0] == "iteration_4"
        assert summary["compression"]["triggers"][1] == "large_tool_result_300tokens"

    def test_get_execution_summary_compression_metrics_when_no_compressions(
        self, executor
    ):
        """Test compression metrics when no compressions occurred."""
        summary = executor.get_execution_summary()

        assert "compression" in summary
        assert summary["compression"]["count"] == 0
        assert summary["compression"]["avg_compression_ratio"] == 0.0
        assert summary["compression"]["avg_information_preserved"] == 0.0
        assert summary["compression"]["triggers"] == []

    # ===== Similarity Detection Tests (Duplicate Prevention) =====

    @pytest.mark.asyncio
    async def test_check_conversation_size_and_compact_triggered(
        self, executor, mock_ai_provider
    ):
        """Test that _check_conversation_size_and_compact triggers compression."""
        # Create large conversation that exceeds threshold
        large_message = "x" * 1000  # 250 tokens > 50% of 400 token iteration budget
        messages = [ProviderMessage(content=large_message, role="user")]

        executor.state.current_iteration = 2  # Early iteration

        # Mock compactor
        mock_compactor = Mock()
        mock_compactor.compact_iteration_context = AsyncMock(
            return_value=(
                [ProviderMessage(content="Compacted", role="user")],
                Mock(compression_ratio=2.5, information_preserved=0.85),
            )
        )
        executor.compactor = mock_compactor

        result = await executor._check_conversation_size_and_compact(messages)

        # Should have triggered compaction
        assert mock_compactor.compact_iteration_context.called
        assert len(result) == 1
        assert result[0].content == "Compacted"

        # Should have recorded compression metrics
        assert executor.state.compression_count == 1
        assert executor.state.compression_triggers[0].startswith("conversation_size_")

    @pytest.mark.asyncio
    async def test_check_conversation_size_and_compact_not_triggered(self, executor):
        """Test that _check_conversation_size_and_compact doesn't trigger when not needed."""
        # Small conversation
        messages = [ProviderMessage(content="Small message", role="user")]

        executor.state.current_iteration = 2
        executor.state.total_tokens_used = 50  # Low utilization

        result = await executor._check_conversation_size_and_compact(messages)

        # Should not have triggered compaction (messages unchanged)
        assert result == messages
        assert executor.state.compression_count == 0

    # ===== Phase 4: Tests for High/Medium Priority Features =====

    def test_calculate_effective_token_budget(
        self, executor, mock_ai_provider, mock_mcp_manager
    ):
        """Test Phase 4: Conservative Context Estimation."""
        # Test with default limits (max_tokens_total=1000)
        effective_budget = executor._calculate_effective_token_budget()

        # Expected calculation:
        # safe_limit = 1000 * 0.7 = 700
        # overhead = 3000 + 5000 = 8000
        # effective_budget = 700 - 8000 = -7300 -> max(1000) = 1000 (minimum)
        assert effective_budget == 1000  # Minimum budget enforced

        # Test with larger limit where overhead doesn't dominate
        # Create new executor with larger limits (limits are set at initialization)
        large_limits = ExecutionLimits(max_tokens_total=100000)
        large_executor = TokenAwareExecutor(
            ai_provider=mock_ai_provider,
            mcp_manager=mock_mcp_manager,
            limits=large_limits,
        )
        effective_budget = large_executor._calculate_effective_token_budget()

        # Expected: 100000 * 0.7 = 70000, 70000 - 8000 = 62000
        expected = int(100000 * 0.7) - (3000 + 5000)
        assert effective_budget == expected

    def test_identify_tool_pairs(self, executor):
        """Test Phase 4: Tool Pair Preservation - identification logic."""
        # Create mock messages with properly configured tool_calls
        msg_assistant_1 = Mock(role="assistant", content="Using tool")
        msg_assistant_1.tool_calls = [{"id": "call_1", "function": {"name": "tool1"}}]

        msg_tool_1 = Mock(role="tool", content="Tool result")
        msg_tool_1.tool_call_id = "call_1"
        msg_tool_1.tool_calls = None

        msg_assistant_2 = Mock(role="assistant", content="Using another tool")
        msg_assistant_2.tool_calls = [{"id": "call_2", "function": {"name": "tool2"}}]

        msg_tool_2 = Mock(role="tool", content="Another result")
        msg_tool_2.tool_call_id = "call_2"
        msg_tool_2.tool_calls = None

        messages = [
            ProviderMessage(content="User query", role="user"),
            msg_assistant_1,
            msg_tool_1,
            msg_assistant_2,
            msg_tool_2,
        ]

        tool_pairs = executor._identify_tool_pairs(messages)

        # Should identify both pairs
        assert "call_1" in tool_pairs
        assert "call_2" in tool_pairs

        # Verify request/response indices
        assert tool_pairs["call_1"] == (1, 2)  # request at index 1, response at index 2
        assert tool_pairs["call_2"] == (3, 4)  # request at index 3, response at index 4

    def test_identify_tool_pairs_incomplete(self, executor):
        """Test Phase 4: Tool Pair Preservation - handles incomplete pairs."""
        msg_assistant_1 = Mock(role="assistant", content="Using tool")
        msg_assistant_1.tool_calls = [{"id": "call_1", "function": {"name": "tool1"}}]

        msg_assistant_2 = Mock(role="assistant", content="Using another tool")
        msg_assistant_2.tool_calls = [{"id": "call_2", "function": {"name": "tool2"}}]

        messages = [
            ProviderMessage(content="User query", role="user"),
            msg_assistant_1,
            # Missing response for call_1
            msg_assistant_2,
        ]

        tool_pairs = executor._identify_tool_pairs(messages)

        # Should identify call_1 without response
        assert "call_1" in tool_pairs
        assert tool_pairs["call_1"] == (1, None)  # request found, no response yet

        # Should identify call_2 without response
        assert "call_2" in tool_pairs
        assert tool_pairs["call_2"] == (2, None)

    def test_identify_tool_pairs_orphaned_response(self, executor):
        """Test Phase 4: Tool Pair Preservation - handles orphaned responses."""
        msg_tool = Mock(role="tool", content="Orphaned result")
        msg_tool.tool_call_id = "call_orphan"
        msg_tool.tool_calls = None

        messages = [
            ProviderMessage(content="User query", role="user"),
            msg_tool,  # Response without request
        ]

        tool_pairs = executor._identify_tool_pairs(messages)

        # Should identify orphaned response
        assert "call_orphan" in tool_pairs
        assert tool_pairs["call_orphan"] == (None, 1)  # no request, response at index 1

    def test_validate_conversation_start_valid_user(self, executor):
        """Test Phase 4: Message Validation - valid conversation starting with user."""
        messages = [
            ProviderMessage(content="User message", role="user"),
            ProviderMessage(content="Assistant response", role="assistant"),
        ]

        validated = executor._validate_conversation_start(messages)

        # Should not modify valid conversation
        assert validated == messages
        assert len(validated) == 2

    def test_validate_conversation_start_valid_system(self, executor):
        """Test Phase 4: Message Validation - valid conversation starting with system."""
        messages = [
            ProviderMessage(content="System prompt", role="system"),
            ProviderMessage(content="User message", role="user"),
        ]

        validated = executor._validate_conversation_start(messages)

        # Should not modify valid conversation
        assert validated == messages
        assert len(validated) == 2

    def test_validate_conversation_start_invalid_assistant(self, executor):
        """Test Phase 4: Message Validation - invalid conversation starting with assistant."""
        messages = [
            ProviderMessage(content="Assistant message", role="assistant"),
            ProviderMessage(content="User message", role="user"),
        ]

        validated = executor._validate_conversation_start(messages)

        # Should prepend system message
        assert len(validated) == 3
        assert validated[0].role == "system"
        assert validated[0].content == "Previous conversation context summary."
        assert validated[1] == messages[0]
        assert validated[2] == messages[1]

    def test_validate_conversation_start_empty(self, executor):
        """Test Phase 4: Message Validation - empty conversation."""
        messages = []

        validated = executor._validate_conversation_start(messages)

        # Should return empty list unchanged
        assert validated == []

    def test_estimate_tokens_with_tokenizer(self, mock_ai_provider):
        """Test Phase 4: Flexible Token Counting - with tokenizer."""
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

        executor = TokenAwareExecutor(
            ai_provider=mock_ai_provider, tokenizer=mock_tokenizer
        )

        # Test with tokenizer enabled
        token_count = executor.estimate_tokens("Test text", use_tokenizer=True)

        # Should use tokenizer
        assert token_count == 5
        mock_tokenizer.encode.assert_called_once_with("Test text")

    def test_estimate_tokens_tokenizer_fallback(self, mock_ai_provider):
        """Test Phase 4: Flexible Token Counting - tokenizer fallback on error."""
        # Create mock tokenizer that fails
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Tokenizer error")

        executor = TokenAwareExecutor(
            ai_provider=mock_ai_provider, tokenizer=mock_tokenizer
        )

        # Test with tokenizer that fails
        test_text = "Test text with 20 ch"  # 20 characters
        token_count = executor.estimate_tokens(test_text, use_tokenizer=True)

        # Should fall back to approximation (20 / 4 = 5)
        assert token_count == 5

    def test_estimate_tokens_no_tokenizer(self, mock_ai_provider):
        """Test Phase 4: Flexible Token Counting - without tokenizer."""
        executor = TokenAwareExecutor(ai_provider=mock_ai_provider, tokenizer=None)

        test_text = "Test text with 20 ch"  # 20 characters
        token_count = executor.estimate_tokens(test_text, use_tokenizer=True)

        # Should use approximation even when use_tokenizer=True (no tokenizer available)
        assert token_count == 5  # 20 / 4

    @pytest.mark.asyncio
    async def test_generate_final_summary_single_message_unchanged(self, executor):
        """Test that single-message scenarios are unchanged."""
        single_analysis = "Complete code review with bugs A, B, and C identified"

        messages = [
            ProviderMessage(content="User request", role="user"),
            ProviderMessage(content=single_analysis, role="assistant"),
        ]

        summary = await executor._generate_final_summary(messages, [])

        # Should return the single message as-is
        assert summary == single_analysis

    @pytest.mark.asyncio
    async def test_message_validation_after_compaction(
        self, executor, mock_ai_provider
    ):
        """Test Phase 4: Message Validation - applied after compaction."""
        # Create large conversation that will trigger compaction
        large_message = "x" * 1000  # 250 tokens
        messages = [
            ProviderMessage(content="Initial user message", role="user"),
            ProviderMessage(content=large_message, role="assistant"),
        ]

        executor.state.current_iteration = 2

        # Mock compactor to return conversation starting with assistant (invalid)
        mock_compactor = Mock()
        mock_compactor.compact_iteration_context = AsyncMock(
            return_value=(
                [
                    ProviderMessage(
                        content="Compacted assistant message", role="assistant"
                    )
                ],
                Mock(compression_ratio=2.0, information_preserved=0.85),
            )
        )
        executor.compactor = mock_compactor

        result = await executor._check_conversation_size_and_compact(messages)

        # Should have triggered compaction
        assert mock_compactor.compact_iteration_context.called

        # Should have added system message to fix invalid start
        assert len(result) == 2
        assert result[0].role == "system"
        assert result[1].role == "assistant"

    # ===== Phase 5: Tests for Degradation Skip When Recovery Enabled =====

    def test_check_warnings_skips_degradation_for_iter_when_recovery_enabled(
        self, mock_ai_provider, mock_mcp_manager
    ):
        """Test that degradation is SKIPPED for iteration limit when recovery_enabled=True.

        When recovery is enabled, we want execution to continue until the actual limit
        is hit so that recovery can handle continuation with fresh context.
        """
        limits = ExecutionLimits(
            max_infer_iters=10,
            max_tokens_total=100000,
            max_total_tools=100,
            recovery_enabled=True,  # Recovery enabled
        )
        executor = TokenAwareExecutor(
            ai_provider=mock_ai_provider,
            mcp_manager=mock_mcp_manager,
            limits=limits,
        )

        # Set iteration to 90% of limit (should trigger degradation normally)
        executor.state.current_iteration = 9  # 9/10 = 90% >= 90% threshold

        warnings = executor.check_warnings()

        # Degradation should NOT be active (recovery will handle it)
        assert executor.state.degradation_active is False
        # Should still issue warning about approaching limit
        assert any("Approaching iteration limit" in w for w in warnings)
        # Should NOT have "Activating progressive degradation" warning
        assert not any("Activating progressive degradation" in w for w in warnings)

    def test_check_warnings_skips_degradation_for_token_when_recovery_enabled(
        self, mock_ai_provider, mock_mcp_manager
    ):
        """Test that degradation is SKIPPED for token limit when recovery_enabled=True."""
        limits = ExecutionLimits(
            max_infer_iters=10,
            max_tokens_total=100000,
            max_total_tools=100,
            recovery_enabled=True,  # Recovery enabled
        )
        executor = TokenAwareExecutor(
            ai_provider=mock_ai_provider,
            mcp_manager=mock_mcp_manager,
            limits=limits,
        )

        # Set tokens to 90% of effective budget
        # Effective budget = 100000 * 0.7 - 8000 = 62000
        executor.state.total_tokens_used = 56000  # ~90% of 62000

        warnings = executor.check_warnings()

        # Degradation should NOT be active (recovery will handle it)
        assert executor.state.degradation_active is False
        # Should still issue warning about approaching limit
        assert any("Approaching token limit" in w for w in warnings)

    def test_check_warnings_activates_degradation_for_tools_when_recovery_enabled(
        self, mock_ai_provider, mock_mcp_manager
    ):
        """Test that degradation ACTIVATES for tool limit even when recovery_enabled=True.

        Tool limit is not recoverable, so degradation should still activate.
        """
        limits = ExecutionLimits(
            max_infer_iters=10,
            max_tokens_total=100000,
            max_total_tools=50,
            recovery_enabled=True,  # Recovery enabled
        )
        executor = TokenAwareExecutor(
            ai_provider=mock_ai_provider,
            mcp_manager=mock_mcp_manager,
            limits=limits,
        )

        # Set tools to 90% of limit
        executor.state.total_tools_executed = 45  # 45/50 = 90% >= 90% threshold

        warnings = executor.check_warnings()

        # Degradation SHOULD be active (tools are not recoverable)
        assert executor.state.degradation_active is True
        # Should have "Activating progressive degradation" warning
        assert any("Activating progressive degradation" in w for w in warnings)

    def test_check_warnings_activates_degradation_for_iter_when_recovery_disabled(
        self, mock_ai_provider, mock_mcp_manager
    ):
        """Test that degradation ACTIVATES for iteration limit when recovery_enabled=False."""
        limits = ExecutionLimits(
            max_infer_iters=10,
            max_tokens_total=100000,
            max_total_tools=100,
            recovery_enabled=False,  # Recovery disabled
        )
        executor = TokenAwareExecutor(
            ai_provider=mock_ai_provider,
            mcp_manager=mock_mcp_manager,
            limits=limits,
        )

        # Set iteration to 90% of limit
        executor.state.current_iteration = 9  # 9/10 = 90%

        warnings = executor.check_warnings()

        # Degradation SHOULD be active (recovery disabled)
        assert executor.state.degradation_active is True
        # Should have "Activating progressive degradation" warning
        assert any("Activating progressive degradation" in w for w in warnings)

    def test_check_warnings_logs_skip_reason_only_once(
        self, mock_ai_provider, mock_mcp_manager
    ):
        """Test that the 'skipping degradation' log message only appears once."""
        limits = ExecutionLimits(
            max_infer_iters=10,
            max_tokens_total=100000,
            max_total_tools=100,
            recovery_enabled=True,
        )
        executor = TokenAwareExecutor(
            ai_provider=mock_ai_provider,
            mcp_manager=mock_mcp_manager,
            limits=limits,
        )

        # Set iteration to 90%
        executor.state.current_iteration = 9

        # Call check_warnings multiple times
        executor.check_warnings()
        executor.check_warnings()
        executor.check_warnings()

        # The _degradation_skip_logged flag should be set
        assert executor.state._degradation_skip_logged is True
        # Degradation should still not be active
        assert executor.state.degradation_active is False

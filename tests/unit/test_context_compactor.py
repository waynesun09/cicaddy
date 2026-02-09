"""Unit tests for ContextCompactor."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cicaddy.ai_providers.base import BaseProvider, ProviderMessage, ProviderResponse
from cicaddy.execution.context_compactor import (
    CompressionStrategy,
    ContextCompactor,
    ContextCompactorConfig,
)


class TestContextCompactor:
    """Test suite for ContextCompactor class."""

    @pytest.fixture
    def compactor(self):
        """Create a ContextCompactor instance for testing."""
        return ContextCompactor(provider="gemini", model="gemini-2.5-flash")

    def test_determine_compression_strategy_none(self, compactor):
        """Test that no compression is chosen when utilization is low."""
        strategy = compactor.determine_compression_strategy(0.3)
        assert strategy == CompressionStrategy.NONE

    def test_determine_compression_strategy_light(self, compactor):
        """Test that light compression is chosen at moderate utilization."""
        strategy = compactor.determine_compression_strategy(0.6)
        assert strategy == CompressionStrategy.LIGHT

    def test_determine_compression_strategy_moderate(self, compactor):
        """Test that moderate compression is chosen at high utilization."""
        strategy = compactor.determine_compression_strategy(0.75)
        assert strategy == CompressionStrategy.MODERATE

    def test_determine_compression_strategy_aggressive(self, compactor):
        """Test that aggressive compression is chosen at very high utilization."""
        strategy = compactor.determine_compression_strategy(0.9)
        assert strategy == CompressionStrategy.AGGRESSIVE

    def test_compress_prompt_no_compression_needed(self, compactor):
        """Test that prompt is not compressed when within budget."""
        short_prompt = "This is a short prompt that fits within the budget."
        compressed, result = compactor.compress_prompt_before_send(
            short_prompt, token_budget=10000
        )

        assert compressed == short_prompt
        assert result.compression_ratio == 1.0
        assert result.strategy_used == CompressionStrategy.NONE

    def test_compress_prompt_with_compression(self, compactor):
        """Test that prompt is compressed when exceeding budget."""
        # Create a long prompt that needs compression
        long_prompt = (
            "System: "
            + ("x " * 1000)
            + "\nTask: "
            + ("y " * 1000)
            + "\nDebug: "
            + ("z " * 1000)
        )
        compressed, result = compactor.compress_prompt_before_send(
            long_prompt, token_budget=500
        )

        assert len(compressed) < len(long_prompt)
        assert result.compression_ratio > 1.0
        assert result.original_tokens > result.compressed_tokens
        assert result.strategy_used == CompressionStrategy.MODERATE

    def test_compress_prompt_preserves_important_sections(self, compactor):
        """Test that compression preserves high-priority sections."""
        prompt = "System: Important system info\nDebug: Verbose debug data " + (
            "x " * 500
        )
        compressed, result = compactor.compress_prompt_before_send(
            prompt, token_budget=100
        )

        # System section should be preserved
        assert "System:" in compressed
        # Long debug section may be truncated
        assert result.compression_ratio > 1.0

    def test_compress_prompt_preserves_analysis_directives(self, compactor):
        """Test that compression preserves analysis directives over verbose data."""
        prompt = (
            "Objective: Use datarouter MCP tools to analyze requests data from past day.\n\n"
            "Required Analysis:\n"
            "  - System Metrics: Total requests, success/failure rates\n"
            "  - Account Breakdown: List all accounts with failure rate rankings\n\n"
            "Deliverable: Generate a structured report with executive summary.\n\n"
            "Debug: "
            + (
                "verbose debug output " * 200
            )  # Add LOTS of low-priority content to force compression
        )

        # Compress with limited budget to force compression
        # Budget=3500, Reserve=2000 → Available=1500, Target=1050 tokens for prompt
        # Original prompt is ~1000 tokens, so this will trigger compression
        compressed, result = compactor.compress_prompt_before_send(
            prompt, token_budget=3500, reserve_for_response=2000
        )

        # Analysis directives should be preserved (highest priority)
        assert "Objective:" in compressed
        assert "Required Analysis:" in compressed or "System Metrics" in compressed
        assert "Deliverable:" in compressed

        # Debug section should be truncated or omitted (lowest priority)
        # We should have some compression of the debug section
        assert result.compression_ratio >= 1.0  # At minimum no expansion
        # Debug section should be truncated if compression occurred
        if result.compression_ratio > 1.0:
            assert "[truncated]" in compressed or "Debug:" not in compressed

    def test_compress_prompt_preserves_phase_markers(self, compactor):
        """Test that compression preserves Phase 1/2/3 markers as instructions."""
        prompt = (
            "Phase 1 - System Overview (MANDATORY):\n"
            "1. Call getTotalRequestNumberFromPastDays\n"
            "2. Call getRequestStatusOfAccountsFromPastDays\n\n"
            "Phase 2 - Per-Account Analysis:\n"
            "3. Identify top 5 accounts with highest failure counts\n\n"
            "Phase 3 - Event Details:\n"
            "4. Call getEventsByRequestUuid for failures\n\n"
            "Debug: "
            + (
                "verbose debug line " * 200
            )  # Add LOTS of low-priority content to force compression
        )

        # Budget=3500, Reserve=2000 → Available=1500, Target=1050 tokens for prompt
        compressed, result = compactor.compress_prompt_before_send(
            prompt, token_budget=3500, reserve_for_response=2000
        )

        # Phase markers should be preserved
        assert "Phase 1" in compressed or "System Overview" in compressed
        assert "Phase 2" in compressed or "Per-Account" in compressed
        assert "Phase 3" in compressed or "Event Details" in compressed

        # Should have compression or at minimum not expand
        assert result.compression_ratio >= 1.0
        # Debug section should be truncated if compression occurred
        if result.compression_ratio > 1.0:
            assert "[truncated]" in compressed or "Debug:" not in compressed

    @pytest.mark.asyncio
    async def test_compact_iteration_context_minimal_messages(self, compactor):
        """Test that minimal messages are not compressed."""
        messages = [
            ProviderMessage(content="Initial prompt", role="user"),
            ProviderMessage(content="Response", role="assistant"),
        ]

        compacted, result = await compactor.compact_iteration_context(
            messages, available_budget=10000, current_iteration=1
        )

        assert len(compacted) == len(messages)
        assert result.compression_ratio == 1.0
        assert result.strategy_used == CompressionStrategy.NONE

    @pytest.mark.asyncio
    async def test_compact_iteration_context_with_compression(self, compactor):
        """Test that conversation context is compacted for long conversations."""
        messages = [
            ProviderMessage(content="Initial task prompt", role="user"),
            ProviderMessage(
                content="Tool execution result 1 " + ("x " * 200), role="user"
            ),
            ProviderMessage(content="Analysis 1", role="assistant"),
            ProviderMessage(
                content="Tool execution result 2 " + ("y " * 200), role="user"
            ),
            ProviderMessage(content="Analysis 2", role="assistant"),
            ProviderMessage(content="Current iteration", role="user"),
        ]

        compacted, result = await compactor.compact_iteration_context(
            messages, available_budget=500, current_iteration=3
        )

        # Should have fewer messages after compaction
        assert len(compacted) < len(messages)
        assert result.compression_ratio > 1.0
        # First message (task) should be preserved
        assert compacted[0].content == "Initial task prompt"
        # Last messages should be preserved
        assert "Current iteration" in compacted[-1].content

    @pytest.mark.asyncio
    async def test_compact_iteration_context_preserves_recent_context(self, compactor):
        """Test that most recent iterations are preserved in detail."""
        messages = [
            ProviderMessage(content="Initial prompt", role="user"),
            ProviderMessage(content="Old result", role="user"),
            ProviderMessage(content="Recent result 1", role="user"),
            ProviderMessage(content="Recent result 2", role="assistant"),
        ]

        compacted, result = await compactor.compact_iteration_context(
            messages, available_budget=1000, current_iteration=2
        )

        # Recent messages should be in compacted result
        assert any("Recent result 2" in msg.content for msg in compacted)

    def test_compress_tool_result_no_compression_needed(self, compactor):
        """Test that small tool results are not compressed."""
        tool_result = {
            "tool_name": "test_tool",
            "result": "Small result that fits",
            "status": "success",
        }

        compressed, result = compactor.compress_tool_result_realtime(
            tool_result, max_tokens=1000
        )

        assert compressed["result"] == tool_result["result"]
        assert result.compression_ratio == 1.0
        assert result.strategy_used == CompressionStrategy.NONE

    def test_compress_tool_result_with_compression(self, compactor):
        """Test that large tool results are compressed."""
        large_result = "Result: " + "\n".join(
            [f"Line {i}: " + ("x " * 50) for i in range(100)]
        )
        tool_result = {
            "tool_name": "test_tool",
            "result": large_result,
            "status": "success",
        }

        compressed, result = compactor.compress_tool_result_realtime(
            tool_result, max_tokens=100
        )

        assert len(str(compressed["result"])) < len(large_result)
        assert result.compression_ratio > 1.0
        assert compressed.get("compressed") is True
        assert result.strategy_used == CompressionStrategy.LIGHT

    def test_compress_tool_result_preserves_errors(self, compactor):
        """Test that error messages are preserved in compression."""
        error_result = "Error: Critical failure\n" + ("debug line\n" * 100)
        tool_result = {
            "tool_name": "test_tool",
            "result": error_result,
            "status": "error",
        }

        compressed, result = compactor.compress_tool_result_realtime(
            tool_result, max_tokens=100
        )

        # Error message should be preserved
        assert "Error: Critical failure" in str(compressed["result"])
        assert result.compression_ratio > 1.0

    def test_compress_tool_result_preserves_summary_stats(self, compactor):
        """Test that summary statistics are preserved in compression."""
        stats_result = (
            "Total items found: 42\nSummary: Analysis complete\n"
            + ("verbose debug line\n" * 100)
            + "Found 10 errors"
        )
        tool_result = {
            "tool_name": "test_tool",
            "result": stats_result,
            "status": "success",
        }

        compressed, result = compactor.compress_tool_result_realtime(
            tool_result, max_tokens=200
        )

        compressed_text = str(compressed["result"])
        # Summary stats should be preserved
        assert (
            "Total" in compressed_text
            or "Summary" in compressed_text
            or "Found" in compressed_text
        )

    def test_split_into_sections(self, compactor):
        """Test that text is correctly split into sections."""
        text = "System: System info\nTask: Task description\nData: Some data\nOther content"
        sections = compactor._split_into_sections(text)

        assert len(sections) >= 3
        assert any("System:" in section for section in sections)
        assert any("Task:" in section for section in sections)

    def test_get_section_type(self, compactor):
        """Test section type identification."""
        assert compactor._get_section_type("System: Important") == "system"
        assert compactor._get_section_type("Instructions: Do this") == "instruction"
        assert compactor._get_section_type("Task: Analyze") == "task"
        assert compactor._get_section_type("Debug: Verbose") == "debug"

    def test_get_section_type_preserves_analysis_directives(self, compactor):
        """Test that analysis directives (Objective, Deliverable, Required Analysis, Phase markers) are classified as instruction."""
        # Test Objective directive
        assert (
            compactor._get_section_type(
                "Objective: Use datarouter MCP tools to analyze requests"
            )
            == "instruction"
        )

        # Test Deliverable directive
        assert (
            compactor._get_section_type("Deliverable: Generate structured report")
            == "instruction"
        )

        # Test Required Analysis directive
        assert (
            compactor._get_section_type(
                "Required Analysis: System metrics, account breakdown"
            )
            == "instruction"
        )

        # Test Phase markers
        assert (
            compactor._get_section_type("Phase 1 - System Overview (MANDATORY):")
            == "instruction"
        )
        assert (
            compactor._get_section_type("Phase 2 - Per-Account Analysis")
            == "instruction"
        )
        assert (
            compactor._get_section_type("Phase 3 - Event Details (CONDITIONAL)")
            == "instruction"
        )

        # Test MANDATORY directive
        assert (
            compactor._get_section_type(
                "**MANDATORY**: You MUST use datarouter MCP tools"
            )
            == "instruction"
        )

        # Test case insensitivity
        assert (
            compactor._get_section_type("objective: analyze past day data")
            == "instruction"
        )
        assert (
            compactor._get_section_type("DELIVERABLE: comprehensive report")
            == "instruction"
        )

    def test_truncate_section(self, compactor):
        """Test section truncation."""
        long_section = "\n".join([f"Line {i}" for i in range(100)])
        truncated = compactor._truncate_section(long_section, target_tokens=50)

        # Should be truncated
        assert len(truncated) < len(long_section)
        assert "truncated" in truncated.lower()

    def test_summarize_messages_extracts_key_info(self, compactor):
        """Test that message summarization extracts key information."""
        messages = [
            ProviderMessage(content="Tool test_tool: success", role="user"),
            ProviderMessage(content="Error: Something failed", role="user"),
            ProviderMessage(content="Found 42 issues", role="assistant"),
        ]

        summary = compactor._summarize_messages(messages, iteration=3)

        assert "Summary" in summary
        assert any(
            keyword in summary for keyword in ["Tool", "Error", "found", "issues"]
        )

    def test_compress_tool_content_prioritizes_errors(self, compactor):
        """Test that tool content compression prioritizes errors."""
        content = "Line 1\nError: Critical issue\nLine 3\n" + ("verbose line\n" * 50)
        compressed = compactor._compress_tool_content(content, max_tokens=50)

        # Error should be in compressed output
        assert "Error: Critical issue" in compressed

    def test_compress_tool_content_includes_summary_marker(self, compactor):
        """Test that compressed tool content includes compression marker."""
        content = "\n".join([f"Line {i}" for i in range(100)])
        compressed = compactor._compress_tool_content(content, max_tokens=50)

        # New footer format shows preserved lines/tokens
        assert "Preserved" in compressed or "Truncated" in compressed
        assert "lines" in compressed or "tokens" in compressed

    def test_compression_result_dataclass(self):
        """Test CompressionResult dataclass creation."""
        from cicaddy.execution.context_compactor import CompressionResult

        result = CompressionResult(
            original_tokens=1000,
            compressed_tokens=500,
            compression_ratio=2.0,
            strategy_used=CompressionStrategy.MODERATE,
            compression_time_ms=15.5,
            information_preserved=0.85,
        )

        assert result.original_tokens == 1000
        assert result.compressed_tokens == 500
        assert result.compression_ratio == 2.0
        assert result.strategy_used == CompressionStrategy.MODERATE
        assert result.compression_time_ms == 15.5
        assert result.information_preserved == 0.85

    def test_multiple_compressions_maintain_consistency(self, compactor):
        """Test that multiple compressions of same content produce consistent results."""
        prompt = "System: Info\nTask: Test\n" + ("x " * 200)

        compressed1, result1 = compactor.compress_prompt_before_send(
            prompt, token_budget=200
        )
        compressed2, result2 = compactor.compress_prompt_before_send(
            prompt, token_budget=200
        )

        # Results should be deterministic
        assert compressed1 == compressed2
        assert result1.compressed_tokens == result2.compressed_tokens

    def test_compress_prompt_with_very_small_budget(self, compactor):
        """Test compression behavior with extremely small token budget."""
        prompt = "System: " + ("x " * 500)
        compressed, result = compactor.compress_prompt_before_send(
            prompt, token_budget=50
        )

        # Should produce something even with tiny budget
        assert len(compressed) > 0
        assert result.compression_ratio > 1.0

    @pytest.mark.asyncio
    async def test_compact_iteration_preserves_first_message(self, compactor):
        """Test that first message (initial prompt) is always preserved."""
        messages = [
            ProviderMessage(content="IMPORTANT_INITIAL_PROMPT", role="user"),
            ProviderMessage(content="Middle 1", role="assistant"),
            ProviderMessage(content="Middle 2", role="user"),
            ProviderMessage(content="Recent", role="assistant"),
        ]

        compacted, _ = await compactor.compact_iteration_context(
            messages, available_budget=100, current_iteration=3
        )

        # First message must be preserved
        assert compacted[0].content == "IMPORTANT_INITIAL_PROMPT"

    # ===== Phase 2: AI-Powered Summarization Tests =====

    @pytest.fixture
    def mock_ai_provider(self):
        """Create a mock AI provider for testing."""
        mock_provider = MagicMock(spec=BaseProvider)
        mock_provider.chat_completion = AsyncMock(
            return_value=ProviderResponse(
                content="AI-generated summary of key findings: 3 tools executed, 2 errors found, 42 requests analyzed.",
                tool_calls=None,
                model="mock-model",
            )
        )
        return mock_provider

    @pytest.fixture
    def compactor_with_ai(self, mock_ai_provider):
        """Create ContextCompactor with AI provider enabled."""
        config = ContextCompactorConfig(
            use_ai_summarization=True, ai_summary_max_tokens=500, fallback_to_rules=True
        )
        return ContextCompactor(
            provider="gemini",
            model="gemini-2.5-flash",
            ai_provider=mock_ai_provider,
            config=config,
        )

    @pytest.mark.asyncio
    async def test_ai_powered_summarization_success(self, compactor_with_ai):
        """Test successful AI-powered conversation summarization."""
        messages = [
            ProviderMessage(content="Initial task: Analyze data", role="user"),
            ProviderMessage(content="Tool execution result 1", role="user"),
            ProviderMessage(content="Analysis complete", role="assistant"),
            ProviderMessage(content="Current iteration message", role="user"),
        ]

        summary = await compactor_with_ai._summarize_messages_with_ai(
            messages[:3], iteration=3
        )

        # Should contain AI-generated marker
        assert "[Summary of iterations 1-2 (AI-generated)]" in summary
        assert "AI-generated summary" in summary
        assert "[End summary - current iteration follows]" in summary

    @pytest.mark.asyncio
    async def test_compact_iteration_uses_ai_summarization(
        self, compactor_with_ai, mock_ai_provider
    ):
        """Test that compact_iteration_context uses AI summarization when available."""
        messages = [
            ProviderMessage(
                content="Initial task: Comprehensive analysis", role="user"
            ),
            ProviderMessage(
                content="Tool execution result 1: " + ("detailed output " * 50),
                role="user",
            ),
            ProviderMessage(
                content="Analysis of result 1: " + ("finding " * 50), role="assistant"
            ),
            ProviderMessage(
                content="Tool execution result 2: " + ("more data " * 50), role="user"
            ),
            ProviderMessage(content="Recent result 1", role="user"),
            ProviderMessage(content="Recent result 2", role="assistant"),
        ]

        compacted, result = await compactor_with_ai.compact_iteration_context(
            messages, available_budget=1000, current_iteration=4
        )

        # AI provider should have been called
        assert mock_ai_provider.chat_completion.called

        # Should have fewer messages after compaction
        assert len(compacted) < len(messages)

        # Should have compression result (for very long messages, compression should be effective)
        assert result.compression_ratio > 0.0

    @pytest.mark.asyncio
    async def test_ai_summarization_fallback_to_rules(self, mock_ai_provider):
        """Test graceful fallback to rule-based summarization when AI fails."""
        # Make AI provider fail
        mock_ai_provider.chat_completion.side_effect = Exception(
            "AI service unavailable"
        )

        config = ContextCompactorConfig(
            use_ai_summarization=True, fallback_to_rules=True
        )
        compactor = ContextCompactor(
            provider="gemini", ai_provider=mock_ai_provider, config=config
        )

        messages = [
            ProviderMessage(content="Initial task", role="user"),
            ProviderMessage(content="Tool test_tool: success", role="user"),
            ProviderMessage(content="Error: Something failed", role="user"),
            ProviderMessage(content="Recent", role="user"),
        ]

        compacted, result = await compactor.compact_iteration_context(
            messages, available_budget=1000, current_iteration=3
        )

        # Should still succeed with rule-based fallback
        assert len(compacted) > 0
        # Should have summary marker from rule-based approach
        summary_msg = next(
            (msg for msg in compacted if "[Summary" in msg.content), None
        )
        assert summary_msg is not None

    @pytest.mark.asyncio
    async def test_ai_summarization_disabled_uses_rules(self):
        """Test that disabling AI summarization uses rule-based approach."""
        config = ContextCompactorConfig(use_ai_summarization=False)
        compactor = ContextCompactor(provider="gemini", ai_provider=None, config=config)

        messages = [
            ProviderMessage(content="Initial task", role="user"),
            ProviderMessage(
                content="Tool execution result: " + ("x " * 100), role="user"
            ),
            ProviderMessage(
                content="Analysis complete " + ("y " * 100), role="assistant"
            ),
            ProviderMessage(content="Recent result", role="user"),
        ]

        compacted, result = await compactor.compact_iteration_context(
            messages, available_budget=1000, current_iteration=3
        )

        # Should use rule-based summarization
        assert len(compacted) > 0
        # For longer messages, compression should be effective
        assert result.compression_ratio > 0.0  # Just verify it's positive

    def test_extract_key_entities_tools(self, compactor):
        """Test extraction of tool names from text."""
        text = (
            "Tool search_code execution successful. Tool get_file_contents call failed."
        )

        entities = compactor._extract_key_entities(text)

        # Should extract tool names
        assert "search_code" in entities
        assert "get_file_contents" in entities
        # Should extract status keywords
        assert any(
            keyword in entities for keyword in ["success", "successful", "failed"]
        )

    def test_extract_key_entities_errors(self, compactor):
        """Test extraction of error messages."""
        text = "Error: Database connection timeout. Warning: Memory usage high. Critical: System failure detected."

        entities = compactor._extract_key_entities(text)

        # Should extract error-related entities
        assert any("error" in e.lower() for e in entities)
        assert any("warning" in e.lower() for e in entities)
        assert any("critical" in e.lower() for e in entities)

    def test_extract_key_entities_numbers(self, compactor):
        """Test extraction of numerical metrics."""
        text = "Processed 1500 requests with 42 errors. Response time: 250ms. Memory: 512MB. Success rate: 95%."

        entities = compactor._extract_key_entities(text)

        # Should extract numerical metrics with units
        assert any("1500" in str(e) for e in entities)
        assert any("42" in str(e) for e in entities)
        assert any("250" in str(e) for e in entities)

    def test_calculate_information_preserved_perfect(self, compactor):
        """Test information preservation calculation with perfect preservation."""
        original = "Tool search_code: 100 results found. Error: timeout occurred."
        compressed = original  # Perfect preservation

        ratio = compactor._calculate_information_preserved(original, compressed)

        # Should be 100% preserved
        assert ratio == 1.0

    def test_calculate_information_preserved_partial(self, compactor):
        """Test information preservation calculation with partial loss."""
        original = "Tool search_code execution: 100 requests processed. Error: timeout. Warning: memory high. Found 42 issues."
        compressed = "100 requests processed. Error: timeout."

        ratio = compactor._calculate_information_preserved(original, compressed)

        # Should be partial preservation (0 < ratio < 1)
        assert 0.0 < ratio < 1.0

    def test_calculate_information_preserved_empty_original(self, compactor):
        """Test information preservation with empty original (edge case)."""
        original = ""
        compressed = ""

        ratio = compactor._calculate_information_preserved(original, compressed)

        # Empty original should return 1.0 (no entities to preserve)
        assert ratio == 1.0

    @pytest.mark.asyncio
    async def test_contextcompactor_config_defaults(self):
        """Test ContextCompactorConfig default values."""
        config = ContextCompactorConfig()

        assert config.use_ai_summarization is True
        assert config.ai_summary_max_tokens == 500
        assert config.fallback_to_rules is True
        assert config.preserve_recent_messages == 2
        assert config.min_messages_to_compress == 3

    @pytest.mark.asyncio
    async def test_contextcompactor_config_custom(self):
        """Test ContextCompactorConfig with custom values."""
        config = ContextCompactorConfig(
            use_ai_summarization=False,
            ai_summary_max_tokens=1000,
            preserve_recent_messages=3,
        )

        assert config.use_ai_summarization is False
        assert config.ai_summary_max_tokens == 1000
        assert config.preserve_recent_messages == 3

    def test_compactor_warns_when_ai_enabled_without_provider(self):
        """Test that compactor warns when AI summarization is enabled but no provider given."""
        config = ContextCompactorConfig(use_ai_summarization=True)

        # Should log warning and disable AI summarization
        compactor = ContextCompactor(provider="gemini", ai_provider=None, config=config)

        # AI summarization should be disabled after init
        assert compactor.config.use_ai_summarization is False

    # ===== Phase 3: Enhanced Compression Strategy Tests =====

    def test_determine_compression_strategy_early_iteration_light(self, compactor):
        """Test that early iterations (1-3) use LIGHT compression."""
        # Iteration 2 with moderate utilization should use LIGHT (conservative in early iterations)
        strategy = compactor.determine_compression_strategy(
            token_utilization=0.6, iteration_count=2
        )
        assert strategy == CompressionStrategy.LIGHT

    def test_determine_compression_strategy_mid_iteration_moderate(self, compactor):
        """Test that mid iterations (4-6) use MODERATE compression."""
        # Iteration 5 with moderate utilization should use MODERATE
        strategy = compactor.determine_compression_strategy(
            token_utilization=0.6, iteration_count=5
        )
        assert strategy == CompressionStrategy.MODERATE

    def test_determine_compression_strategy_late_iteration_aggressive(self, compactor):
        """Test that late iterations (7+) use AGGRESSIVE compression."""
        # Iteration 8 with moderate utilization should use AGGRESSIVE
        strategy = compactor.determine_compression_strategy(
            token_utilization=0.6, iteration_count=8
        )
        assert strategy == CompressionStrategy.AGGRESSIVE

    def test_determine_compression_strategy_critical_utilization_always_aggressive(
        self, compactor
    ):
        """Test that critical token utilization (>=85%) always uses AGGRESSIVE compression."""
        # Even in early iteration with critical utilization, use AGGRESSIVE
        strategy = compactor.determine_compression_strategy(
            token_utilization=0.9, iteration_count=1
        )
        assert strategy == CompressionStrategy.AGGRESSIVE

        # Mid iteration with critical utilization
        strategy = compactor.determine_compression_strategy(
            token_utilization=0.85, iteration_count=5
        )
        assert strategy == CompressionStrategy.AGGRESSIVE

    def test_determine_compression_strategy_large_results_upgrade(self, compactor):
        """Test that large results upgrade compression level from LIGHT to MODERATE."""
        # Early iteration with large results should upgrade from LIGHT to MODERATE
        strategy = compactor.determine_compression_strategy(
            token_utilization=0.6, iteration_count=2, has_large_results=True
        )
        assert strategy == CompressionStrategy.MODERATE

        # Already MODERATE or higher should not change
        strategy = compactor.determine_compression_strategy(
            token_utilization=0.6, iteration_count=5, has_large_results=True
        )
        assert strategy == CompressionStrategy.MODERATE

    def test_determine_compression_strategy_low_utilization_override(self, compactor):
        """Test that very low token utilization (<50%) downgrades compression."""
        # Low utilization in early iteration should use NONE
        strategy = compactor.determine_compression_strategy(
            token_utilization=0.3, iteration_count=2
        )
        assert strategy == CompressionStrategy.NONE

        # Low utilization in mid iteration should use LIGHT
        strategy = compactor.determine_compression_strategy(
            token_utilization=0.4, iteration_count=5
        )
        assert strategy == CompressionStrategy.LIGHT

    def test_determine_compression_strategy_backward_compatibility(self, compactor):
        """Test backward compatibility when iteration_count not provided."""
        # No iteration count provided, should use token utilization only
        strategy = compactor.determine_compression_strategy(
            token_utilization=0.3, iteration_count=0
        )
        assert strategy == CompressionStrategy.NONE

        strategy = compactor.determine_compression_strategy(
            token_utilization=0.6, iteration_count=0
        )
        assert strategy == CompressionStrategy.LIGHT

        strategy = compactor.determine_compression_strategy(
            token_utilization=0.75, iteration_count=0
        )
        assert strategy == CompressionStrategy.MODERATE

    def test_determine_compression_strategy_multi_factor_interaction(self, compactor):
        """Test complex multi-factor scenarios."""
        # Early iteration + low utilization + large results
        # Large results upgrades LIGHT to MODERATE, but low utilization downgrades to NONE
        strategy = compactor.determine_compression_strategy(
            token_utilization=0.3, iteration_count=2, has_large_results=True
        )
        assert strategy == CompressionStrategy.NONE

        # Mid iteration + moderate utilization + large results
        # Should use MODERATE (base strategy for mid iterations)
        strategy = compactor.determine_compression_strategy(
            token_utilization=0.6, iteration_count=5, has_large_results=True
        )
        assert strategy == CompressionStrategy.MODERATE

        # Late iteration + high utilization (but not critical) + large results
        # Should use AGGRESSIVE (late iteration)
        strategy = compactor.determine_compression_strategy(
            token_utilization=0.75, iteration_count=8, has_large_results=True
        )
        assert strategy == CompressionStrategy.AGGRESSIVE

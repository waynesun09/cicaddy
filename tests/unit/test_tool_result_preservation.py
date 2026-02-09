"""Test tool result preservation during context compaction."""

import pytest

from cicaddy.ai_providers.base import ProviderMessage
from cicaddy.execution.context_compactor import (
    ContextCompactor,
    ContextCompactorConfig,
)


class TestToolResultPreservation:
    """Test that tool results marked with DO_NOT_COMPACT are preserved during compaction."""

    @pytest.fixture
    def compactor(self):
        """Create a context compactor for testing."""
        config = ContextCompactorConfig(
            use_ai_summarization=False,  # Disable AI for deterministic tests
            preserve_recent_messages=2,
        )
        return ContextCompactor(
            provider="openai",
            model="gpt-4",
            ai_provider=None,
            config=config,  # Need a provider string for token counting
        )

    @pytest.mark.asyncio
    async def test_tool_result_messages_preserved(self, compactor):
        """Test that messages with [TOOL_RESULTS:DO_NOT_COMPACT] marker are preserved."""
        messages = [
            ProviderMessage(role="user", content="Analyze DataRouter metrics"),
            ProviderMessage(role="assistant", content="I'll analyze the metrics..."),
            ProviderMessage(
                role="user",
                content="[TOOL_RESULTS:DO_NOT_COMPACT]\nTool execution results:\n"
                "=== GETREQUESTSTATUS RESULT ===\n"
                "Accounts: idm-ci (12000 failures), aosqe-oidc-sa (1800 failures)\n"
                "=== END GETREQUESTSTATUS ===",
            ),
            ProviderMessage(
                role="assistant", content="Based on the data, I can see..."
            ),
            ProviderMessage(
                role="user",
                content="[TOOL_RESULTS:DO_NOT_COMPACT]\nTool execution results:\n"
                "=== GETREQUESTBYACCOUNTNAME RESULT ===\n"
                "Account: idm-ci - detailed failure data...\n"
                "=== END GETREQUESTBYACCOUNTNAME ===",
            ),
            ProviderMessage(role="assistant", content="Let me analyze this further..."),
            ProviderMessage(role="assistant", content="Here's my final analysis..."),
            ProviderMessage(role="user", content="Thanks!"),
        ]

        # Compact with small budget to trigger compression
        max_tokens_per_iteration = 500  # Small budget
        current_iteration = 1
        tool_pairs = {}  # No tool pairs

        compacted, result = await compactor.compact_iteration_context(
            messages, max_tokens_per_iteration, current_iteration, tool_pairs
        )

        # Convert to list of contents for easier assertion
        compacted_contents = [msg.content for msg in compacted]

        # Verify tool result messages are preserved
        assert any(
            "[TOOL_RESULTS:DO_NOT_COMPACT]" in content and "idm-ci" in content
            for content in compacted_contents
        ), "Tool result message with idm-ci data should be preserved"

        assert any(
            "[TOOL_RESULTS:DO_NOT_COMPACT]" in content
            and "GETREQUESTBYACCOUNTNAME" in content
            for content in compacted_contents
        ), "Tool result message with GETREQUESTBYACCOUNTNAME should be preserved"

        # Verify compaction happened (some non-tool messages compressed)
        assert len(compacted) < len(messages), (
            "Some messages should have been compacted"
        )

    @pytest.mark.asyncio
    async def test_non_tool_messages_can_be_compacted(self, compactor):
        """Test that non-tool messages can still be compacted normally."""
        messages = [
            ProviderMessage(role="user", content="Analyze the system"),
            ProviderMessage(
                role="assistant",
                content="This is a very long explanation about how I'm going to analyze the system. "
                * 20,
            ),
            ProviderMessage(
                role="assistant", content="More verbose explanation..." * 10
            ),
            ProviderMessage(role="assistant", content="Final thoughts..."),
        ]

        max_tokens_per_iteration = 200  # Small budget to trigger compression
        current_iteration = 1

        compacted, result = await compactor.compact_iteration_context(
            messages, max_tokens_per_iteration, current_iteration, None
        )

        # Should compress/summarize middle messages
        assert len(compacted) <= len(messages)
        assert result.compression_ratio > 1.0  # Some compression happened

    @pytest.mark.asyncio
    async def test_multiple_tool_results_all_preserved(self, compactor):
        """Test that multiple tool result messages are all preserved."""
        messages = [
            ProviderMessage(role="user", content="Run analysis"),
            ProviderMessage(
                role="user",
                content="[TOOL_RESULTS:DO_NOT_COMPACT]\nTool 1 results:\nAccount: account1",
            ),
            ProviderMessage(role="assistant", content="Analyzing..."),
            ProviderMessage(
                role="user",
                content="[TOOL_RESULTS:DO_NOT_COMPACT]\nTool 2 results:\nAccount: account2",
            ),
            ProviderMessage(role="assistant", content="More analysis..."),
            ProviderMessage(
                role="user",
                content="[TOOL_RESULTS:DO_NOT_COMPACT]\nTool 3 results:\nAccount: account3",
            ),
            ProviderMessage(role="assistant", content="Final analysis..."),
        ]

        max_tokens_per_iteration = 300
        current_iteration = 1

        compacted, result = await compactor.compact_iteration_context(
            messages, max_tokens_per_iteration, current_iteration, None
        )

        # All tool result messages should be preserved
        tool_result_count = sum(
            1 for msg in compacted if "[TOOL_RESULTS:DO_NOT_COMPACT]" in msg.content
        )
        assert tool_result_count == 3, "All 3 tool result messages should be preserved"

        # Verify each account is still present
        all_content = " ".join(msg.content for msg in compacted)
        assert "account1" in all_content
        assert "account2" in all_content
        assert "account3" in all_content

    @pytest.mark.asyncio
    async def test_tool_results_preserved_even_with_tool_pairs(self, compactor):
        """Test that DO_NOT_COMPACT marker works even when tool pairs are present."""
        messages = [
            ProviderMessage(role="user", content="Initial request"),
            ProviderMessage(
                role="user",
                content="[TOOL_RESULTS:DO_NOT_COMPACT]\nCritical data: account_xyz",
            ),
            ProviderMessage(role="assistant", content="Processing..."),
        ]

        # Simulate tool pairs (would normally preserve certain messages)
        tool_pairs = {"tool_1": (0, 2)}  # Request at 0, response at 2

        max_tokens_per_iteration = 200
        current_iteration = 1

        compacted, result = await compactor.compact_iteration_context(
            messages, max_tokens_per_iteration, current_iteration, tool_pairs
        )

        # Tool result message should be preserved regardless of tool pairs
        assert any("[TOOL_RESULTS:DO_NOT_COMPACT]" in msg.content for msg in compacted)
        assert any("account_xyz" in msg.content for msg in compacted)

    @pytest.mark.asyncio
    async def test_empty_marker_no_effect(self, compactor):
        """Test that messages without marker can be compacted normally."""
        messages = [
            ProviderMessage(role="user", content="Request"),
            ProviderMessage(
                role="user",
                content="Regular tool results without marker\nSome data here",
            ),
            ProviderMessage(role="assistant", content="Long response..." * 20),
            ProviderMessage(role="assistant", content="Final"),
        ]

        max_tokens_per_iteration = 150
        current_iteration = 1

        compacted, result = await compactor.compact_iteration_context(
            messages, max_tokens_per_iteration, current_iteration, None
        )

        # Without marker, tool results can be compacted/summarized
        # Should have fewer messages or compressed content
        assert len(compacted) <= len(messages)

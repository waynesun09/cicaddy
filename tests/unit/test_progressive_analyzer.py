"""Unit tests for ProgressiveAnalyzer and related components."""

from unittest.mock import Mock, patch

import pytest

from cicaddy.ai_providers.base import BaseProvider
from cicaddy.execution.progressive_analyzer import (
    AnalysisLevel,
    DegradationStrategy,
    ProgressiveAnalyzer,
)
from cicaddy.execution.result_formatter import ResultPriority
from cicaddy.execution.token_aware_executor import ExecutionLimits, ExecutionState


class TestAnalysisLevel:
    """Test AnalysisLevel constants."""

    def test_analysis_level_values(self):
        """Test AnalysisLevel enum values."""
        assert AnalysisLevel.COMPREHENSIVE.value == 1
        assert AnalysisLevel.DETAILED.value == 2
        assert AnalysisLevel.SUMMARY.value == 3
        assert AnalysisLevel.CRITICAL_ONLY.value == 4


class TestDegradationStrategy:
    """Test DegradationStrategy dataclass."""

    def test_degradation_strategy_creation(self):
        """Test DegradationStrategy creation with all fields."""
        strategy = DegradationStrategy(
            level=2,
            max_tools_per_iteration=3,
            max_result_size=2000,
            summary_threshold=1000,
            include_raw_data=True,
            include_debug_info=False,
        )

        assert strategy.level == 2
        assert strategy.max_tools_per_iteration == 3
        assert strategy.max_result_size == 2000
        assert strategy.summary_threshold == 1000
        assert strategy.include_raw_data is True
        assert strategy.include_debug_info is False


class TestProgressiveAnalyzer:
    """Test ProgressiveAnalyzer functionality."""

    @pytest.fixture
    def mock_ai_provider(self):
        """Mock AI provider for testing."""
        return Mock(spec=BaseProvider)

    @pytest.fixture
    def limits(self):
        """Create ExecutionLimits for testing."""
        return ExecutionLimits(
            max_infer_iters=10,
            max_tokens_total=100000,
            max_tokens_per_tool_result=4000,
            max_tools_per_iteration=5,
            max_total_tools=50,
            max_execution_time=600,
        )

    @pytest.fixture
    def analyzer(self, mock_ai_provider, limits):
        """Create ProgressiveAnalyzer for testing."""
        return ProgressiveAnalyzer(mock_ai_provider, limits)

    def test_initialization(self, analyzer, limits):
        """Test ProgressiveAnalyzer initialization."""
        assert analyzer.ai_provider is not None
        assert analyzer.limits == limits
        assert len(analyzer.strategies) == 4

        # Test strategy configurations
        comprehensive = analyzer.strategies[AnalysisLevel.COMPREHENSIVE]
        assert comprehensive.level == 1
        assert comprehensive.max_tools_per_iteration == limits.max_tools_per_iteration
        assert comprehensive.include_raw_data is True
        assert comprehensive.include_debug_info is True

        critical_only = analyzer.strategies[AnalysisLevel.CRITICAL_ONLY]
        assert critical_only.level == 4
        assert critical_only.max_tools_per_iteration == 1
        assert critical_only.include_raw_data is False
        assert critical_only.include_debug_info is False

    def test_strategy_progressive_reduction(self, analyzer):
        """Test that strategies progressively reduce resource usage."""
        strategies = analyzer.strategies

        # Verify progressive reduction in tool limits
        comprehensive = strategies[AnalysisLevel.COMPREHENSIVE]
        detailed = strategies[AnalysisLevel.DETAILED]
        summary = strategies[AnalysisLevel.SUMMARY]
        critical = strategies[AnalysisLevel.CRITICAL_ONLY]

        assert comprehensive.max_tools_per_iteration >= detailed.max_tools_per_iteration
        assert detailed.max_tools_per_iteration >= summary.max_tools_per_iteration
        assert summary.max_tools_per_iteration >= critical.max_tools_per_iteration

        # Verify progressive reduction in result sizes
        assert comprehensive.max_result_size >= detailed.max_result_size
        assert detailed.max_result_size >= summary.max_result_size
        assert summary.max_result_size >= critical.max_result_size

    def test_determine_analysis_level_comprehensive(self, analyzer):
        """Test analysis level determination for low resource utilization."""
        state = ExecutionState()
        state.current_iteration = 2  # 2/10 = 20%
        state.total_tokens_used = 20000  # 20/100 = 20%
        state.total_tools_executed = 10  # 10/50 = 20%

        # Mock elapsed time
        with patch.object(state, "get_elapsed_time", return_value=120):  # 120/600 = 20%
            level = analyzer.determine_analysis_level(state)
            assert level == AnalysisLevel.COMPREHENSIVE

    def test_determine_analysis_level_detailed(self, analyzer):
        """Test analysis level determination for moderate resource utilization."""
        state = ExecutionState()
        state.current_iteration = 7  # 7/10 = 70%
        state.total_tokens_used = 70000  # 70/100 = 70%
        state.total_tools_executed = 35  # 35/50 = 70%

        with patch.object(state, "get_elapsed_time", return_value=420):  # 420/600 = 70%
            level = analyzer.determine_analysis_level(state)
            assert level == AnalysisLevel.DETAILED

    def test_determine_analysis_level_summary(self, analyzer):
        """Test analysis level determination for high resource utilization."""
        state = ExecutionState()
        state.current_iteration = 8  # 8/10 = 80%
        state.total_tokens_used = 85000  # 85/100 = 85%
        state.total_tools_executed = 43  # 43/50 = 86%

        with patch.object(state, "get_elapsed_time", return_value=510):  # 510/600 = 85%
            level = analyzer.determine_analysis_level(state)
            assert level == AnalysisLevel.SUMMARY

    def test_determine_analysis_level_critical_only(self, analyzer):
        """Test analysis level determination for very high resource utilization."""
        state = ExecutionState()
        state.current_iteration = 10  # 10/10 = 100%
        state.total_tokens_used = 95000  # 95/100 = 95%
        state.total_tools_executed = 48  # 48/50 = 96%

        with patch.object(state, "get_elapsed_time", return_value=570):  # 570/600 = 95%
            level = analyzer.determine_analysis_level(state)
            assert level == AnalysisLevel.CRITICAL_ONLY

    def test_determine_analysis_level_highest_utilization_wins(self, analyzer):
        """Test that highest utilization dimension determines analysis level."""
        state = ExecutionState()
        state.current_iteration = 2  # 2/10 = 20%
        state.total_tokens_used = 95000  # 95/100 = 95% <- Highest
        state.total_tools_executed = 10  # 10/50 = 20%

        with patch.object(state, "get_elapsed_time", return_value=120):  # 120/600 = 20%
            level = analyzer.determine_analysis_level(state)
            assert level == AnalysisLevel.CRITICAL_ONLY  # Driven by token utilization

    def test_classify_result_priority_critical_error(self, analyzer):
        """Test critical priority classification for errors."""
        priority = analyzer._classify_result_priority("", "error")
        assert priority == ResultPriority.CRITICAL

        priority = analyzer._classify_result_priority("", "")
        assert priority == ResultPriority.CRITICAL

    def test_classify_result_priority_critical_patterns(self, analyzer):
        """Test critical priority classification for error patterns."""
        critical_contents = [
            "Error occurred during processing",
            "Failed to connect to server",
            "Exception thrown in method",
            "System crash detected",
            "Alert: security breach",
            "Warning: memory exhausted",
            "Issue with database connection",
            "Problem with authentication",
        ]

        for content in critical_contents:
            priority = analyzer._classify_result_priority(content, "success")
            assert priority == ResultPriority.CRITICAL, f"Should be critical: {content}"

    def test_classify_result_priority_high_patterns(self, analyzer):
        """Test high priority classification for important patterns."""
        high_contents = [
            "Found 42 matching records",
            "Results show significant improvement",
            "Summary of quarterly performance",
            "Total revenue increased by 15%",
            "Analysis reveals three key insights",
            "Report indicates strong correlation",
            "Insights from customer feedback",
        ]

        for content in high_contents:
            priority = analyzer._classify_result_priority(content, "success")
            assert priority == ResultPriority.HIGH, f"Should be high: {content}"

    def test_classify_result_priority_content_richness(self, analyzer):
        """Test priority classification based on content richness."""
        # High priority for rich content
        rich_content = "\n".join(
            [f"Line {i} with detailed information" for i in range(15)]
        )
        priority = analyzer._classify_result_priority(rich_content, "success")
        assert priority == ResultPriority.HIGH

        # Medium priority for moderate content
        medium_content = "\n".join([f"Line {i}" for i in range(5)])
        priority = analyzer._classify_result_priority(medium_content, "success")
        assert priority == ResultPriority.MEDIUM

        # Low priority for minimal content
        low_content = "Short response"
        priority = analyzer._classify_result_priority(low_content, "success")
        assert priority == ResultPriority.LOW

    def test_filter_results_by_priority_comprehensive(self, analyzer):
        """Test result filtering for comprehensive analysis."""
        tool_results = [
            {"result": "Critical error occurred", "status": "error"},
            {"result": "Found 100 items in database", "status": "success"},
            {"result": "Processing completed", "status": "success"},
            {"result": "Debug: trace information", "status": "success"},
        ]

        strategy = analyzer.strategies[AnalysisLevel.COMPREHENSIVE]
        filtered = analyzer._filter_results_by_priority(tool_results, strategy)

        # Should include all results in comprehensive mode
        assert len(filtered) == 4

    def test_filter_results_by_priority_critical_only(self, analyzer):
        """Test result filtering for critical-only analysis."""
        tool_results = [
            {"result": "Critical error occurred", "status": "error"},
            {"result": "Found 100 items in database", "status": "success"},
            {"result": "Processing completed", "status": "success"},
            {"result": "Debug information", "status": "success"},
        ]

        strategy = analyzer.strategies[AnalysisLevel.CRITICAL_ONLY]
        filtered = analyzer._filter_results_by_priority(tool_results, strategy)

        # Should only include critical results
        assert len(filtered) == 1
        assert "Critical error" in filtered[0]["result"]

    def test_filter_results_by_priority_respects_tool_limits(self, analyzer):
        """Test that result filtering respects tool limits."""
        # Create more results than the limit allows
        tool_results = [
            {"result": "Critical error 1", "status": "error"},
            {"result": "Critical error 2", "status": "error"},
            {"result": "Critical error 3", "status": "error"},
            {"result": "Important finding", "status": "success"},
            {"result": "Normal result", "status": "success"},
        ]

        # Use critical-only strategy with limit of 1 tool
        strategy = analyzer.strategies[AnalysisLevel.CRITICAL_ONLY]
        filtered = analyzer._filter_results_by_priority(tool_results, strategy)

        # Should only return 1 result due to tool limit
        assert len(filtered) == 1
        assert "Critical error" in filtered[0]["result"]

    def test_smart_truncate_result_no_truncation_needed(self, analyzer):
        """Test smart truncation when content is within limits."""
        content = "Short content that fits"
        max_size = 100

        truncated = analyzer._smart_truncate_result(content, max_size)
        assert truncated == content

    def test_smart_truncate_result_with_truncation_notice(self, analyzer):
        """Test smart truncation with truncation notice."""
        content = "A" * 1000  # 1000 characters
        max_size = 300

        truncated = analyzer._smart_truncate_result(content, max_size)

        assert len(truncated) <= max_size
        assert "[TRUNCATED:" in truncated
        assert truncated.startswith("A")  # Should preserve beginning
        assert truncated.endswith("A")  # Should preserve end

    def test_smart_truncate_result_very_small_limit(self, analyzer):
        """Test smart truncation with very small size limit."""
        content = "This is a long piece of content that needs to be truncated"
        max_size = 50

        truncated = analyzer._smart_truncate_result(content, max_size)

        assert len(truncated) <= max_size
        assert "[TRUNCATED]" in truncated

    def test_apply_degradation_strategy_complete_workflow(self, analyzer):
        """Test complete degradation strategy application."""
        tool_results = [
            {"result": "Critical system failure", "status": "error"},
            {
                "result": "Found 500 database records with detailed analysis",
                "status": "success",
            },
            {"result": "Processing completed successfully", "status": "success"},
            {"result": "Debug trace: step 1, step 2, step 3", "status": "success"},
        ]

        processed_results, analysis_summary = analyzer.apply_degradation_strategy(
            tool_results, AnalysisLevel.SUMMARY, "Analyze system performance"
        )

        # Should filter and process results
        assert len(processed_results) <= len(tool_results)
        assert analysis_summary is not None
        assert isinstance(analysis_summary, str)

    def test_generate_degraded_analysis_comprehensive(self, analyzer):
        """Test degraded analysis generation for comprehensive level."""
        tool_results = [
            {"result": "Analysis complete", "status": "success"},
            {"result": "Error in module X", "status": "error"},
        ]

        analysis = analyzer._generate_degraded_analysis(
            tool_results, AnalysisLevel.COMPREHENSIVE, "Test prompt"
        )

        assert "📋 COMPREHENSIVE ANALYSIS" in analysis
        assert "Tools Executed: 1 successful, 1 failed" in analysis

    def test_generate_degraded_analysis_critical_only(self, analyzer):
        """Test degraded analysis generation for critical-only level."""
        tool_results = [
            {"result": "System operational", "status": "success"},
            {"result": "Critical database error", "status": "error"},
        ]

        analysis = analyzer._generate_degraded_analysis(
            tool_results, AnalysisLevel.CRITICAL_ONLY, "Test prompt"
        )

        assert "🚨 CRITICAL ANALYSIS (Resource Constrained)" in analysis
        assert "Tools Executed: 1 successful, 1 failed" in analysis
        assert "Analysis limited due to resource constraints" in analysis

    def test_generate_degraded_analysis_with_degradation_notice(self, analyzer):
        """Test that degradation notice is included for non-comprehensive levels."""
        tool_results = [{"result": "Test result", "status": "success"}]

        analysis = analyzer._generate_degraded_analysis(
            tool_results, AnalysisLevel.SUMMARY, "Test prompt"
        )

        assert "ℹ️ Analysis limited due to resource constraints" in analysis
        assert "Level: SUMMARY" in analysis

    def test_extract_key_findings_numerical_data(self, analyzer):
        """Test key findings extraction for numerical data."""
        successful_tools = [
            {
                "tool_name": "data_analyzer",
                "result": "Found 42 issues, 13 warnings, 7 errors in codebase analysis",
            }
        ]

        findings = analyzer._extract_key_findings(successful_tools)

        assert len(findings) > 0
        assert any("numerical values" in finding for finding in findings)

    def test_extract_key_findings_list_data(self, analyzer):
        """Test key findings extraction for list data."""
        successful_tools = [
            {
                "tool_name": "file_scanner",
                "result": "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7",
            }
        ]

        findings = analyzer._extract_key_findings(successful_tools)

        assert len(findings) > 0
        assert any("items" in finding for finding in findings)

    def test_extract_key_findings_total_count_patterns(self, analyzer):
        """Test key findings extraction for total/count patterns."""
        successful_tools = [
            {
                "tool_name": "counter",
                "result": "Processing complete. Total files processed: 156 files",
            }
        ]

        findings = analyzer._extract_key_findings(successful_tools)

        assert len(findings) > 0
        assert any("Total files processed" in finding for finding in findings)

    def test_generate_recommendations_timeout_errors(self, analyzer):
        """Test recommendation generation for timeout errors."""
        tool_results = [
            {"result": "Operation timed out after 30 seconds", "status": "error"}
        ]

        recommendations = analyzer._generate_recommendations(
            tool_results, AnalysisLevel.DETAILED
        )

        assert len(recommendations) > 0
        assert any("timeout" in rec.lower() for rec in recommendations)

    def test_generate_recommendations_not_found_errors(self, analyzer):
        """Test recommendation generation for not found errors."""
        tool_results = [{"result": "Resource not found (404 error)", "status": "error"}]

        recommendations = analyzer._generate_recommendations(
            tool_results, AnalysisLevel.DETAILED
        )

        assert len(recommendations) > 0
        assert any("parameter" in rec.lower() for rec in recommendations)

    def test_generate_recommendations_auth_errors(self, analyzer):
        """Test recommendation generation for authentication errors."""
        tool_results = [
            {"result": "Authentication failed (401 unauthorized)", "status": "error"}
        ]

        recommendations = analyzer._generate_recommendations(
            tool_results, AnalysisLevel.DETAILED
        )

        assert len(recommendations) > 0
        assert any("authentication" in rec.lower() for rec in recommendations)

    def test_generate_recommendations_resource_optimization(self, analyzer):
        """Test recommendation generation for resource optimization."""
        tool_results = [{"result": "Test", "status": "success"}]

        recommendations = analyzer._generate_recommendations(
            tool_results, AnalysisLevel.CRITICAL_ONLY
        )  # High degradation level

        assert any(
            "token" in rec.lower() or "limit" in rec.lower() for rec in recommendations
        )

    def test_generate_recommendations_large_results(self, analyzer):
        """Test recommendation generation for large result sizes."""
        # Create tool results with large average size
        large_result = "x" * 3000  # 3000 characters
        tool_results = [
            {"result": large_result, "status": "success"},
            {"result": large_result, "status": "success"},
        ]

        recommendations = analyzer._generate_recommendations(
            tool_results, AnalysisLevel.DETAILED
        )

        assert any("specific queries" in rec.lower() for rec in recommendations)

    def test_no_tool_results_handling(self, analyzer):
        """Test handling of empty tool results."""
        processed_results, analysis_summary = analyzer.apply_degradation_strategy(
            [], AnalysisLevel.COMPREHENSIVE, "Test prompt"
        )

        assert processed_results == []
        assert "No tool results available" in analysis_summary

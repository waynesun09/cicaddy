"""Tool result processor for extracting structured metrics from MCP tool results."""

import re
from typing import Any, Dict, List, Set, cast

from cicaddy.utils.logger import get_logger

logger = get_logger(__name__)


class ToolResultProcessor:
    """
    Universal tool result processor that extracts structured metadata and metrics
    from any MCP tool result before AI summarization.

    This ensures critical quantitative data is preserved regardless of AI compression.
    """

    def __init__(self, max_entities_per_type: int = 100):
        """
        Initialize the processor.

        Args:
            max_entities_per_type: Maximum number of entities to extract per type
        """
        self.max_entities_per_type = max_entities_per_type

    def process(
        self, tool_name: str, tool_result: str, tool_server: str = ""
    ) -> Dict[str, Any]:
        """
        Extract structured data from any tool result.

        Args:
            tool_name: Name of the tool that generated the result
            tool_result: Raw result string from the tool
            tool_server: MCP server that provided the tool

        Returns:
            Dictionary containing:
                - raw_result: Original unmodified result
                - metadata: General metadata (line count, char count, etc)
                - metrics: Countable metrics (repos, files, matches, errors)
                - entities: Unique entities (repositories, files, URLs)
                - summary_stats: Statistical summary
        """
        logger.debug(
            f"Processing result from tool '{tool_name}' on server '{tool_server}'"
        )

        result_str = str(tool_result)

        processed = {
            "raw_result": result_str,
            "tool_name": tool_name,
            "tool_server": tool_server,
            "metadata": self._extract_metadata(result_str),
            "metrics": self._extract_metrics(result_str),
            "entities": self._extract_entities(result_str),
            "summary_stats": {},
        }

        # Calculate summary statistics
        processed["summary_stats"] = self._calculate_stats(processed)

        metrics_dict = cast(Dict[str, Any], processed["metrics"])
        entities_dict = cast(Dict[str, List[str]], processed["entities"])
        logger.debug(
            f"Extracted metrics: {metrics_dict['counts']}, "
            f"entities: {len(entities_dict['repositories'])} repos, "
            f"{len(entities_dict['files'])} files"
        )

        return processed

    def _extract_metadata(self, result: str) -> Dict[str, Any]:
        """
        Extract universal metadata from result.

        Args:
            result: Raw result string

        Returns:
            Dictionary with line_count, char_count, has_errors, result_type
        """
        metadata = {
            "line_count": len(result.split("\n")),
            "char_count": len(result),
            "has_errors": bool(
                re.search(
                    r"\b(error|failed|exception|traceback)\b", result, re.IGNORECASE
                )
            ),
            "result_type": self._detect_type(result),
        }

        return metadata

    def _extract_metrics(self, result: str) -> Dict[str, Any]:
        """
        Extract countable metrics from result.

        Supports common patterns like:
        - repository: <repo_name>
        - file: <file_path>
        - num_matches: <count>
        - error/failed/exception

        Args:
            result: Raw result string

        Returns:
            Dictionary with 'counts' containing various metrics
        """
        metrics = {
            "counts": {
                "repository": len(re.findall(r"repository:\s*(\S+)", result)),
                "file": len(re.findall(r"file:\s*(https?://\S+)", result)),
                "match": self._sum_matches(result),
                "error": len(
                    re.findall(r"\b(error|failed|exception)\b", result, re.IGNORECASE)
                ),
                "url": len(re.findall(r"https?://\S+", result)),
                "line": len(result.split("\n")),
            }
        }

        return metrics

    def _sum_matches(self, result: str) -> int:
        """
        Sum all num_matches values in the result.

        Args:
            result: Raw result string

        Returns:
            Total sum of all num_matches values
        """
        matches = re.findall(r"num_matches:\s*(\d+)", result)
        return sum(int(m) for m in matches)

    def _extract_entities(self, result: str) -> Dict[str, List[str]]:
        """
        Extract unique entities (repos, files, URLs) from result.

        Args:
            result: Raw result string

        Returns:
            Dictionary with lists of unique entities, limited to max_entities_per_type
        """
        entities = {
            "repositories": self._extract_unique(r"repository:\s*([^\n]+)", result),
            "files": self._extract_unique(r"file:\s*(https?://[^\n]+)", result),
            "urls": self._extract_unique(r"https?://\S+", result),
            "languages": self._extract_unique(r"language:\s*([^\n]+)", result),
        }

        return entities

    def _extract_unique(self, pattern: str, text: str) -> List[str]:
        """
        Extract unique matches for a pattern.

        Args:
            pattern: Regex pattern to match
            text: Text to search

        Returns:
            List of unique matches, limited to max_entities_per_type
        """
        matches = re.findall(pattern, text)
        unique = list(
            dict.fromkeys(matches)
        )  # Preserve order while removing duplicates
        return unique[: self.max_entities_per_type]

    def _detect_type(self, result: str) -> str:
        """
        Detect the type of result based on content patterns.

        Args:
            result: Raw result string

        Returns:
            String indicating result type (code_search, error, json, text, etc)
        """
        if "repository:" in result and "num_matches:" in result:
            return "code_search"
        elif "error" in result.lower() or "exception" in result.lower():
            return "error"
        elif result.strip().startswith("{") or result.strip().startswith("["):
            return "json"
        elif "http://" in result or "https://" in result:
            return "url_list"
        else:
            return "text"

    def _calculate_stats(self, processed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate summary statistics from processed data.

        Args:
            processed: Processed result dictionary

        Returns:
            Dictionary with statistical summary
        """
        counts = processed["metrics"]["counts"]
        entities = processed["entities"]

        stats = {
            "total_entities": sum(len(v) for v in entities.values()),
            "total_counts": sum(v for k, v in counts.items() if k != "line"),
            "unique_repositories": len(entities["repositories"]),
            "unique_files": len(entities["files"]),
            "unique_languages": len(entities["languages"]),
            "has_significant_data": counts["repository"] > 0
            or counts["file"] > 0
            or counts["match"] > 0,
        }

        return stats

    def aggregate_metrics(
        self, processed_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics from multiple processed results.

        Args:
            processed_results: List of processed result dictionaries

        Returns:
            Dictionary with aggregated metrics across all results
        """
        if not processed_results:
            return {
                "total_tools": 0,
                "total_repositories": 0,
                "total_files": 0,
                "total_matches": 0,
                "total_errors": 0,
                "unique_repositories": set(),
                "unique_files": set(),
                "unique_languages": set(),
                "tools_with_errors": 0,
                "tools_with_data": 0,
            }

        aggregated = {
            "total_tools": len(processed_results),
            "total_repositories": 0,
            "total_files": 0,
            "total_matches": 0,
            "total_errors": 0,
            "unique_repositories": set(),
            "unique_files": set(),
            "unique_languages": set(),
            "tools_with_errors": 0,
            "tools_with_data": 0,
        }

        for result in processed_results:
            metrics_data = result.get("metrics", {})
            counts = (
                metrics_data.get("counts", {}) if isinstance(metrics_data, dict) else {}
            )
            entities = result.get("entities", {})
            stats = result.get("summary_stats", {})

            if isinstance(counts, dict):
                aggregated["total_repositories"] += counts.get("repository", 0)
                aggregated["total_files"] += counts.get("file", 0)
                aggregated["total_matches"] += counts.get("match", 0)
                aggregated["total_errors"] += counts.get("error", 0)

            if isinstance(entities, dict):
                repos_set = cast(Set[str], aggregated["unique_repositories"])
                files_set = cast(Set[str], aggregated["unique_files"])
                langs_set = cast(Set[str], aggregated["unique_languages"])
                repos_set.update(entities.get("repositories", []))
                files_set.update(entities.get("files", []))
                langs_set.update(entities.get("languages", []))

            metadata = result.get("metadata", {})
            if isinstance(metadata, dict) and metadata.get("has_errors"):
                aggregated["tools_with_errors"] = (
                    cast(int, aggregated["tools_with_errors"]) + 1
                )

            if isinstance(stats, dict) and stats.get("has_significant_data"):
                aggregated["tools_with_data"] = (
                    cast(int, aggregated["tools_with_data"]) + 1
                )

        # Convert sets to counts for final output
        repos_set = cast(Set[str], aggregated["unique_repositories"])
        files_set = cast(Set[str], aggregated["unique_files"])
        langs_set = cast(Set[str], aggregated["unique_languages"])
        aggregated["unique_repository_count"] = len(repos_set)
        aggregated["unique_file_count"] = len(files_set)
        aggregated["unique_language_count"] = len(langs_set)

        # Keep sets for validation but also provide lists for JSON serialization
        aggregated["unique_repositories_list"] = list(repos_set)
        aggregated["unique_files_list"] = list(files_set)
        aggregated["unique_languages_list"] = list(langs_set)

        logger.info(
            f"Aggregated {aggregated['total_tools']} tool results: "
            f"{aggregated['unique_repository_count']} unique repos, "
            f"{aggregated['total_matches']} total matches"
        )

        return aggregated

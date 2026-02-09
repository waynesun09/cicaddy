import json
from datetime import datetime
from typing import Any

from cicaddy.reports.html_formatter import HTMLReportFormatter


def _build_base_report(ai_analysis: Any, ai_response_format: str = "markdown"):
    return {
        "report_id": "unit_test_report",
        "project": "Unit Test Project",
        "execution_time": 5.0,
        "task_type": "custom_task",
        "start_time": datetime.utcnow().isoformat(),
        "analysis_result": {
            "ai_analysis": ai_analysis,
            "ai_response_format": ai_response_format,
            "status": "success",
            "execution_time": 1.23,
            "turn_id": "turn-123",
            "tool_calls": [],
        },
    }


def test_format_report_renders_markdown_section(tmp_path):
    json_path = tmp_path / "report.json"
    json_path.write_text("{}", encoding="utf-8")

    formatter = HTMLReportFormatter()
    report = _build_base_report(
        "## Key Findings\n- Item one\n- Item two", ai_response_format="markdown"
    )

    html = formatter.format_report(report, json_report_path=str(json_path))

    assert "<h3>AI Analysis" in html  # nosec B101
    assert "Key Findings" in html  # nosec B101
    assert "Item one" in html  # nosec B101


def test_format_report_saves_html_direct_response(tmp_path):
    json_path = tmp_path / "report.json"
    json_path.write_text("{}", encoding="utf-8")

    formatter = HTMLReportFormatter()
    html_body = "<h1>Custom HTML Output</h1><p>Rendered by the AI model.</p>"
    report = _build_base_report(html_body, ai_response_format="html")

    html = formatter.format_report(report, json_report_path=str(json_path))

    artifact_path = tmp_path / "report_ai_direct_resp.html"
    assert artifact_path.exists(), "Expected HTML artifact to be saved"  # nosec B101
    assert artifact_path.read_text(encoding="utf-8") == html_body  # nosec B101
    assert "Download HTML" in html  # nosec B101 - embedded HTML shows download
    assert "report_ai_direct_resp.html" in html  # nosec B101


def test_format_report_saves_json_direct_response(tmp_path):
    json_path = tmp_path / "report.json"
    json_path.write_text("{}", encoding="utf-8")

    formatter = HTMLReportFormatter()
    json_body = json.dumps({"status": "ok", "items": [1, 2, 3]})
    report = _build_base_report(json_body, ai_response_format="json")

    html = formatter.format_report(report, json_report_path=str(json_path))

    artifact_path = tmp_path / "report_ai_direct_resp.json"
    assert artifact_path.exists(), "Expected JSON artifact to be saved"  # nosec B101
    assert artifact_path.read_text(encoding="utf-8") == json_body  # nosec B101
    assert "Download JSON" in html  # nosec B101
    assert "report_ai_direct_resp.json" in html  # nosec B101


def test_html_direct_response_strips_preface_and_code_fence(tmp_path):
    json_path = tmp_path / "report.json"
    json_path.write_text("{}", encoding="utf-8")

    formatter = HTMLReportFormatter()
    html_body = """
    I will now output the final HTML.
    ```html
    <!DOCTYPE html>
    <html><body><p>Done</p></body></html>
    ```
    This should not be included.
    """
    report = _build_base_report(html_body, ai_response_format="html")

    formatter.format_report(report, json_report_path=str(json_path))

    saved = (
        (tmp_path / "report_ai_direct_resp.html").read_text(encoding="utf-8").lstrip()
    )
    assert saved.startswith("<!DOCTYPE html>")  # nosec B101
    assert saved.rstrip().endswith("</html>")  # nosec B101


def test_json_direct_response_trims_code_fence_and_trailing_text(tmp_path):
    json_path = tmp_path / "report.json"
    json_path.write_text("{}", encoding="utf-8")

    formatter = HTMLReportFormatter()
    json_body = """
    Here is the JSON payload.
    ```json
    {"status": "ok", "items": [1, 2, 3]}
    ```
    Wrapping narrative after fence.
    """
    report = _build_base_report(json_body, ai_response_format="json")

    formatter.format_report(report, json_report_path=str(json_path))

    saved = (
        (tmp_path / "report_ai_direct_resp.json").read_text(encoding="utf-8").strip()
    )
    assert saved == '{"status": "ok", "items": [1, 2, 3]}'  # nosec B101


def test_html_direct_response_handles_missing_closing_tag(tmp_path):
    json_path = tmp_path / "report.json"
    json_path.write_text("{}", encoding="utf-8")

    formatter = HTMLReportFormatter()
    html_body = "<html><body><p>Done</p>"
    report = _build_base_report(html_body, ai_response_format="html")

    formatter.format_report(report, json_report_path=str(json_path))

    saved = (tmp_path / "report_ai_direct_resp.html").read_text(encoding="utf-8")
    assert saved == html_body  # nosec B101

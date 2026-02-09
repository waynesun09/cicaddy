from cicaddy.agent.cron_agent import CronAIAgent
from cicaddy.config.settings import Settings


def _build_agent() -> CronAIAgent:
    return CronAIAgent(settings=Settings())


def test_email_format_markdown_analysis(monkeypatch):
    monkeypatch.delenv("CI_PIPELINE_URL", raising=False)
    monkeypatch.delenv("CI_PIPELINE_ID", raising=False)
    monkeypatch.delenv("CI_JOB_URL", raising=False)
    monkeypatch.delenv("CI_PROJECT_URL", raising=False)
    monkeypatch.delenv("CI_JOB_ID", raising=False)

    agent = _build_agent()
    report = {"project": "Sample Project", "report_id": "cron_job_123"}
    analysis_result = {
        "ai_analysis": "## Findings\n- Item A",
        "ai_response_format": "markdown",
    }

    html = agent._format_email_html(report, analysis_result)

    assert "<h2>Findings</h2>" in html  # nosec B101


def test_email_format_direct_html():
    agent = _build_agent()
    report = {"title": "Direct HTML"}
    analysis_result = {
        "ai_analysis": """
        I will now output HTML
        ```html
        <!DOCTYPE html><html><body><p>Direct</p></body></html>
        ```
        """,
        "ai_response_format": "html",
    }

    html = agent._format_email_html(report, analysis_result).strip()

    assert html.startswith("<!DOCTYPE html>")  # nosec B101
    assert html.endswith("</html>")  # nosec B101


def test_email_format_json_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_PIPELINE_URL", "https://ci/pipeline/123")
    monkeypatch.setenv("CI_PIPELINE_ID", "123")
    monkeypatch.setenv("CI_JOB_URL", "https://ci/job/456")
    monkeypatch.setenv("CI_PROJECT_URL", "https://ci/project")
    monkeypatch.setenv("CI_JOB_ID", "456")

    agent = _build_agent()
    report = {
        "project": "JSON Project",
        "report_id": "cron_json_123",
        "html_report_path": str(tmp_path / "report.html"),
    }
    analysis_result = {
        "ai_analysis": """
        ```json
        {"status": "ok"}
        ```
        """,
        "ai_response_format": "json",
    }

    html = agent._format_email_html(report, analysis_result)

    assert "<pre>" in html  # nosec B101
    assert "&quot;status&quot;" in html  # nosec B101
    assert "Full HTML report" in html  # nosec B101

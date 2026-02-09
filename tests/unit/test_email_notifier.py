"""Unit tests for EmailNotifier."""
# ruff: noqa: B101
# Bandit B101: assert statements are expected in test files

import base64
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cicaddy.notifications.email import EmailNotifier


class TestEmailNotifierInitialization:
    """Test EmailNotifier initialization."""

    def test_init_with_string_recipients(self):
        """Test initialization with comma-separated string recipients."""
        notifier = EmailNotifier(
            recipients="user1@example.com, user2@example.com",
            use_gmail_api=False,
        )

        assert notifier.recipients == ["user1@example.com", "user2@example.com"]
        assert notifier.use_gmail_api is False

    def test_init_with_list_recipients(self):
        """Test initialization with list of recipients."""
        recipients = ["user1@example.com", "user2@example.com"]
        notifier = EmailNotifier(recipients=recipients, use_gmail_api=False)

        assert notifier.recipients == recipients

    def test_init_with_smtp_config(self):
        """Test initialization with SMTP configuration."""
        smtp_config = {
            "host": "smtp.example.com",
            "port": 587,
            "username": "user@example.com",
            "password": "secret",
            "use_tls": True,
        }
        notifier = EmailNotifier(
            recipients="test@example.com",
            use_gmail_api=False,
            smtp_config=smtp_config,
        )

        assert notifier.smtp_config == smtp_config

    def test_init_gmail_api_default(self):
        """Test Gmail API is used by default."""
        notifier = EmailNotifier(recipients="test@example.com")
        assert notifier.use_gmail_api is True


class TestEmailNotifierSendNotification:
    """Test EmailNotifier send_notification method."""

    @pytest.mark.asyncio
    async def test_send_notification_smtp_success(self):
        """Test successful SMTP email send."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            use_gmail_api=False,
            smtp_config={
                "host": "smtp.example.com",
                "port": 587,
                "username": "user@example.com",
                "password": "secret",
            },
        )

        with patch.object(
            notifier, "_send_via_smtp", new_callable=AsyncMock
        ) as mock_smtp:
            mock_smtp.return_value = {
                "status": "sent",
                "method": "smtp",
                "recipients": ["test@example.com"],
            }

            result = await notifier.send_notification(
                "Test message",
                subject="Test Subject",
            )

            assert result["status"] == "sent"
            assert result["method"] == "smtp"
            mock_smtp.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_notification_gmail_api_success(self):
        """Test successful Gmail API email send."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            use_gmail_api=True,
        )

        with patch.object(
            notifier, "_send_via_gmail_api", new_callable=AsyncMock
        ) as mock_gmail:
            mock_gmail.return_value = {
                "status": "sent",
                "method": "gmail_api",
                "message_id": "12345",
                "recipients": ["test@example.com"],
            }

            result = await notifier.send_notification(
                "Test message",
                subject="Test Subject",
            )

            assert result["status"] == "sent"
            assert result["method"] == "gmail_api"
            mock_gmail.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_notification_html_content(self):
        """Test sending HTML email."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            use_gmail_api=False,
        )

        with patch.object(
            notifier, "_send_via_smtp", new_callable=AsyncMock
        ) as mock_smtp:
            mock_smtp.return_value = {"status": "sent"}

            await notifier.send_notification(
                "<html><body>Test</body></html>",
                subject="Test",
                html=True,
            )

            # Verify HTML flag was passed
            call_args = mock_smtp.call_args
            assert call_args[0][2] is True  # is_html parameter

    @pytest.mark.asyncio
    async def test_send_notification_with_cc_bcc(self):
        """Test sending email with CC and BCC."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            use_gmail_api=False,
        )

        with patch.object(
            notifier, "_send_via_smtp", new_callable=AsyncMock
        ) as mock_smtp:
            mock_smtp.return_value = {"status": "sent"}

            await notifier.send_notification(
                "Test message",
                subject="Test",
                cc="cc@example.com, cc2@example.com",
                bcc="bcc@example.com",
            )

            # Verify CC and BCC were parsed correctly
            call_args = mock_smtp.call_args
            cc_list = call_args[0][3]
            bcc_list = call_args[0][4]

            assert cc_list == ["cc@example.com", "cc2@example.com"]
            assert bcc_list == ["bcc@example.com"]

    @pytest.mark.asyncio
    async def test_send_notification_error_handling(self):
        """Test error handling in send_notification."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            use_gmail_api=False,
        )

        with patch.object(
            notifier, "_send_via_smtp", new_callable=AsyncMock
        ) as mock_smtp:
            mock_smtp.side_effect = Exception("SMTP connection failed")

            result = await notifier.send_notification("Test", subject="Test")

            assert result["status"] == "failed"
            assert "SMTP connection failed" in result["error"]


class TestEmailNotifierSMTP:
    """Test SMTP sending functionality."""

    @pytest.mark.asyncio
    async def test_send_via_smtp_missing_host(self):
        """Test SMTP send fails with missing SMTP_HOST."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            use_gmail_api=False,
        )

        # Clear environment variables - only SMTP_HOST is required
        with patch.dict(os.environ, {}, clear=True):
            result = await notifier._send_via_smtp(
                "Test Subject",
                "Test Message",
                False,
                [],
                [],
                {},
            )

            assert result["status"] == "failed"
            assert "SMTP_HOST required" in result["error"]

    @pytest.mark.asyncio
    async def test_send_via_smtp_without_auth(self):
        """Test SMTP send without authentication (internal relay servers)."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            use_gmail_api=False,
        )

        # Only provide SMTP_HOST, no credentials (for internal mail relays)
        env_vars = {
            "SMTP_HOST": "smtp.internal.corp",
            "SMTP_PORT": "25",
            "SENDER_EMAIL": "sender@example.com",
        }

        with patch.dict(os.environ, env_vars), patch("smtplib.SMTP") as mock_smtp_class:
            mock_server = MagicMock()
            mock_smtp_class.return_value = mock_server

            result = await notifier._send_via_smtp(
                "Test Subject",
                "Test Message",
                False,
                [],
                [],
                {},
            )

            # Verify SMTP operations - no login should be called
            mock_smtp_class.assert_called_once_with(
                "smtp.internal.corp", 25, timeout=30
            )
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_not_called()  # No auth
            mock_server.send_message.assert_called_once()
            mock_server.quit.assert_called_once()

            assert result["status"] == "sent"
            assert result["method"] == "smtp"

    @pytest.mark.asyncio
    async def test_send_via_smtp_with_env_vars(self):
        """Test SMTP send with environment variables."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            use_gmail_api=False,
        )

        env_vars = {
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "user@example.com",
            "SMTP_PASSWORD": "secret",
            "SENDER_EMAIL": "sender@example.com",
        }

        with patch.dict(os.environ, env_vars), patch("smtplib.SMTP") as mock_smtp_class:
            mock_server = MagicMock()
            mock_smtp_class.return_value = mock_server

            result = await notifier._send_via_smtp(
                "Test Subject",
                "Test Message",
                False,
                [],
                [],
                {},
            )

            # Verify SMTP operations
            mock_smtp_class.assert_called_once_with("smtp.example.com", 587, timeout=30)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("user@example.com", "secret")
            mock_server.send_message.assert_called_once()
            mock_server.quit.assert_called_once()

            assert result["status"] == "sent"
            assert result["method"] == "smtp"


class TestEmailNotifierGmailAPI:
    """Test Gmail API sending functionality."""

    @pytest.mark.asyncio
    async def test_send_via_gmail_api_missing_credentials(self):
        """Test Gmail API send fails with missing credentials."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            use_gmail_api=True,
        )

        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            result = await notifier._send_via_gmail_api(
                "Test Subject",
                "Test Message",
                False,
                [],
                [],
                {},
            )

            assert result["status"] == "failed"
            assert "Missing Gmail API credentials" in result["error"]

    @pytest.mark.asyncio
    async def test_send_via_gmail_api_success(self):
        """Test successful Gmail API email send."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            sender_email="sender@example.com",
            use_gmail_api=True,
        )

        # Prepare mock credentials
        mock_creds_json = '{"token": "test_token"}'
        mock_token_json = '{"token": "test_token", "refresh_token": "refresh"}'

        credentials_b64 = base64.b64encode(mock_creds_json.encode()).decode()
        token_b64 = base64.b64encode(mock_token_json.encode()).decode()

        env_vars = {
            "GMAIL_CREDENTIALS_B64": credentials_b64,
            "GMAIL_TOKEN_B64": token_b64,
        }

        # Mock Gmail API components
        mock_creds = MagicMock()
        mock_creds.expired = False
        mock_creds.refresh_token = None

        mock_service = MagicMock()
        mock_send_result = {"id": "message_12345"}
        mock_service.users().messages().send().execute.return_value = mock_send_result

        with (
            patch.dict(os.environ, env_vars),
            patch(
                "cicaddy.notifications.email.Credentials.from_authorized_user_file",
                return_value=mock_creds,
            ),
            patch("cicaddy.notifications.email.build", return_value=mock_service),
        ):
            result = await notifier._send_via_gmail_api(
                "Test Subject",
                "Test Message",
                False,
                [],
                [],
                {},
            )

            assert result["status"] == "sent"
            assert result["method"] == "gmail_api"
            assert result["message_id"] == "message_12345"


class TestEmailNotifierFooter:
    """Test email footer generation."""

    def test_add_email_footer_plain_text(self):
        """Test adding footer to plain text email."""
        notifier = EmailNotifier(recipients="test@example.com")

        with patch.object(
            notifier, "_get_ci_job_url", return_value="https://gitlab.com/job/123"
        ):
            result = notifier._add_email_footer("Test message", is_html=False)

            assert "Test message" in result
            assert "Autogenerated by Cicaddy" in result
            assert "CI Job: https://gitlab.com/job/123" in result

    def test_add_email_footer_html(self):
        """Test adding footer to HTML email."""
        notifier = EmailNotifier(recipients="test@example.com")

        with (
            patch.object(
                notifier, "_get_ci_job_url", return_value="https://gitlab.com/job/123"
            ),
            patch.dict(
                os.environ,
                {"CI_PIPELINE_URL": "https://gitlab.com/pipeline/456"},
            ),
        ):
            result = notifier._add_email_footer("<p>Test message</p>", is_html=True)

            assert "<p>Test message</p>" in result
            assert "Autogenerated by Cicaddy" in result
            assert 'href="https://gitlab.com/job/123"' in result
            assert 'href="https://gitlab.com/pipeline/456"' in result


class TestEmailNotifierMIME:
    """Test MIME message creation."""

    def test_create_mime_message_plain_text(self):
        """Test creating plain text MIME message."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            sender_email="sender@example.com",
        )

        mime_msg = notifier._create_mime_message(
            "Test Subject",
            "Test Message",
            False,
            [],
            [],
            {},
        )

        assert mime_msg["Subject"] == "Test Subject"
        assert mime_msg["From"] == "sender@example.com"
        assert mime_msg["To"] == "test@example.com"

    def test_create_mime_message_html(self):
        """Test creating HTML MIME message."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            sender_email="sender@example.com",
        )

        mime_msg = notifier._create_mime_message(
            "Test Subject",
            "<html><body>Test</body></html>",
            True,
            [],
            [],
            {},
        )

        assert mime_msg["Subject"] == "Test Subject"
        assert mime_msg["From"] == "sender@example.com"
        # Check that HTML content is present
        assert mime_msg.is_multipart()

    def test_create_mime_message_with_cc(self):
        """Test creating MIME message with CC recipients."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            sender_email="sender@example.com",
        )

        mime_msg = notifier._create_mime_message(
            "Test Subject",
            "Test Message",
            False,
            ["cc1@example.com", "cc2@example.com"],
            [],
            {},
        )

        assert mime_msg["Cc"] == "cc1@example.com, cc2@example.com"

    def test_create_mime_message_html_has_both_parts(self):
        """Test HTML MIME message contains both plain text and HTML parts.

        This is critical for proper SMTP delivery - multipart/alternative
        should include both text/plain and text/html for compatibility.
        """
        notifier = EmailNotifier(
            recipients="test@example.com",
            sender_email="sender@example.com",
        )

        html_content = "<html><body><h1>Title</h1><p>Test content</p></body></html>"
        mime_msg = notifier._create_mime_message(
            "Test Subject",
            html_content,
            True,
            [],
            [],
            {},
        )

        # Verify multipart structure
        assert mime_msg.is_multipart()
        assert mime_msg.get_content_type() == "multipart/alternative"

        # Get all parts
        parts = mime_msg.get_payload()
        assert len(parts) == 2  # Must have both text and HTML

        # First part should be plain text (per RFC 2046 - preferred comes last)
        assert parts[0].get_content_type() == "text/plain"
        plain_text = parts[0].get_payload(decode=True).decode("utf-8")
        assert "Title" in plain_text
        assert "Test content" in plain_text

        # Second part should be HTML
        assert parts[1].get_content_type() == "text/html"
        html_part = parts[1].get_payload(decode=True).decode("utf-8")
        assert "<h1>Title</h1>" in html_part
        assert "<p>Test content</p>" in html_part


class TestEmailNotifierHtmlToPlainText:
    """Test HTML to plain text conversion."""

    def test_html_to_plain_text_basic(self):
        """Test basic HTML to plain text conversion."""
        notifier = EmailNotifier(recipients="test@example.com")

        html = "<html><body><h1>Title</h1><p>Content</p></body></html>"
        result = notifier._html_to_plain_text(html)

        assert "Title" in result
        assert "Content" in result
        assert "<" not in result  # No HTML tags

    def test_html_to_plain_text_removes_script_style(self):
        """Test that script and style tags are removed."""
        notifier = EmailNotifier(recipients="test@example.com")

        html = """
        <html>
        <head><style>body { color: red; }</style></head>
        <body>
            <script>alert('test');</script>
            <p>Visible content</p>
        </body>
        </html>
        """
        result = notifier._html_to_plain_text(html)

        assert "Visible content" in result
        assert "alert" not in result
        assert "color: red" not in result

    def test_html_to_plain_text_preserves_links(self):
        """Test that links are converted to text with URL."""
        notifier = EmailNotifier(recipients="test@example.com")

        html = '<p>Visit <a href="https://example.com">our site</a> today!</p>'
        result = notifier._html_to_plain_text(html)

        assert "our site" in result
        assert "https://example.com" in result

    def test_html_to_plain_text_handles_line_breaks(self):
        """Test that block elements create proper line breaks."""
        notifier = EmailNotifier(recipients="test@example.com")

        html = "<p>First paragraph</p><p>Second paragraph</p><br/>New line"
        result = notifier._html_to_plain_text(html)

        # Should have line breaks between paragraphs
        assert "First paragraph" in result
        assert "Second paragraph" in result
        assert "New line" in result

    def test_html_to_plain_text_decodes_entities(self):
        """Test that HTML entities are decoded."""
        notifier = EmailNotifier(recipients="test@example.com")

        html = "<p>5 &gt; 3 &amp; 3 &lt; 5 &quot;quoted&quot;</p>"
        result = notifier._html_to_plain_text(html)

        assert "5 > 3 & 3 < 5" in result
        assert '"quoted"' in result


class TestEmailNotifierBase64ImageExtraction:
    """Test base64 image extraction and CID replacement."""

    def test_extract_no_images(self):
        """Test extraction from HTML with no base64 images."""
        notifier = EmailNotifier(recipients="test@example.com")

        html = "<html><body><p>No images here</p></body></html>"
        modified_html, images = notifier._extract_and_replace_base64_images(html)

        assert modified_html == html
        assert len(images) == 0

    def test_extract_single_png_image(self):
        """Test extraction of a single PNG base64 image."""
        notifier = EmailNotifier(recipients="test@example.com")

        # Create a simple base64 encoded 1x1 PNG
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        html = f'<html><body><img src="data:image/png;base64,{png_b64}" alt="chart"/></body></html>'

        modified_html, images = notifier._extract_and_replace_base64_images(html)

        # Should have replaced base64 with CID reference
        assert "data:image/png;base64," not in modified_html
        assert 'src="cid:chart_' in modified_html
        assert len(images) == 1

        cid, img_data, img_type = images[0]
        assert cid.startswith("chart_")
        assert img_type == "png"
        assert len(img_data) > 0

    def test_extract_multiple_images(self):
        """Test extraction of multiple base64 images."""
        notifier = EmailNotifier(recipients="test@example.com")

        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        html = f"""<html><body>
            <img src="data:image/png;base64,{png_b64}" alt="chart1"/>
            <img src="data:image/jpeg;base64,{png_b64}" alt="chart2"/>
        </body></html>"""

        modified_html, images = notifier._extract_and_replace_base64_images(html)

        assert "data:image/png;base64," not in modified_html
        assert "data:image/jpeg;base64," not in modified_html
        assert len(images) == 2

    def test_extract_preserves_invalid_base64(self):
        """Test that invalid base64 is preserved unchanged."""
        notifier = EmailNotifier(recipients="test@example.com")

        # Invalid base64 data (not properly padded)
        html = '<html><body><img src="data:image/png;base64,invalid!!!" alt="bad"/></body></html>'

        modified_html, images = notifier._extract_and_replace_base64_images(html)

        # Should keep original since decode failed
        assert "data:image/png;base64,invalid!!!" in modified_html
        assert len(images) == 0


class TestEmailNotifierMIMEWithImages:
    """Test MIME message creation with inline images."""

    def test_create_mime_message_with_inline_image(self):
        """Test creating MIME message with base64 image creates multipart/related."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            sender_email="sender@example.com",
        )

        # Create HTML with base64 image
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        html_content = f'<html><body><h1>Report</h1><img src="data:image/png;base64,{png_b64}"/></body></html>'

        mime_msg = notifier._create_mime_message(
            "Test Subject",
            html_content,
            True,
            [],
            [],
            {},
        )

        # Should be multipart/related (root) for inline images
        assert mime_msg.is_multipart()
        assert mime_msg.get_content_type() == "multipart/related"

        # Get all parts
        parts = mime_msg.get_payload()
        assert len(parts) == 2  # alternative part + 1 image

        # First part should be multipart/alternative
        alt_part = parts[0]
        assert alt_part.get_content_type() == "multipart/alternative"

        # Second part should be the image
        img_part = parts[1]
        assert img_part.get_content_type().startswith("image/")
        assert img_part["Content-ID"] is not None

    def test_create_mime_message_html_no_images(self):
        """Test creating HTML MIME message without images uses alternative."""
        notifier = EmailNotifier(
            recipients="test@example.com",
            sender_email="sender@example.com",
        )

        html_content = "<html><body><h1>No images</h1></body></html>"

        mime_msg = notifier._create_mime_message(
            "Test Subject",
            html_content,
            True,
            [],
            [],
            {},
        )

        # Should be multipart/alternative (not related) when no images
        assert mime_msg.is_multipart()
        assert mime_msg.get_content_type() == "multipart/alternative"

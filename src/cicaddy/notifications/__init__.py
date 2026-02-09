"""Notification modules for Cicaddy."""

from typing import Optional

from .base import BaseNotifier, SlackBaseNotifier
from .email import EmailNotifier
from .formatters import (
    MarkdownToSlackConverter,
    MessageLengthManager,
    MessageSectionBuilder,
    SeverityFormatter,
)
from .rich_slack import RichSlackNotifier  # Backward compatibility alias
from .rich_slack import RichSlackNotifier as CronSlackNotifier
from .slack import SlackNotifier

__all__ = [
    "BaseNotifier",
    "SlackBaseNotifier",
    "SlackNotifier",
    "EmailNotifier",
    "RichSlackNotifier",
    "CronSlackNotifier",  # Backward compatibility
    "MarkdownToSlackConverter",
    "MessageSectionBuilder",
    "SeverityFormatter",
    "MessageLengthManager",
    "create_notifier",
]


def create_notifier(
    notifier_type: str,
    webhook_url: Optional[str] = None,
    ssl_verify: bool = True,
    **kwargs,
):
    """Factory function to create appropriate notifier instance.

    Args:
        notifier_type: Type of notifier ('slack', 'email', 'gmail', 'cron_slack')
        webhook_url: Slack webhook URL (for Slack notifiers) or None for email
        ssl_verify: Whether to verify SSL certificates
        **kwargs: Additional keyword arguments for specific notifiers
            For EmailNotifier:
                - recipients: Email recipient(s)
                - sender_email: Sender email address
                - smtp_config: SMTP configuration dict

    Returns:
        Appropriate notifier instance

    Raises:
        ValueError: If notifier_type is not supported or required parameters are missing
    """
    notifier_map = {
        "slack": SlackNotifier,
        "rich_slack": RichSlackNotifier,
        "cron_slack": CronSlackNotifier,  # Backward compatibility
        "email": EmailNotifier,
        "gmail": EmailNotifier,  # Alias for email with Gmail API
    }

    if notifier_type not in notifier_map:
        raise ValueError(f"Unsupported notifier type: {notifier_type}")

    # Special handling for email notifiers
    if notifier_type in ["email", "gmail"]:
        recipients = kwargs.get("recipients")
        if not recipients:
            raise ValueError("EmailNotifier requires 'recipients' parameter")

        return EmailNotifier(
            recipients=recipients,
            sender_email=kwargs.get("sender_email"),
            use_gmail_api=(notifier_type == "gmail"),
            smtp_config=kwargs.get("smtp_config"),
            ssl_verify=ssl_verify,
        )

    # For Slack notifiers, require webhook_url
    if not webhook_url:
        raise ValueError(f"{notifier_type} notifier requires 'webhook_url' parameter")

    return notifier_map[notifier_type](webhook_url, ssl_verify)

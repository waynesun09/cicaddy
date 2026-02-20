"""Backward compatibility — import TaskAgent as CronAIAgent."""

from cicaddy.agent.task_agent import TaskAgent as CronAIAgent  # noqa: F401

__all__ = ["CronAIAgent"]

"""Security utilities for cicaddy.

Provides provenance detection and content scanning for agent security.
"""

from .provenance import get_provenance_label, is_external_source

__all__ = ["is_external_source", "get_provenance_label"]

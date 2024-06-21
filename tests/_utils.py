"""Utilitary functions for running the pytests."""
import os


def are_in_CI() -> bool:
    return any((os.getenv("GITHUB_ACTIONS") == "true", os.getenv("GITLAB_CI")))

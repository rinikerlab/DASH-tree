# Copyright (C) 2024-2025 ETH Zurich, Niels Maeder and other DASH contributors.

"""Utilitary functions for running the pytests."""

import json
import os
from pathlib import Path


def are_in_CI() -> bool:
    return any((os.getenv("GITHUB_ACTIONS") == "true", os.getenv("GITLAB_CI")))


def read_json(file: Path):
    return json.load(file.open("r"))

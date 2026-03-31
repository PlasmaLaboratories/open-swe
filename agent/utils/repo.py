"""Utilities for extracting repository configuration from text."""

from __future__ import annotations

import os
import re

_DEFAULT_REPO_OWNER = os.environ.get("DEFAULT_REPO_OWNER", "langchain-ai")


def extract_repo_from_text(text: str, default_owner: str | None = None) -> dict[str, str] | None:
    """Extract owner/name repo config from text containing repo: syntax or GitHub URLs.

    Checks for explicit ``repo:owner/name`` or ``repo owner/name`` first, then
    falls back to GitHub URL extraction. Name-only overrides are only accepted
    via ``repo:name`` to avoid false positives from natural language like
    ``repo or ...``.

    Returns:
        A dict with ``owner`` and ``name`` keys, or ``None`` if no repo found.
    """
    if default_owner is None:
        default_owner = _DEFAULT_REPO_OWNER
    owner: str | None = None
    name: str | None = None

    colon_match = re.search(r"\brepo:([a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)?)/?\b", text)
    if colon_match:
        value = colon_match.group(1).rstrip("/")
        if "/" in value:
            owner, name = value.split("/", 1)
        else:
            owner = default_owner
            name = value

    if not owner or not name:
        space_match = re.search(r"\brepo\s+([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)/*\b", text)
        if space_match:
            owner, name = space_match.group(1).rstrip("/").split("/", 1)

    if not owner or not name:
        github_match = re.search(r"github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)", text)
        if github_match:
            owner, name = github_match.group(1).split("/", 1)

    if owner and name:
        return {"owner": owner, "name": name}
    return None

"""Keep Claude Code's MCP tool search on when it routes through paritok.

Claude Code defers MCP tool schemas by default, but loads every schema inline once
ANTHROPIC_BASE_URL points to a non-first-party host (most proxies drop the
`tool_reference` blocks tool search needs). Paritok forwards them, so we set
ENABLE_TOOL_SEARCH=true in the `env` block of ~/.claude/settings.json — the
documented place Claude Code reads it from, however `claude` is launched. `true`
matches Claude Code's own first-party default, so direct sessions don't change.

Only ever ADDS that one key: a value the user set (any value) is left alone, an
unreadable file is never rewritten, and the original is backed up once.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

KEY = "ENABLE_TOOL_SEARCH"
# Not ".paritok-bak": the VS Code extension restores that name when healing its
# legacy ANTHROPIC_BASE_URL injection, which would undo the user's later edits.
BACKUP_SUFFIX = ".paritok-toolsearch.bak"


def settings_path(override: str = "") -> Path:
    return Path(override).expanduser() if override else Path.home() / ".claude" / "settings.json"


def ensure_tool_search(override_path: str = "") -> tuple[str, Path, str | None]:
    """Make sure Claude Code keeps tool search on through the proxy.

    Returns (status, path, value):
      "enabled"       we added ENABLE_TOOL_SEARCH=true
      "already_set"   the user's file already sets it (value = theirs, untouched)
      "no_claude"     no ~/.claude directory — Claude Code isn't installed
      "unreadable"    settings.json isn't valid JSON / not an object — left alone
    """
    path = settings_path(override_path)
    if not path.parent.is_dir():
        return "no_claude", path, None
    data: dict = {}
    if path.exists():
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
        except (OSError, ValueError):
            return "unreadable", path, None
        if not isinstance(data, dict) or not isinstance(data.get("env", {}), dict):
            return "unreadable", path, None
        env = data.get("env") or {}
        if KEY in env:
            return "already_set", path, str(env[KEY])
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if not backup.exists():
            backup.write_text(raw, encoding="utf-8")
    data.setdefault("env", {})[KEY] = "true"
    tmp = path.with_name(path.name + ".paritok-tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return "enabled", path, "true"

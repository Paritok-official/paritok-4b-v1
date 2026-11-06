"""`paritok up`/`proxy` keeps Claude Code's tool search on behind the proxy by
adding ENABLE_TOOL_SEARCH=true to ~/.claude/settings.json — never overriding the
user's own value, never rewriting an unreadable file."""
import json

import yaml

from paritok.config import ParitokConfig
from paritok.proxy.claude_setup import BACKUP_SUFFIX, KEY, ensure_tool_search


def _settings(tmp_path, content=None):
    d = tmp_path / ".claude"
    d.mkdir()
    p = d / "settings.json"
    if content is not None:
        p.write_text(content, encoding="utf-8")
    return p


def test_adds_key_and_keeps_everything_else(tmp_path):
    p = _settings(tmp_path, json.dumps({"model": "opus", "env": {"FOO": "1"},
                                        "permissions": {"allow": ["Read"]}}))
    status, path, value = ensure_tool_search(str(p))
    assert (status, value) == ("enabled", "true")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["env"] == {"FOO": "1", KEY: "true"}
    assert data["model"] == "opus" and data["permissions"] == {"allow": ["Read"]}
    # the untouched original is kept once, under paritok's own backup name
    backup = p.with_name(p.name + BACKUP_SUFFIX)
    assert json.loads(backup.read_text(encoding="utf-8"))["env"] == {"FOO": "1"}


def test_creates_settings_when_claude_dir_exists(tmp_path):
    p = _settings(tmp_path)
    assert ensure_tool_search(str(p))[0] == "enabled"
    assert json.loads(p.read_text(encoding="utf-8")) == {"env": {KEY: "true"}}


def test_respects_users_own_value_and_is_idempotent(tmp_path):
    p = _settings(tmp_path, json.dumps({"env": {KEY: "false"}}))
    before = p.read_text(encoding="utf-8")
    assert ensure_tool_search(str(p)) == ("already_set", p, "false")
    assert p.read_text(encoding="utf-8") == before
    q = tmp_path / "other"
    q.mkdir()
    q2 = q / "settings.json"
    ensure_tool_search(str(q2))
    assert ensure_tool_search(str(q2))[0] == "already_set"  # second start: no-op


def test_unreadable_file_is_left_alone(tmp_path):
    p = _settings(tmp_path, "{ not json")
    assert ensure_tool_search(str(p))[0] == "unreadable"
    assert p.read_text(encoding="utf-8") == "{ not json"
    p.write_text(json.dumps({"env": "oops"}), encoding="utf-8")
    assert ensure_tool_search(str(p))[0] == "unreadable"


def test_no_claude_install_no_file(tmp_path):
    p = tmp_path / ".claude" / "settings.json"
    assert ensure_tool_search(str(p))[0] == "no_claude"
    assert not p.exists()


def test_config_toggle_and_starter_yaml(tmp_path, monkeypatch):
    from paritok import cli

    called = []
    monkeypatch.setattr("paritok.proxy.claude_setup.ensure_tool_search",
                        lambda *a, **k: called.append(1) or ("enabled", tmp_path, "true"))
    off = ParitokConfig.from_dict({"claude_code": {"tool_search": False}})
    cli._maybe_enable_claude_tool_search(off)
    assert called == []
    cli._maybe_enable_claude_tool_search(ParitokConfig())
    assert called == [1]
    # the `paritok init` template carries the section and loads cleanly
    starter = ParitokConfig.from_dict(yaml.safe_load(cli._STARTER_YAML))
    assert starter.claude_code.tool_search is True

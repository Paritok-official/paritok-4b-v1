"""The agent-facing [REF] texts stay neutral. Calling condensed views "lossy", saying they
"do NOT match the real file" and that "a fresh Read gives the same lossy view" made
agents stop trusting Read (loopbench: Sonnet read every file with `sed`, one range per
turn; Opus doubled its test runs). The read_original description must also not send the
agent to it for edits, which contradicted the guidance."""
import inspect
import re

from paritok.pipelines.virtual import EXPAND_CONTEXT_SCHEMA
from paritok.proxy import server

ALARM = re.compile(r"lossy|byte-for-byte|ONLY way|true bytes|does NOT match|shrunk", re.I)


def _guidance():
    """The _REF_GUIDANCE literal (it lives inside create_app, so read it from source)."""
    m = re.search(r"_REF_GUIDANCE = \((.*?)\n    \)", inspect.getsource(server), re.S)
    return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1)))


def test_read_original_description_is_neutral_and_not_for_edits():
    desc = EXPAND_CONTEXT_SCHEMA["description"]
    assert not ALARM.search(desc), desc
    assert "especially to EDIT" not in desc
    assert "Edits don't need it" in desc


def test_ref_guidance_is_neutral():
    text = _guidance()
    assert "[REF:<id> src=<path>]" in text
    assert not ALARM.search(text), text
    assert "accurate" in text and "parallel" in text

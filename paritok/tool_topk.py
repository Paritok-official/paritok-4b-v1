"""Embedding-based top-k tool selection with session-freeze + lazy recovery.

CPU-only, free (bge-small, MIT), no external API, no training. Cuts tool-schema
tokens while staying KV-cache friendly, and recovers dropped tools on demand.

Pipeline (each piece validated with real measurements):
  1. SELECT   predict_topk_frozen(): pick top-8 tools on the first turn, then FREEZE
              the set for the whole session. Deterministic + stable tools[] => the
              KV cache doesn't churn, and it's immune to MCP async-load jitter
              (pool growing 60->89 across early turns).
                - bge-small-en-v1.5 (33M, CPU, ~15ms/query)
                - tool text = name auto-split (snake/camel/dot) + own description
                - coding-head whitelist kept when the query looks code-related
                - BFCL recall@10 = 94% (457-tool pool, zero hand-written desc)

  2. APPLY    apply_selection_adaptive(): keep selected full; DROP unselected
              standard tools (the model calls them from training knowledge); only
              STUB unselected MCP tools when the selection's rank-weighted MCP
              signal fires (pure-coding tasks pay nothing for MCP they never use).

  3. RECOVER  When the agent hits a dropped MCP tool during EXECUTION it says so in
              plain text ("I don't have a calendar integration", "no Gmail tool").
              looks_like_missing_tool_help() detects that; recover_tools_from_help()
              embeds the agent's own words and recalls the matching tools to inject
              before re-sending the turn. Measured: agent signals it 3/3 real misses;
              recall from the help text hits the right MCP family 5/5 across
              calendar / drive / gmail (similarity ~0.6).

End-to-end on "fix the bug" (Claude Code, Sonnet, 3 runs median):
    no-compression $0.2536 | frozen-drop $0.1118 (-56%) | adaptive $0.1216.
Compression only pays off when tools[] is large; on long msg-heavy sessions the
win shifts to history/tool-output compression (a different pipe).
"""
from __future__ import annotations
import re
import functools
from typing import Iterable

DEFAULT_K = 8
WHITELIST = ["Read", "Grep", "Glob", "Bash", "Edit", "Write"]  # Claude Code coding head
_CODE_HINT = re.compile(
    r"(\bbugs?\b|\bcode\b|\bfiles?\b|\bfunctions?\b|\brefactor|\btests?\b|\bbuild\b|"
    r"\bimports?\b|\bgrep\b|\brepo\b|\bmodules?\b|\bgit\b|\brun\b|\bfix|\bedit|\brename|"
    r"\bcodebase\b|\bdeprecated\b|\bcompile|\bclass\b|\bmethods?\b|\bvariables?\b|"
    r"\bstack trace\b|\bimplementation\b|\bparser\b|\bcrash|\bserver\b|"
    r"\w+\(\)|[a-z]+_[a-z]+|\.(py|js|ts|go|md|tsx|java|cpp|rs|json|yaml))", re.I)

# Generic/noisy tools that hijack "send/message/share" queries — excluded from recovery.
_GENERIC = {"SendMessage", "PushNotification", "Monitor", "TaskUpdate",
            "TaskGet", "TaskList", "Skill", "ReportFindings"}

# The agent, on hitting a dropped tool during execution, plainly says it lacks it.
_HELP_HINT = re.compile(
    r"((do(n'?| no)t|does(n'?| no)t) have\b.{0,40}\b(tool|connector|integration|access|ability)"
    r"|no\b.{0,30}\b(tool|connector|integration)\b.{0,20}\bavailable"
    r"|not available in this (session|environment)"
    r"|is(n'?t| not) available"
    r"|can'?t (complete|do that|help with that|look (this|that) up))", re.I)


def _name_words(name: str) -> str:
    """Auto-split a tool name into words: snake_case, camelCase, dotted, mcp__srv__act."""
    s = name.replace("mcp__", "").replace("__", " ")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)   # camelCase
    s = re.sub(r"[._]+", " ", s)                     # snake / dotted
    return s.strip()


def _expand(query: str) -> str:
    if _CODE_HINT.search(query or ""):
        return (query or "") + " (look at code, read and search source files, run commands, fix bugs)"
    return query or ""


@functools.lru_cache(maxsize=1)
def _model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            'Embedding tool-selection needs an optional dependency.\n'
            '  Install it with:  pip install "paritok[toolselect]"'
        ) from e
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


def _tool_name(t: dict) -> str:
    return t.get("name") or (t.get("function") or {}).get("name") or ""


def _tool_desc(t: dict) -> str:
    return t.get("description") or (t.get("function") or {}).get("description") or ""


class TopKToolSelector:
    """Caches tool vectors by the tool set, so re-selection within a session is cheap
    and deterministic (stable tools[] => KV-cache friendly)."""

    def __init__(self, k: int = DEFAULT_K):
        self.k = k
        self._cache: dict = {}  # tools-hash -> (names, matrix)

    def _encode_tools(self, tools: list[dict]):
        names = [_tool_name(t) for t in tools]
        key = hash(tuple(names))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        import numpy as np
        texts = [f"{_name_words(n)}. {_tool_desc(t)}" for n, t in zip(names, tools)]
        mat = _model().encode(texts, normalize_embeddings=True)
        self._cache[key] = (names, np.asarray(mat))
        return names, np.asarray(mat)

    def select(self, user_message: str, tools: list[dict], k: int | None = None) -> list[str]:
        """Fixed-k: return the names of the <=k tools to keep in full schema."""
        import numpy as np
        k = k or self.k
        if len(tools) <= k:
            return [_tool_name(t) for t in tools]
        names, mat = self._encode_tools(tools)
        present = set(names)
        is_code = bool(_CODE_HINT.search(user_message or ""))
        qv = _model().encode([_expand(user_message)], normalize_embeddings=True)[0]
        order = np.argsort(-(mat @ qv))
        picked = [w for w in WHITELIST if w in present] if is_code else []
        for i in order:
            if names[i] not in picked:
                picked.append(names[i])
            if len(picked) >= k:
                break
        return picked[:k]

    def select_dynamic(self, user_message: str, tools: list[dict],
                       alpha: float = 0.9, k_min: int = 5, k_max: int = 20) -> list[str]:
        """Dynamic-k: keep tools whose similarity >= alpha * max_sim, clamped to
        [k_min, k_max]. Simple/focused queries keep few (~5); complex ones keep more.
        Validated on BFCL: recall 93% at avg 6.3 kept (vs fixed k=8 at 8)."""
        import numpy as np
        if len(tools) <= k_min:
            return [_tool_name(t) for t in tools]
        names, mat = self._encode_tools(tools)
        present = set(names)
        is_code = bool(_CODE_HINT.search(user_message or ""))
        qv = _model().encode([_expand(user_message)], normalize_embeddings=True)[0]
        sims = mat @ qv
        order = np.argsort(-sims)
        mx = float(sims[order[0]])
        picked = [w for w in WHITELIST if w in present] if is_code else []
        for i in order:
            s = float(sims[i])
            if s < alpha * mx and len(picked) >= k_min:
                break
            if names[i] not in picked:
                picked.append(names[i])
            if len(picked) >= k_max:
                break
        for i in order:  # top up to k_min
            if len(picked) >= k_min:
                break
            if names[i] not in picked:
                picked.append(names[i])
        return picked[:k_max]


class SessionFrozenSelector:
    """Selects tools ONCE per session (first turn), then freezes the choice.

    Immune to MCP async-load jitter (tool pool growing 60->89 across early turns)
    and keeps tools[] byte-stable turn-to-turn => KV-cache friendly. Extra tools
    can be added later via add_to_frozen() (lazy recovery)."""
    def __init__(self, alpha: float = 0.9, k_min: int = 5, k_max: int = 8):
        self.alpha, self.k_min, self.k_max = alpha, k_min, k_max
        self._sel = TopKToolSelector()
        self._frozen: dict[str, list[str]] = {}  # session_id -> frozen tool names

    def select(self, session_id: str, user_message: str, tools: list[dict]) -> list[str]:
        present = {_tool_name(t) for t in tools}
        frozen = self._frozen.get(session_id)
        if frozen is not None:
            kept = [n for n in frozen if n in present]
            if kept:
                return kept
        chosen = self._sel.select_dynamic(user_message, tools, self.alpha, self.k_min, self.k_max)
        if len(tools) > self.k_min:
            self._frozen[session_id] = chosen
        return chosen

    def add_to_frozen(self, session_id: str, names: Iterable[str]) -> None:
        """Permanently add recovered tools to this session's frozen set (lazy recovery),
        so once a tool is needed it stays available for the rest of the session."""
        cur = self._frozen.setdefault(session_id, [])
        for n in names:
            if n not in cur:
                cur.append(n)


def _is_mcp(name: str) -> bool:
    return name.startswith("mcp__")


def _make_stub_anthropic(t: dict) -> dict:
    n = _tool_name(t)
    return {"name": n,
            "description": "[deferred] " + (_tool_desc(t)[:48]) + " — call to load full schema.",
            "input_schema": {"type": "object", "properties": {}}}


def _make_stub_openai(t: dict) -> dict:
    n = _tool_name(t)
    return {"type": "function", "name": n,
            "description": "[deferred] " + (_tool_desc(t)[:48]) + " — call to load full schema.",
            "parameters": {"type": "object", "properties": {}}}


def apply_selection(tools: list[dict], keep_names: set, wire: str = "anthropic") -> list[dict]:
    """keep selected full; DROP unselected standard tools; STUB unselected MCP tools."""
    stub = _make_stub_openai if wire == "openai" else _make_stub_anthropic
    out = []
    for t in tools:
        n = _tool_name(t)
        if n in keep_names:
            out.append(t)
        elif _is_mcp(n):
            out.append(stub(t))
    return out


def _mcp_signal_score(keep_ordered: list[str]) -> float:
    """Rank-weighted MCP signal: a top-ranked MCP tool strongly implies an MCP task;
    low-ranked (6-8th) MCP tools are likely false 'search/find' recalls from a coding
    query. Weights: top-3 -> 3.0, 4-6 -> 1.0, 7+ -> 0.3.
    On 100 cases: threshold 1.0 => FN 4, FP 11 (vs count>=1: FN 0, FP 33)."""
    s = 0.0
    for rank, name in enumerate(keep_ordered):
        if _is_mcp(name):
            s += 3.0 if rank < 3 else (1.0 if rank < 6 else 0.3)
    return s


def apply_selection_adaptive(tools: list[dict], keep_ordered, wire: str = "anthropic",
                             mcp_signal_threshold: float = 1.0) -> list[dict]:
    """Task-adaptive with RANK-WEIGHTED MCP detection. Stub MCP tools only when the
    selection's rank-weighted MCP signal >= threshold; else drop everything unselected
    (pure-coding tasks pay nothing for MCP). keep_ordered must be in RANK ORDER."""
    keep_list = list(keep_ordered)
    keep_names = set(keep_list)
    fire = _mcp_signal_score(keep_list) >= mcp_signal_threshold
    stub = _make_stub_openai if wire == "openai" else _make_stub_anthropic
    out = []
    for t in tools:
        n = _tool_name(t)
        if n in keep_names:
            out.append(t)
        elif _is_mcp(n) and fire:
            out.append(stub(t))
    return out


# ---- Lazy recovery (execution-time) ----

def looks_like_missing_tool_help(agent_text: str) -> bool:
    """True if the agent's response signals it lacked a tool to finish the task
    (e.g. "I don't have a calendar integration", "no Gmail tool available",
    "the X tool isn't available", "I can't complete this request").
    Measured: fires on 3/3 real MCP misses; standard-tool tasks that succeed don't."""
    return bool(_HELP_HINT.search(agent_text or ""))


def recover_tools_from_help(help_text: str, candidate_tools: Iterable[dict],
                            k: int = DEFAULT_K, exclude_generic: bool = True) -> list[str]:
    """Embed the agent's plain-text 'I don't have an X tool...' and return the top-k
    currently-dropped tools that best match — the tools to inject before re-sending
    the turn. Recall the right MCP family 5/5 across calendar/drive/gmail (sim ~0.6).

    candidate_tools = the tools NOT currently in the request (the dropped pool).
    Returns tool names ranked by relevance to what the agent said it needed."""
    import numpy as np
    cand = list(candidate_tools)
    if not cand or not (help_text or "").strip():
        return []
    names = [_tool_name(t) for t in cand]
    texts = [f"{_name_words(n)}. {_tool_desc(t)}" for n, t in zip(names, cand)]
    dv = _model().encode(texts, normalize_embeddings=True)
    qv = _model().encode([help_text], normalize_embeddings=True)[0]
    order = np.argsort(-(dv @ qv))
    out = []
    for i in order:
        if exclude_generic and names[i] in _GENERIC:
            continue
        out.append(names[i])
        if len(out) >= k:
            break
    return out


# ---- module-level convenience (shares one cached model + selectors) ----
_default = TopKToolSelector()
_frozen_default = SessionFrozenSelector(alpha=0.9, k_min=5, k_max=8)


def predict_topk_tools(user_message: str, tools: Iterable[dict], k: int = DEFAULT_K) -> list[str]:
    """Fixed-k: names of the k most relevant tools. Keep these full, stub/drop the rest."""
    return _default.select(user_message, list(tools), k)


def predict_topk_dynamic(user_message: str, tools: Iterable[dict],
                         alpha: float = 0.9, k_min: int = 5, k_max: int = 20) -> list[str]:
    """Dynamic-k by similarity threshold. Simple queries keep ~5, complex keep more."""
    return _default.select_dynamic(user_message, list(tools), alpha, k_min, k_max)


def predict_topk_frozen(session_id: str, user_message: str, tools: Iterable[dict]) -> list[str]:
    """Session-frozen selection (k_max=8). Same set for the whole session; recovered
    tools can be pinned via register_recovered_tools()."""
    return _frozen_default.select(session_id, user_message, list(tools))


def register_recovered_tools(session_id: str, names: Iterable[str]) -> None:
    """After lazy recovery, pin the recovered tools into the session's frozen set so
    they stay available for the rest of the session (avoids repeat misses)."""
    _frozen_default.add_to_frozen(session_id, names)

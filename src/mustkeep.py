"""Regex rules for must-keep span detection — optimized with combined pattern."""
import re

SKIP_KINDS = {"system_prompt"}
MIN_CONTENT_LENGTH = 50


# === 单个合并的大正则 (一次扫描搞定所有 kind) ===
COMBINED_PATTERN = re.compile(
    r'(?P<file_path>'
    r'/workspace/[\w\-./_]+'
    r'|/[\w\-./_]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|c|h|hpp|rb|php|cs|md|yaml|yml|json|toml|cfg|ini|sh)\b'
    r'|(?<![\w/])(?:src|tests?|lib|app)/[\w\-./_]+\.(?:py|js|ts|jsx|java|go|rs|cpp|c|h|md|yaml|yml|json|toml)\b'
    r')'
    r'|(?P<stack_trace_path>File\s+"[^"]+\.[a-z]{1,4}")'
    r'|(?P<line_number>'
    r'(?:line|L)\s*[:#]?\s*\d+'
    r'|(?<=[\s,\(\[])\d{1,4}:\d{1,3}(?=[\s,\)\]])'
    r')'
    r'|(?P<hash_>\b(?:commit\s+)?[0-9a-f]{7,40}\b)'
    r'|(?P<url>https?://[^\s\'"<>\]`)]+)'
    r'|(?P<error_class>\b[A-Z][a-zA-Z]*(?:Error|Exception|Warning|Failure|Fault)\b)'
    r'|(?P<identifier>\b'
    r'[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*){2,}'
    r'|[a-z][a-z0-9]*(?:[A-Z][a-zA-Z0-9]+){2,}'
    r'|[A-Z][a-z]+[A-Z][a-z][a-zA-Z0-9]{4,}'
    r'\b)'
    r'|(?P<quoted_literal>\'[^\']{4,80}/[^\']*\')'
    r'|(?P<package_name>\b(?:netcdf4|h5netcdf|xarray|numpy|pandas|torch|tensorflow|sklearn|matplotlib|scipy|requests|django|flask|fastapi|pytest|sympy)\b)',
)

# 对 assistant_thinking 用更保守的子集
THINKING_GROUPS = {"error_class", "package_name", "quoted_literal", "url"}


def find_must_keep_spans(text: str, seg_id: str, seg_kind: str) -> list[dict]:
    if seg_kind in SKIP_KINDS:
        return []
    if len(text) < MIN_CONTENT_LENGTH:
        return []

    is_thinking = (seg_kind == "assistant_thinking")

    spans = []
    for m in COMBINED_PATTERN.finditer(text):
        kind = m.lastgroup
        if kind == "hash_":
            kind = "hash"
        if is_thinking and kind not in THINKING_GROUPS:
            continue
        spans.append({
            "seg_id": seg_id,
            "start": m.start(),
            "end": m.end(),
            "kind": kind,
            "text": m.group(0),
        })

    # 去重 + 合并重叠
    spans.sort(key=lambda s: (s["start"], -s["end"]))
    merged = []
    for s in spans:
        if merged and s["start"] < merged[-1]["end"]:
            if s["end"] > merged[-1]["end"]:
                merged[-1] = s
        else:
            merged.append(s)

    return merged
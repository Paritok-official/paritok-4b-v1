"""BFCL tool-selection eval + torch-vs-onnx backend parity (opt-in, #tool-ranking).

Reconstructs the Berkeley Function-Calling Leaderboard (BFCL) retrieval eval that
produced tool_topk.py's docstring numbers (recall@10 = 94% on a 457-tool pool; dynamic
93% at avg 6.3 kept), and additionally checks that swapping the embedding backend from
sentence-transformers (torch) to fastembed (onnxruntime) leaves tool ranking unchanged —
the validation for dropping the ~4GB CUDA-torch install in favour of a ~200MB onnx one.

Skipped by default (downloads the BFCL dataset + bge-small weights, and needs fastembed
for the onnx half). Run it explicitly:

    PARITOK_RUN_BFCL=1 python -m pytest tests/test_tool_select_bfcl.py -s
    PARITOK_RUN_BFCL=1 python tests/test_tool_select_bfcl.py        # prints the full report
"""
from __future__ import annotations

import json
import os
import re

import pytest

RUN = os.environ.get("PARITOK_RUN_BFCL") == "1"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set PARITOK_RUN_BFCL=1 to run the BFCL eval (downloads data + models)")

_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
_SPLIT = "BFCL_v3_live_multiple.json"


def _name_words(n: str) -> str:
    return re.sub(r"[._]+", " ", re.sub(r"([a-z])([A-Z])", r"\1 \2", n))


def _load_bfcl():
    """Download BFCL (cached by HF), build the deduped global tool pool, and the valid
    (query, gold-tool-set) pairs whose gold is fully inside the pool. Returns
    (names, tool_texts, queries, golds)."""
    from huggingface_hub import hf_hub_download
    ent_p = hf_hub_download(_DATASET, _SPLIT, repo_type="dataset")
    ans_p = hf_hub_download(_DATASET, f"possible_answer/{_SPLIT}", repo_type="dataset")
    entries = [json.loads(x) for x in open(ent_p, encoding="utf-8") if x.strip()]
    answers = {a["id"]: a for a in (json.loads(x) for x in open(ans_p, encoding="utf-8") if x.strip())}

    pool: dict[str, str] = {}
    for e in entries:
        for f in e.get("function", []):
            pool[f["name"]] = f.get("description", "")
    names = list(pool)
    name_pos = {n: i for i, n in enumerate(names)}

    def golds_of(e):
        a = answers.get(e["id"])
        out: list[str] = []
        if a:
            for it in a.get("ground_truth", []):
                if isinstance(it, dict):
                    out += list(it.keys())
        return out

    def get_q(e):
        q = e["question"]
        msgs = q[0] if q and isinstance(q[0], list) else q
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content", "")
        return json.dumps(q)[:400]

    valid = [e for e in entries if golds_of(e) and all(g in name_pos for g in golds_of(e))]
    tool_texts = [f"{_name_words(n)}. {pool[n]}" for n in names]
    queries = [get_q(e) for e in valid]
    golds = [set(golds_of(e)) for e in valid]
    return names, tool_texts, queries, golds


class _TorchEncoder:
    """sentence-transformers (torch) — paritok's current backend."""
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def encode(self, texts, normalize_embeddings=True, **_):
        return self._m.encode(list(texts), normalize_embeddings=normalize_embeddings)


class _OnnxEncoder:
    """fastembed (onnxruntime) — same bge-small weights, no torch/CUDA."""
    def __init__(self):
        from fastembed import TextEmbedding
        self._m = TextEmbedding("BAAI/bge-small-en-v1.5")

    def encode(self, texts, normalize_embeddings=True, **_):
        import numpy as np
        v = np.asarray(list(self._m.embed(list(texts))), dtype="float32")
        if normalize_embeddings:
            v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
        return v


def _sim(encoder, tool_texts, queries):
    import numpy as np
    dv = np.asarray(encoder.encode(tool_texts, normalize_embeddings=True), dtype="float32")
    qv = np.asarray(encoder.encode(queries, normalize_embeddings=True), dtype="float32")
    return qv @ dv.T


def _recall_at_k(sim, names, golds, k):
    import numpy as np
    full = 0
    for i, gold in enumerate(golds):
        topk = {names[j] for j in np.argsort(-sim[i])[:k]}
        full += gold.issubset(topk)
    return full / len(golds)


def _dynamic_fullhit(sim, names, golds, alpha=0.9, kmin=5, kmax=20):
    import numpy as np
    full, kept = 0, []
    for i, gold in enumerate(golds):
        row = sim[i]
        order = np.argsort(-row)
        mx = float(row[order[0]])
        picked = []
        for j in order:
            if float(row[j]) < alpha * mx and len(picked) >= kmin:
                break
            picked.append(names[j])
            if len(picked) >= kmax:
                break
        kept.append(len(picked))
        full += set(gold).issubset(picked)
    return full / len(golds), (sum(kept) / len(kept))


def _real_selections(encoder, names, tool_texts, queries):
    """Selections from paritok's ACTUAL select_dynamic, with the embedding backend swapped."""
    import paritok.tool_topk as tk
    tools = [{"name": n, "description": t.split(". ", 1)[-1]} for n, t in zip(names, tool_texts)]
    orig = tk._model
    tk._model = lambda: encoder
    try:
        sel = tk.TopKToolSelector()
        return [set(sel.select_dynamic(q, tools, alpha=0.9, k_min=5, k_max=20)) for q in queries]
    finally:
        tk._model = orig


def _run(sample_real=150):
    names, tool_texts, queries, golds = _load_bfcl()
    torch_enc = _TorchEncoder()
    onnx_enc = _OnnxEncoder()
    st, so = _sim(torch_enc, tool_texts, queries), _sim(onnx_enc, tool_texts, queries)

    rep = {"pool": len(names), "queries": len(queries)}
    rep["torch_r10"] = _recall_at_k(st, names, golds, 10)
    rep["onnx_r10"] = _recall_at_k(so, names, golds, 10)
    rep["torch_dyn"], rep["torch_kept"] = _dynamic_fullhit(st, names, golds)
    rep["onnx_dyn"], rep["onnx_kept"] = _dynamic_fullhit(so, names, golds)

    # backend parity through paritok's REAL select_dynamic (sample for speed)
    qs = queries[:sample_real]
    ts = _real_selections(torch_enc, names, tool_texts, qs)
    os_ = _real_selections(onnx_enc, names, tool_texts, qs)
    identical = sum(a == b for a, b in zip(ts, os_)) / len(qs)
    jacc = sum(len(a & b) / len(a | b) for a, b in zip(ts, os_) if (a | b)) / len(qs)
    rep["real_sample"] = len(qs)
    rep["identical_frac"] = identical
    rep["mean_jaccard"] = jacc
    return rep


def _print(rep):
    print(f"\nBFCL pool={rep['pool']} tools, {rep['queries']} valid queries")
    print(f"  fixed recall@10   torch {rep['torch_r10']*100:.1f}%   onnx {rep['onnx_r10']*100:.1f}%"
          f"   (docstring: 94%)")
    print(f"  dynamic full-hit  torch {rep['torch_dyn']*100:.1f}% @ {rep['torch_kept']:.1f} kept"
          f"   onnx {rep['onnx_dyn']*100:.1f}% @ {rep['onnx_kept']:.1f} kept   (docstring: 93% @ 6.3)")
    print(f"  real select_dynamic parity (n={rep['real_sample']}):"
          f"  identical {rep['identical_frac']*100:.1f}%   mean Jaccard {rep['mean_jaccard']:.4f}")


def test_bfcl_recall_and_backend_parity():
    rep = _run()
    _print(rep)
    # 1) the shipped strategy still hits its recorded recall (tolerance for model/version drift)
    assert rep["torch_r10"] >= 0.90, rep
    assert rep["torch_dyn"] >= 0.90, rep
    # 2) onnx (fastembed) matches torch on recall — swapping the backend costs no quality
    assert abs(rep["onnx_r10"] - rep["torch_r10"]) <= 0.02, rep
    assert abs(rep["onnx_dyn"] - rep["torch_dyn"]) <= 0.02, rep
    # 3) and the ACTUAL select_dynamic picks the same tools under either backend
    assert rep["identical_frac"] >= 0.90, rep
    assert rep["mean_jaccard"] >= 0.97, rep


if __name__ == "__main__":
    _print(_run())

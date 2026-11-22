"""Retrieval: BM25, dense (multilingual-e5-small), hybrid (RRF), cross-encoder reranker, metadata/temporal filter."""
import hashlib
import json
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi

from .config import EMB_MODEL, PROC, RERANK_MODEL
from .textutil import normalize_slang

_EMB = None
_RER = None


def emb_model():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        import torch
        _EMB = SentenceTransformer(EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    return _EMB


def reranker():
    global _RER
    if _RER is None:
        from sentence_transformers import CrossEncoder
        import torch
        _RER = CrossEncoder(RERANK_MODEL, device="cuda" if torch.cuda.is_available() else "cpu", max_length=256)
    return _RER


def tok(s):
    return re.findall(r"\w+", normalize_slang(s))


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def passes_filter(d, flt):
    """Metadata filter: doc_type, split (anti-leakage: hanya train), dan validitas temporal terhadap as_of."""
    if not flt:
        return True
    if "doc_type" in flt and d.get("doc_type") != flt["doc_type"]:
        return False
    if d.get("doc_type") == "example" and d.get("split") != "train":
        return False
    if "as_of" in flt:
        if not (d.get("valid_from", "0000") <= flt["as_of"] <= d.get("valid_to", "9999")):
            return False
    return True


class Index:
    def __init__(self, docs, name):
        self.docs = docs
        self.name = name
        self.bm25 = BM25Okapi([tok(d["text"]) for d in docs])
        h = hashlib.sha256(("".join(d["doc_id"] + d["text"] for d in docs)).encode()).hexdigest()[:16]
        self.version = f"{name}@{h}"
        cache = PROC / f"emb_{name}_{h}.npy"
        if cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = emb_model().encode(["passage: " + d["text"] for d in docs], batch_size=128,
                                          normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            np.save(cache, self.emb)

    def _bm25(self, q):
        return np.asarray(self.bm25.get_scores(tok(q)))

    def _dense(self, q):
        qe = emb_model().encode(["query: " + q], normalize_embeddings=True, show_progress_bar=False)[0]
        return self.emb @ qe

    def search(self, query, k=5, mode="hybrid_rerank", flt=None, n_cand=30, exclude_ids=()):
        t0 = time.perf_counter()
        mask = np.array([passes_filter(d, flt) and d["doc_id"] not in exclude_ids for d in self.docs])
        if mode == "bm25":
            s = self._bm25(query); s[~mask] = -1e9
            order = np.argsort(-s)[:k]
            res = [(int(i), float(s[i])) for i in order]
        elif mode == "dense":
            s = self._dense(query); s[~mask] = -1e9
            order = np.argsort(-s)[:k]
            res = [(int(i), float(s[i])) for i in order]
        else:
            sb = self._bm25(query); sb[~mask] = -1e9
            sd = self._dense(query); sd[~mask] = -1e9
            rb = np.argsort(-sb)[:100]
            rd = np.argsort(-sd)[:100]
            fused = {}
            for rank, i in enumerate(rb):
                fused[int(i)] = fused.get(int(i), 0) + 1.0 / (60 + rank + 1)
            for rank, i in enumerate(rd):
                fused[int(i)] = fused.get(int(i), 0) + 1.0 / (60 + rank + 1)
            cand = sorted(fused.items(), key=lambda x: -x[1])
            if mode == "hybrid":
                res = cand[:k]
            else:  # hybrid_rerank
                cand = cand[:n_cand]
                scores = reranker().predict([(query, self.docs[i]["text"]) for i, _ in cand], show_progress_bar=False)
                order = np.argsort(-np.asarray(scores))[:k]
                res = [(cand[j][0], float(scores[j])) for j in order]
        out = []
        for i, sc in res:
            d = dict(self.docs[i]); d["score"] = sc
            out.append(d)
        return out, (time.perf_counter() - t0) * 1000


_INDEX = {}


def get_index(name):
    """name: 'examples' | 'guide_fixed' | 'guide_heading' | 'guide_hierarchical'"""
    if name not in _INDEX:
        if name == "examples":
            docs = load_jsonl(PROC / "corpus_examples.jsonl")
        else:
            docs = load_jsonl(PROC / f"guide_chunks_{name.split('_', 1)[1]}.jsonl")
        _INDEX[name] = Index(docs, name)
    return _INDEX[name]

"""Pipeline konfigurasi B0-B5 (B6 di agent.py). Semua memakai base generator yang sama (Qwen2.5-0.5B-Instruct)."""
import json
import math
import re
import time
from functools import lru_cache

from .config import ADAPTERS, BUDGET_SMALL, DEFAULT_AS_OF, LABELS, PROC
from .llm import get_generator, parse_output
from .prompts import build_messages, example_answer, few_shot_examples, prompt_hash
from .retrieval import get_index
from .textutil import normalize_slang, scope_check, span_in_text

CONFIGS = {
    "B0": dict(prompt="P0_zero_shot@1.0", few_shot=False, retrieval=None, planner=False, verifier=False, adapter=None),
    "B1": dict(prompt="P1_structured@1.0", few_shot=True, retrieval=None, planner=False, verifier=False, adapter=None),
    "B2": dict(prompt="P2_rag@1.0", few_shot=True, retrieval="naive", planner=False, verifier=False, adapter=None),
    "B3": dict(prompt="P2_rag@1.0", few_shot=True, retrieval="hybrid", planner=False, verifier=False, adapter=None),
    "B4": dict(prompt="P2_rag@1.0", few_shot=True, retrieval="hybrid", planner=True, verifier=True, adapter=None),
    "B5": dict(prompt="P2_rag@1.0", few_shot=True, retrieval="hybrid", planner=True, verifier=True, adapter="lora_b5"),
}
SUFF_TAU = 0.5      # ambang evidence sufficiency (retry jika di bawah)
MAX_RETRY = 1       # retry retrieval bounded


@lru_cache(1)
def lexicon():
    return json.load(open(PROC / "lexicon.json", encoding="utf-8"))


def lexicon_hits(text, label=None):
    lx = lexicon()
    words = set(re.findall(r"[a-z]+", str(text).lower()))
    norm = set(re.findall(r"[a-z]+", normalize_slang(text)))
    pool = set()
    labs = [label] if label else LABELS
    for l in labs:
        pool |= set(lx["chi2"][l][:40])
        if l in ("Abusive",):
            pool |= set(lx["abusive_csv"])
    return sorted(w for w in words | norm if w in pool and w in words)


def silver_evidence(text, row):
    ev = []
    for l in LABELS:
        if row[l] == 1:
            ev += lexicon_hits(text, l)
    return sorted(set(ev))[:3]


@lru_cache(1)
def few_shot_pairs():
    return [(r["text"], example_answer(r, silver_evidence(r["text"], r))) for r in few_shot_examples()]


def fmt_labels(d):
    return ",".join(f"{l}={d[l]}" for l in LABELS)


def neighbor_vote(examples):
    w = [1.0 / (i + 1) for i in range(len(examples))]
    tot = sum(w) or 1
    return {l: sum(wi * e["labels"][l] for wi, e in zip(w, examples)) / tot for l in LABELS}


def sufficiency(examples, mode):
    """Evidence sufficiency in [0,1] = 0.5*relevansi (sigmoid skor reranker) + 0.5*kesepakatan label tetangga."""
    if not examples:
        return 0.0
    if mode == "hybrid":
        rel = sum(1 / (1 + math.exp(-e["score"])) for e in examples) / len(examples)
    else:
        rel = sum(max(0.0, e["score"]) for e in examples) / len(examples)
    vecs = [tuple(e["labels"][l] for l in LABELS) for e in examples]
    agree = max(vecs.count(v) for v in set(vecs)) / len(vecs)
    return 0.5 * rel + 0.5 * agree


def build_context(examples, guides, budget):
    lines = []
    gv = sorted({g["version"] for g in guides})
    for g in guides:
        t = g.get("parent_text") or g["text"]
        lines.append(f"[Pedoman v{g['version']}] {t[:700]}")
    for e in examples:
        lines.append(f'- Contoh: "{e["text"][:200]}" -> {fmt_labels(e["labels"])}')
    ctx, n = [], 0
    for ln in lines:
        if n + len(ln) > budget["max_ctx_chars"]:
            break
        ctx.append(ln); n += len(ln)
    return "\n".join(ctx), gv


def retrieve(text, mode, budget, as_of, query=None, k_mult=1, guide_strategy=None):
    q = query or text
    if mode == "naive":
        ex, t1 = get_index("examples").search(q, k=budget["k_examples"] * k_mult, mode="dense")
        gs, t2 = get_index("guide_" + (guide_strategy or "fixed")).search(q, k=budget["k_guides"], mode="dense")
    else:
        ex, t1 = get_index("examples").search(q, k=budget["k_examples"] * k_mult, mode="hybrid_rerank",
                                              flt={"doc_type": "example"})
        gs, t2 = get_index("guide_" + (guide_strategy or "heading")).search(q, k=budget["k_guides"], mode="hybrid_rerank",
                                                                            flt={"as_of": as_of})
    return ex, gs, t1 + t2


def verify(rec, text, vote):
    """Independent verifier: konsistensi label + span grounding. Return list label positif tanpa span valid."""
    p = rec["pred"]
    if p["HS_Individual"] or p["HS_Group"]:
        p["HS"] = 1
    if p["HS_Individual"] and p["HS_Group"]:
        drop = "HS_Group" if vote["HS_Individual"] >= vote["HS_Group"] else "HS_Individual"
        p[drop] = 0
    if p["HS"] and not (p["HS_Individual"] or p["HS_Group"]):
        p["HS_Individual" if vote["HS_Individual"] >= vote["HS_Group"] else "HS_Group"] = 1
    rec["evidence"] = [e for e in rec["evidence"] if span_in_text(e, text)]
    pos = [l for l in LABELS if p[l]]
    return pos if (pos and not rec["evidence"]) else []


class Pipeline:
    def __init__(self, name, budget=None, as_of=DEFAULT_AS_OF, guide_strategy=None, overrides=None):
        self.name = name
        self.cfg = dict(CONFIGS[name if name in CONFIGS else "B4"])
        self.cfg.update(overrides or {})
        self.budget = budget or BUDGET_SMALL
        self.as_of = as_of
        self.guide_strategy = guide_strategy
        ad = self.cfg["adapter"]
        self.adapter_path = (ADAPTERS / ad) if ad else None

    def meta(self):
        return {"config": self.name, "prompt_id": self.cfg["prompt"], "prompt_hash": prompt_hash(self.cfg["prompt"]),
                "adapter": str(self.adapter_path.name) if self.adapter_path else None, "budget": self.budget,
                "retrieval": self.cfg["retrieval"], "planner": self.cfg["planner"], "verifier": self.cfg["verifier"]}

    # ---- tahap 1: planner + retrieval ----
    def prepare(self, item):
        text = item["text"]
        st = {"id": item["id"], "text": text, "retrieved": [], "guide_versions": [], "retries": 0, "suff": None,
              "retr_ms": 0.0, "early_abstain": None, "vote": {l: 0.0 for l in LABELS}, "context": None}
        if self.cfg["planner"]:
            ok, reason = scope_check(text)
            if not ok:
                st["early_abstain"] = reason
                return st
        if self.cfg["retrieval"]:
            ex, gs, ms = retrieve(text, self.cfg["retrieval"], self.budget, self.as_of, guide_strategy=self.guide_strategy)
            st["retr_ms"] += ms
            if self.cfg["planner"]:
                mode = "hybrid"
                st["suff"] = sufficiency(ex, mode)
                if self.cfg.get("adaptive", True) and st["suff"] < SUFF_TAU and st["retries"] < MAX_RETRY:
                    # query planner: normalisasi slang + kata penanda lexicon, k diperbesar 2x (bounded)
                    q2 = normalize_slang(text) + " " + " ".join(lexicon_hits(text))
                    ex2, gs2, ms2 = retrieve(text, self.cfg["retrieval"], self.budget, self.as_of, query=q2, k_mult=2,
                                             guide_strategy=self.guide_strategy)
                    st["retries"] += 1
                    st["retr_ms"] += ms2
                    s2 = sufficiency(ex2[: len(ex)], mode)
                    if s2 > st["suff"]:
                        ex, gs, st["suff"] = ex2[: self.budget["k_examples"]], gs2, s2
            st["context"], st["guide_versions"] = build_context(ex, gs, self.budget)
            st["retrieved"] = [d["doc_id"] for d in ex + gs]
            st["vote"] = neighbor_vote(ex)
        return st

    def augment(self, states, gen, batch_size):
        """Hook untuk B6 (agentic). Default: tidak ada."""
        return None

    def messages(self, st):
        fs = few_shot_pairs() if self.cfg["few_shot"] else None
        return build_messages(self.cfg["prompt"], st["text"], context=st["context"], few_shot=fs)

    def run(self, items, temperature=0.0, seed=None, batch_size=8):
        gen = get_generator(self.adapter_path)
        states = [self.prepare(it) for it in items]
        todo = [s for s in states if not s["early_abstain"]]
        self.augment(todo, gen, batch_size)
        outs = gen.generate([self.messages(s) for s in todo], batch_size=batch_size, temperature=temperature, seed=seed)
        recs = {}
        for s, o in zip(todo, outs):
            pred, ok = parse_output(o["text"])
            recs[s["id"]] = {"raw_output": o["text"], "parse_ok": ok, "pred": {l: pred[l] for l in LABELS},
                             "evidence": pred["evidence"], "abstain": pred["abstain"],
                             "abstain_reason": "model_abstain" if pred["abstain"] else None,
                             "prompt_tokens": o["prompt_tokens"], "completion_tokens": o["completion_tokens"],
                             "llm_ms": o["latency_ms"], "n_llm_calls": 1}
        # ---- verifier + bounded repair (B4/B5) ----
        if self.cfg["verifier"]:
            need = []
            for s in todo:
                r = recs[s["id"]]
                r["pred_before_verify"] = dict(r["pred"])
                r["evidence_before_verify"] = list(r["evidence"])
                r["unsupported_before"] = len([l for l in LABELS if r["pred"][l]]) if not [
                    e for e in r["evidence"] if span_in_text(e, s["text"])] else 0
                if verify(r, s["text"], s["vote"]):
                    need.append(s)
            if need:
                msgs = [build_messages("P_repair@1.0", s["text"], labels=", ".join(
                    l for l in LABELS if recs[s["id"]]["pred"][l])) for s in need]
                rep = gen.generate(msgs, batch_size=batch_size, max_new_tokens=40)
                for s, o in zip(need, rep):
                    r = recs[s["id"]]
                    r["n_llm_calls"] += 1
                    r["prompt_tokens"] += o["prompt_tokens"]; r["completion_tokens"] += o["completion_tokens"]
                    r["llm_ms"] += o["latency_ms"]
                    m = re.search(r"\{.*\}", o["text"], re.S)
                    spans = []
                    if m:
                        try:
                            spans = json.loads(m.group(0)).get("evidence", [])
                        except Exception:
                            spans = re.findall(r'"([^"]+)"', m.group(0))[1:]
                    r["evidence"] = [str(x) for x in spans if span_in_text(x, s["text"])][:5]
                    if not r["evidence"]:
                        for l in LABELS:
                            if r["pred"][l]:
                                hits = lexicon_hits(s["text"], l)
                                if hits:
                                    r["evidence"] += hits[:2]
                                elif s["vote"][l] < 0.5:
                                    r["pred"][l] = 0      # klaim tak didukung -> dibuang
                        r["evidence"] = sorted(set(r["evidence"]))
                        verify(r, s["text"], s["vote"])
            for s in todo:
                r = recs[s["id"]]
                n_dis = sum(abs(r["pred"][l] - s["vote"][l]) >= 0.7 for l in LABELS)
                if not r["abstain"] and s["suff"] is not None and s["suff"] < 0.3 and n_dis >= 2:
                    r["abstain"], r["abstain_reason"] = True, "conflicting_evidence"
        # ---- rakit record ----
        out = []
        for s in states:
            base = {"id": s["id"], **self.meta(), "model_id": "Qwen/Qwen2.5-0.5B-Instruct", "temperature": temperature,
                    "seed": seed, "as_of": self.as_of, "guideline_versions": s["guide_versions"],
                    "retrieved_ids": s["retrieved"], "retries": s["retries"], "sufficiency": s["suff"],
                    "retr_ms": s["retr_ms"]}
            if s["early_abstain"]:
                base.update({"pred": {l: 0 for l in LABELS}, "evidence": [], "abstain": True,
                             "abstain_reason": s["early_abstain"], "raw_output": None, "parse_ok": True,
                             "prompt_tokens": 0, "completion_tokens": 0, "llm_ms": 0.0, "n_llm_calls": 0,
                             "scores": {l: 0.0 for l in LABELS}})
            else:
                r = recs[s["id"]]
                base.update(r)
                if self.cfg["planner"]:
                    base["scores"] = {l: round(0.5 * r["pred"][l] + 0.5 * s["vote"][l], 4) for l in LABELS}
                else:
                    base["scores"] = {l: float(r["pred"][l]) for l in LABELS}
            for k in ("tool_log", "tool_stats", "temporal_check"):
                if k in s:
                    base[k] = s[k]
            base["latency_ms"] = base["retr_ms"] + base["llm_ms"] + s.get("agent_ms", 0.0)
            base["prompt_tokens"] += s.get("agent_prompt_tokens", 0)
            base["completion_tokens"] += s.get("agent_completion_tokens", 0)
            base["n_llm_calls"] += s.get("agent_calls", 0)
            out.append(base)
        return out

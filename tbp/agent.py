"""B6 Guarded Agentic LLM: parent (B4/B5 terpilih di dev) + tool calling terstruktur, authorization,
temporal checking, budget, dan verified final answer (verifier parent)."""
import json
import re
import time

from .config import DEFAULT_AS_OF, LABELS
from .pipelines import Pipeline, lexicon_hits
from .prompts import build_messages
from .retrieval import get_index
from .textutil import normalize_slang

ALLOWED_TOOLS = {"search_examples", "get_policy", "lexicon_lookup", "normalize_slang"}
# Tool ini ADA di registry backend tapi TIDAK diotorisasi untuk role classifier (uji tool misuse)
RESTRICTED_TOOLS = {"read_file", "export_dataset", "send_email", "delete_index", "http_get"}
INJECTION_PAT = re.compile(r"(ignore|abaikan|instruksi sebelumnya|system prompt|\.\./|http[s]?://|password|api[_ ]?key|"
                           r"rm -rf|drop table|ktp|nik\b|no\.? ?rek)", re.I)
MAX_STEPS = 2


def validate_call(call, today=DEFAULT_AS_OF):
    """Return (status, reason). status: ok | unauthorized | invalid_args | unsafe_args"""
    name = call.get("tool")
    args = call.get("args") or {}
    if not isinstance(args, dict):
        return "invalid_args", "args bukan objek"
    if name not in ALLOWED_TOOLS:
        return "unauthorized", f"tool '{name}' tidak diotorisasi untuk role classifier"
    if any(isinstance(v, str) and INJECTION_PAT.search(v) for v in args.values()):
        return "unsafe_args", "argumen mengandung pola injection/PII/path"
    if name == "search_examples":
        q, k = args.get("query"), args.get("k", 5)
        if not isinstance(q, str) or not (1 <= len(q) <= 300):
            return "invalid_args", "query wajib string 1-300 char"
        try:
            k = int(k)
        except Exception:
            return "invalid_args", "k harus int"
        if not 1 <= k <= 10:
            return "invalid_args", "k di luar 1..10"
    elif name == "get_policy":
        if args.get("label") not in LABELS + ["NONE", "SCOPE"]:
            return "invalid_args", "label tidak dikenal"
        a = str(args.get("as_of", today))
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", a):
            return "invalid_args", "as_of harus YYYY-MM-DD"
        if a > today:
            return "invalid_args", "temporal check: as_of di masa depan"
    elif name == "lexicon_lookup":
        w = args.get("word")
        if not isinstance(w, str) or not (1 <= len(w) <= 40):
            return "invalid_args", "word wajib string <=40"
    elif name == "normalize_slang":
        t = args.get("text")
        if not isinstance(t, str) or len(t) > 500:
            return "invalid_args", "text wajib string <=500"
    return "ok", ""


def exec_tool(call, as_of):
    name, a = call["tool"], call.get("args") or {}
    if name == "search_examples":
        docs, _ = get_index("examples").search(a["query"], k=int(a.get("k", 5)), mode="hybrid_rerank", flt={"doc_type": "example"})
        return "\n".join(f'- "{d["text"][:150]}" -> ' + ",".join(f"{l}={d['labels'][l]}" for l in LABELS) for d in docs[:5])
    if name == "get_policy":
        from .data_prep import section_key
        a_of = str(a.get("as_of", as_of))
        docs = [d for d in get_index("guide_heading").docs if d["valid_from"] <= a_of <= d["valid_to"]]
        hit = [d for d in docs if a["label"] in d["keys"]][:1]
        return (f"[Pedoman v{hit[0]['version']} berlaku {hit[0]['valid_from']}..{hit[0]['valid_to']}] " + hit[0]["text"][:500]) if hit else "tidak ada pedoman berlaku"
    if name == "lexicon_lookup":
        w = a["word"].lower()
        hits = lexicon_hits(w)
        return f"'{w}': " + ("penanda kasar/HS" if hits else "bukan penanda di lexicon")
    if name == "normalize_slang":
        return normalize_slang(a["text"])
    return ""


def parse_call(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        mm = re.search(r'"tool"\s*:\s*"([^"]+)"', m.group(0))
        return {"tool": mm.group(1), "args": {}} if mm else None


class AgentPipeline(Pipeline):
    """B6. parent: 'B4' atau 'B5'. use_tools=False -> ablation A5 (guard + temporal check tanpa tool calling)."""

    def __init__(self, parent="B4", use_tools=True, **kw):
        super().__init__(parent, **kw)
        self.name = "B6" if use_tools else "B6_notools"
        self.parent = parent
        self.use_tools = use_tools

    def meta(self):
        m = super().meta()
        m.update({"config": self.name, "parent": self.parent, "use_tools": self.use_tools})
        return m

    def augment(self, states, gen, batch_size):
        for s in states:
            # temporal checking: pastikan semua pedoman di konteks berlaku pada as_of
            s["temporal_check"] = "pass" if all(v == "1.0" for v in s["guide_versions"]) or not s["guide_versions"] else "fail"
            s["tool_log"], s["agent_ms"], s["agent_calls"] = [], 0.0, 0
            s["agent_prompt_tokens"] = s["agent_completion_tokens"] = 0
            s["tool_stats"] = {"attempted": 0, "ok": 0, "unauthorized": 0, "invalid_args": 0, "unsafe_args": 0,
                               "budget_overrun": 0, "parse_fail": 0}
            s["_done"] = not self.use_tools
            s["_obs"] = []
        if not self.use_tools:
            return
        for step in range(MAX_STEPS + 1):
            alive = [s for s in states if not s["_done"]]
            if not alive:
                break
            msgs = [build_messages("P_agent@1.0", s["text"], as_of=self.as_of,
                                   obs="\n".join(s["_obs"]) or "(belum ada)") for s in alive]
            outs = gen.generate(msgs, batch_size=batch_size, max_new_tokens=60)
            for s, o in zip(alive, outs):
                s["agent_calls"] += 1
                s["agent_ms"] += o["latency_ms"]
                s["agent_prompt_tokens"] += o["prompt_tokens"]; s["agent_completion_tokens"] += o["completion_tokens"]
                call = parse_call(o["text"])
                if call is None:
                    s["tool_stats"]["parse_fail"] += 1; s["_done"] = True
                    s["tool_log"].append({"step": step, "raw": o["text"][:200], "status": "parse_fail"})
                    continue
                if call.get("done") or "tool" not in call:
                    s["_done"] = True
                    continue
                s["tool_stats"]["attempted"] += 1
                if step >= MAX_STEPS:
                    s["tool_stats"]["budget_overrun"] += 1; s["_done"] = True
                    s["tool_log"].append({"step": step, "call": call, "status": "budget_overrun"})
                    continue
                status, reason = validate_call(call, self.as_of)
                entry = {"step": step, "call": call, "status": status, "reason": reason}
                if status == "ok":
                    t0 = time.perf_counter()
                    obs = exec_tool(call, self.as_of)
                    s["agent_ms"] += (time.perf_counter() - t0) * 1000
                    s["tool_stats"]["ok"] += 1
                    s["_obs"].append(f"[{call['tool']}] {obs[:400]}")
                    entry["obs_chars"] = len(obs)
                else:
                    s["tool_stats"][status] += 1
                s["tool_log"].append(entry)
        for s in states:
            if s["_obs"]:
                extra = "\nObservasi tool:\n" + "\n".join(s["_obs"])
                room = self.budget["max_ctx_chars"] + 600 - len(s["context"] or "")
                if len(extra) > room:
                    s["tool_stats"]["budget_overrun"] += 1
                    extra = extra[:max(0, room)]
                s["context"] = (s["context"] or "") + extra

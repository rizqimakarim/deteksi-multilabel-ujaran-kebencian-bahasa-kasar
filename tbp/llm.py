"""Wrapper generator (Qwen2.5-0.5B-Instruct) dengan batching, LoRA adapter opsional, log token & latensi."""
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GEN_MODEL, LABELS, MAX_NEW_TOKENS

_CACHE = {}


def bf16_ok():
    """bf16 native hanya di GPU compute capability >= 8 (Ampere+). T4 (7.5) -> False."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def infer_dtype(dev):
    """bf16 di Ampere+ (mis. RTX 2050/30xx/A100), fp16 di T4/V100, fp32 di CPU. Override: env TBP_DTYPE=bf16|fp16|fp32."""
    import os
    forced = os.environ.get("TBP_DTYPE")
    if forced:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[forced]
    if dev != "cuda":
        return torch.float32
    return torch.bfloat16 if bf16_ok() else torch.float16


class Generator:
    def __init__(self, adapter_path=None):
        self.model_id = GEN_MODEL
        self.adapter = str(adapter_path) if adapter_path else None
        self.tok = AutoTokenizer.from_pretrained(GEN_MODEL)
        self.tok.padding_side = "left"
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = infer_dtype(dev)
        self.dtype = str(dtype).replace("torch.", "")
        model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, dtype=dtype).to(dev)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model = model.merge_and_unload()
        model.eval()
        self.model = model
        self.device = dev

    def chat_prompt(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, list_of_messages, batch_size=8, temperature=0.0, seed=None, max_new_tokens=MAX_NEW_TOKENS):
        """Return list of dict(text, prompt_tokens, completion_tokens, latency_ms) (latency = porsi waktu batch)."""
        prompts = [self.chat_prompt(m) for m in list_of_messages]
        out = [None] * len(prompts)
        # urutkan berdasarkan panjang untuk efisiensi padding
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            if seed is not None:
                torch.manual_seed(seed + b)
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=0.9)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            gen = self.model.generate(**enc, **kw)
            if self.device == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            new = gen[:, enc["input_ids"].shape[1]:]
            for j, i in enumerate(idx):
                ids = new[j]
                n_comp = int((ids != self.tok.eos_token_id).sum())
                out[i] = {"text": self.tok.decode(ids, skip_special_tokens=True),
                          "prompt_tokens": int(enc["attention_mask"][j].sum()),
                          "completion_tokens": n_comp, "latency_ms": dt / len(idx)}
        return out


def get_generator(adapter_path=None):
    key = str(adapter_path)
    if key not in _CACHE:
        _CACHE.clear()  # hanya satu model di VRAM 4GB
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        _CACHE[key] = Generator(adapter_path)
    return _CACHE[key]


def parse_output(text):
    """Parse output contract. Return (dict, ok). Gagal parse -> semua 0, ok=False."""
    res = {l: 0 for l in LABELS}
    res.update({"evidence": [], "abstain": False})
    m = re.search(r"\{.*\}", text, re.S)
    ok = False
    if m:
        s = m.group(0)
        try:
            d = json.loads(s)
            ok = True
        except Exception:
            d = {}
            for l in LABELS + ["abstain"]:
                mm = re.search(r'"%s"\s*:\s*(true|false|1|0)' % l, s)
                if mm:
                    d[l] = mm.group(1)
                    ok = True
            mm = re.search(r'"evidence"\s*:\s*\[(.*?)\]', s, re.S)
            if mm:
                d["evidence"] = re.findall(r'"([^"]*)"', mm.group(1))
        for l in LABELS:
            v = d.get(l, 0)
            res[l] = 1 if str(v).lower() in ("1", "true", "yes") else 0
        ev = d.get("evidence", [])
        if isinstance(ev, str):
            ev = [ev]
        res["evidence"] = [str(e) for e in ev if isinstance(e, (str, int, float))][:5]
        res["abstain"] = str(d.get("abstain", False)).lower() in ("1", "true", "yes")
    return res, ok

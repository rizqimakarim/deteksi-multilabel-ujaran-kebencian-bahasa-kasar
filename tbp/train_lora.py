"""LoRA fine-tuning untuk B5 (base generator sama: Qwen2.5-0.5B-Instruct).
Format input = prompt B4 (P2_rag + few-shot + konteks retrieval hybrid, self-excluded); target = output contract
(label gold TRAIN + silver evidence dari lexicon train).  Jalankan: python -m tbp.train_lora
"""
import json
import math
import os
import random
import time

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ADAPTERS, BUDGET_SMALL, DEFAULT_AS_OF, GEN_MODEL, LABELS, PROC, SEED
from .pipelines import build_context, few_shot_pairs, silver_evidence
from .prompts import build_messages, example_answer
from .retrieval import get_index

N_TRAIN = 1600
N_VAL = 80
LORA_CFG = dict(r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
LR, EPOCHS, GRAD_ACC, MAX_LEN = 2e-4, 1, 8, 1280


def make_examples(rows, exclude_self=True):
    ex_idx, g_idx = get_index("examples"), get_index("guide_heading")
    fs = few_shot_pairs()
    out = []
    for r in rows:
        ex, _ = ex_idx.search(r["text"], k=BUDGET_SMALL["k_examples"], mode="hybrid_rerank", flt={"doc_type": "example"},
                              exclude_ids={r["id"]} if exclude_self else ())
        gs, _ = g_idx.search(r["text"], k=BUDGET_SMALL["k_guides"], mode="hybrid_rerank", flt={"as_of": DEFAULT_AS_OF})
        ctx, _ = build_context(ex, gs, BUDGET_SMALL)
        msgs = build_messages("P2_rag@1.0", r["text"], context=ctx, few_shot=fs)
        out.append((msgs, example_answer(r, silver_evidence(r["text"], r))))
    return out


def encode(tok, msgs, target):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + t_ids)[-MAX_LEN:]
    labels = ([-100] * len(p_ids) + t_ids)[-MAX_LEN:]
    return torch.tensor([ids]), torch.tensor([labels])


@torch.no_grad()
def eval_loss(model, tok, data):
    model.eval()
    tot = 0
    for msgs, tgt in data:
        ids, lab = encode(tok, msgs, tgt)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            tot += model(input_ids=ids.cuda(), labels=lab.cuda()).loss.float().item()
    model.train()
    return tot / len(data)


# Presisi: bf16 murni di GPU Ampere+ (RTX 2050); di T4/V100 (tanpa bf16) -> bobot fp32 + autocast fp16 + GradScaler.
AMP = False


def main():
    random.seed(SEED); torch.manual_seed(SEED)
    tr = pd.read_json(PROC / "train.jsonl", lines=True)
    fs_texts = {t for t, _ in few_shot_pairs()}
    tr = tr[~tr.text.isin(fs_texts)].sample(frac=1, random_state=SEED)
    train_rows = tr.iloc[:N_TRAIN].to_dict("records")
    val_rows = tr.iloc[N_TRAIN:N_TRAIN + N_VAL].to_dict("records")       # held-out dari TRAIN (bukan test)
    dev = pd.read_json(PROC / "dev.jsonl", lines=True)
    dev_rows = dev[dev.in_scope].iloc[:60].to_dict("records")
    max_steps = int(os.environ.get("TBP_MAX_STEPS", "0"))      # >0 hanya untuk smoke test (tidak dipakai di run resmi)
    if max_steps:
        train_rows, val_rows, dev_rows = train_rows[:max_steps * GRAD_ACC], val_rows[:4], dev_rows[:4]
    t0 = time.time()
    train_data, val_data, dev_data = make_examples(train_rows), make_examples(val_rows), make_examples(dev_rows, False)
    prep_s = time.time() - t0

    global AMP
    from .llm import bf16_ok
    AMP = (not bf16_ok()) or os.environ.get("TBP_DTYPE") == "fp16"   # TBP_DTYPE=fp16 memaksa jalur T4 (uji lokal)
    precision = "fp32 weights + fp16 autocast (AMP)" if AMP else "bf16"
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, dtype=torch.float32 if AMP else torch.bfloat16).cuda()
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA_CFG))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    n_steps = math.ceil(len(train_data) * EPOCHS / GRAD_ACC)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / n_steps))
    torch.cuda.reset_peak_memory_stats()
    log = {"config": {"base_model": GEN_MODEL, "method": f"LoRA ({precision}, bukan QLoRA)", **LORA_CFG, "lr": LR, "epochs": EPOCHS,
                      "grad_accum": GRAD_ACC, "max_len": MAX_LEN, "n_train": len(train_data), "n_val": len(val_data),
                      "seed": SEED, "trainable_params": trainable, "total_params": total,
                      "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
           "steps": [], "eval": []}
    log["eval"].append({"step": 0, "val_loss": eval_loss(model, tok, val_data), "dev_loss": eval_loss(model, tok, dev_data)})
    print(log["eval"][-1], flush=True)
    model.train()
    step, run_loss, t_train = 0, 0.0, time.time()
    order = list(range(len(train_data))) * EPOCHS
    random.shuffle(order)
    for i, j in enumerate(order):
        ids, lab = encode(tok, *train_data[j])
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            loss = model(input_ids=ids.cuda(), labels=lab.cuda()).loss.float() / GRAD_ACC
        scaler.scale(loss).backward()
        run_loss += loss.item()
        if (i + 1) % GRAD_ACC == 0 or i == len(order) - 1:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            step += 1
            log["steps"].append({"step": step, "train_loss": run_loss, "lr": sched.get_last_lr()[0]})
            run_loss = 0.0
            if step % 10 == 0:
                print(log["steps"][-1], flush=True)
            if step % 50 == 0 or step == n_steps:
                log["eval"].append({"step": step, "val_loss": eval_loss(model, tok, val_data),
                                    "dev_loss": eval_loss(model, tok, dev_data)})
                print(log["eval"][-1], flush=True)
    name = "lora_smoketest" if max_steps else "lora_b5"
    out = ADAPTERS / name
    model.save_pretrained(out)
    size = sum(os.path.getsize(out / f) for f in os.listdir(out))
    log["resources"] = {"train_seconds": round(time.time() - t_train, 1), "data_prep_seconds": round(prep_s, 1),
                        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "adapter_bytes": size}
    json.dump(log, open(ADAPTERS / f"{name}_training_log.json", "w"), indent=1)
    print(log["resources"])


if __name__ == "__main__":
    main()

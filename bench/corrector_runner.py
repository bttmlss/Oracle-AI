#!/usr/bin/env python3
"""Corrector v1 measurement run (2026-11-14, Adrian's go).

Same 20 State/Ledger problems, same per-problem seeds as replication
seed 1 (20261022 + 1000 + idx), temp 0.7, max_tokens 768 — so the
number compares directly against repl_base_s1 (27/100 across seeds,
seed-1 baseline) and repl_hook_s1 (the margin controller, 16/20 on
seed 1). No tracker, no answer key, no logit surgery, no rollback:
the only augmentation is the corrector nudging the next input
embedding when the state is flat. Ledger: corrector_s1.jsonl,
resume-safe at problem granularity. Question: does a model that is
informed through its own coordinates self-correct in its own text?
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corrector import Corrector
from hooked_model import HookedModel
from queue_local import BASE_PROMPT, append_jsonl
from repl_runner import done_ids, score

BASE = os.path.dirname(os.path.abspath(__file__))
SEED0 = 20261022


def main():
    problems = json.load(open(os.path.join(BASE, "state_problems.json")))
    model = HookedModel(cache_dir=os.path.join(BASE, "hf_cache"))
    print("[corrector s1] model loaded", flush=True)
    emb = model.model.get_input_embeddings().weight.detach()
    corrector = Corrector(emb)
    led = os.path.join(BASE, "corrector_s1.jsonl")
    done = done_ids(led)
    for idx, p in enumerate(problems):
        if p["id"] in done:
            continue
        torch.manual_seed(SEED0 + 1000 + idx)
        prompt = BASE_PROMPT.format(problem=p["text"])
        r = model.generate(prompt, max_tokens=768, temperature=0.7,
                           corrector=corrector)
        nudges = [iv for iv in r["interventions"]
                  if iv["type"] == "corrector_nudge"]
        row = {"seed": 1, "problem_id": p["id"], "arm": "corrector",
               "response_text": r["text"],
               "final_correct": score(r["text"], p),
               "n_tokens": r["n_sampled"], "seconds": r["seconds"],
               "n_nudges": len(nudges), "nudges": nudges}
        append_jsonl(row, led)
        print(f"[corrector s1] {p['id']} ok={row['final_correct']} "
              f"nudges={len(nudges)} tok={row['n_tokens']} "
              f"{row['seconds']}s", flush=True)
    rows = [json.loads(l) for l in open(led)]
    print(f"[corrector s1] TOTAL "
          f"{sum(r['final_correct'] for r in rows)}/20  "
          f"nudges={sum(r['n_nudges'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()

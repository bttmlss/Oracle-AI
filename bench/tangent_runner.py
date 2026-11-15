#!/usr/bin/env python3
"""Tangent-model measurement run, v2.

Same 20 problems, same per-problem seeds (20261022 + 1000 + idx), temp 0.7,
max_tokens 768 as the corrector / margin-hook / baseline seed-1 runs, so the
score compares directly. v2 adds named ARMS so the shared-space question can
be answered with data instead of argued:

  legacy        pre-norm h, templated-prompt target, unit step 0.5
                (exactly the v1 design in the README)
  postnorm      final-RMSNorm'd h (the vector that makes the logits)
  postnorm_user postnorm + target built from the user problem text only
                (no system block / template tokens)

Usage (from bench/):
  python tangent_runner.py --diagnose            # ~1 min: norms + cosines
  python tangent_runner.py --arm legacy
  python tangent_runner.py --arm postnorm
  python tangent_runner.py --arm postnorm_user
  python tangent_runner.py --summary             # compare all ledgers

Each arm writes tangent_<arm>_s1.jsonl (resume-safe per problem).
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from bench_common import append_jsonl, done_ids, get_base_prompt, score  # noqa: E402

SEED0 = 20261022
ARMS = {
    "legacy":        dict(h_space="pre_norm",  target="templated"),
    "postnorm":      dict(h_space="post_norm", target="templated"),
    "postnorm_user": dict(h_space="post_norm", target="user_only"),
}
REFERENCE = {"baseline": "5/20", "corrector": "9/20", "margin-hook": "16/20"}


def load_problems():
    return json.load(open(os.path.join(BASE, "state_problems.json")))


def build_tangent(model, prompt, target):
    from tangent import Tangent
    emb = model.model.get_input_embeddings().weight.detach()
    if target == "templated":
        ids = model._encode_prompt(prompt)
    else:
        ids = model.tok(prompt, add_special_tokens=False,
                        return_tensors="pt").input_ids
    return Tangent(emb, ids)


def run_arm(arm, args):
    import torch
    from hooked_model import HookedModel
    cfg = ARMS[arm]
    base_prompt, src = get_base_prompt(args.allow_fallback_prompt)
    print(f"[tangent {arm}] prompt source: {src}", flush=True)
    problems = load_problems()
    model = HookedModel(cache_dir=os.path.join(BASE, "hf_cache"))
    led = os.path.join(BASE, f"tangent_{arm}_s1.jsonl")
    done = done_ids(led)
    for idx, p in enumerate(problems):
        if args.limit and idx >= args.limit:
            break
        if p["id"] in done:
            continue
        torch.manual_seed(SEED0 + 1000 + idx)
        prompt = base_prompt.format(problem=p["text"])
        tangent = build_tangent(model, prompt, cfg["target"])
        r = model.generate(prompt, max_tokens=768, temperature=0.7,
                           corrector=tangent, h_space=cfg["h_space"])
        nudges = [iv for iv in r["interventions"]
                  if iv["type"] == tangent.label]
        row = {"seed": 1, "problem_id": p["id"], "arm": f"tangent:{arm}",
               "prompt_source": src,
               "n_target_tokens": tangent.n_target_tokens,
               "response_text": r["text"],
               "final_correct": score(r["text"], p),
               "n_tokens": r["n_sampled"], "seconds": r["seconds"],
               "n_nudges": len(nudges), "nudges": nudges}
        append_jsonl(row, led)
        print(f"[tangent {arm}] {p['id']} ok={row['final_correct']} "
              f"nudges={len(nudges)} tok={row['n_tokens']} "
              f"{row['seconds']}s", flush=True)
    summarize([led])


def diagnose(args):
    """Cheap pre-flight: one problem, 40 tokens, each arm. Prints the numbers
    that decide whether 'steer toward the prompt' is what a run would do."""
    import torch
    from hooked_model import HookedModel
    base_prompt, _ = get_base_prompt(args.allow_fallback_prompt)
    p = load_problems()[0]
    model = HookedModel(cache_dir=os.path.join(BASE, "hf_cache"))
    emb = model.model.get_input_embeddings().weight.detach().float()
    rown = emb.norm(dim=1)
    print(f"embedding rows: mean norm {rown.mean():.3f}, "
          f"median {rown.median():.3f}, max {rown.max():.3f}")
    prompt = base_prompt.format(problem=p["text"])
    for arm, cfg in ARMS.items():
        torch.manual_seed(SEED0 + 1000)
        t = build_tangent(model, prompt, cfg["target"])
        r = model.generate(prompt, max_tokens=40, temperature=0.7,
                           corrector=t, h_space=cfg["h_space"])
        nz = [iv for iv in r["interventions"] if iv["type"] == t.label]
        if not nz:
            print(f"{arm}: no nudges")
            continue

        def avg(k):
            return st.mean(iv[k] for iv in nz)
        print(f"{arm:14s} |T|={t.target_norm:.3f} |h|={avg('h_norm'):.2f} "
              f"|err|={avg('error_norm'):.2f} "
              f"cos(h,T)={avg('target_cos'):+.3f} "
              f"cos(err,-h)={avg('cos_err_neg_h'):+.3f}")
    print("\nRead it: cos(err,-h) near +1 means T - h is basically -h, so "
          "the nudge pushes AGAINST the current state and the prompt target "
          "barely matters. |h| >> |T| is the cause.")


def summarize(paths):
    ref = {}
    cpath = os.path.join(BASE, "corrector_s1_ec2.jsonl")
    if os.path.exists(cpath):
        for l in open(cpath):
            r = json.loads(l)
            ref[r["problem_id"]] = r["final_correct"]
    print("\nreference (seed 1):", REFERENCE)
    for path in paths:
        rows = [json.loads(l) for l in open(path)]
        if not rows:
            continue
        ok = sum(r["final_correct"] for r in rows)
        line = (f"{os.path.basename(path)}: {ok}/{len(rows)}  "
                f"nudges={sum(r['n_nudges'] for r in rows)}")
        both = [(r["final_correct"], ref[r["problem_id"]]) for r in rows
                if r["problem_id"] in ref]
        if both:
            fixed = sum(a and not b for a, b in both)
            broke = sum(b and not a for a, b in both)
            line += f"  vs corrector: +{fixed} fixed / -{broke} broken"
        print(line)
        ev = [iv for r in rows for iv in r["nudges"] if "error_norm" in iv]
        if ev:
            half = len(ev) // 2
            a = st.mean(i["error_norm"] for i in ev[:half])
            b = st.mean(i["error_norm"] for i in ev[half:])
            c = st.mean(i["cos_err_neg_h"] for i in ev)
            print(f"  error_norm first half {a:.2f} -> second half {b:.2f}; "
                  f"mean cos(err,-h) {c:+.3f}")
    print("n=20, one seed: differences of 1-3 problems are inside noise.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-fallback-prompt", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summarize(sorted(glob.glob(os.path.join(BASE, "tangent_*_s1.jsonl"))))
    elif args.diagnose:
        diagnose(args)
    elif args.arm:
        run_arm(args.arm, args)
    else:
        ap.error("pick --arm, --diagnose or --summary")


if __name__ == "__main__":
    main()

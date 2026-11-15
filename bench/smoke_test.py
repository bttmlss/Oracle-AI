#!/usr/bin/env python3
"""Smoke test for the Rung 2 bench — proves each wire works, mechanically.

WIRE 1 (tap):        per-token log has prob + entropy + top5 for every token.
WIRE 3a (abort):     a tracker stops the stream at a chosen token.
WIRE 3b (rollback):  a tracker rolls back to an earlier token, splices
                     correction text in, and generation continues from it.
WIRE 3c (logits):    a logit hook bans the model's greedy first choice;
                     the sampled stream must route around it.

Writes bench/smoke_results.json. Run with bench/venv/bin/python.
"""
import json
import os
import resource
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hooked_model import HookedModel, last_line_start

BASE = os.path.dirname(os.path.abspath(__file__))
results = {}


def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def main():
    import time
    t0 = time.time()
    m = HookedModel(cache_dir=os.path.join(BASE, "hf_cache"))
    results["load_seconds"] = round(time.time() - t0, 1)
    results["rss_mb_after_load"] = rss_mb()
    print(f"loaded in {results['load_seconds']}s, RSS {rss_mb()} MB",
          flush=True)

    # WIRE 1 — capture
    r = m.generate("What is 17 + 25? Answer with just the number.",
                   max_tokens=12, temperature=0.0)
    ok = all(e["prob"] is not None and e["entropy"] is not None
             and len(e["top5"]) == 5 for e in r["token_log"])
    results["wire1_capture"] = {
        "pass": bool(ok and r["n_sampled"] > 0),
        "n_sampled": r["n_sampled"], "text": r["text"],
        "seconds": r["seconds"],
        "tok_per_s": round(r["n_sampled"] / max(r["seconds"], 0.01), 2),
        "first_tokens": [(e["str"], round(e["prob"], 3),
                          round(e["entropy"], 2))
                         for e in r["token_log"][:8]]}
    print("WIRE1:", results["wire1_capture"]["pass"], r["text"][:60],
          flush=True)

    # WIRE 3a — abort at token 12
    def aborter(state):
        if state["i"] >= 11:
            return ("abort",)
        return None
    r2 = m.generate("Write a story about a lighthouse keeper.",
                    max_tokens=80, temperature=0.0, on_token=aborter)
    results["wire3a_abort"] = {
        "pass": r2["stopped_reason"] == "tracker_abort"
                and r2["n_sampled"] == 12,
        "stopped_reason": r2["stopped_reason"],
        "n_sampled": r2["n_sampled"]}
    print("WIRE3a:", results["wire3a_abort"], flush=True)

    # WIRE 3b — rollback + splice, snapped to a line boundary (Argon trap 3)
    fired = {"done": False}

    def roller(state):
        if state["i"] >= 13 and not fired["done"]:
            fired["done"] = True
            keep = last_line_start(m.tok, state["gen_ids"])
            return ("rollback", keep,
                    "[Augmentation note: the keeper's name is Mara. "
                    "Continue the story using that name.] ")
        return None
    r3 = m.generate("Write a story about a lighthouse keeper.",
                    max_tokens=70, temperature=0.0, on_token=roller)
    results["wire3b_rollback"] = {
        "pass": len(r3["interventions"]) == 1
                and "Mara" in r3["text"]
                and any(e["str"] == "[ROLLBACK]" for e in r3["token_log"]),
        "interventions": r3["interventions"],
        "text_tail": r3["text"][-200:]}
    print("WIRE3b:", results["wire3b_rollback"]["pass"], flush=True)

    # WIRE 3c — ban the greedy first token
    base = m.generate("The capital of France is", max_tokens=4,
                      temperature=0.0)
    banned_id = base["token_log"][0]["id"]
    banned_str = base["token_log"][0]["str"]

    def banner(logits, state):
        logits[0, banned_id] = -1e9
        return logits
    hooked = m.generate("The capital of France is", max_tokens=4,
                        temperature=0.0, logit_hook=banner)
    results["wire3c_logit_ban"] = {
        "pass": hooked["token_log"][0]["id"] != banned_id
                and all(e["id"] != banned_id for e in hooked["token_log"]),
        "banned": banned_str, "baseline_text": base["text"],
        "hooked_text": hooked["text"]}
    print("WIRE3c:", results["wire3c_logit_ban"], flush=True)

    # WIRE 3d — rollback state equality (Argon trap 2 check): greedy,
    # roll back 20 -> 10 with an EMPTY splice; the regenerated tokens
    # 10..29 must be byte-identical to the uninterrupted pass.
    ra = m.generate("Write a story about a lighthouse keeper.",
                    max_tokens=30, temperature=0.0)
    fired2 = {"done": False}

    def roller2(state):
        if state["i"] >= 19 and not fired2["done"]:
            fired2["done"] = True
            return ("rollback", 10, "")
        return None
    rb = m.generate("Write a story about a lighthouse keeper.",
                    max_tokens=40, temperature=0.0, on_token=roller2)
    a_ids = [e["id"] for e in ra["token_log"] if e["prob"] is not None]
    b_post = []
    seen_marker = False
    for e in rb["token_log"]:
        if e["str"] == "[ROLLBACK]":
            seen_marker = True
            continue
        if seen_marker and e["prob"] is not None:
            b_post.append(e["id"])
    if ra["n_sampled"] == 30 and seen_marker:
        eq_pass = b_post[:20] == a_ids[10:30]
    else:
        eq_pass = None  # inconclusive (early EOS / marker missing)
    results["wire3d_rollback_state_equality"] = {
        "pass": eq_pass,
        "a_ids_10_29": a_ids[10:30], "b_post_rollback_ids": b_post[:20]}
    print("WIRE3d:", results["wire3d_rollback_state_equality"]["pass"],
          flush=True)

    results["rss_mb_final"] = rss_mb()
    results["all_pass"] = all([
        results["wire1_capture"]["pass"], results["wire3a_abort"]["pass"],
        results["wire3b_rollback"]["pass"],
        results["wire3c_logit_ban"]["pass"],
        results["wire3d_rollback_state_equality"]["pass"] is True])
    with open(os.path.join(BASE, "smoke_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("ALL PASS:", results["all_pass"], "| RSS", rss_mb(), "MB",
          flush=True)


if __name__ == "__main__":
    main()

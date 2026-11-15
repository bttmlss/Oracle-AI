#!/usr/bin/env python3
"""Generate the 20 State/Ledger problems (BENCH-PLAN task family).

Each problem: initial state (A, B, C) + 7 single-variable operations
(+, -, *2, /2 with exact divisibility; values kept in [0, 999]).
Ground truth = the exact state after each operation, computed here.
Output: state_problems.json with per-step prescribed (var, op, delta)
and the true state sequence. Deterministic (fixed seed).
"""
import json
import os
import random

BASE = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(20261021)

STORIES = [
    ("A warehouse register tracks three bins", "units"),
    ("A vault ledger tracks three accounts", "credits"),
    ("A greenhouse log tracks three water tanks", "liters"),
    ("A workshop board tracks three parts bins", "parts"),
    ("A harbor office tracks three fuel depots", "barrels"),
]


def gen_problem(idx):
    story, unit = STORIES[idx % len(STORIES)]
    state = {v: rng.randrange(20, 400) for v in "ABC"}
    init = dict(state)
    ops = []
    for _ in range(7):
        var = rng.choice("ABC")
        cur = state[var]
        choices = ["add", "sub"]
        if cur * 2 <= 999:
            choices.append("mul2")
        if cur % 2 == 0 and cur > 0:
            choices.append("div2")
        kind = rng.choice(choices)
        if kind == "add":
            d = rng.randrange(5, 120)
            state[var] = cur + d
            ops.append((var, "+", d, f"Add {d} to {var}"))
        elif kind == "sub":
            d = rng.randrange(5, cur + 1) if cur >= 5 else cur
            state[var] = cur - d
            ops.append((var, "-", d, f"Subtract {d} from {var}"))
        elif kind == "mul2":
            state[var] = cur * 2
            ops.append((var, "*", 2, f"Double {var}"))
        else:
            state[var] = cur // 2
            ops.append((var, "/", 2, f"Halve {var}"))
    # recompute the true state sequence step by step
    seq = []
    s = dict(init)
    for var, op, d, _ in ops:
        if op == "+":
            s[var] += d
        elif op == "-":
            s[var] -= d
        elif op == "*":
            s[var] *= d
        else:
            s[var] //= d
        seq.append(dict(s))
    lines = [f"{story}: A, B, and C. Initially A={init['A']}, "
             f"B={init['B']}, C={init['C']} ({unit}). Apply these "
             f"operations in order:"]
    for i, (_, _, _, text) in enumerate(ops, 1):
        lines.append(f"{i}. {text}.")
    return {"id": f"S{idx+1:02d}", "text": "\n".join(lines),
            "initial": init,
            "ops": [{"var": v, "op": o, "delta": d} for v, o, d, _ in ops],
            "states": seq, "final": seq[-1]}


def main():
    problems = [gen_problem(i) for i in range(20)]
    with open(os.path.join(BASE, "state_problems.json"), "w") as f:
        json.dump(problems, f, indent=1)
    print(f"wrote {len(problems)} problems")
    print(problems[0]["text"])
    print("S01 true states:", problems[0]["states"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stand-ins for the helpers tangent_runner.py imports from modules that were
not in the zip (queue_local.py, repl_runner.py).

- append_jsonl / done_ids: trivial, resume-safe ledger helpers.
- score: RECONSTRUCTED. Parses the last "State: A=.., B=.., C=.." line and
  compares it to the problem's true final state. Validated against
  corrector_s1_ec2.jsonl: agrees on 20/20 rows (9 correct / 11 incorrect).
- BASE_PROMPT: NOT reconstructable from the zip. The real one lives in
  queue_local.py on the bench box. get_base_prompt() prefers the real module
  and only falls back to a guess when told to (allow_fallback=True), because
  a different prompt silently breaks comparability with the 5/20, 9/20 and
  16/20 reference runs.
"""
import json
import os
import re

_STATE_RE = re.compile(
    r"State:\s*A\s*=\s*(-?\d+)\s*,\s*B\s*=\s*(-?\d+)\s*,\s*C\s*=\s*(-?\d+)")

# Guess only: inferred from the corrector ledger's output format
# ("Step k: ... | State: A=.., B=.., C=.."). Not the original wording.
FALLBACK_BASE_PROMPT = (
    "{problem}\n\n"
    "After each operation, write one line in exactly this format:\n"
    "Step k: <var> = <computation> | State: A=<a>, B=<b>, C=<c>")


def append_jsonl(row, path):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def done_ids(path):
    """Problem ids already in the ledger (resume-safe at problem level)."""
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["problem_id"])
    return out


def score(text, problem):
    """True iff the LAST 'State:' line equals the problem's true final state."""
    m = _STATE_RE.findall(text)
    if not m:
        return False
    a, b, c = map(int, m[-1])
    f = problem["final"]
    return (a, b, c) == (f["A"], f["B"], f["C"])


def get_base_prompt(allow_fallback=False):
    try:
        from queue_local import BASE_PROMPT
        return BASE_PROMPT, "queue_local"
    except ImportError:
        if not allow_fallback:
            raise SystemExit(
                "queue_local.BASE_PROMPT not found. Copy queue_local.py next "
                "to this file (needed for comparability with the reference "
                "runs), or pass --allow-fallback-prompt to use a guessed "
                "prompt (results then NOT comparable).")
        return FALLBACK_BASE_PROMPT, "FALLBACK-GUESS"

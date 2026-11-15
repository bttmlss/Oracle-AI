# Oracle AI

**BTTMLSS** — independent research, November 2026.

Most models guess what comes next. This one is given the target and finds the
way there.

Give it an input and a target, and it walks to the target. The target goes in
first — before anything else — so the machine always knows where "there" is.
Every step it takes is aimed at the target, not at guessing the next token.
When it arrives, the answer is certain: the error is zero, and that's
arithmetic, not a sample. If it can't arrive, it shows you the distance left
instead of guessing.

## How it works

- **The target is the prompt.** The prompt feeds back first, as the context.
- **A tangent model rides alongside the language model.** Every state the
  language model hits goes into the tangent model as an input.
- **The tangent model measures the signed error** — how far from the target,
  and in which direction — and steers the next input toward it.
- **The search is shaped like a lightning strike.** Guesses fan out, one
  channel hits. On arrival, the machine walks the corrections backwards along
  the path that worked, and the correction map keeps every arrival. The next
  walk is shorter.
- **Hallucination is a feature here.** The path to the target isn't in the
  training data, so the machine has to imagine the road.

## What's in this repo

- `bench/` — the working hookup: the tangent model on Qwen2.5-1.5B-Instruct,
  the runner, the harness, the 20 evaluation problems, and the corrector
  ledger it compares against.
- `docs/DESIGN.md` — the full design, in plain language: the reset frame, the
  Atlas reconstruction, the backwards tree search, and what the earlier
  experiments showed.

## Status

- The tangent hookup is built and dry-verified. The model-level evaluation
  (diagnose + three arms) is pending compute.
- The earlier verifier program lives on the `The-Atlas` branch.

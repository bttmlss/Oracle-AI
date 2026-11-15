# Tangent-model hookup — Atlas in tandem with Qwen2.5-1.5B-Instruct (v2)

## v2 changes (this revision)
- **Finding: `h` is pre-final-norm, so the README's shared-space claim did not hold.**
  The harness hook captures the last decoder layer's output, before Qwen's final
  RMSNorm. Logits are `lm_head(norm(h))`, not `h·Eᵀ`, so v1's `h` carries a
  different (and very likely much larger) scale than the embedding rows that
  make up the target. If |h| >> |T|, then `T − h ≈ −h` and the nudge pushes
  against the current state with the prompt barely mattering. I could not run the
  model here, so this is an architectural inference; `--diagnose` measures it
  (|h|, |T|, cos(err, −h)) in about a minute.
- **Harness dropped the tangent diagnostics.** `error_norm` / `target_cos` were
  computed but never written to the ledger, so a run could not show whether the
  error shrank. Now logged, plus `h_norm`, `target_norm`, `cos_err_neg_h`.
- **Ledger label** is now per-controller (`tangent_nudge`), not hardcoded.
- **Hook leak fixed**: the hidden-state hook is removed in a `finally`.
- **Device-safe** embedding add; `h_space` option (`pre_norm` = v1, `post_norm`).
- **Missing imports**: `tangent_runner.py` needed `queue_local` / `repl_runner`,
  which were not in the zip. `bench_common.py` supplies `score` (reconstructed,
  20/20 agreement with the corrector ledger), `done_ids`, `append_jsonl`.
  `BASE_PROMPT` cannot be reconstructed: copy your `queue_local.py` alongside,
  or pass `--allow-fallback-prompt` (then NOT comparable to 5/9/16 of 20).
- **Arms** so the open question is testable: `legacy` (v1 exactly), `postnorm`,
  `postnorm_user` (target from the problem text only, no template tokens).
- `tangent.py` options (defaults = v1): `mode="proportional"`, `beta` (derivative
  term, tangent-paper sign convention), `exclude_ids`. Proportional only bites
  once |error| falls below `step_size`, so calibrate `gain` from `--diagnose`.
- Originals kept as `*.orig.py`. `test_tangent.py` holds model-free checks (needs torch).

Not verified here: no torch/model in the authoring sandbox. Everything was
compile-checked; `bench_common`, the summary path and the scorer were run.

## Run order
```
cd bench
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./venv/bin/python test_tangent.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./venv/bin/python tangent_runner.py --diagnose
./venv/bin/python tangent_runner.py --arm legacy      # v1 as specced
./venv/bin/python tangent_runner.py --arm postnorm
./venv/bin/python tangent_runner.py --arm postnorm_user
./venv/bin/python tangent_runner.py --summary
```

---
# Original README (v1)

# Tangent-model hookup — Atlas in tandem with Qwen2.5-1.5B-Instruct

## What this is
The LLM runs unchanged. A second model — the tangent model — rides alongside it:
every state the LLM hits goes into the tangent model as an input, and the tangent
model steers the next input toward the target. The target is the prompt itself.

## Files
- `tangent.py` — the tangent model (the new code)
- `tangent_runner.py` — evaluation runner, mirrors `corrector_runner.py`
- `hooked_model.py` — the generation harness the tangent model plugs into
- `corrector.py`, `corrector_runner.py` — the earlier prototype this mirrors
- `state_problems.json` — the 20 evaluation problems (+ `gen_state_problems.py`)
- `corrector_s1_ec2.jsonl` — the corrector run's ledger (for score comparison)
- `download_model.py`, `smoke_test.py` — model fetch + smoke test

## How it works
- Target **T = mean of the prompt tokens' embedding rows** (the prompt exactly as
  the model sees it, chat-templated ids).
- Each generation step, the tangent model takes the final hidden state **h** as the
  current state, computes the signed error **T − h**, and returns
  `delta = 0.5 · unit(error)`, applied to the NEXT input embedding through the
  harness's existing `corrector=` channel.
- Fire rule: every step while the error is nonzero (continuous steering, unlike
  the corrector's flatness gate).
- No answer key, no logit surgery, no weight changes, no reward.

## Design calls to review
1. **h as the current state.** The harness's `decide(h, logits)` offers the final
   hidden state but not the current token id, so `h` is used as "the vector the
   LLM just produced." Qwen2.5 ties embeddings, so h and the embedding rows
   share one space (logits = h·Eᵀ). The literal last token's embedding row
   would have required changing the harness.
2. **Ledger label.** The harness hardcodes the intervention type string
   `"corrector_nudge"`, so tangent nudges carry that label in the ledger.
3. **Every-step firing.** The error is essentially never exactly zero, so this
   steers continuously. Whether that is right (vs. a gate) is an open question.

## Verification status
Model-free dry checks passed: target equals the manual mean of the prompt rows;
delta norm exactly 0.5 and exactly parallel to the signed error (cos = 1.0);
all harness-required info keys present; zero-error path returns no delta;
`step_size` dial respected; `score()` agrees with all 9 correct / 11 incorrect
rows of the corrector ledger; tokenizer loads from local cache offline.

The 2-token model generation run and the full 20-problem run were NOT attempted:
the box has ~0.67 GB free and a running load needs ~4–4.5 GB.

## To run
```
cd bench
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./venv/bin/python tangent_runner.py
```
Resume-safe per problem, ~40–60 s/problem on EC2-class hardware. Compares directly
against baseline 5/20, corrector 9/20, margin-hook 16/20.

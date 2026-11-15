#!/usr/bin/env python3
"""HookedModel — the Rung 2 bench harness (2026-10-20).

An open-weights model running in OUR process, with the augmentation wired
into the generation loop itself:

- WIRE 1 (tap): every sampled token is captured as it is produced — its
  text, the probability the model assigned it, the entropy of the full
  distribution at that step, and the top-5 alternatives.
- WIRE 2 (calculation): a tracker callback inspects the stream per token
  and maintains whatever running state the task defines (the trajectory).
- WIRE 3 (intervention): the tracker (or a logit hook) can act back
  mid-generation: ABORT the stream; ROLL BACK to an earlier token, SPLICE
  correction text into the context, and continue from the splice; or
  MODIFY THE LOGITS before sampling (suppress/boost specific tokens).
  The model's weights are never touched.

Rollback is a logical rollback: the generated token list is truncated to
the chosen index, the splice tokens are appended, and the KV cache is
rebuilt with one forward pass over the rebuilt sequence, whose final
logits seed the next sample. Exact at the sequence level.
"""
import time

import torch
import torch.nn.functional as F


def last_line_start(tok, gen_ids):
    """Token index just after the most recent newline in the generated
    text (0 if none). Rollback targets should snap to this boundary:
    truncating mid-line leaves subword scars — a tokenization the model
    would never produce itself (Argon audit, 2026-10-20)."""
    for k in range(len(gen_ids), 0, -1):
        if tok.decode(gen_ids[:k]).endswith("\n"):
            return k
    return 0


class HookedModel:
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct",
                 cache_dir=None, dtype=torch.bfloat16, warmup=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id,
                                                 cache_dir=cache_dir)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, cache_dir=cache_dir)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, cache_dir=cache_dir)
        self.model.eval()
        self.eos = self.tok.eos_token_id
        if warmup:
            # Burn the cold start (lazy kernel setup + weight page-in cost
            # ~133 s on a first 40-token call) on a 2-token dummy so real
            # generations time cleanly. Argon audit, 2026-10-21.
            try:
                self.generate("Warm up.", max_tokens=2, temperature=0.0)
            except Exception:
                pass

    def _encode_prompt(self, prompt, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) \
            + [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        return self.tok(text, return_tensors="pt").input_ids

    def generate(self, *args, **kwargs):
        """Public entry. Guarantees the hidden-state hook is removed even if
        generation raises (v1 leaked the hook on any exception, so a crashed
        problem left a stale hook on every later call)."""
        self._hook_handle = None
        try:
            return self._generate(*args, **kwargs)
        finally:
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None

    @torch.inference_mode()
    def _generate(self, prompt, max_tokens=200, temperature=0.0,
                  system=None, on_token=None, logit_hook=None,
                  max_rollbacks=3, corrector=None, h_space="pre_norm"):
        """Generate with the augmentation wired in.

        on_token(state) -> None | ('abort',) | ('rollback', keep_n, splice)
          state keys: i (sampled count), text (generated text so far),
          token_id, token_str, prob, entropy, log.
          keep_n = number of generated tokens to KEEP before the splice.
        logit_hook(logits, state) -> modified logits (pre-sampling).
        corrector: a Corrector (bench/corrector.py) or None. When set,
          each step's final hidden state + logits are offered to
          corrector.decide(); a returned delta is added to the NEXT
          input token's embedding, so it flows through the model's own
          layers and KV cache. No logits are touched, no answer key is
          used — the nudge rides on the model's own top-2.
        h_space: which hidden state decide() receives. "pre_norm" (v1
          behaviour) is the last decoder layer's output, BEFORE the final
          RMSNorm. "post_norm" applies the model's final norm first, giving
          the vector that is actually multiplied by the tied embedding matrix
          to make the logits (logits = lm_head(norm(h))). Only post_norm
          supports the "h lives in the embedding/logit space" reading.
        A corrector may define .label (str) to name its ledger entries and
        may return extra diagnostic keys in info; all scalar extras are
        written into each intervention record.
        """
        if h_space not in ("pre_norm", "post_norm"):
            raise ValueError(f"h_space must be pre_norm|post_norm: {h_space!r}")
        ids = self._encode_prompt(prompt, system)
        n_prompt = ids.shape[1]
        gen_ids = []
        log = []
        interventions = []
        rollbacks = 0
        sampled = 0
        stopped = "max_tokens"
        t0 = time.time()

        # Corrector wiring: capture the final hidden state at the last
        # position on every forward pass, via a hook on the last
        # decoder layer (cheap — one vector per pass, no full
        # hidden-states dump).
        hook_handle = None
        h_last = [None]
        pending_delta = [None]

        def _capture(module, inputs, output):
            # transformers 4.x layers return a tuple (hidden, ...);
            # 5.x layers return the hidden-states tensor directly.
            h = output[0] if isinstance(output, tuple) else output
            h_last[0] = h[:, -1, :].detach().clone()

        if corrector is not None:
            try:
                last_layer = self.model.model.layers[-1]
            except AttributeError:
                last_layer = self.model.get_decoder().layers[-1]
            hook_handle = last_layer.register_forward_hook(_capture)
            self._hook_handle = hook_handle
            if h_space == "post_norm":
                try:
                    final_norm = self.model.model.norm
                except AttributeError:
                    final_norm = self.model.get_decoder().norm

        out = self.model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()

        while sampled < max_tokens:
            state = {"i": sampled, "text": self.tok.decode(gen_ids),
                     "log": log, "gen_ids": list(gen_ids)}
            step_logits = logit_hook(logits.clone(), state) \
                if logit_hook is not None else logits
            if corrector is not None and h_last[0] is not None:
                h_in = h_last[0][0]
                if h_space == "post_norm":
                    h_in = final_norm(h_in.unsqueeze(0))[0]
                delta, info = corrector.decide(h_in, step_logits[0])
                if delta is not None:
                    pending_delta[0] = delta
                    rec = {
                        "type": getattr(corrector, "label",
                                        "corrector_nudge"),
                        "after_sampled_token": sampled,
                        "margin": info["margin"],
                        "entropy": info["entropy"],
                        "top1": self.tok.decode([info["top1"]]),
                        "runner_up": self.tok.decode([info["runner_up"]]),
                        "delta_norm": info["delta_norm"],
                    }
                    # Extra scalar diagnostics (error_norm, target_cos, ...)
                    # v1 dropped these, so the ledger could not show whether
                    # the error ever shrank.
                    for k, v in info.items():
                        if k not in rec and k not in (
                                "top1", "runner_up", "fired") \
                                and isinstance(v, (int, float)):
                            rec[k] = v
                    interventions.append(rec)
            probs = F.softmax(step_logits, dim=-1)
            entropy = float(-(probs * probs.clamp_min(1e-12).log())
                            .sum().item())
            top = torch.topk(probs, 5)
            if temperature and temperature > 0:
                dist = F.softmax(step_logits / temperature, dim=-1)
                nxt = int(torch.multinomial(dist, 1).item())
            else:
                nxt = int(torch.argmax(step_logits, dim=-1).item())
            entry = {"i": sampled, "id": nxt,
                     "str": self.tok.decode([nxt]),
                     "prob": float(probs[0, nxt].item()),
                     "entropy": entropy,
                     "top5": [(self.tok.decode([int(t)]), float(p))
                              for t, p in zip(top.indices[0],
                                              top.values[0])]}
            log.append(entry)
            gen_ids.append(nxt)
            sampled += 1
            state.update({"token_id": nxt, "token_str": entry["str"],
                          "prob": entry["prob"], "entropy": entropy,
                          "text": self.tok.decode(gen_ids)})

            if on_token is not None:
                action = on_token(state)
                if action and action[0] == "abort":
                    stopped = "tracker_abort"
                    break
                if action and action[0] == "rollback" and \
                        rollbacks < max_rollbacks:
                    _, keep_n, splice = action
                    rollbacks += 1
                    pending_delta[0] = None  # nudge dies with the token
                    interventions.append({
                        "type": "rollback",
                        "after_sampled_token": sampled,
                        "at_text": state["text"],
                        "keep_n": keep_n, "splice": splice,
                        "discarded": self.tok.decode(gen_ids[keep_n:]),
                    })
                    splice_ids = self.tok(
                        splice, add_special_tokens=False)["input_ids"]
                    gen_ids = gen_ids[:keep_n] + list(splice_ids)
                    full = torch.cat(
                        [ids, torch.tensor([gen_ids], dtype=torch.long)],
                        dim=1)
                    rebuilt = self.model(input_ids=full, use_cache=True)
                    past = rebuilt.past_key_values
                    logits = rebuilt.logits[:, -1, :].float()
                    log.append({"i": sampled, "str": "[ROLLBACK]",
                                "prob": None, "entropy": None, "top5": [],
                                "keep_n": keep_n, "splice": splice})
                    continue
            if nxt == self.eos:
                stopped = "eos"
                break
            if pending_delta[0] is not None:
                # Vector-channel nudge: the next token enters through an
                # embedding shifted toward the model's own runner-up, so
                # the correction flows through the model's own layers
                # and KV cache instead of editing its output.
                emb_mod = self.model.get_input_embeddings()
                dev = emb_mod.weight.device
                emb = emb_mod(torch.tensor([[nxt]], device=dev)) \
                    + pending_delta[0].to(device=dev,
                                          dtype=emb_mod.weight.dtype)
                pending_delta[0] = None
                step = self.model(inputs_embeds=emb,
                                  past_key_values=past, use_cache=True)
            else:
                step = self.model(input_ids=torch.tensor([[nxt]]),
                                  past_key_values=past, use_cache=True)
            past = step.past_key_values
            logits = step.logits[:, -1, :].float()

        if hook_handle is not None:
            hook_handle.remove()
        return {"text": self.tok.decode(gen_ids), "token_log": log,
                "interventions": interventions, "stopped_reason": stopped,
                "n_prompt_tokens": int(n_prompt), "n_sampled": sampled,
                "seconds": round(time.time() - t0, 2)}

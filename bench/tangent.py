#!/usr/bin/env python3
"""Tangent model, v2 (upgrade of tangent.py).

Defaults reproduce v1 EXACTLY (unit-norm step of 0.5 toward the prompt mean,
fired every step). Everything new is opt-in, so v1 numbers stay comparable.

What changed and why
--------------------
1. Diagnostics. info now carries error_norm, h_norm, target_norm, target_cos,
   and cos_err_neg_h = cos(error, -h). If cos_err_neg_h is near 1, the
   "steering toward the prompt" is really "pushing against h" (h is so much
   bigger than the target that T - h ~ -h). That is the check on the shared-
   space assumption; see README "Findings". The harness (patched) now writes
   all of these into the ledger, so the controller's own trajectory is
   inspectable (v1 computed error_norm/target_cos but the harness dropped them).
2. mode="proportional": delta = gain * error, clipped to norm <= step_size.
   Unit mode is bang-bang: it never backs off near the target and cannot
   converge, only chatter. Proportional shrinks as the error shrinks.
3. beta: optional derivative term on the error, u = gain*e + beta*(e - e_prev),
   same sign convention as the tangent paper (e = target - actual). The
   previous error is per-instance state, so build one Tangent per problem.
4. exclude_ids: drop token ids (chat-template / special tokens) from the
   target mean. The v1 target averages the whole templated prompt, including
   the system block and <|im_start|>/<|im_end|> markers, which are identical
   for every problem and pull T toward a problem-independent point.
5. Input validation (shape/device/dtype) so a wrong h fails loudly.
"""
import torch
import torch.nn.functional as F


class Tangent:
    label = "tangent_nudge"  # read by the patched harness for the ledger type

    def __init__(self, embed_matrix, prompt_ids, step_size=0.5,
                 mode="unit", gain=1.0, beta=0.0, exclude_ids=None):
        if mode not in ("unit", "proportional"):
            raise ValueError(f"mode must be 'unit' or 'proportional', got {mode!r}")
        self.emb = embed_matrix
        self.step_size = float(step_size)
        self.mode = mode
        self.gain = float(gain)
        self.beta = float(beta)
        self._prev_error = None

        ids = torch.as_tensor(prompt_ids).reshape(-1).to(embed_matrix.device)
        if exclude_ids:
            drop = torch.as_tensor(sorted(set(exclude_ids)),
                                   device=ids.device)
            keep = ~torch.isin(ids, drop)
            if int(keep.sum()) == 0:
                raise ValueError("exclude_ids removed every prompt token")
            ids = ids[keep]
        with torch.no_grad():
            self.target = self.emb[ids].float().mean(dim=0)  # (hidden,)
        self.target_norm = float(torch.norm(self.target))
        self.n_target_tokens = int(ids.numel())

    def decide(self, h, logits):
        """h: (hidden,) current state. logits: (vocab,) pre-sampling logits.
        Returns (delta_or_None, info); info always has the keys the harness
        requires (margin, entropy, top1, runner_up) plus diagnostics."""
        if h.dim() != 1 or h.shape[0] != self.target.shape[0]:
            raise ValueError(f"h must be ({self.target.shape[0]},), "
                             f"got {tuple(h.shape)}")
        logits = logits.float()
        h = h.float().to(self.target.device)
        top2 = torch.topk(logits, 2)
        margin = float(top2.values[0] - top2.values[1])
        probs = F.softmax(logits, dim=-1)
        ent = float(-(probs * probs.clamp_min(1e-12).log()).sum())
        error = self.target - h
        err_norm = float(torch.norm(error))
        info = {
            "margin": margin, "entropy": ent,
            "top1": int(top2.indices[0]), "runner_up": int(top2.indices[1]),
            "fired": False,
            "error_norm": err_norm,
            "h_norm": float(torch.norm(h)),
            "target_norm": self.target_norm,
            "target_cos": float(F.cosine_similarity(
                h.unsqueeze(0), self.target.unsqueeze(0)).item()),
            "cos_err_neg_h": float(F.cosine_similarity(
                error.unsqueeze(0), (-h).unsqueeze(0)).item()),
        }
        prev, self._prev_error = self._prev_error, error
        if err_norm < 1e-8:
            return None, info

        if self.mode == "unit":
            u = error / err_norm
            if self.beta and prev is not None:
                u = u + self.beta * (error - prev) / err_norm
            delta = self.step_size * u
        else:  # proportional, clipped
            u = self.gain * error
            if self.beta and prev is not None:
                u = u + self.beta * (error - prev)
            n = float(torch.norm(u))
            delta = u if n <= self.step_size else u * (self.step_size / n)

        dn = float(torch.norm(delta))
        if dn < 1e-12:
            return None, info
        info["fired"] = True
        info["delta_norm"] = dn
        return delta, info

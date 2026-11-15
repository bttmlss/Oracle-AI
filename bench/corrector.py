#!/usr/bin/env python3
"""The corrector, v1 (2026-11-14, Adrian's go).

Job: read the model's own pre-collapse state per token generation and,
when the state is flat (undecided), nudge the state toward the model's
own strongest alternative — so the model continues from a corrected
state it can itself see and use.

What it is NOT:
- Not the old controller. No answer key, no ground truth, no logit
  surgery, no rollback. The tracker in stepc_margin_runner knew the
  right answer in advance and edited the output (grabbed f). The
  corrector uses only the model's own distribution (grabbed x).
- Not a trained model. v1 is geometric: the nudge direction is the
  difference of the tied embedding rows for the model's top-2 tokens.
  Qwen2.5 ties input and output embeddings, so that direction lives in
  the model's own coordinate space. A learned corrector (a small net
  trained on (hidden state -> correction) pairs) is v2, after we have
  a dataset of nudged trajectories to learn from.

Mechanism (per generation step):
  1. Read h (final hidden state at the last position) and the logits.
  2. Flatness test from the hidden-geometry probe (seed 1 hooked,
     27 erred vs 114 correct): erred margin 2.15, correct 8.34;
     erred entropy 0.49, correct 0.03. Fire when margin < 3.0 AND
     entropy > 0.2 — the band where the alternatives are still alive.
  3. delta = alpha * unit(emb[runner_up] - emb[top1]). Applied by the
     harness to the NEXT input token's embedding, so it flows through
     the model's own layers and into the KV cache. The sampled token
     itself is untouched; what changes is the state the model reads
     that token through. The test is whether the model then
     self-corrects in its own text.

Dials (my choices, not truths): margin_thresh 3.0, entropy_thresh 0.2,
alpha 0.5. All overridable; the runner logs every firing so the dials
can be judged from the data.
"""
import torch
import torch.nn.functional as F


class Corrector:
    def __init__(self, embed_matrix, margin_thresh=3.0,
                 entropy_thresh=0.2, alpha=0.5):
        """embed_matrix: the model's tied embedding weight, shape
        (vocab, hidden). Rows are token coordinates in the model's own
        space; the nudge direction is a difference of two rows."""
        self.emb = embed_matrix
        self.margin_thresh = margin_thresh
        self.entropy_thresh = entropy_thresh
        self.alpha = alpha

    def decide(self, h, logits):
        """h: hidden vector (hidden,) — unused by the v1 rule except
        for the record; the firing decision rides on the logits'
        flatness, the 1-D shadow the probe proved separable.
        logits: (vocab,) pre-sampling logits for the next token.
        Returns (delta_or_None, info)."""
        logits = logits.float()
        top2 = torch.topk(logits, 2)
        # Margin in LOGIT units, as the probe measured it (erred 2.15,
        # correct 8.34). A probability margin would max out at 1.0 and
        # make the threshold vacuous.
        margin = float(top2.values[0] - top2.values[1])
        probs = F.softmax(logits, dim=-1)
        ent = float(-(probs * probs.clamp_min(1e-12).log()).sum())
        top1 = int(top2.indices[0])
        runner = int(top2.indices[1])
        info = {"margin": margin, "entropy": ent, "top1": top1,
                "runner_up": runner, "fired": False}
        if margin >= self.margin_thresh or ent <= self.entropy_thresh:
            return None, info
        direction = self.emb[runner].float() - self.emb[top1].float()
        n = float(torch.norm(direction))
        if n < 1e-8:
            return None, info
        delta = self.alpha * direction / n
        info["fired"] = True
        info["delta_norm"] = float(torch.norm(delta))
        return delta, info

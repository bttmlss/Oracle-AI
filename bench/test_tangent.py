#!/usr/bin/env python3
"""Model-free checks for tangent.py v2. Needs torch only (no model).
Run: python test_tangent.py   (NOT run in the authoring sandbox: no torch there.)"""
import torch
from tangent import Tangent

torch.manual_seed(0)
V, H = 50, 16
E = torch.randn(V, H) * 0.05
ids = torch.tensor([3, 3, 7, 9, 11])
logits = torch.randn(V)


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


# v1 behaviour preserved
t = Tangent(E, ids)
check("target == manual mean", torch.allclose(t.target, E[ids].mean(0), atol=1e-6))
h = torch.randn(H)
d, info = t.decide(h, logits)
check("unit delta norm 0.5", abs(float(d.norm()) - 0.5) < 1e-5)
check("delta parallel to error",
      float(torch.nn.functional.cosine_similarity(d, t.target - h, dim=0)) > 0.9999)
d0, i0 = t.decide(t.target.clone(), logits)
check("zero error -> no delta", d0 is None and not i0["fired"])
for k in ("margin", "entropy", "top1", "runner_up", "delta_norm", "error_norm",
          "h_norm", "target_norm", "target_cos", "cos_err_neg_h"):
    check(f"info has {k}", k in info)

# the scale-mismatch signature: huge h => error ~ -h
big = torch.randn(H) * 200
_, ib = Tangent(E, ids).decide(big, logits)
check("|h|>>|T| gives cos(err,-h)~1", ib["cos_err_neg_h"] > 0.999)

# new options
t2 = Tangent(E, ids, exclude_ids=[3])
check("exclude_ids drops those rows",
      torch.allclose(t2.target, E[torch.tensor([7, 9, 11])].mean(0), atol=1e-6)
      and t2.n_target_tokens == 3)
tp = Tangent(E, ids, mode="proportional", gain=0.1, step_size=0.5)
dp, _ = tp.decide(h, logits)
check("proportional never exceeds cap", float(dp.norm()) <= 0.5 + 1e-6)
tb = Tangent(E, ids, beta=0.5)
d1, _ = tb.decide(h, logits)
d2, _ = tb.decide(h * 0.5, logits)
check("beta uses previous error", not torch.allclose(d2 / d2.norm(),
      (tb.target - h * 0.5) / (tb.target - h * 0.5).norm(), atol=1e-4))
try:
    t.decide(torch.randn(H + 1), logits)
    check("bad h shape raises", False)
except ValueError:
    check("bad h shape raises", True)
print("ALL PASS")

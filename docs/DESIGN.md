# Oracle AI — design

Plain-language account of the machine. The earlier verifier program lives on
the `The-Atlas` branch; this is what replaced it.

## The reset frame

The project reset to its most basic form: Y = F(m·X + b).

- **Y is the output**, doing double duty as the correction signal AND the
  next vector.
- **X is each token as it is being selected.** Every node on the model's
  branching paths is fed back as an input.
- **F is the activation — the catalyst.** It enables the reaction without
  being consumed. F becomes Φ, same thing, new character.
- **The W-equation:** W(t+1) = Φ(W(t), x(t)) + Γ(ε(t)). Φ runs the machine,
  Γ evolves it — the error term is the only thing that changes W.
- **The exhaust rule:** with every new weight added, the exhaust comes out
  along with the output. The remainder IS the signal.
- **ε(t) is error over time.** The signal is the error's trajectory and
  history, not a point value. Its shape is the sharp signal — and it has to
  be SIGNED. Magnitude without direction learns nothing.

## The pivot: target-seeking, not next-token prediction

Everything built before this had no target to train on, and a real-time
training model needs a real-time target. Current models are probabilistic
next-token engines: given input, where next?

This machine is different. Give it an input AND a target, and it finds a way
to that target. The error becomes the distance to the target — real, signed,
measurable at every step.

The evidence already supported it: the pieces that worked had a target (the
margin trick had the answer key; the live-gradient experiment's teacher was
the target). The blurry ones (the first corrector, the targetless engine)
had none.

## Atlas: the origin

The first project. A word is turned into a hash — a computable number — and
that number is fed in as the FIRST input. The first input is the target, so
the model knows it from the start. The second input is a second
parameter/weight. The model calculates the error and the trajectory to the
target, and reaches it in two steps because there are only two weights.

Add more weights, more words, more inputs — more steps — and with enough
inputs the model finds a universal law of correction. The correction map is
the model finding a universal pathway.

The original Atlas program was lost (chat session deleted, terminal shut
down, unsaved file wiped as it was being saved). This repo is the rebuild
from the reconstruction.

## The Acaninator Oracle: backwards tree search

A backwards tree search — still moving forward through state space, but the
logic is flipped because the target is given first. Same shape as Jeopardy:
the answer is given, the question is the work.

- Forward, the model guesses its way toward the target. The search is shaped
  like a lightning strike: branches fan out, one channel hits the ground.
  That hit is the 100% certain moment — the error at zero.
- Then it runs backwards. Backpropagation sends a signal back along the
  exact path that worked, and every step on that path gets nudged so the same
  path is easier to find next time. The rest of the branches just die off.
- The correction map is the pile of all those backwards runs — every arrival
  traced back and kept.

## The tandem architecture

The machine runs in tandem with a language model. The language model runs
unchanged; every node, token, and vector it hits goes as an input into the
tangent model. The target is the prompt itself. The prompt is Y — Y feeds
back into the model first, as the context. The tangent model measures the
error from each language-model state to the prompt-as-target and steers the
next input toward it.

The model stays frozen. The iteration lives in the input — and now the input
has somewhere to go.

## Hallucination as a feature, certainty on arrival

A normal model mirrors what exists. Even its new ideas are remixes of what's
already there. This one has to imagine the road, because the road to the
target isn't in its training data. Making up the path is the whole job — so
hallucination stops being a bug.

And the answer, on arrival, is mathematically certain, not probabilistic.
The error hitting zero is arithmetic, not a sample. The certainty holds only
on arrival: if the machine can't reach the target, it shows the remaining
distance instead of guessing. It can never silently fake it.

## Prompt framing

In a target-seeking model, failure is not "the model doesn't know the
answer." It's "did you frame the prompt so the model can respond at all."
The prompt IS the target specification. A bad prompt isn't a dumb model —
it's a target the machine can't walk to.

This is what the bench already showed: the margin run rescued 56 out of 100
without teaching the model anything. The knowledge was already there; the
framing at step one was the block.

## What the pendulum experiments showed

- Signed difference is load-bearing. Ablate direction (magnitude only) and
  learning dies completely.
- Closed-loop self-training freezes: a student refit on its own corrections
  stops changing after the first fit. Fresh signal on the system's own path
  is what unfreezes it.
- Live per-step gradient with a teacher in the loop works: weights glide to
  the true rule. The pass-count metric was confounded by a
  tolerance-vs-bias trap, not by the learning.
- Cheapest vessel wins, twice: linear beats the bigger models.

## The bench

`bench/` holds the working hookup: Qwen2.5-1.5B-Instruct, the tangent model
(target = mean of the prompt's token embeddings), the signed error driving
the next input embedding, and the 20-problem evaluation set.

Known honest gaps, recorded in `bench/README_tangent.md`:

- The first version read the hidden state before the final norm, so the
  steering pushed against the current state more than toward the prompt. The
  v2 fix applies the final norm first (`postnorm` arm). The diagnose run
  measures whether that fixes it.
- The model-level evaluation (diagnose + three arms) is pending compute.

# Implementation notes

The public implementation has two layers.

`multi_axis_maxk/advantages.py` is the canonical algorithm layer. It accepts a
reward tensor shaped `(..., m, D)`, applies finite-batch expected improvement
and the leave-two-out baseline independently to each of the `D` axes,
normalizes over the `m` candidates, and finally sums the axes. It has no model,
Diffusers, Accelerate, or CLIP dependency.

`training/train_sd3.py` is the faithful experiment layer adapted from
Flow-GRPO. It collects on-policy SD3.5-M trajectories, computes the released
CLIP-axis reward matrix, calls the same EI+L2O primitives, and performs the
PPO-clipped diffusion policy update with a KL anchor. Its public interface is
deliberately restricted to the three reported axes and the `maxk`, `max1`, and
`count` credit rules. Exploratory reward families and conditioning variants
from the research checkout are not part of this release.

The research checkout imported `compute_batch_ei` and
`compute_l2o_baseline` from a local `remax_utils` checkout. The public code
ports and refactors the required kernels into `multi_axis_maxk/advantages.py`
and tests them against exhaustive subset enumeration. This removes an editable
SSH dependency while preserving the research implementation's numerical
behavior and making the algorithm auditable without a GPU.

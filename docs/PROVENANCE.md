# Code provenance

This public tree was extracted from the committed state
`8d68c76df4a4c72dfe94fcc265b60e31a67c7983` of the authors' research fork on
2026-08-12. Uncommitted files and experiment artifacts in that checkout were
deliberately excluded.

## Flow-GRPO-derived paths

The following paths retain or adapt the SD3 training mechanics from the
official MIT-licensed [Flow-GRPO](https://github.com/yifan123/flow_grpo)
codebase:

- `training/train_sd3.py`;
- `flow_grpo/diffusers_patch/`;
- the CLIP scoring and EMA helpers under `flow_grpo/`.

The corresponding upstream license is preserved at
`licenses/FLOW_GRPO_LICENSE`. The research fork began from Flow-GRPO's public
Git history and then accumulated the paper-specific multi-reward and max@K
changes. See `THIRD_PARTY_NOTICES.md` for citation and license details.

The CLIP quality scorer was additionally adapted from `RE-N-Y/imscore`, as
marked in its source header. The optional aesthetic scorer follows the pinned
Improved Aesthetic Predictor reference implementation. Neither project ships
model weights in this repository.

## Max@K estimator

The research environment resolved its estimator from an internal `remax_utils`
checkout at commit `8edf8ea2d337cbcedcf6374ceddd95a7eff788c0`. The EI and L2O
kernels in `multi_axis_maxk/advantages.py` are adapted from that implementation.
The public module renames and documents the estimator, adds input validation
and multi-axis composition, and removes the runtime dependency on a separate
checkout. Its tests compare EI and L2O against exhaustive without-replacement
subset enumeration. The complete internal source tree is not vendored.

## Paper-owned components

The multi-axis composition, fixed prompt-axis definitions, compact paper
trainer, training recipes, prompt datasets, evaluation aggregation,
documentation, and tests are the paper authors' release components. Generated
samples, checkpoints, tracking logs, exploratory reward families, cluster
launchers, credentials, and local model caches are not included.

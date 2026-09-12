# Third-party notices

This repository contains adapted code from the following projects.

The SD3 training code was adapted from the authors' Flow-GRPO research fork
at revision `8d68c76df4a4c72dfe94fcc265b60e31a67c7983`.
The EI and leave-two-out kernels in `multi_axis_maxk/advantages.py` were
adapted from the authors' `remax_utils` implementation at revision
`8edf8ea2d337cbcedcf6374ceddd95a7eff788c0`; no separate checkout is required.

## Flow-GRPO

- Project: **Flow-GRPO: Training Flow Matching Models via Online RL**
- Upstream: <https://github.com/yifan123/flow_grpo>
- License: MIT
- Copyright: 2025 Jie Liu

The SD3 sampling, log-probability, prompt-encoding, EMA, and PPO/GRPO training
paths under `flow_grpo/` and `training/` originate from or are adapted from
Flow-GRPO. The upstream MIT license is reproduced in
`licenses/FLOW_GRPO_LICENSE`.

Flow-GRPO itself acknowledges code from `ddpo-pytorch` and Hugging Face
`diffusers`. The retained patch files preserve their source headers. Diffusers
code is Apache-2.0 licensed, and DDPO is MIT licensed; copies are included as
`licenses/APACHE-2.0` and `licenses/DDPO_LICENSE`.

## Imscore

`flow_grpo/clip_scorer.py` retains a small adapted preprocessing and scoring
path from [RE-N-Y/imscore](https://github.com/RE-N-Y/imscore). The upstream
repository grants unrestricted use in its `LISCENSE` file. The adaptation was
inherited through the authors' Flow-GRPO research fork and is identified in the
source header. Its license text is reproduced in `licenses/IMSCORE_LICENSE`.

## Improved Aesthetic Predictor

The optional aesthetic evaluation follows Christoph Schuhmann's
[improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor).
Its checkpoint is downloaded on demand from the pinned upstream revision and
is not included in this repository. The adapted inference code is Apache-2.0
licensed; that license is reproduced in `licenses/APACHE-2.0`. Model weights
may have separate terms, which users should review before use.

## GenEval prompts

- Project: **GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment**
- Upstream: <https://github.com/djghosh13/geneval>
- License: MIT

`data/geneval_single_object/train.txt` and `test.txt` reproduce the 80
single-object prompts of GenEval as plain text. No GenEval code or evaluation
weights are included.

## Pick-a-Pic prompts

The color-coverage evaluation uses the Pick-a-Pic prompt list distributed with
Flow-GRPO (`dataset/pickscore/test.txt`). It is not redistributed here;
`scripts/fetch_pickapic_prompts.sh` downloads it from upstream and verifies its
checksum. Pick-a-Pic is described in Kirstain et al. (2023); consult the
dataset's own terms before further redistribution.

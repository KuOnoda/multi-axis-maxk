# Third-party notices

This repository contains adapted code from the following projects.

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

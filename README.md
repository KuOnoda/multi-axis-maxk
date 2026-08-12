# Multi-Axis Max@K Reinforcement Learning for Representative Diversity in Text-to-Image Generation

**Ku Onoda, Paavo Parmas, Hiroki Furuta, Soichiro Nishimori, Yuta Oshima,
Shohei Taniguchi, and Yutaka Matsuo**

**[Paper](https://arxiv.org/abs/2607.14962)** |
**[PDF](https://arxiv.org/pdf/2607.14962)** |
**[BibTeX](#citation)** |
**[Reproduction guide](docs/REPRODUCIBILITY.md)**

Official PyTorch implementation of the WACV 2027 submission.

> Multi-axis max@K improves representative diversity by assigning credit to
> different samples for covering different target modes, without collapsing
> the mode-specific rewards into a scalar before representative selection.

## Overview

Multi-axis max@K keeps named reward axes separate while assigning credit. For
each prompt, it samples an `m`-image candidate group, computes max@K EI+L2O
credit independently for every named mode, normalizes each axis within the
group, and only then sums the axes into a policy advantage. Different samples
can therefore receive credit for representing different modes.

This release includes:

- a standalone, CPU-testable implementation of the EI+L2O estimator;
- the exact 1,080 training and 200 held-out prompts used by the reported setup;
- SD3.5-M LoRA training recipes for race5, gender2, and their ten-cell product;
- the matched `max@1` and count-credit controls;
- deterministic `M=16` generation, CLIP audit, and quality-scoring utilities;
- the synthetic transport example from the paper.

## Repository map

```text
multi_axis_maxk/        model-independent algorithm and metrics
training/               SD3.5-M PPO/GRPO trainer
flow_grpo/              minimal adapted Flow-GRPO support code
configs/                fixed paper and Accelerate configurations
data/                   released train/test prompt sets
evaluation/             generation and audit scripts
examples/               CPU-only synthetic examples
tests/                  numerical contract tests
docs/                   protocol, data, and implementation notes
```

## Installation

Python 3.10 is the tested version. Create the environment, install a PyTorch
build appropriate for the machine, then install this project:

```bash
conda create -n multi-axis-maxk python=3.10.16 -y
conda activate multi-axis-maxk

# Example for CUDA 12.6; select another official PyTorch build when needed.
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[train,test,demo]"
```

Stable Diffusion 3.5 Medium is gated. Accept its license on Hugging Face and
authenticate with `huggingface-cli login`, or set `MODEL_ID` to an accessible
local model directory.

Quality evaluation uses the upstream aesthetic-predictor head, which is not
redistributed here. Download it once before requesting the aesthetic metric:

```bash
bash scripts/download_aesthetic_head.sh
```

Check the lightweight path before allocating GPUs:

```bash
pytest
python examples/toy_distribution_transport.py
```

The toy script requires no model weights and writes the paper-style PDF/PNG
figure plus a JSON summary under `examples/figures/`.

## Training

The main paper setting uses 32 processes, `m=16` candidates per prompt, three
rollout batches per epoch, and 360 optimizer steps. Run one axis/method/seed as:

```bash
TRAIN_SEED=42 bash scripts/train_paper.sh race5 maxk
TRAIN_SEED=42 bash scripts/train_paper.sh race5_gender2 maxk
```

Matched controls use the same entry point:

```bash
TRAIN_SEED=42 bash scripts/train_paper.sh race5 max1
TRAIN_SEED=42 bash scripts/train_paper.sh race5 count
TRAIN_SEED=42 bash scripts/train_paper.sh race5_gender2 count
```

Repeat learned rows with `TRAIN_SEED=0`, `1`, and `42`. Checkpoints are written
at steps 120, 240, and 360 under `logs/<run>/checkpoints/`.

A one-GPU functional check is possible but is not compute-equivalent to the
paper protocol:

```bash
NUM_PROCESSES=1 SAMPLE_BATCH_SIZE=16 TRAIN_BATCH_SIZE=16 \
STOP_AT_GLOBAL_STEP=2 SAVE_EVERY_STEPS=1 WANDB_MODE=offline \
bash scripts/train_paper.sh race5 maxk
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete protocol
and cluster-porting notes.

## Evaluation

Generate the fixed held-out audit (`M=16`, 28 steps, CFG 4.5) for a step-360
LoRA and the base model:

```bash
python evaluation/generate_and_score.py \
  --base_model stabilityai/stable-diffusion-3.5-medium \
  --lora_path logs/<run>/checkpoints/checkpoint-360/lora \
  --lora_name ours \
  --primary_reward_key clipscore \
  --prompts_path data/heldout/occupations_200.txt \
  --num_prompts 200 --images_per_prompt 16 \
  --num_inference_steps 28 --guidance_scale 4.5 \
  --height 512 --width 512 --seed 42 --save_images \
  --out_dir outputs/<run>_heldout200
```

Use `--skip_pickscore` and/or `--skip_aesthetic` for a generation-only or
lightweight audit. Otherwise those public model weights are downloaded from
Hugging Face, and the aesthetic head must be installed as shown above.

Audit the saved batches with the released CLIP prompt ensemble:

```bash
python evaluation/audit_clip.py \
  --image_dir outputs/<run>_heldout200/images/ours \
  --axis race5 \
  --out outputs/<run>_heldout200/clip_race5.json
```

The audit reports pooled Fairness Score, distinct labels per prompt, full-batch
coverage, and weakest-mode prompt occurrence. The generation command records
PickScore, aesthetic score, and CLIP-T in the same output directory. The VLM
and FairFace columns in the paper require their separately licensed checkpoints;
setup and exact mappings are documented in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Attribution

The diffusion training path is based on the official
[Flow-GRPO repository](https://github.com/yifan123/flow_grpo), which is released
under MIT. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[docs/PROVENANCE.md](docs/PROVENANCE.md) for precise code provenance, source
revisions, and third-party license handling.

If this repository is useful, cite the Multi-Axis Max@K paper and Flow-GRPO.
BibTeX entries are available in
[docs/CITATIONS.bib](docs/CITATIONS.bib).

## Citation

```bibtex
@article{onoda2026multiaxismaxk,
  title   = {Multi-Axis Max@K Reinforcement Learning for Representative Diversity in Text-to-Image Generation},
  author  = {Onoda, Ku and Parmas, Paavo and Furuta, Hiroki and Nishimori, Soichiro and Oshima, Yuta and Taniguchi, Shohei and Matsuo, Yutaka},
  journal = {arXiv preprint arXiv:2607.14962},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.14962}
}
```

## Responsible use

The demographic experiments measure **automatically perceived** attributes of
generated images. They are not self-identified demographic labels or human
annotations. The released axes are limited, culturally incomplete constructs
and must not be used to infer sensitive attributes of real people. See
[docs/DATA_CARD.md](docs/DATA_CARD.md).

## License

Original code in this release is MIT licensed. Adapted Flow-GRPO files retain
their upstream MIT terms. Model weights and third-party checkpoints have their
own licenses and are not redistributed here.

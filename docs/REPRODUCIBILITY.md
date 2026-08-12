# Reproducibility protocol

## Reported learned rows

| Item | Value |
|---|---|
| Base generator | Stable Diffusion 3.5 Medium |
| Adaptation | Transformer LoRA, rank 32, alpha 32, Gaussian init |
| LoRA targets | `to_q`, `to_k`, `to_v`, `to_out.0` |
| Training prompts | 1,080 neutral-person prompts |
| Candidate group | `m=16` per prompt |
| Denoising | 10 steps, guidance 1, 512×512 |
| Representative window | race5 `k=5`; gender2 `k=2`; product `k=10` |
| Axis weights | all ones |
| Optimizer | AdamW, lr `1e-4`, betas `(0.9, 0.999)` |
| Weight decay / eps | `1e-4` / `1e-8` |
| PPO clip / KL beta | `1e-5` / `0.05` |
| Gradient clip | 1.0 |
| Precision | fp16 |
| Rollout schedule | 3 batches per epoch, 1 inner epoch |
| Reporting checkpoint | step 360 |
| Seeds | 0, 1, 42 |

The reference jobs used 32 GPU processes, local rollout batch 8, and three
rollout batches per epoch. This creates 48 prompt groups per epoch and 360
updates over 120 epochs. Reducing GPU count while holding all local dimensions
fixed changes the number of prompt groups and is not a compute-matched run.

## Reward definitions

Race5 uses the five CLIP prompt-ensemble prototypes in
`flow_grpo/clip_prompt_reward.py`. Each prompt embedding is L2-normalized,
embeddings are averaged within a class and normalized again, and image/class
cosines are converted to probabilities with CLIP's learned logit scale and
temperature 1.0.

Gender2 uses its two prompt-ensemble prototypes. The ten-cell product reward is
the outer product of race5 and gender2 probabilities in race-major order.

## Held-out generation

- prompts: `data/heldout/occupations_200.txt`;
- 16 images per prompt;
- 28 inference steps;
- CFG 4.5;
- 512×512;
- deterministic seed `base_seed + prompt_index * 10000 + image_index`.

Always generate once and reuse exactly those image files for every probe and
quality metric.

PickScore and the CLIP backbone are resolved from local `models/` directories
when present and otherwise from their public Hugging Face model IDs. The
aesthetic linear head is intentionally not vendored; install the pinned
upstream file with `bash scripts/download_aesthetic_head.sh`, or set
`AESTHETIC_MODEL_PATH` to an existing copy. Record all resolved model revisions
and file checksums with the evaluation output.

## Independent probes

The reward-side CLIP audit is implemented in `evaluation/audit_clip.py`.
The reported independent audits additionally use:

- **Qwen2.5-VL-Instruct**: independent yes/no questions per mode, scored from
  next-token Yes/No logits at temperature 1.0. Scores are not renormalized
  across modes; the top label is their argmax.
- **FairFace ResNet-34**: official 7-race checkpoint after YOLOv8-face detection
  (confidence 0.30, IoU 0.45, size 640, largest face, 20% padding). East and
  Southeast Asian are merged; Middle Eastern mass is dropped and the retained
  race5 mass renormalized. Product scores use the outer product with FairFace
  binary-gender probabilities.

Those checkpoints are not redistributed because they have separate download
and usage terms. Record model revision/checksum, detector revision/checksum,
CUDA/PyTorch versions, and the image-manifest checksum in every audit run.

## Hardware portability

The original captured environment was Linux aarch64, CUDA 12.6, PyTorch 2.6.0,
Diffusers 0.33.1, Transformers 4.40.0, Accelerate 1.4.0, and PEFT 0.10.0.
Use the same Python package versions where possible, but install the appropriate
official PyTorch wheel for the target architecture. Avoid copying the original
environment's editable SSH dependencies; this repository replaces the only
required max@K dependency with `multi_axis_maxk/advantages.py`.

## Reproducibility tiers

1. `pytest` and the toy example: CPU only, minutes.
2. One-GPU two-step check: validates model loading, reward scoring, grouping,
   backpropagation, and checkpoint writing.
3. One reported training row: 32-process reference protocol.
4. Main tables: all learned methods × three seeds, held-out generation, three
   probes, and quality scoring on the same image manifests.

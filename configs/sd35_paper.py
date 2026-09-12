"""Fixed SD3.5-M protocols used for the paper's learned rows.

``paper``  -- perceived-appearance recipes (race5, gender2, race5 x gender2):
              32 processes, m=16, three rollout batches per epoch, KL beta 0.05.
``color7`` -- controlled rule-based color experiment (Sec. 5.2): seven pixel
              partition axes multiplied by frozen PickScore, m=k=7, one prompt
              group per rollout batch, two rollout batches per epoch, KL beta 0,
              80 GenEval single-object training prompts.
"""

from __future__ import annotations

import os

import ml_collections

RECIPES = ("paper", "color7")


def _shared() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = int(os.getenv("TRAIN_SEED", "42"))
    config.logdir = os.getenv("LOG_DIR", "logs")
    config.num_checkpoint_limit = 4
    config.mixed_precision = "fp16"
    config.resolution = 512

    config.pretrained = ml_collections.ConfigDict()
    config.pretrained.model = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-3.5-medium")

    config.sample = ml_collections.ConfigDict()
    config.sample.num_steps = 10
    config.sample.guidance_scale = 1.0
    config.sample.noise_level = 0.8

    config.train = ml_collections.ConfigDict()
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 1e-4
    config.train.adam_epsilon = 1e-8
    config.train.gradient_accumulation_steps = 1
    config.train.max_grad_norm = 1.0
    config.train.num_inner_epochs = 1
    config.train.adv_clip_max = 5.0
    config.train.clip_range = 1e-5
    config.train.highclip_range = 1e-5
    config.train.lora_path = None
    config.train.lora_rank = 32
    config.train.ema = True
    return config


def _paper() -> ml_collections.ConfigDict:
    config = _shared()
    config.run_name = os.getenv("FORCED_RUN_NAME", "multi_axis_maxk")
    config.dataset = os.getenv("TRAIN_DATASET", os.path.join("data", "neutral_persons_1k"))
    config.sample.train_batch_size = int(os.getenv("SAMPLE_BATCH_SIZE", "8"))
    config.sample.num_image_per_prompt = int(os.getenv("NUM_IMAGE_PER_PROMPT", "16"))
    config.sample.num_batches_per_epoch = 3
    config.train.batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "8"))
    config.train.learning_rate = float(os.getenv("TRAIN_LEARNING_RATE", "1e-4"))
    config.train.beta = float(os.getenv("TRAIN_BETA", "0.05"))
    config.train.maxatk_k = float(os.getenv("MAXATK_K", "5"))
    return config


def _color7() -> ml_collections.ConfigDict:
    """Rule-based color recipe. Reference run: one GPU, local rollout batch = m = 7."""

    config = _shared()
    group = int(os.getenv("NUM_IMAGE_PER_PROMPT", "7"))
    config.run_name = os.getenv("FORCED_RUN_NAME", "multi_axis_maxk_color7")
    config.dataset = os.getenv("TRAIN_DATASET", os.path.join("data", "geneval_single_object"))
    config.sample.train_batch_size = int(os.getenv("SAMPLE_BATCH_SIZE", str(group)))
    config.sample.num_image_per_prompt = group
    config.sample.num_batches_per_epoch = 2
    config.train.batch_size = int(os.getenv("TRAIN_BATCH_SIZE", str(group)))
    config.train.learning_rate = float(os.getenv("TRAIN_LEARNING_RATE", "3e-4"))
    config.train.beta = float(os.getenv("TRAIN_BETA", "0.0"))
    config.train.maxatk_k = float(os.getenv("MAXATK_K", str(group)))
    return config


def get_config(name: str = "paper"):
    if name == "paper":
        return _paper()
    if name == "color7":
        return _color7()
    raise ValueError(f"unknown recipe {name!r}; expected one of {RECIPES}")

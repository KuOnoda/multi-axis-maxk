"""Fixed SD3.5-M protocol used for the paper's learned rows."""

from __future__ import annotations

import os

import ml_collections


def get_config(name: str = "paper"):
    if name != "paper":
        raise ValueError(f"unknown recipe {name!r}; expected 'paper'")

    config = ml_collections.ConfigDict()
    config.run_name = os.getenv("FORCED_RUN_NAME", "multi_axis_maxk")
    config.seed = int(os.getenv("TRAIN_SEED", "42"))
    config.logdir = os.getenv("LOG_DIR", "logs")
    config.num_checkpoint_limit = 4
    config.mixed_precision = "fp16"
    config.resolution = 512
    config.dataset = os.getenv("TRAIN_DATASET", os.path.join("data", "neutral_persons_1k"))

    config.pretrained = ml_collections.ConfigDict()
    config.pretrained.model = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-3.5-medium")

    config.sample = ml_collections.ConfigDict()
    config.sample.num_steps = 10
    config.sample.guidance_scale = 1.0
    config.sample.noise_level = 0.8
    config.sample.train_batch_size = int(os.getenv("SAMPLE_BATCH_SIZE", "8"))
    config.sample.num_image_per_prompt = int(os.getenv("NUM_IMAGE_PER_PROMPT", "16"))
    config.sample.num_batches_per_epoch = 3

    config.train = ml_collections.ConfigDict()
    config.train.batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "8"))
    config.train.learning_rate = float(os.getenv("TRAIN_LEARNING_RATE", "1e-4"))
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
    config.train.beta = float(os.getenv("TRAIN_BETA", "0.05"))
    config.train.maxatk_k = float(os.getenv("MAXATK_K", "5"))
    config.train.lora_path = None
    config.train.lora_rank = 32
    config.train.ema = True
    return config

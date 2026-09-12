#!/usr/bin/env python3
"""Paper trainer for multi-axis max@K on Stable Diffusion 3.5 Medium.

This is the intentionally narrow public training path. It supports the three
released CLIP axes, the seven rule-based color axes of the controlled color
experiment, and the max@K, max@1, and hard-count credit rules used in the
paper. Sampling and PPO replay follow the Flow-GRPO SD3 implementation.
"""

from __future__ import annotations

import contextlib
import datetime
import os
from collections import defaultdict
from pathlib import Path

import torch
from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
from ml_collections import config_flags
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from flow_grpo.clip_prompt_reward import (
    ClipPromptRewardBundleScorer,
    expand_clip_prompt_reward_keys,
    is_clip_prompt_reward_key,
)
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.pixel_partition_reward import (
    PixelPartitionRewardScorer,
    expand_pixel_partition_reward_keys,
    is_pixel_partition_reward_key,
)
from multi_axis_maxk.advantages import advantages_from_group_ids
from multi_axis_maxk.sampling import DistributedKRepeatSampler

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/sd35_paper.py:paper", "Training configuration.")
logger = get_logger(__name__)


class TextPromptDataset(Dataset):
    """Line-delimited prompt dataset with stable integer identifiers."""

    def __init__(self, directory: str, split: str = "train"):
        self.path = Path(directory) / f"{split}.txt"
        self.prompts = [line.strip() for line in self.path.read_text().splitlines() if line.strip()]
        if not self.prompts:
            raise ValueError(f"no prompts found in {self.path}")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict:
        return {"prompt": self.prompts[index], "index": index}

    @staticmethod
    def collate_fn(rows: list[dict]) -> tuple[list[str], list[int]]:
        return [row["prompt"] for row in rows], [row["index"] for row in rows]


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


REWARD_FAMILIES = ("clip_prompt", "pixel_partition")
ANCHOR_KEYS = ("none", "pickscore")


def parse_recipe(config) -> tuple[list[str], str, str, int]:
    """Resolve the released reward family, credit rule, and window from the environment.

    ``STOCHASTIC_REWARD_KEYS`` holds exactly one released alias: a CLIP axis
    (``clip_prompt_basis_*``) or the rule-based color basis
    (``pixel_partition_basis``). ``k == m`` is accepted because the reported
    color runs used ``m = k = 7``; see ``multi_axis_maxk.advantages`` for the
    baseline used in that case.
    """

    aliases = [item.strip() for item in os.getenv("STOCHASTIC_REWARD_KEYS", "").split(",")]
    aliases = [item for item in aliases if item]
    if len(aliases) != 1:
        raise ValueError("STOCHASTIC_REWARD_KEYS must contain one released reward alias")
    reward_keys = expand_clip_prompt_reward_keys(aliases)
    if reward_keys and all(is_clip_prompt_reward_key(key) for key in reward_keys):
        family = "clip_prompt"
    else:
        reward_keys = expand_pixel_partition_reward_keys(aliases)
        if reward_keys and all(is_pixel_partition_reward_key(key) for key in reward_keys):
            family = "pixel_partition"
        else:
            raise ValueError(f"unsupported paper reward axis: {aliases[0]!r}")
    mode = os.getenv("ADVANTAGE_MODE", "coverage").strip().lower()
    if mode not in {"coverage", "count"}:
        raise ValueError("the paper trainer supports ADVANTAGE_MODE=coverage or count")
    k_value = _env_float("MAXATK_K", float(config.train.maxatk_k))
    if int(k_value) != k_value:
        raise ValueError(f"MAXATK_K must be integer-valued, got {k_value}")
    k = int(k_value)
    m = _env_int("NUM_IMAGE_PER_PROMPT", int(config.sample.num_image_per_prompt))
    if mode == "coverage" and k > 1 and not 2 <= k <= m:
        raise ValueError(f"max@K credit requires 2 <= k <= m; got k={k}, m={m}")
    return reward_keys, family, mode, k


def parse_anchor() -> str:
    """Return the frozen quality anchor multiplied into every reward axis, or ``"none"``.

    The controlled color experiment multiplies each rule-based color score by a
    frozen PickScore value before max@K credit (``REWARD_ANCHOR_KEY=pickscore``).
    The anchor is not a target axis and is never given to the credit rule on its
    own. Only multiplicative combination is released.
    """

    anchor = os.getenv("REWARD_ANCHOR_KEY", "none").strip().lower() or "none"
    if anchor not in ANCHOR_KEYS:
        raise ValueError(f"REWARD_ANCHOR_KEY must be one of {ANCHOR_KEYS}, got {anchor!r}")
    combine = os.getenv("REWARD_ANCHOR_COMBINE_MODE", "multiply").strip().lower()
    if combine != "multiply":
        raise ValueError("the public trainer releases REWARD_ANCHOR_COMBINE_MODE=multiply only")
    return anchor


def parse_m_eq_k_baseline() -> str:
    baseline = os.getenv("MAXK_M_EQ_K_BASELINE", "group_mean").strip().lower()
    if baseline not in {"group_mean", "none"}:
        raise ValueError("MAXK_M_EQ_K_BASELINE must be 'group_mean' or 'none'")
    return baseline


def compute_text_embeddings(prompts, text_encoders, tokenizers, device):
    with torch.no_grad():
        prompt_embeds, pooled = encode_prompt(
            text_encoders, tokenizers, prompts, max_sequence_length=128
        )
    return prompt_embeds.to(device), pooled.to(device)


def compute_log_prob(transformer, pipeline, sample, timestep_index, config):
    prediction = transformer(
        hidden_states=sample["latents"][:, timestep_index],
        timestep=sample["timesteps"][:, timestep_index],
        encoder_hidden_states=sample["prompt_embeds"],
        pooled_projections=sample["pooled_prompt_embeds"],
        return_dict=False,
    )[0]
    return sde_step_with_logprob(
        pipeline.scheduler,
        prediction.float(),
        sample["timesteps"][:, timestep_index],
        sample["latents"][:, timestep_index].float(),
        prev_sample=sample["next_latents"][:, timestep_index].float(),
        noise_level=config.sample.noise_level,
    )


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def save_checkpoint(
    transformer,
    global_step,
    accelerator,
    ema,
    trainable_parameters,
):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output = (
            Path(accelerator.project_dir) / "checkpoints" / f"checkpoint-{global_step}" / "lora"
        )
        output.mkdir(parents=True, exist_ok=True)
        if ema is not None:
            ema.copy_ema_to(trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(output)
        if ema is not None:
            ema.copy_temp_to(trainable_parameters)
        logger.info(f"saved {output}")
    accelerator.wait_for_everyone()


def main(_):
    config = FLAGS.config
    reward_keys, reward_family, advantage_mode, k = parse_recipe(config)
    anchor_key = parse_anchor()
    m_eq_k_baseline = parse_m_eq_k_baseline()

    config.seed = _env_int("TRAIN_SEED", int(config.seed))
    config.sample.num_image_per_prompt = _env_int(
        "NUM_IMAGE_PER_PROMPT", int(config.sample.num_image_per_prompt)
    )
    config.train.learning_rate = _env_float(
        "TRAIN_LEARNING_RATE", float(config.train.learning_rate)
    )
    config.train.beta = _env_float("TRAIN_BETA", float(config.train.beta))
    stop_at_step = _env_int("STOP_AT_GLOBAL_STEP", 360)
    save_every = _env_int("SAVE_EVERY_STEPS", 120)
    run_name = os.getenv("FORCED_RUN_NAME", config.run_name).strip()
    if not run_name:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    train_timesteps = int(config.sample.num_steps)
    project = ProjectConfiguration(
        project_dir=str(Path(config.logdir) / run_name),
        total_limit=int(config.num_checkpoint_limit),
    )
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=project,
        gradient_accumulation_steps=(
            int(config.train.gradient_accumulation_steps) * train_timesteps
        ),
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=os.getenv("WANDB_PROJECT", "multi-axis-maxk"),
            config={
                "reward_keys": reward_keys,
                "reward_family": reward_family,
                "anchor_key": anchor_key,
                "m_eq_k_baseline": m_eq_k_baseline,
                "advantage_mode": advantage_mode,
                "k": k,
                "m": int(config.sample.num_image_per_prompt),
                "seed": config.seed,
                "learning_rate": config.train.learning_rate,
                "kl_beta": config.train.beta,
            },
            init_kwargs={"wandb": {"name": run_name}},
        )

    set_seed(config.seed, device_specific=True)
    logger.info(
        f"paper recipe: family={reward_family}, mode={advantage_mode}, axes={len(reward_keys)}, "
        f"anchor={anchor_key}, k={k}, m={config.sample.num_image_per_prompt}, "
        f"processes={accelerator.num_processes}"
    )

    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=2,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Denoising",
    )
    for module in (
        pipeline.vae,
        pipeline.text_encoder,
        pipeline.text_encoder_2,
        pipeline.text_encoder_3,
        pipeline.transformer,
    ):
        module.requires_grad_(False)

    inference_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(accelerator.mixed_precision, torch.float32)
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    lora_config = LoraConfig(
        r=int(config.train.lora_rank),
        lora_alpha=int(config.train.lora_rank),
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    lora_path = os.getenv("LORA_PATH", "").strip() or config.train.lora_path
    if lora_path:
        pipeline.transformer = PeftModel.from_pretrained(
            pipeline.transformer, lora_path, is_trainable=True
        )
    else:
        pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)
    transformer = pipeline.transformer
    trainable_parameters = [
        parameter for parameter in transformer.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("no trainable LoRA parameters found")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    ema = EMAModuleWrapper(trainable_parameters) if config.train.ema else None

    dataset = TextPromptDataset(config.dataset, split="train")
    sampler = DistributedKRepeatSampler(
        dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=config.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )
    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    iterator = iter(dataloader)
    if reward_family == "clip_prompt":
        reward_scorer = ClipPromptRewardBundleScorer(accelerator.device)
    else:
        reward_scorer = PixelPartitionRewardScorer()
    anchor_fn = None
    if anchor_key == "pickscore":
        from flow_grpo.rewards import pickscore_score

        anchor_fn = pickscore_score(accelerator.device)
    autocast = contextlib.nullcontext

    negative_embed, negative_pooled = compute_text_embeddings(
        [""], text_encoders, tokenizers, accelerator.device
    )
    global_step = 0
    epoch = 0

    while global_step < stop_at_step:
        transformer.eval()
        rollout_batches = []
        for batch_index in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: rollout",
            disable=not accelerator.is_local_main_process,
        ):
            sampler_epoch = epoch * config.sample.num_batches_per_epoch + batch_index
            sampler.set_epoch(sampler_epoch)
            prompts, prompt_indices = next(iterator)
            prompt_embeds, pooled_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, accelerator.device
            )
            batch_size = len(prompts)
            with torch.no_grad(), autocast():
                images, latent_history, old_log_probs = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_embeds,
                    negative_prompt_embeds=negative_embed.repeat(batch_size, 1, 1),
                    negative_pooled_prompt_embeds=negative_pooled.repeat(batch_size, 1),
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                )
            scores = reward_scorer.score_keys(images, prompts, reward_keys)
            rewards = torch.stack([torch.as_tensor(scores[key]) for key in reward_keys], dim=-1).to(
                accelerator.device, dtype=torch.float32
            )
            anchor_values = None
            if anchor_fn is not None:
                anchor_raw, _ = anchor_fn(images, prompts, None)
                anchor_values = torch.as_tensor(anchor_raw).to(
                    accelerator.device, dtype=torch.float32
                )
                # Paper Appendix (rule-based color rewards): every axis score is
                # multiplied by the frozen anchor before max@K credit.
                rewards = rewards * anchor_values[:, None]
            latents = torch.stack(latent_history, dim=1)
            group_ids = torch.tensor(
                [sampler_epoch * len(dataset) + index for index in prompt_indices],
                device=accelerator.device,
                dtype=torch.long,
            )
            rollout_batches.append(
                {
                    "group_ids": group_ids,
                    "rewards": rewards,
                    "anchor": (
                        anchor_values
                        if anchor_values is not None
                        else torch.zeros(batch_size, device=accelerator.device)
                    ),
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_embeds,
                    "timesteps": pipeline.scheduler.timesteps.repeat(batch_size, 1),
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": torch.stack(old_log_probs, dim=1),
                }
            )

        samples = {
            key: torch.cat([batch[key] for batch in rollout_batches], dim=0)
            for key in rollout_batches[0]
        }
        gathered_ids = accelerator.gather(samples.pop("group_ids")).cpu()
        gathered_rewards = accelerator.gather(samples.pop("rewards")).cpu()
        gathered_anchor = accelerator.gather(samples.pop("anchor")).cpu()
        gathered_advantages = advantages_from_group_ids(
            gathered_ids,
            gathered_rewards,
            candidates_per_group=config.sample.num_image_per_prompt,
            k=k,
            method="maxk" if advantage_mode == "coverage" else "count",
            m_eq_k_baseline=m_eq_k_baseline,
        )
        local_count = samples["latents"].shape[0]
        samples["advantages"] = gathered_advantages.reshape(accelerator.num_processes, local_count)[
            accelerator.process_index
        ].to(accelerator.device)
        samples["advantages"] = samples["advantages"][:, None].expand(-1, train_timesteps)

        if accelerator.is_main_process:
            group_order = torch.argsort(gathered_ids, stable=True)
            grouped_rewards = gathered_rewards[group_order].reshape(
                -1, config.sample.num_image_per_prompt, gathered_rewards.shape[-1]
            )
            accelerator.log(
                {
                    "epoch": epoch,
                    "reward/mean": float(gathered_rewards.mean()),
                    "reward/axis_batch_max_mean": float(grouped_rewards.amax(dim=1).mean()),
                    "reward/anchor_mean": float(gathered_anchor.mean()),
                    "advantage/abs_mean": float(gathered_advantages.abs().mean()),
                },
                step=global_step,
            )

        total_batch = samples["latents"].shape[0]
        for inner_epoch in range(config.train.num_inner_epochs):
            permutation = torch.randperm(total_batch, device=accelerator.device)
            samples = {key: value[permutation] for key, value in samples.items()}
            batch_size = int(config.train.batch_size)
            transformer.train()
            metrics = defaultdict(list)
            for start in tqdm(
                range(0, total_batch, batch_size),
                desc=f"Epoch {epoch}.{inner_epoch}: train",
                disable=not accelerator.is_local_main_process,
            ):
                sample = {key: value[start : start + batch_size] for key, value in samples.items()}
                for timestep_index in range(train_timesteps):
                    with accelerator.accumulate(transformer), autocast():
                        _, log_prob, current_mean, std = compute_log_prob(
                            transformer, pipeline, sample, timestep_index, config
                        )
                        reference_mean = None
                        if config.train.beta > 0:
                            unwrapped = unwrap_model(transformer, accelerator)
                            with torch.no_grad(), unwrapped.disable_adapter():
                                _, _, reference_mean, _ = compute_log_prob(
                                    unwrapped, pipeline, sample, timestep_index, config
                                )

                        advantage = sample["advantages"][:, timestep_index].clamp(
                            -config.train.adv_clip_max, config.train.adv_clip_max
                        )
                        log_ratio = (log_prob - sample["log_probs"][:, timestep_index]).clamp(
                            -20.0, 20.0
                        )
                        ratio = torch.exp(log_ratio)
                        lower = 1.0 - config.train.clip_range
                        upper = 1.0 + config.train.highclip_range
                        policy_loss = torch.maximum(
                            -advantage * ratio,
                            -advantage * ratio.clamp(lower, upper),
                        ).mean()
                        kl_loss = policy_loss.new_zeros(())
                        if reference_mean is not None:
                            kl_loss = (
                                (current_mean - reference_mean)
                                .square()
                                .mean(dim=(1, 2, 3), keepdim=True)
                                / (2 * std.square())
                            ).mean()
                        loss = policy_loss + config.train.beta * kl_loss
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    metrics["loss"].append(loss.detach())
                    metrics["policy_loss"].append(policy_loss.detach())
                    metrics["kl_loss"].append(kl_loss.detach())
                    metrics["approx_kl"].append(
                        0.5
                        * (log_prob.detach() - sample["log_probs"][:, timestep_index])
                        .square()
                        .mean()
                    )
                    metrics["clip_fraction"].append(
                        ((ratio.detach() < lower) | (ratio.detach() > upper)).float().mean()
                    )

                    if accelerator.sync_gradients:
                        global_step += 1
                        if ema is not None:
                            ema.step(trainable_parameters, global_step)
                        reduced = {
                            f"train/{name}": accelerator.reduce(
                                torch.stack(values).mean(), reduction="mean"
                            ).item()
                            for name, values in metrics.items()
                        }
                        if accelerator.is_main_process:
                            accelerator.log(reduced, step=global_step)
                        metrics.clear()
                        if save_every > 0 and global_step % save_every == 0:
                            save_checkpoint(
                                transformer, global_step, accelerator, ema, trainable_parameters
                            )
                        if global_step >= stop_at_step:
                            break
                if global_step >= stop_at_step:
                    break
            if global_step >= stop_at_step:
                break
        epoch += 1

    if save_every <= 0 or global_step % save_every:
        save_checkpoint(transformer, global_step, accelerator, ema, trainable_parameters)
    accelerator.end_training()


if __name__ == "__main__":
    app.run(main)

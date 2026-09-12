#!/usr/bin/env python3
"""Generate a fixed base/LoRA audit set and compute image-quality metrics.

Demographic coverage is intentionally evaluated in a separate pass with
``evaluation/audit_clip.py`` so every probe can reuse exactly the same images.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from tqdm import tqdm

try:
    from peft import PeftModel
except ImportError as e:
    raise ImportError("peft is required to load LoRA adapters.") from e

from flow_grpo import rewards as flow_rewards

MetricFn = Callable[[list[Image.Image], list[str], list[dict]], np.ndarray]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _get_rank_world(args_rank: int | None, args_world_size: int | None) -> tuple[int, int]:
    rank = args_rank
    world_size = args_world_size
    if rank is None:
        env_rank = os.getenv("OMPI_COMM_WORLD_RANK")
        if env_rank is not None and env_rank.isdigit():
            rank = int(env_rank)
    if world_size is None:
        env_ws = os.getenv("OMPI_COMM_WORLD_SIZE")
        if env_ws is not None and env_ws.isdigit():
            world_size = int(env_ws)
    return rank or 0, world_size or 1


def _resolve_dtype(precision: str) -> torch.dtype:
    precision = precision.lower()
    if precision in ("fp16", "float16"):
        return torch.float16
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def _load_text_prompts(path: str, num_prompts: int, seed: int) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        prompts_all = [line.strip() for line in f if line.strip()]
    if num_prompts > 0 and num_prompts < len(prompts_all):
        rng = random.Random(seed)
        indices = list(range(len(prompts_all)))
        rng.shuffle(indices)
        prompts_all = [prompts_all[i] for i in indices[:num_prompts]]
    return [
        {
            "prompt": prompt,
            "metadata": {},
            "tag": "all",
        }
        for prompt in prompts_all
    ]


def _shard_list(xs: Sequence[Any], *, rank: int, world_size: int) -> list[Any]:
    if world_size <= 1:
        return list(xs)
    n = len(xs)
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return list(xs[start:end])


def _autocast_context(device: str, pipe_dtype: torch.dtype):
    if device.startswith("cuda") and pipe_dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=pipe_dtype)
    return torch.autocast(device_type="cpu", enabled=False)


def _load_pipeline(base_model: str, device: str, dtype: torch.dtype, disable_bar: bool):
    pipe = StableDiffusion3Pipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe.safety_checker = None
    if disable_bar:
        pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    try:
        pipe.vae.to(device=device, dtype=torch.float32)
        _orig_decode = pipe.vae.decode

        def _upcast_decode(z, **kwargs):
            return _orig_decode(z.to(pipe.vae.dtype), **kwargs)

        pipe.vae.decode = _upcast_decode
    except Exception:
        pass
    return pipe


def _apply_lora(pipe, lora_path: str, device: str):
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)
    try:
        pipe.transformer.set_adapter("default")
    except Exception:
        pass
    pipe.transformer.to(device)
    return pipe


def _wrap_reward_factory(
    key: str,
    *,
    device: str,
) -> MetricFn:
    score_functions = {
        "pickscore": flow_rewards.pickscore_score,
        "aesthetic": flow_rewards.aesthetic_score,
        "clipscore": flow_rewards.clip_score,
    }
    if key not in score_functions:
        raise ValueError(
            f"Unsupported quality metric {key!r}; choose from {sorted(score_functions)}"
        )
    factory = score_functions[key]
    fn = factory(device) if "device" in factory.__code__.co_varnames else factory()

    def _metric(images: list[Image.Image], prompts: list[str], metadata: list[dict]) -> np.ndarray:
        scores, _ = fn(images, prompts, metadata)
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        return np.asarray(scores, dtype=np.float32)

    return _metric


def _summarize_values(vals: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "median": float(np.median(vals)),
    }


def _generate_and_evaluate(
    *,
    pipe,
    prompt_rows: list[dict],
    metric_fns: dict[str, MetricFn],
    metric_keys: list[str],
    images_per_prompt: int,
    gen_batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    base_seed: int,
    pipe_dtype: torch.dtype,
    save_images_dir: str | None,
    model_name: str,
    prompt_idx_offset: int,
) -> list[dict]:
    results: list[dict] = []
    device = str(getattr(pipe, "device", "cuda"))
    if device == "meta":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for local_idx, row in enumerate(
        tqdm(
            prompt_rows,
            desc=f"{model_name}: prompts",
            dynamic_ncols=True,
            disable=os.getenv("DISABLE_TQDM", "0") == "1",
        )
    ):
        global_idx = prompt_idx_offset + local_idx
        prompt = row["prompt"]
        metadata = dict(row.get("metadata", {}))
        tag = str(row.get("tag", "all"))
        seeds = [base_seed + global_idx * 10_000 + j for j in range(images_per_prompt)]

        all_images: list[Image.Image] = []
        for start in range(0, len(seeds), gen_batch_size):
            chunk_seeds = seeds[start : start + gen_batch_size]
            generators = [
                torch.Generator(device=device).manual_seed(seed_val) for seed_val in chunk_seeds
            ]
            with torch.inference_mode(), _autocast_context(device, pipe_dtype):
                out = pipe(
                    prompt=[prompt] * len(chunk_seeds),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generators,
                    output_type="pil",
                )
            all_images.extend(out.images)

        prompts_batch = [prompt] * len(all_images)
        metadata_batch = [metadata] * len(all_images)
        scores_per_key: dict[str, np.ndarray] = {}
        for key, fn in metric_fns.items():
            scores_per_key[key] = fn(all_images, prompts_batch, metadata_batch)

        for image_idx, (img, seed_val) in enumerate(zip(all_images, seeds, strict=False)):
            record: dict[str, Any] = {
                "prompt_idx": global_idx,
                "prompt": prompt,
                "tag": tag,
                "image_idx": image_idx,
                "seed": seed_val,
            }
            for key in metric_keys:
                record[key] = float(scores_per_key[key][image_idx])
            results.append(record)

            if save_images_dir:
                out_dir = Path(save_images_dir) / model_name / tag
                out_dir.mkdir(parents=True, exist_ok=True)
                img_path = (
                    out_dir / f"prompt_{global_idx:04d}_img_{image_idx:03d}_seed_{seed_val}.png"
                )
                img.save(img_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper-core eval bundle.")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--lora_name", type=str, default="lora")
    parser.add_argument("--skip_base", action="store_true")
    parser.add_argument(
        "--primary_reward_key",
        choices=("clipscore", "pickscore", "aesthetic"),
        default="clipscore",
    )
    parser.add_argument("--prompts_path", type=str, default="data/heldout/occupations_200.txt")
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--prompt_seed", type=int, default=0)
    parser.add_argument("--images_per_prompt", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--gen_batch_size", type=int, default=2)
    parser.add_argument("--pipe_precision", type=str, default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_pickscore", action="store_true")
    parser.add_argument("--skip_aesthetic", action="store_true")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--parts_dir", type=str, default="")
    parser.add_argument("--merge_timeout_sec", type=int, default=3600)
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    rank, world_size = _get_rank_world(args.rank, args.world_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe_dtype = _resolve_dtype(args.pipe_precision)

    repo_root = _repo_root()
    prompts_path = (
        str((repo_root / args.prompts_path).resolve())
        if not os.path.isabs(args.prompts_path)
        else args.prompts_path
    )
    lora_path = (
        str((repo_root / args.lora_path).resolve())
        if not os.path.isabs(args.lora_path)
        else args.lora_path
    )

    prompt_rows_all = _load_text_prompts(prompts_path, args.num_prompts, args.prompt_seed)

    prompt_rows_shard = _shard_list(prompt_rows_all, rank=rank, world_size=world_size)
    n_total = len(prompt_rows_all)
    base_count = n_total // world_size
    rem = n_total % world_size
    prompt_idx_offset = rank * base_count + min(rank, rem)

    out_dir = args.out_dir or str(
        repo_root
        / "outputs"
        / "paper_core_eval"
        / os.getenv("RUN_ID", f"manual_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    parts_dir = args.parts_dir or os.path.join(out_dir, "parts")
    Path(parts_dir).mkdir(parents=True, exist_ok=True)
    save_images_dir = os.path.join(out_dir, "images") if args.save_images else None

    metric_keys = []
    default_quality_keys = []
    if not args.skip_pickscore:
        default_quality_keys.append("pickscore")
    if not args.skip_aesthetic:
        default_quality_keys.append("aesthetic")
    for key in [args.primary_reward_key, *default_quality_keys]:
        if key not in metric_keys:
            metric_keys.append(key)
    metric_fns = {
        key: _wrap_reward_factory(
            key,
            device=device,
        )
        for key in metric_keys
    }

    if rank == 0:
        print(f"[INFO] rank={rank}/{world_size} device={device}")
        print(f"[INFO] primary_reward_key={args.primary_reward_key}")
        print(f"[INFO] metric_keys={metric_keys}")
        print(f"[INFO] prompts={len(prompt_rows_all)} (shard={len(prompt_rows_shard)})")
        print(f"[INFO] images_per_prompt={args.images_per_prompt}")
        print(f"[INFO] lora_path={lora_path}")
        print(f"[INFO] out_dir={out_dir}")

    all_model_results: dict[str, list[dict]] = {}
    t0 = time.time()

    gen_kwargs = dict(
        metric_fns=metric_fns,
        metric_keys=metric_keys,
        images_per_prompt=args.images_per_prompt,
        gen_batch_size=args.gen_batch_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        base_seed=args.seed,
        pipe_dtype=pipe_dtype,
        save_images_dir=save_images_dir,
        prompt_idx_offset=prompt_idx_offset,
    )

    if not args.skip_base:
        if rank == 0:
            print(f"[INFO] Loading base model: {args.base_model}")
        base_pipe = _load_pipeline(args.base_model, device, pipe_dtype, args.disable_progress_bar)
        base_pipe.transformer.eval()
        all_model_results["base"] = _generate_and_evaluate(
            pipe=base_pipe,
            prompt_rows=prompt_rows_shard,
            model_name="base",
            **gen_kwargs,
        )
        del base_pipe
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if rank == 0:
        print(f"[INFO] Loading LoRA model: {args.base_model} + {lora_path}")
    lora_pipe = _load_pipeline(args.base_model, device, pipe_dtype, args.disable_progress_bar)
    lora_pipe = _apply_lora(lora_pipe, lora_path, device)
    lora_pipe.transformer.eval()
    all_model_results[args.lora_name] = _generate_and_evaluate(
        pipe=lora_pipe,
        prompt_rows=prompt_rows_shard,
        model_name=args.lora_name,
        **gen_kwargs,
    )
    del lora_pipe
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    part_data = {
        "rank": rank,
        "world_size": world_size,
        "elapsed_sec": elapsed,
        "metric_keys": metric_keys,
        "primary_reward_key": args.primary_reward_key,
        "models": all_model_results,
    }
    part_path = Path(parts_dir) / f"rank-{rank:05d}.json"
    part_tmp = Path(parts_dir) / f"rank-{rank:05d}.json.tmp"
    with open(part_tmp, "w", encoding="utf-8") as f:
        json.dump(part_data, f, ensure_ascii=False)
    os.replace(part_tmp, part_path)
    if rank == 0:
        print(f"[INFO] Wrote part: {part_path}")

    if world_size > 1 and rank != 0:
        return

    if world_size > 1:
        expected = [Path(parts_dir) / f"rank-{r:05d}.json" for r in range(world_size)]
        t_wait = time.time()
        while True:
            missing = [p for p in expected if not p.exists()]
            if not missing:
                break
            if time.time() - t_wait > args.merge_timeout_sec:
                raise TimeoutError(
                    f"Timed out waiting for parts. Missing: {[str(p) for p in missing]}"
                )
            time.sleep(5)

        merged: dict[str, list[dict]] = {}
        for p in expected:
            with open(p, encoding="utf-8") as f:
                part = json.load(f)
            for model_name, records in part["models"].items():
                merged.setdefault(model_name, []).extend(records)
    else:
        merged = all_model_results

    summary: dict[str, Any] = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_model": args.base_model,
            "lora_path": lora_path,
            "lora_name": args.lora_name,
            "primary_reward_key": args.primary_reward_key,
            "metric_keys": metric_keys,
            "prompts_path": prompts_path,
            "num_prompts": len(prompt_rows_all),
            "prompt_seed": args.prompt_seed,
            "images_per_prompt": args.images_per_prompt,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "seed": args.seed,
            "world_size": world_size,
        },
        "models": {},
    }

    for model_name, records in merged.items():
        model_summary: dict[str, Any] = {"total_images": len(records)}
        overall: dict[str, Any] = {}
        for key in metric_keys:
            vals = np.array([r[key] for r in records], dtype=np.float32)
            overall[key] = _summarize_values(vals)
        model_summary["overall"] = overall

        per_prompt_records: dict[int, list[dict]] = {}
        for record in records:
            per_prompt_records.setdefault(int(record["prompt_idx"]), []).append(record)

        prompt_stats = []
        for prompt_idx in sorted(per_prompt_records.keys()):
            recs = per_prompt_records[prompt_idx]
            ps: dict[str, Any] = {
                "prompt_idx": prompt_idx,
                "prompt": recs[0]["prompt"],
                "tag": recs[0].get("tag", "all"),
            }
            for key in metric_keys:
                arr = np.array([r[key] for r in recs], dtype=np.float32)
                ps[key] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "q25": float(np.percentile(arr, 25)),
                    "median": float(np.median(arr)),
                    "q75": float(np.percentile(arr, 75)),
                    "range": float(arr.max() - arr.min()),
                    "values": arr.tolist(),
                }
            prompt_stats.append(ps)
        model_summary["per_prompt"] = prompt_stats
        summary["models"][model_name] = model_summary

        print(f"\n{'=' * 60}")
        print(f"  {model_name}  ({len(records)} images)")
        print(f"{'=' * 60}")
        for key in metric_keys:
            stats = overall[key]
            print(
                f"  {key}: mean={stats['mean']:.4f}  std={stats['std']:.4f}"
                f"  min={stats['min']:.4f}  max={stats['max']:.4f}"
            )

    out_json = os.path.join(out_dir, "eval_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Wrote JSON: {out_json}")

    out_csv = os.path.join(out_dir, "eval_per_image.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "prompt_idx", "tag", "prompt", "image_idx", "seed"] + metric_keys)
        for model_name, records in merged.items():
            for record in records:
                row = [
                    model_name,
                    record["prompt_idx"],
                    record.get("tag", "all"),
                    record["prompt"],
                    record["image_idx"],
                    record["seed"],
                ]
                row += [f"{record[key]:.6f}" for key in metric_keys]
                writer.writerow(row)
    print(f"[INFO] Wrote CSV: {out_csv}")
    print(f"[INFO] Done. Total wall-clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

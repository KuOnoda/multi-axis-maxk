#!/usr/bin/env python3
"""Score saved images with the seven rule-based color axes and report ``C_pix@M``.

Expects the ``prompt_<i>_img_<j>_seed_<s>.png`` layout written by
``evaluation/generate_and_score.py --save_images``. For every prompt the first
``--m`` images (by image index) form the evaluation batch; the metric is the
mean over prompts and axes of the per-axis batch maximum, Eq. (pixel-batchmax)
in the paper. The rule-based scores need no model weights and run on CPU.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from flow_grpo.pixel_partition_reward import (
    PARTITION_MODES,
    PartitionParams,
    partition_fractions,
)
from multi_axis_maxk.metrics import axiswise_batch_max_coverage

FILENAME = re.compile(r"prompt_(\d+)_img_(\d+)_seed_(\d+)\.png$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--m", type=int, default=12, help="batch size M used for C_pix@M")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    params = PartitionParams.from_env()
    paths = sorted(args.image_dir.rglob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no PNG images under {args.image_dir}")

    per_prompt: dict[int, list[tuple[int, np.ndarray, str]]] = defaultdict(list)
    for path in paths:
        match = FILENAME.search(path.name)
        if match is None:
            raise ValueError(f"unexpected image filename: {path.name}")
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        fractions = partition_fractions(image, params)
        per_prompt[int(match.group(1))].append((int(match.group(2)), fractions, str(path)))

    rows = []
    batches = []
    for prompt_idx in sorted(per_prompt):
        entries = sorted(per_prompt[prompt_idx], key=lambda item: item[0])
        if len(entries) < args.m:
            raise ValueError(
                f"prompt {prompt_idx} has {len(entries)} images but --m={args.m} were requested"
            )
        batches.append(np.stack([entry[1] for entry in entries[: args.m]], axis=0))
        for image_idx, fractions, path in entries:
            rows.append(
                {
                    "path": path,
                    "prompt_idx": prompt_idx,
                    "image_idx": image_idx,
                    "scores": fractions.tolist(),
                    "top_mode": PARTITION_MODES[int(fractions.argmax())],
                }
            )

    scores = np.stack(batches, axis=0)  # (P, M, D)
    per_axis_batch_max = scores.max(axis=1).mean(axis=0)
    per_axis_image_mean = scores.reshape(-1, scores.shape[-1]).mean(axis=0)
    summary = {
        "image_dir": str(args.image_dir),
        "num_prompts": int(scores.shape[0]),
        "m": int(args.m),
        "modes": list(PARTITION_MODES),
        "partition_params": {
            "tau": params.tau,
            "alpha": params.alpha,
            "beta": params.beta,
            "gamma": params.gamma,
        },
        "c_pix_at_m": axiswise_batch_max_coverage(scores, args.m),
        "per_axis_batch_max_mean": dict(
            zip(PARTITION_MODES, per_axis_batch_max.tolist(), strict=True)
        ),
        "per_axis_image_mean": dict(
            zip(PARTITION_MODES, per_axis_image_mean.tolist(), strict=True)
        ),
        "images": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    per_axis = ", ".join(
        f"{mode}={value:.3f}" for mode, value in summary["per_axis_batch_max_mean"].items()
    )
    print(
        f"C_pix@{args.m} = {summary['c_pix_at_m']:.4f} over {summary['num_prompts']} prompts; "
        f"{per_axis}"
    )


if __name__ == "__main__":
    main()

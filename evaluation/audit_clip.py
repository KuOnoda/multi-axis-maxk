#!/usr/bin/env python3
"""Audit saved images with the fixed CLIP prompt ensemble used in training."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from flow_grpo.clip_prompt_reward import ClipPromptRewardBundleScorer
from multi_axis_maxk.metrics import fairness_score

AXIS_KEYS = {
    "race5": ("race5_v4", 5),
    "gender2": ("gender2_ens", 2),
    "race5_gender2": ("race5_v4_gender2_ens", 10),
}
FILENAME = re.compile(r"prompt_(\d+)_img_(\d+)_seed_(\d+)\.png$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--axis", choices=sorted(AXIS_KEYS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    axis_name, mode_count = AXIS_KEYS[args.axis]
    keys = [f"clip_prompt_{axis_name}_{i}" for i in range(mode_count)]
    paths = sorted(args.image_dir.rglob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no PNG images under {args.image_dir}")

    scorer = ClipPromptRewardBundleScorer(args.device)
    rows = []
    for start in range(0, len(paths), args.batch_size):
        chunk = paths[start : start + args.batch_size]
        images = [Image.open(path).convert("RGB") for path in chunk]
        scored = scorer.score_keys(images, [""] * len(images), keys)
        matrix = np.stack([scored[key] for key in keys], axis=1)
        for path, values in zip(chunk, matrix, strict=True):
            match = FILENAME.search(path.name)
            if match is None:
                raise ValueError(f"unexpected image filename: {path.name}")
            rows.append(
                {
                    "path": str(path),
                    "prompt_idx": int(match.group(1)),
                    "scores": values.tolist(),
                    "top_mode": int(values.argmax()),
                }
            )

    counts = np.bincount([row["top_mode"] for row in rows], minlength=mode_count)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_idx"]].append(row["top_mode"])
    distinct = [len(set(labels)) for labels in by_prompt.values()]
    full = [len(set(labels)) == mode_count for labels in by_prompt.values()]
    occurrence = [
        float(np.mean([mode in labels for labels in by_prompt.values()]))
        for mode in range(mode_count)
    ]
    summary = {
        "axis": args.axis,
        "n_images": len(rows),
        "n_prompts": len(by_prompt),
        "top_label_counts": counts.tolist(),
        "top_label_frequencies": (counts / counts.sum()).tolist(),
        "fairness_score": fairness_score(counts),
        "mean_distinct_modes_per_prompt": float(np.mean(distinct)),
        "full_batch_coverage": float(np.mean(full)),
        "per_mode_prompt_occurrence": occurrence,
        "weakest_mode_prompt_occurrence": min(occurrence),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_image": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

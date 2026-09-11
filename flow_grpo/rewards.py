"""Small quality-reward factories used by the public evaluation scripts."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _pil_to_tensor(images, *, normalize: bool) -> torch.Tensor:
    if isinstance(images, torch.Tensor):
        tensor = images.detach()
    else:
        tensor = torch.stack(
            [
                torch.as_tensor(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)
                for image in images
            ]
        )
    if normalize:
        return tensor.float() / 255.0 if tensor.dtype == torch.uint8 else tensor.float()
    return tensor


def aesthetic_score(device="cuda"):
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).to(device)

    def score(images, prompts, metadata):
        del prompts, metadata
        return scorer(_pil_to_tensor(images, normalize=False)), {}

    return score


def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def score(images, prompts, metadata):
        del metadata
        return scorer(_pil_to_tensor(images, normalize=True), prompts), {}

    return score


def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def score(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            array = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = [Image.fromarray(item.transpose(1, 2, 0)) for item in array]
        return scorer(prompts, images), {}

    return score

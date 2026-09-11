"""CLIP prompt-ensemble rewards for perceived-appearance target modes.

Each alias expands to one reward key per class:

  - ``clip_prompt_basis_gender2`` -> ``clip_prompt_gender2_0..1``
  - ``clip_prompt_basis_gender2_ens`` -> ``clip_prompt_gender2_ens_0..1``
  - ``clip_prompt_basis_race5``   -> ``clip_prompt_race5_0..4``
  - ``clip_prompt_basis_race5_v4`` -> ``clip_prompt_race5_v4_0..4``
  - ``clip_prompt_basis_race5_gender2`` -> ``clip_prompt_race5_gender2_0..9``
  - ``clip_prompt_basis_race5_v4_gender2`` -> ``clip_prompt_race5_v4_gender2_0..9``
  - ``clip_prompt_basis_race5_v4_gender2_ens`` -> ``clip_prompt_race5_v4_gender2_ens_0..9``

By default each reward is the softmax probability within the prompt set.  This
keeps the reward discriminative for a 2-way or 5-way classifier; raw cosine can
be selected with ``CLIP_PROMPT_REWARD_MODE=cosine`` for ablations.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

CLIP_PROMPT_PREFIX = "clip_prompt_"
CLIP_PROMPT_WILDCARD_PREFIX = "clip_prompt_basis_"


@dataclass(frozen=True)
class ClipPromptAxis:
    modes: tuple[str, ...]
    prompts: tuple[str, ...]
    prompt_ensemble: tuple[tuple[str, ...], ...] | None = None


@dataclass(frozen=True)
class ClipPromptCompositeAxis:
    factors: tuple[str, str]
    modes: tuple[str, ...]


AXES: dict[str, ClipPromptAxis] = {
    "gender2": ClipPromptAxis(
        modes=("male", "female"),
        prompts=(
            "A photo of a male",
            "A photo of a female",
        ),
    ),
    # Gender prompt ensemble used by the paper recipes.
    "gender2_ens": ClipPromptAxis(
        modes=("male", "female"),
        prompts=(),
        prompt_ensemble=(
            (
                "a photo of a male person",
                "a portrait of a male person",
                "a close-up portrait of a man",
                "a headshot of a man",
                "a photo of a man",
                "a face photo of a male adult",
            ),
            (
                "a photo of a female person",
                "a portrait of a female person",
                "a close-up portrait of a woman",
                "a headshot of a woman",
                "a photo of a woman",
                "a face photo of a female adult",
            ),
        ),
    ),
    # Single-prompt variant; paper recipes use the race5_v4 ensemble below.
    "race5": ClipPromptAxis(
        modes=("white", "black", "asian", "indian", "latino"),
        prompts=(
            "A photo of a White person",
            "A photo of a Black person",
            "A photo of an Asian",
            "A photo of an Indian",
            "A photo of a Latino",
        ),
    ),
    # Each class embedding averages normalized prompt embeddings and is then
    # normalized again. Asian denotes East/Southeast Asian; Indian denotes
    # South Asian in this evaluation label set.
    "race5_v4": ClipPromptAxis(
        modes=("white", "black", "asian", "indian", "latino"),
        prompts=(),
        prompt_ensemble=(
            (
                "a photo of a White person",
                "a portrait of a White face",
                "a photo of a Caucasian person",
                "a portrait of a Caucasian face",
                "a photo of a European person",
                "a portrait of a European face",
                "a headshot of a White person",
                "a photo of a light-skinned European person",
            ),
            (
                "a photo of a Black person",
                "a portrait of a Black face",
                "a photo of an African person",
                "a photo of an African American person",
                "a portrait of a Black African person",
                "a headshot of a Black person",
                "a photo of a person of African descent",
                "a portrait of a dark-skinned Black person",
            ),
            (
                "a photo of an East Asian person",
                "a portrait of an East Asian face",
                "a photo of a Southeast Asian person",
                "a portrait of a Southeast Asian face",
                "a headshot of an East Asian person",
                "a photo of a person from East Asia",
            ),
            (
                "a photo of an Indian person",
                "a portrait of an Indian face",
                "a photo of a South Asian person",
                "a headshot of a South Asian person",
                "a photo of a Bangladeshi person",
                "a photo of a person from the Indian subcontinent",
            ),
            (
                "a photo of a Latino person",
                "a portrait of a Latino face",
                "a photo of a Hispanic person",
                "a portrait of a Hispanic face",
                "a photo of a Latin American person",
                "a headshot of a Hispanic person",
            ),
        ),
    ),
}


COMPOSITE_AXES: dict[str, ClipPromptCompositeAxis] = {
    # Product of marginal CLIP prompt probabilities. Mode order is
    # race-major: white_male, white_female, black_male, ...
    "race5_gender2": ClipPromptCompositeAxis(
        factors=("race5", "gender2"),
        modes=tuple(
            f"{race}_{gender}" for race in AXES["race5"].modes for gender in AXES["gender2"].modes
        ),
    ),
    "race5_v4_gender2": ClipPromptCompositeAxis(
        factors=("race5_v4", "gender2"),
        modes=tuple(
            f"{race}_{gender}"
            for race in AXES["race5_v4"].modes
            for gender in AXES["gender2"].modes
        ),
    ),
    "race5_v4_gender2_ens": ClipPromptCompositeAxis(
        factors=("race5_v4", "gender2_ens"),
        modes=tuple(
            f"{race}_{gender}"
            for race in AXES["race5_v4"].modes
            for gender in AXES["gender2_ens"].modes
        ),
    ),
}


def clip_prompt_axis_modes(axis_name: str) -> tuple[str, ...]:
    if axis_name in AXES:
        return AXES[axis_name].modes
    if axis_name in COMPOSITE_AXES:
        return COMPOSITE_AXES[axis_name].modes
    raise KeyError(f"Unknown clip_prompt axis: {axis_name!r}")


def is_clip_prompt_reward_key(key: str) -> bool:
    """True iff ``key`` is an expanded ``clip_prompt_<axis>_<mode_idx>`` key."""
    if not key.startswith(CLIP_PROMPT_PREFIX):
        return False
    if key.startswith(CLIP_PROMPT_WILDCARD_PREFIX):
        return False
    axis_name, _, mode_str = key[len(CLIP_PROMPT_PREFIX) :].rpartition("_")
    if axis_name not in AXES and axis_name not in COMPOSITE_AXES:
        return False
    try:
        mode_idx = int(mode_str)
    except ValueError:
        return False
    return 0 <= mode_idx < len(clip_prompt_axis_modes(axis_name))


def parse_clip_prompt_reward_key(key: str) -> tuple[str, int]:
    """Parse ``clip_prompt_<axis>_<mode_idx>``."""
    if not is_clip_prompt_reward_key(key):
        raise ValueError(f"Not a clip_prompt reward key: {key!r}")
    axis_name, _, mode_str = key[len(CLIP_PROMPT_PREFIX) :].rpartition("_")
    return axis_name, int(mode_str)


def expand_clip_prompt_reward_keys(keys: Sequence[str]) -> list[str]:
    """Expand ``clip_prompt_basis_<axis>`` aliases; pass other keys through."""
    expanded: list[str] = []
    for key in keys:
        if key.startswith(CLIP_PROMPT_WILDCARD_PREFIX):
            axis_name = key[len(CLIP_PROMPT_WILDCARD_PREFIX) :]
            if axis_name in AXES or axis_name in COMPOSITE_AXES:
                expanded.extend(
                    f"{CLIP_PROMPT_PREFIX}{axis_name}_{i}"
                    for i in range(len(clip_prompt_axis_modes(axis_name)))
                )
                continue
        expanded.append(key)
    seen: set[str] = set()
    deduped: list[str] = []
    for key in expanded:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _to_nchw_float(images) -> torch.Tensor:
    """Coerce trainer image batch to ``(B, 3, H, W)`` float in [0, 1]."""
    if isinstance(images, Image.Image):
        arr = np.asarray(images.convert("RGB"), dtype=np.float32)[None] / 255.0
        return torch.from_numpy(arr).permute(0, 3, 1, 2).clamp(0, 1)
    if isinstance(images, torch.Tensor):
        x = images.detach().cpu().to(torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        if float(x.max()) > 1.0 + 1e-3:
            x = x / 255.0
        return x.clamp(0, 1)
    if isinstance(images, list | tuple) and images and isinstance(images[0], Image.Image):
        arr = np.stack(
            [np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0 for im in images],
            axis=0,
        )
        return torch.from_numpy(arr).permute(0, 3, 1, 2).clamp(0, 1)
    x = np.asarray(images, dtype=np.float32)
    if x.ndim == 3:
        x = x[None]
    if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
        x = x.transpose(0, 3, 1, 2)
    if x.max() > 1.0 + 1e-3:
        x = x / 255.0
    return torch.from_numpy(x).clamp(0, 1)


class ClipPromptRewardBundleScorer:
    """Batch scorer for the fixed-prompt CLIP reward keys.

    ``score_keys`` groups requested keys by axis, scores each axis, and extracts
    the requested mode columns.
    """

    def __init__(self, device):
        from flow_grpo.clip_scorer import ClipScorer

        self.device = device
        self.scorer = ClipScorer(device)
        self.reward_mode = os.getenv("CLIP_PROMPT_REWARD_MODE", "prob").strip().lower()
        if self.reward_mode not in {"prob", "cosine", "logit"}:
            raise ValueError(
                "CLIP_PROMPT_REWARD_MODE must be one of "
                "{'prob', 'cosine', 'logit'}, got "
                f"{self.reward_mode!r}"
            )
        self.temperature = float(os.getenv("CLIP_PROMPT_TEMPERATURE", "1.0"))
        if self.temperature <= 0:
            raise ValueError("CLIP_PROMPT_TEMPERATURE must be > 0")
        self._text_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _text_embeds(self, axis_name: str) -> torch.Tensor:
        cached = self._text_cache.get(axis_name)
        if cached is not None:
            return cached
        spec = AXES[axis_name]
        if spec.prompt_ensemble is None:
            prompt_groups = tuple((prompt,) for prompt in spec.prompts)
        else:
            prompt_groups = spec.prompt_ensemble
        if len(prompt_groups) != len(spec.modes):
            raise ValueError(
                f"clip_prompt axis {axis_name!r} has {len(spec.modes)} modes "
                f"but {len(prompt_groups)} prompt groups"
            )

        flat_prompts = [prompt for group in prompt_groups for prompt in group]
        texts = self.scorer.processor(
            text=flat_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            embeds = self.scorer.model.get_text_features(
                input_ids=texts["input_ids"],
                attention_mask=texts["attention_mask"],
            )
        embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        if spec.prompt_ensemble is not None:
            class_embeds = []
            offset = 0
            for group in prompt_groups:
                group_embeds = embeds[offset : offset + len(group)]
                offset += len(group)
                class_embed = group_embeds.mean(dim=0)
                class_embed = class_embed / class_embed.norm(p=2).clamp_min(1e-12)
                class_embeds.append(class_embed)
            embeds = torch.stack(class_embeds, dim=0)
        self._text_cache[axis_name] = embeds.detach()
        return self._text_cache[axis_name]

    @torch.no_grad()
    def _axis_scores(self, images, axis_name: str) -> np.ndarray:
        if axis_name in COMPOSITE_AXES:
            if self.reward_mode != "prob":
                raise ValueError(
                    "Composite clip_prompt axes require "
                    "CLIP_PROMPT_REWARD_MODE=prob because they multiply "
                    "marginal probabilities."
                )
            spec = COMPOSITE_AXES[axis_name]
            first = self._axis_scores(images, spec.factors[0])
            second = self._axis_scores(images, spec.factors[1])
            if first.shape[0] == 0:
                return np.zeros((0, len(spec.modes)), dtype=np.float32)
            joint = first[:, :, None] * second[:, None, :]
            return joint.reshape(first.shape[0], -1).astype(np.float32)

        pixels = _to_nchw_float(images)
        if pixels.shape[0] == 0:
            return np.zeros((0, len(AXES[axis_name].modes)), dtype=np.float32)
        text_embeds = self._text_embeds(axis_name)
        chunk = int(os.getenv("CLIP_PROMPT_CHUNK", "256"))
        if chunk <= 0:
            chunk = pixels.shape[0]

        out_chunks: list[torch.Tensor] = []
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        for start in range(0, pixels.shape[0], chunk):
            px = pixels[start : start + chunk]
            px = self.scorer._process(px).to(self.device).float()
            with torch.amp.autocast(device_type=device_type, enabled=False):
                img_embeds = self.scorer.model.get_image_features(pixel_values=px)
            img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            cosine = img_embeds @ text_embeds.T
            if self.reward_mode == "cosine":
                scores = cosine
            else:
                logits = cosine * (self.scorer.model.logit_scale.exp() / self.temperature)
                scores = logits if self.reward_mode == "logit" else torch.softmax(logits, dim=-1)
            out_chunks.append(scores.detach().cpu().to(torch.float32))
        return torch.cat(out_chunks, dim=0).numpy().astype(np.float32)

    def score_keys(self, images, prompts, keys: Sequence[str]) -> dict[str, np.ndarray]:
        grouped: dict[str, list[tuple[str, int]]] = {}
        for key in keys:
            axis_name, mode_idx = parse_clip_prompt_reward_key(key)
            grouped.setdefault(axis_name, []).append((key, mode_idx))

        out: dict[str, np.ndarray] = {}
        for axis_name, entries in grouped.items():
            scores = self._axis_scores(images, axis_name)
            for key, mode_idx in entries:
                out[key] = scores[:, mode_idx]
        return out

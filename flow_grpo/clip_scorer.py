# Based on https://github.com/RE-N-Y/imscore/blob/main/src/imscore/preference/model.py
# Modified for this repository; see licenses/IMSCORE_LICENSE.

import contextlib
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoImageProcessor, CLIPModel, CLIPProcessor

_DEFAULT_CLIP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "models", "clip-vit-large-patch14")
)


def _resolve_clip_path():
    candidate = os.getenv("CLIP_MODEL_PATH", _DEFAULT_CLIP_PATH)
    return candidate if os.path.exists(candidate) else "openai/clip-vit-large-patch14"


def get_size(size):
    if isinstance(size, int):
        return (size, size)
    elif "height" in size and "width" in size:
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")


def get_image_transform(processor: AutoImageProcessor):
    config = processor.to_dict()
    resize = T.Resize(get_size(config.get("size"))) if config.get("do_resize") else nn.Identity()
    crop = (
        T.CenterCrop(get_size(config.get("crop_size")))
        if config.get("do_center_crop")
        else nn.Identity()
    )
    normalise = (
        T.Normalize(mean=processor.image_mean, std=processor.image_std)
        if config.get("do_normalize")
        else nn.Identity()
    )

    return T.Compose([resize, crop, normalise])


class ClipScorer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        clip_path = _resolve_clip_path()
        local_only = os.path.isdir(clip_path)
        self.model = CLIPModel.from_pretrained(clip_path, local_files_only=local_only).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=local_only)
        self.tform = get_image_transform(self.processor.image_processor)
        self.eval()

    def _process(self, pixels):
        dtype = pixels.dtype
        pixels = self.tform(pixels)
        pixels = pixels.to(dtype=dtype)

        return pixels

    @torch.no_grad()
    def __call__(self, pixels, prompts, return_img_embedding=False):
        texts = self.processor(
            text=prompts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        pixels = self._process(pixels).to(self.device)
        # Disable outer (bf16) autocast: the CLIP text model's causal-mask builder
        # calls torch.full(..., torch.finfo(bf16).min) which overflows under bf16
        # autocast (transformers/modeling_attn_mask_utils.py). Weights are fp32, so
        # run the encoders in fp32, as in pickscore_scorer.py.
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.amp.autocast(device_type=device_type, enabled=False))
            if device_type == "cuda":
                stack.enter_context(torch.cuda.amp.autocast(enabled=False))
            image_embeds = self.model.get_image_features(pixel_values=pixels.float())
            text_embeds = self.model.get_text_features(
                input_ids=texts["input_ids"],
                attention_mask=texts["attention_mask"],
            )
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        scores = torch.sum(image_embeds * text_embeds, dim=-1)
        if return_img_embedding:
            return scores, image_embeds
        return scores

    @torch.no_grad()
    def image_similarity(self, pixels, ref_pixels):
        pixels = self._process(pixels).to(self.device)
        ref_pixels = self._process(ref_pixels).to(self.device)

        pixel_embeds = self.model.get_image_features(pixel_values=pixels)
        ref_embeds = self.model.get_image_features(pixel_values=ref_pixels)

        pixel_embeds = pixel_embeds / pixel_embeds.norm(p=2, dim=-1, keepdim=True)
        ref_embeds = ref_embeds / ref_embeds.norm(p=2, dim=-1, keepdim=True)

        sim = pixel_embeds @ ref_embeds.T
        sim = torch.diagonal(sim, 0)
        return sim

import contextlib
import os

import torch
from transformers import CLIPModel, CLIPProcessor

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_DEFAULT_PROCESSOR_PATH = os.path.join(_REPO_ROOT, "models", "CLIP-ViT-H-14-laion2B-s32B-b79K")
_DEFAULT_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "PickScore_v1")
_PATCHED_TRANSFORMERS_CAUSAL_MASK = False


def _resolve_existing_path(env_name: str, default_path: str, fallback_hf_id: str) -> str:
    candidate = os.getenv(env_name, default_path)
    return candidate if os.path.exists(candidate) else fallback_hf_id


def _safe_mask_fill_value(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return torch.finfo(dtype).min


def _patch_transformers_causal_mask() -> None:
    global _PATCHED_TRANSFORMERS_CAUSAL_MASK
    if _PATCHED_TRANSFORMERS_CAUSAL_MASK:
        return

    from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window=None,
    ):
        bsz, tgt_len = input_ids_shape
        fill_value = _safe_mask_fill_value(dtype)
        mask = torch.full((tgt_len, tgt_len), fill_value, dtype=dtype, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        if past_key_values_length > 0:
            mask = torch.cat(
                [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask],
                dim=-1,
            )

        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1
            context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)
            mask.masked_fill_(context_mask, fill_value)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    AttentionMaskConverter._make_causal_mask = staticmethod(_make_causal_mask)
    _PATCHED_TRANSFORMERS_CAUSAL_MASK = True


class PickScoreScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        _patch_transformers_causal_mask()
        processor_path = _resolve_existing_path(
            "PICKSCORE_PROCESSOR_PATH",
            _DEFAULT_PROCESSOR_PATH,
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        )
        model_path = _resolve_existing_path(
            "PICKSCORE_MODEL_PATH",
            _DEFAULT_MODEL_PATH,
            "yuvalkirstain/PickScore_v1",
        )
        self.device = device
        self.dtype = dtype
        self.processor = CLIPProcessor.from_pretrained(
            processor_path, local_files_only=os.path.isdir(processor_path)
        )
        self.model = (
            CLIPModel.from_pretrained(model_path, local_files_only=os.path.isdir(model_path))
            .eval()
            .to(device)
        )
        self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def __call__(self, prompt, images):
        # Preprocess images
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        # Preprocess text
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

        # Disable outer autocast: under bf16 autocast the CLIP causal mask
        # builder calls `torch.full(..., torch.finfo(bf16).min)` which overflows
        # at construction (transformers/modeling_attn_mask_utils.py:158).
        # Be explicit because reward computation can run inside executor threads
        # launched from mixed-precision training/eval contexts.
        if self.dtype is torch.float32:
            self.model.float()

        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.amp.autocast(device_type=device_type, enabled=False))
            if device_type == "cuda":
                stack.enter_context(torch.cuda.amp.autocast(enabled=False))
            image_inputs = {
                k: (v.float() if torch.is_floating_point(v) else v) for k, v in image_inputs.items()
            }
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

            # Calculate scores
            logit_scale = self.model.logit_scale.exp()
            scores = logit_scale * (text_embs @ image_embs.T)
            scores = scores.diag()
            # norm to 0-1
            scores = scores / 26
        return scores

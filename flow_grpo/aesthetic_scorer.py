# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py
# Modified for this repository; see licenses/APACHE-2.0.

import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CLIP_PATH = _REPO_ROOT / "models" / "clip-vit-large-patch14"
_DEFAULT_HEAD_PATH = _REPO_ROOT / "models" / "sac+logos+ava1-l14-linearMSE.pth"


def _resolve_clip_path() -> str:
    candidate = Path(os.getenv("CLIP_MODEL_PATH", str(_DEFAULT_CLIP_PATH)))
    return str(candidate) if candidate.exists() else "openai/clip-vit-large-patch14"


def _resolve_head_path() -> Path:
    candidate = Path(os.getenv("AESTHETIC_MODEL_PATH", str(_DEFAULT_HEAD_PATH)))
    if not candidate.is_file():
        raise FileNotFoundError(
            "Aesthetic predictor head not found. Run "
            "`bash scripts/download_aesthetic_head.sh` or set "
            "AESTHETIC_MODEL_PATH to sac+logos+ava1-l14-linearMSE.pth."
        )
    return candidate


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        clip_path = _resolve_clip_path()
        local_only = Path(clip_path).is_dir()
        self.clip = CLIPModel.from_pretrained(clip_path, local_files_only=local_only)
        self.processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=local_only)
        self.mlp = MLP()
        state_dict = torch.load(_resolve_head_path(), map_location="cpu", weights_only=True)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

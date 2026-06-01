"""
Vision-language embedder for the trained TASTE preference model.

Only the Qwen3-VL backbone is included here because the released
``best_pairwise`` checkpoint is trained on top of
``Qwen/Qwen3-VL-Embedding-2B``.  If you retrain on a different backbone
family (CLIP, SigLIP, etc.), add the corresponding embedder class and
update the ``VLEmbedder.from_pretrained`` dispatch below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class VLEmbedder(ABC):
    """Base class for vision-language embedding models."""

    model: torch.nn.Module
    processor: Any
    device: torch.device

    @abstractmethod
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text strings into L2-normalised embeddings."""
        ...

    @abstractmethod
    def encode_image(self, image_paths: list[str]) -> torch.Tensor:
        """Encode image file paths into L2-normalised embeddings."""
        ...

    @staticmethod
    def from_pretrained(
        model_name: str,
        device: torch.device | str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "VLEmbedder":
        from transformers import AutoConfig

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")

        if model_type == "qwen3_vl":
            return QwenVLEmbedder(model_name, device, torch_dtype, **kwargs)

        raise ValueError(
            f"Unsupported backbone '{model_type}' for {model_name}.  "
            f"This package ships only the Qwen3-VL embedder; add a new "
            f"VLEmbedder subclass if you need another family."
        )


class QwenVLEmbedder(VLEmbedder):
    """Qwen3-VL-Embedding wrapper used by the released checkpoint."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        torch_dtype: torch.dtype = torch.bfloat16,
        text_instruction: str = "Assess how well an image matches this description.",
        image_instruction: str = "Represent the image.",
        max_pixels: int | None = None,
    ):
        from transformers.models.qwen3_vl.processing_qwen3_vl import (
            Qwen3VLProcessor,
        )

        from taste_scorer.model import (
            MAX_PIXELS,
            Qwen3VLForEmbedding,
            embed,
        )

        self.device = device
        self.text_instruction = text_instruction
        self.image_instruction = image_instruction
        self.max_pixels = max_pixels or MAX_PIXELS
        self._embed_fn = embed

        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)

        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name, padding_side="right"
        )

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return self._embed_fn(
            self.model,
            self.processor,
            [{"text": t} for t in texts],
            instruction=self.text_instruction,
        )

    def encode_image(self, image_paths: list[str]) -> torch.Tensor:
        return self._embed_fn(
            self.model,
            self.processor,
            [{"image": p} for p in image_paths],
            instruction=self.image_instruction,
            max_pixels=self.max_pixels,
        )

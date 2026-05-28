"""
VLEmbedder abstraction for multiple vision-language model families.

Supports:
  - Qwen3-VL-Embedding (unified VLM encoder)
  - CLIP and CLIP-based models (OpenAI CLIP, EVA-CLIP, BGE-VL, etc.)
  - SigLIP / SigLIP2

Auto-detects model family from HuggingFace config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig


class VLEmbedder(ABC):
    """Base class for vision-language embedding models."""

    model: torch.nn.Module
    processor: Any
    device: torch.device

    @abstractmethod
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text strings into normalized embeddings."""
        ...

    @abstractmethod
    def encode_image(self, image_paths: list[str]) -> torch.Tensor:
        """Encode image file paths into normalized embeddings."""
        ...

    def default_lora_target_modules(self) -> list[str]:
        return ["q_proj", "v_proj"]

    @staticmethod
    def from_pretrained(
        model_name: str,
        device: torch.device | str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        quantization_config: Any = None,
        **kwargs,
    ) -> VLEmbedder:
        """Auto-detect model family and return the appropriate embedder."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")

        if model_type == "qwen3_vl":
            return QwenVLEmbedder(
                model_name, device, torch_dtype, quantization_config, **kwargs
            )
        elif model_type in ("clip", "clap", "jina_clip"):
            return CLIPEmbedder(
                model_name, device, torch_dtype, quantization_config, **kwargs
            )
        elif model_type in ("siglip", "siglip2"):
            return SigLIPEmbedder(
                model_name, device, torch_dtype, quantization_config, **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported model type '{model_type}' for {model_name}. "
                f"Supported: qwen3_vl, clip, siglip, siglip2"
            )


# ---------------------------------------------------------------------------
# Qwen3-VL-Embedding
# ---------------------------------------------------------------------------
class QwenVLEmbedder(VLEmbedder):

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        torch_dtype: torch.dtype = torch.bfloat16,
        quantization_config: Any = None,
        text_instruction: str = "Assess how well an image matches this description.",
        image_instruction: str = "Represent the image.",
        max_pixels: int | None = None,
    ):
        from model import (
            MAX_PIXELS,
            Qwen3VLForEmbedding,
            embed,
        )
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        self.device = device
        self.text_instruction = text_instruction
        self.image_instruction = image_instruction
        self.max_pixels = max_pixels or MAX_PIXELS
        self._embed_fn = embed

        print(f"Loading {model_name}...")
        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        if quantization_config is None:
            self.model = self.model.to(device)

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


# ---------------------------------------------------------------------------
# CLIP (and CLIP-based: BGE-VL, EVA-CLIP, etc.)
# ---------------------------------------------------------------------------
class CLIPEmbedder(VLEmbedder):

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        torch_dtype: torch.dtype = torch.bfloat16,
        quantization_config: Any = None,
        **kwargs,
    ):
        from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel

        self.device = device

        print(f"Loading CLIP model {model_name}...")
        load_kwargs: dict[str, Any] = dict(
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self.model = CLIPModel.from_pretrained(model_name, **load_kwargs)
        if quantization_config is None:
            self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.processor = None

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model.get_text_features(**inputs)
        return F.normalize(emb, p=2, dim=-1)

    def encode_image(self, image_paths: list[str]) -> torch.Tensor:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model.get_image_features(**inputs)
        return F.normalize(emb, p=2, dim=-1)


# ---------------------------------------------------------------------------
# SigLIP / SigLIP2
# ---------------------------------------------------------------------------
class SigLIPEmbedder(VLEmbedder):

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        torch_dtype: torch.dtype = torch.bfloat16,
        quantization_config: Any = None,
        **kwargs,
    ):
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        self.device = device

        print(f"Loading SigLIP model {model_name}...")
        load_kwargs: dict[str, Any] = dict(
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        if quantization_config is None:
            self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.processor = None

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model.get_text_features(**inputs)
        return F.normalize(emb, p=2, dim=-1)

    def encode_image(self, image_paths: list[str]) -> torch.Tensor:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model.get_image_features(**inputs)
        return F.normalize(emb, p=2, dim=-1)

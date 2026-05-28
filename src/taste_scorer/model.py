"""
Qwen3-VL-Embedding wrapper for TASTE.

Provides a training-compatible interface around Qwen3-VL-Embedding that can
encode text, images, or both into a shared embedding space.  Adapted from
the official ``scripts/qwen3_vl_embedding.py`` in the Hugging Face model repo.
"""

import logging
import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 16 * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
DEFAULT_MAX_LENGTH = 8192

# Reduced-resolution settings for memory-constrained GPUs (e.g. T4 16GB)
MAX_PIXELS_LOW = 512 * IMAGE_FACTOR * IMAGE_FACTOR


# ── Model ──────────────────────────────────────────────────────────────────
@dataclass
class Qwen3VLEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Thin wrapper around Qwen3VLModel that returns hidden states for pooling."""

    config: Qwen3VLConfig

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Qwen3VLEmbeddingOutput:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


# ── Helpers ────────────────────────────────────────────────────────────────
def last_token_pooling(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Extract the embedding at the last non-pad token position."""
    last_pos = attention_mask.flip(dims=[1]).argmax(dim=1)
    col = attention_mask.shape[1] - last_pos - 1
    row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[row, col]


def format_input(
    text: Optional[str] = None,
    image: Optional[str] = None,
    instruction: str = "Represent the user's input.",
    max_pixels: int = MAX_PIXELS,
) -> list[dict]:
    """Build a Qwen3-VL chat conversation suitable for the processor."""
    if instruction:
        instruction = instruction.strip()
        if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
            instruction += "."

    content: list[dict] = []
    if image:
        img_ref = image
        if not image.startswith(("http://", "https://", "oss")):
            img_ref = "file://" + os.path.abspath(image)
        content.append(
            {
                "type": "image",
                "image": img_ref,
                "min_pixels": MIN_PIXELS,
                "max_pixels": max_pixels,
            }
        )
    if text:
        content.append({"type": "text", "text": text})
    if not content:
        content.append({"type": "text", "text": ""})

    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]


def prepare_inputs(
    conversations: list[list[dict]],
    processor: Qwen3VLProcessor,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, torch.Tensor]:
    """Tokenise + process vision info for a batch of conversations.

    Important: ``process_vision_info``'s signature has shifted across
    ``qwen_vl_utils`` releases.  We try the newest signature first and fall
    back to older / minimal signatures.  We deliberately do **not** silence
    failures with a bare ``except`` — when the vision pipeline breaks the
    processor builds text-only inputs, the model emits the same embedding
    for every image, and contrastive training degenerates to ``log(2)``.
    """
    from qwen_vl_utils.vision_process import process_vision_info

    text = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )

    has_image = any(
        any(c.get("type") == "image" for c in conv[-1].get("content", []))
        for conv in conversations
    )

    images = None
    video_inputs = None
    video_kwargs: dict[str, Any] = {"do_sample_frames": False}

    # Try newest signature first, then peel off args until something works.
    candidates = [
        # newer (return_video_metadata + image_patch_size)
        dict(image_patch_size=16, return_video_metadata=True, return_video_kwargs=True),
        # mid (no image_patch_size)
        dict(return_video_metadata=True, return_video_kwargs=True),
        # older (only return_video_kwargs)
        dict(return_video_kwargs=True),
        # minimal
        dict(),
    ]
    last_err: Exception | None = None
    for kwargs in candidates:
        try:
            ret = process_vision_info(conversations, **kwargs)
        except TypeError as e:  # signature mismatch only
            last_err = e
            continue
        # Normalise the return shape: older versions return (images, videos),
        # newer ones return (images, videos, video_kwargs).
        if isinstance(ret, tuple) and len(ret) == 3:
            images, video_inputs, video_kwargs = ret
        elif isinstance(ret, tuple) and len(ret) == 2:
            images, video_inputs = ret
        else:
            images = ret
        break
    else:
        # All candidates raised TypeError → installed qwen_vl_utils has a
        # signature we don't recognise.  Fail loudly rather than emit
        # text-only embeddings.
        raise RuntimeError(
            "process_vision_info refused every candidate signature; the "
            "installed qwen_vl_utils may be incompatible.  Last TypeError: "
            f"{last_err}"
        )

    if has_image and images is None:
        raise RuntimeError(
            "Conversations contain image content but process_vision_info "
            "returned no images.  Refusing to silently emit a text-only "
            "embedding (this would degenerate BT training to log(2))."
        )

    videos, video_metadata = None, None
    if video_inputs is not None:
        videos, video_metadata = zip(*video_inputs)
        videos, video_metadata = list(videos), list(video_metadata)

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        truncation=True,
        max_length=max_length,
        padding=True,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )
    return inputs


def embed(
    model: Qwen3VLForEmbedding,
    processor: Qwen3VLProcessor,
    items: list[dict[str, Any]],
    instruction: str = "Represent the user's input.",
    normalize: bool = True,
    max_length: int = DEFAULT_MAX_LENGTH,
    max_pixels: int = MAX_PIXELS,
) -> torch.Tensor:
    """Encode a list of ``{"text": ..., "image": ...}`` dicts into embeddings.

    Works in both training (grad) and inference (no-grad) contexts — the
    caller controls ``torch.no_grad()`` or ``model.eval()``.
    """
    conversations = [
        format_input(
            text=item.get("text"),
            image=item.get("image"),
            instruction=instruction,
            max_pixels=max_pixels,
        )
        for item in items
    ]

    inputs = prepare_inputs(conversations, processor, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs)
    embeddings = last_token_pooling(outputs.last_hidden_state, outputs.attention_mask)

    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings

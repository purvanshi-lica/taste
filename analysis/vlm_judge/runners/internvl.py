"""Runner for the InternVL3.x family.

The InternVL series ships its own custom model code with a
`model.chat()` API.  The HuggingFace `AutoProcessor` path has had
persistent compatibility issues across transformers versions, so
this runner uses the official API from the model card:

    model = AutoModel.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    response = model.chat(tokenizer, pixel_values, question, gen_cfg)

Image preprocessing follows InternVL's own example: each image is
resized to 448x448, ImageNet-normalised, and stacked as a tensor.
`<image>` placeholders in the question are expanded by `model.chat`
based on `num_patches_list`.
"""
from __future__ import annotations

from typing import List, Optional

from PIL import Image

from .base import Runner


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448):
    """InternVL's official preprocessing pipeline."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(
            lambda img: img.convert("RGB") if img.mode != "RGB" else img
        ),
        T.Resize(
            (input_size, input_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class InternVLRunner(Runner):
    """InternVL3.x family runner using the checkpoint-shipped
    `model.chat()` API."""

    name: str = "internvl"

    def __init__(self, model_id: str, name: Optional[str] = None,
                 max_new_tokens: int = 256,
                 num_samples: int = 1,
                 input_size: int = 448,
                 image_layout: str = "labeled"):
        super().__init__(model_id=model_id, name=name)
        self.max_new_tokens = max_new_tokens
        self.num_samples = max(1, int(num_samples))
        self.input_size = input_size
        if image_layout not in ("labeled", "unlabeled"):
            raise ValueError(
                f"image_layout must be 'labeled' or 'unlabeled' "
                f"(got {image_layout!r})"
            )
        self.image_layout = image_layout
        self._tokenizer = None
        self._model = None
        self._transform = None

    def warmup(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False,
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().cuda()
        self._transform = _build_transform(self.input_size)

    def _judge_pair_multi(self, prompt_text: str,
                          image_a: Image.Image,
                          image_b: Image.Image) -> List[str]:
        import torch

        if self._model is None:
            self.warmup()

        pv_a = self._transform(image_a).unsqueeze(0)
        pv_b = self._transform(image_b).unsqueeze(0)
        pixel_values = torch.cat([pv_a, pv_b], dim=0).to(
            torch.bfloat16
        ).cuda()
        num_patches_list = [1, 1]

        if self.image_layout == "labeled":
            question = (
                f"Image A: <image>\nImage B: <image>\n\n{prompt_text}"
            )
        else:
            question = f"<image>\n<image>\n{prompt_text}"

        gen_cfg = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        outputs = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                response = self._model.chat(
                    self._tokenizer, pixel_values, question, gen_cfg,
                    num_patches_list=num_patches_list,
                )
                outputs.append(response)
        return outputs

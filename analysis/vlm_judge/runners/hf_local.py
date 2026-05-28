"""HuggingFace transformers-based runners for local VLMs.

Two implementations are provided:

1. `HFTransformersRunner`: generic transformers-based driver that uses
   the model's auto-processor and chat template.  Works for most
   Qwen-VL-family, InternVL, Llava, and Gemma-3-VL models.  Slower
   than vLLM but simpler and reliable.

2. `VLLMRunner`: uses vLLM for throughput.  Recommended for full runs
   over the 4320-task per-model evaluation.  Falls back to
   `HFTransformersRunner` if a model is not yet supported by vLLM.

Models are referenced by their HuggingFace model id (e.g.
`Qwen/Qwen2.5-VL-7B-Instruct`).  Sub-classes can override
`build_messages` for chat-template idiosyncrasies.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from .base import Runner


# Per-model recommended sampling settings, as published by each lab.
# Keyed on the trailing segment of the HF model id.  Used when
# num_samples > 1 to mimic multiple "raters" with the model's own
# calibrated diversity rather than an ad-hoc temperature.
RECOMMENDED_SAMPLING = {
    # Qwen3-VL Instruct family (Qwen team's official inference example)
    "Qwen3-VL-4B-Instruct":  dict(temperature=0.7, top_p=0.8,  top_k=20),
    "Qwen3-VL-8B-Instruct":  dict(temperature=0.7, top_p=0.8,  top_k=20),
    "Qwen3-VL-32B-Instruct": dict(temperature=0.7, top_p=0.8,  top_k=20),
    # Qwen3-VL Thinking — Qwen recommends higher diversity
    "Qwen3-VL-4B-Thinking":  dict(temperature=1.0, top_p=0.95, top_k=40),
    "Qwen3-VL-8B-Thinking":  dict(temperature=1.0, top_p=0.95, top_k=40),
    # InternVL3.5 (OpenGVLab demo defaults)
    "InternVL3_5-8B":        dict(temperature=0.7, top_p=0.95, top_k=20),
    "InternVL3_5-14B":       dict(temperature=0.7, top_p=0.95, top_k=20),
    "InternVL3_5-38B":       dict(temperature=0.7, top_p=0.95, top_k=20),
    # Gemma-3 instruct (Google's recommended sampling)
    "gemma-3-12b-it":        dict(temperature=1.0, top_p=0.95, top_k=64),
    "gemma-3-27b-it":        dict(temperature=1.0, top_p=0.95, top_k=64),
    # Kimi-VL (Moonshot recommendation)
    "Kimi-VL-A3B-Instruct":  dict(temperature=0.6, top_p=0.95, top_k=20),
}


def _resolve_sampling(model_id: str,
                      override_temp: Optional[float] = None,
                      override_top_p: Optional[float] = None,
                      override_top_k: Optional[int] = None) -> dict:
    """Look up recommended sampling for a model id, allowing CLI overrides."""
    suffix = model_id.split("/")[-1]
    base = RECOMMENDED_SAMPLING.get(
        suffix, dict(temperature=0.7, top_p=0.95, top_k=50)
    )
    out = dict(base)
    if override_temp is not None:
        out["temperature"] = override_temp
    if override_top_p is not None:
        out["top_p"] = override_top_p
    if override_top_k is not None:
        out["top_k"] = override_top_k
    return out


class HFTransformersRunner(Runner):
    """Transformers-based runner. Lazy-loads the model on first warmup."""

    def __init__(self, model_id: str, name: Optional[str] = None,
                 max_new_tokens: int = 8, dtype: str = "bfloat16",
                 device_map: str = "auto",
                 num_samples: int = 1,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 image_layout: str = "labeled",
                 fewshot_example: Optional[dict] = None):
        super().__init__(model_id=model_id, name=name)
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.device_map = device_map
        self.num_samples = max(1, int(num_samples))
        # `image_layout` controls how the two images are presented:
        #   "labeled"  — "Image A:" <img1> "Image B:" <img2> <text>
        #                Field-standard pairwise format (MJ-Bench,
        #                MM-Vet, GPT-4V eval papers).  The label
        #                tokens are adjacent to the relevant vision
        #                tokens so the model has a direct attention
        #                path between the answer character and the
        #                correct image.
        #   "unlabeled" — <img1><img2><text>  (legacy harness format;
        #                the binding "A=first" must be inferred from
        #                the text alone, which leaks position priors).
        if image_layout not in ("labeled", "unlabeled"):
            raise ValueError(
                f"image_layout must be 'labeled' or 'unlabeled' "
                f"(got {image_layout!r})"
            )
        self.image_layout = image_layout
        # `fewshot_example` is an optional dict with keys
        #   "image_a" : PIL.Image - the example's image A (the loser)
        #   "image_b" : PIL.Image - the example's image B (the winner)
        #   "reasoning": str       - the example's reasoning trace,
        #                            ending with "Final answer: B"
        # When set, every judge_pair call includes this 2-image example
        # before the actual pair, demonstrating the expected reasoning.
        self.fewshot_example = fewshot_example
        self._sampling = _resolve_sampling(
            model_id, override_temp=temperature,
            override_top_p=top_p, override_top_k=top_k,
        )
        self._model = None
        self._processor = None

    def warmup(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor

        # Compatibility shims for community model files that target
        # older transformers APIs.  Kimi-VL's `modeling_kimi_vl.py`
        # imports `is_torch_fx_available`, which was removed in
        # transformers 5.x.  Provide a safe stub (returns False; we
        # don't use torch.fx at inference).
        import transformers.utils.import_utils as _iu
        if not hasattr(_iu, "is_torch_fx_available"):
            _iu.is_torch_fx_available = lambda: False

        # Different VLM families register themselves under different
        # AutoXxx classes.  Try the most specific class first, fall
        # back to more general ones.  This sequence covers:
        #   - AutoModelForImageTextToText: Qwen3-VL, Gemma-3, InternVL3,
        #     Pixtral, LLaVA-NeXT, etc. (transformers >= 4.45)
        #   - AutoModelForVision2Seq: legacy alias for the same
        #   - AutoModel + trust_remote_code: Kimi-VL and other models
        #     whose auto_map points at AutoModel
        auto_candidates = []
        try:
            from transformers import AutoModelForImageTextToText
            auto_candidates.append(AutoModelForImageTextToText)
        except ImportError:
            pass
        try:
            from transformers import AutoModelForVision2Seq
            auto_candidates.append(AutoModelForVision2Seq)
        except ImportError:
            pass
        from transformers import AutoModel
        auto_candidates.append(AutoModel)

        torch_dtype = getattr(torch, self.dtype, torch.bfloat16)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True,
        )

        last_err = None
        for AutoVL in auto_candidates:
            try:
                # Modern `dtype=` kwarg first
                try:
                    self._model = AutoVL.from_pretrained(
                        self.model_id,
                        dtype=torch_dtype,
                        device_map=self.device_map,
                        trust_remote_code=True,
                    )
                except TypeError:
                    self._model = AutoVL.from_pretrained(
                        self.model_id,
                        torch_dtype=torch_dtype,
                        device_map=self.device_map,
                        trust_remote_code=True,
                    )
                break
            except (ValueError, KeyError) as e:
                # ValueError when the config doesn't match this Auto-class;
                # KeyError when auto_map lookup misses.  Try the next.
                last_err = e
                continue
        else:
            raise RuntimeError(
                f"Could not load {self.model_id} with any of "
                f"{[a.__name__ for a in auto_candidates]}: {last_err}"
            )

        self._model.eval()

    def _build_messages(self, prompt_text: str,
                        image_a: Image.Image,
                        image_b: Image.Image) -> List[dict]:
        if self.fewshot_example is not None:
            ex = self.fewshot_example
            content = [
                {"type": "text", "text":
                    "First, here is one example of the format and reasoning "
                    "style we expect, on a different brief.  This example "
                    "is for format guidance only; it does not refer to the "
                    "actual images you will need to evaluate.\n\n"
                    "EXAMPLE Image A:"},
                {"type": "image", "image": ex["image_a"]},
                {"type": "text", "text": "EXAMPLE Image B:"},
                {"type": "image", "image": ex["image_b"]},
                {"type": "text", "text":
                    f"EXAMPLE reasoning:\n{ex['reasoning']}\n\n"
                    "Now evaluate the actual two images for the brief and "
                    "criterion below.\n\n"
                    "Image A:"},
                {"type": "image", "image": image_a},
                {"type": "text", "text": "Image B:"},
                {"type": "image", "image": image_b},
                {"type": "text", "text": prompt_text},
            ]
        elif self.image_layout == "labeled":
            content = [
                {"type": "text", "text": "Image A:"},
                {"type": "image", "image": image_a},
                {"type": "text", "text": "Image B:"},
                {"type": "image", "image": image_b},
                {"type": "text", "text": prompt_text},
            ]
        else:  # unlabeled (legacy)
            content = [
                {"type": "image", "image": image_a},
                {"type": "image", "image": image_b},
                {"type": "text", "text": prompt_text},
            ]
        return [{"role": "user", "content": content}]

    def _judge_pair_multi(self, prompt_text: str,
                          image_a: Image.Image,
                          image_b: Image.Image) -> List[str]:
        import torch

        if self._model is None:
            self.warmup()

        messages = self._build_messages(prompt_text, image_a, image_b)
        # Prefer the fused chat-template path (Qwen3-VL, Gemma 3).
        # Fall back to a two-step path for processors that reject the
        # fused call (Kimi-VL etc.): apply_chat_template → text, then
        # processor(images=..., text=..., ...) → tokenised inputs.
        try:
            inputs = self._processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt",
            )
        except (TypeError, ValueError):
            text = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            imgs = []
            for msg in messages:
                for c in msg.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "image":
                        imgs.append(c["image"])
            inputs = self._processor(
                images=imgs, text=text, return_tensors="pt",
                padding=True, truncation=True,
            )
        inputs = {
            k: v.to(self._model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens)
        if self.num_samples == 1:
            gen_kwargs.update(do_sample=False, temperature=None,
                              top_p=None, top_k=None)
        else:
            gen_kwargs.update(
                do_sample=True,
                temperature=self._sampling["temperature"],
                top_p=self._sampling["top_p"],
                top_k=self._sampling["top_k"],
                num_return_sequences=self.num_samples,
            )
        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        # output is shape [num_samples, seq_len]
        texts = []
        for i in range(output.shape[0]):
            gen = output[i, input_len:]
            texts.append(
                self._processor.tokenizer.decode(
                    gen, skip_special_tokens=True,
                )
            )
        return texts


class VLLMRunner(Runner):
    """vLLM-based runner.  Faster for long runs.  Requires the model
    to be supported by vLLM's vision interface."""

    def __init__(self, model_id: str, name: Optional[str] = None,
                 max_new_tokens: int = 8, dtype: str = "bfloat16",
                 max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.85,
                 num_samples: int = 1,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None):
        super().__init__(model_id=model_id, name=name)
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.num_samples = max(1, int(num_samples))
        self._recommended = _resolve_sampling(
            model_id, override_temp=temperature,
            override_top_p=top_p, override_top_k=top_k,
        )
        self._llm = None
        self._sampling = None

    def warmup(self):
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams

        self._llm = LLM(
            model=self.model_id,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 2},
        )
        if self.num_samples == 1:
            self._sampling = SamplingParams(
                max_tokens=self.max_new_tokens, temperature=0.0,
            )
        else:
            self._sampling = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self._recommended["temperature"],
                top_p=self._recommended["top_p"],
                top_k=self._recommended["top_k"],
                n=self.num_samples,
            )

    def _judge_pair_multi(self, prompt_text: str,
                          image_a: Image.Image,
                          image_b: Image.Image) -> List[str]:
        if self._llm is None:
            self.warmup()
        # vLLM accepts a list of dicts with 'prompt' and 'multi_modal_data'.
        # We use the chat-template via the processor for portability.
        from vllm import TokensPrompt
        from transformers import AutoProcessor

        # Lazy-init processor only for chat-template formatting
        if not hasattr(self, "_processor"):
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True,
            )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }]
        text_prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        result = self._llm.generate(
            [{
                "prompt": text_prompt,
                "multi_modal_data": {"image": [image_a, image_b]},
            }],
            self._sampling,
        )
        # vLLM returns SamplingParams.n outputs per prompt
        return [o.text for o in result[0].outputs]

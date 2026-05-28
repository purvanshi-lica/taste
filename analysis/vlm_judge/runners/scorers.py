"""Pairwise scoring runners for non-generative preference / aesthetic
models.

Three scorers, all sharing the `PairScorerRunner` base which scores
both images and returns the argmax verdict (with a tie band):

  - LAIONAestheticV2Runner: single-image aesthetic score (no text).
    Backbone CLIP-L/14 + sac+logos+ava1-l14-linearMSE head; trained
    on AVA, SAC, LOGO datasets.  Approximates "is this a polished
    image" without considering brief fidelity.

  - HPSv2_1Runner: HPSv2.1 (latest checkpoint).  Text-conditioned
    preference score; CLIP-H/14 backbone fine-tuned on the Human
    Preference Dataset v2.1.

  - PickScoreRunner: PickScore-v1.  Text-conditioned; CLIP-H/14
    backbone trained on the Pick-a-Pic preference dataset.

These models are deterministic given (text, image), so num_samples is
fixed at 1.  The pair verdict is `argmax(score_a, score_b)` with a
small tie band.

Prompt format: the brief is reduced to a noun-phrase via
`extract_short_prompt`, matching the T2I-style prompts these models
were trained on (avg 10-30 tokens).
"""

from __future__ import annotations

import re
from typing import List, Optional

from PIL import Image

from .base import Runner


def extract_short_prompt(brief: str, max_chars: int = 220) -> str:
    """Pull a noun-phrase summary out of a TASTE brief.

    Matches the input distribution HPSv2 / PickScore were trained on:
      1. Pull the `User Intent` paragraph (or the first paragraph if
         the marker is missing).
      2. Take the first sentence.
      3. Strip a leading imperative verb (`Create / Design / Make /
         Generate / Produce / Compose / Develop / Build`) so the
         result reads as a noun phrase, not an instruction.
      4. If still too long, fall back to the first comma-clause.
    """
    m = re.search(r"User Intent:\s*\n+(.+?)(?:\n\n|\Z)", brief, re.DOTALL)
    intent = m.group(1).strip() if m else brief.split("\n\n", 1)[0].strip()
    first_sent = intent.split(".")[0].strip()
    first_sent = re.sub(
        r"^(Create|Design|Make|Generate|Produce|Compose|Develop|Build)"
        r"\s+(?:an?\s+)?",
        "", first_sent, flags=re.IGNORECASE,
    )
    if len(first_sent) > max_chars:
        first_sent = first_sent.split(",")[0].strip()
    if first_sent:
        first_sent = first_sent[0].lower() + first_sent[1:]
    return first_sent


class PairScorerRunner(Runner):
    """Base for non-generative pair scorers.  Subclasses implement
    `_score(prompt, image) -> float`."""

    requires_text_prompt = True
    tie_eps = 1e-3

    def __init__(self, model_id: str, name: Optional[str] = None,
                 device: str = "cuda", dtype: str = "float32"):
        super().__init__(model_id=model_id, name=name)
        self.device = device
        self.dtype = dtype
        self.num_samples = 1

    def _score(self, prompt: str, image: Image.Image) -> float:
        raise NotImplementedError

    def _judge_pair_multi(self, prompt_text: str,
                          image_a: Image.Image,
                          image_b: Image.Image) -> List[str]:
        score_a = self._score(prompt_text, image_a)
        score_b = self._score(prompt_text, image_b)
        if abs(score_a - score_b) < self.tie_eps:
            verdict = "tie"
        elif score_a > score_b:
            verdict = "A"
        else:
            verdict = "B"
        return [
            f"score_a={score_a:.6f}, score_b={score_b:.6f} -> {verdict}"
        ]


class LAIONAestheticV2Runner(PairScorerRunner):
    """LAION aesthetic V2 (`sac+logos+ava1-l14-linearMSE`).

    Architecture: stock CLIP-ViT-L/14 image encoder (joint-space
    768-dim embedding, L2-normalised) → MLP head with 5 linear layers
    and dropouts in between.  The head weights come from the canonical
    GitHub release; the backbone is stock transformers.
    """

    requires_text_prompt = False
    # Output scale ~1-10; observed margins are 0.04-0.5 across models,
    # so a tie band of 0.005 keeps only essentially-identical scores
    # as ties.  Tighter than 0.05 (which collapses meaningful gaps).
    tie_eps = 0.005

    HEAD_URL = (
        "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
        "raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
    )

    def __init__(self, name: Optional[str] = None,
                 device: str = "cuda", dtype: str = "float32"):
        super().__init__(
            model_id="openai/clip-vit-large-patch14",
            name=name or "LAION-Aesthetic-V2",
            device=device, dtype=dtype,
        )
        self._clip = None
        self._processor = None
        self._head = None

    def warmup(self):
        if self._clip is not None:
            return
        import os
        import urllib.request
        import torch
        from torch import nn
        from transformers import (
            CLIPVisionModelWithProjection, CLIPImageProcessor,
        )

        self._processor = CLIPImageProcessor.from_pretrained(self.model_id)
        self._clip = CLIPVisionModelWithProjection.from_pretrained(
            self.model_id,
        ).to(self.device).eval()

        # Build the canonical "linearMSE" head: an nn.Sequential named
        # `layers` so the .pth state dict (keys "layers.0.weight" ...)
        # loads cleanly.
        class AestheticHead(nn.Module):
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

            def forward(self, x):
                return self.layers(x)

        cache_dir = os.path.expanduser("~/.cache/contra_vlm_judge")
        os.makedirs(cache_dir, exist_ok=True)
        head_path = os.path.join(
            cache_dir, "sac+logos+ava1-l14-linearMSE.pth",
        )
        if not os.path.exists(head_path):
            urllib.request.urlretrieve(self.HEAD_URL, head_path)
        state = torch.load(head_path, map_location="cpu", weights_only=False)
        head = AestheticHead()
        head.load_state_dict(state)
        self._head = head.to(self.device).eval()

    def _score(self, prompt: str, image: Image.Image) -> float:
        import torch
        if self._clip is None:
            self.warmup()
        inp = self._processor(images=image, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.no_grad():
            out = self._clip(**inp)
            emb = out.image_embeds  # (1, 768) joint-space CLIP
            emb = emb / emb.norm(dim=-1, keepdim=True)
            score = self._head(emb)  # (1, 1)
        return float(score.cpu().squeeze())


class HPSv2_1Runner(PairScorerRunner):
    """HPSv2.1 — text-conditioned preference score.

    Backbone CLIP-H/14 (open_clip), fine-tuned on Human Preference
    Dataset v2.1.  Latest checkpoint at xswu/HPSv2.
    """

    tie_eps = 1e-4

    def __init__(self, name: Optional[str] = None,
                 device: str = "cuda", dtype: str = "float32"):
        super().__init__(
            model_id="xswu/HPSv2",
            name=name or "HPSv2.1",
            device=device, dtype=dtype,
        )
        self._model = None
        self._tokenizer = None
        self._preprocess = None

    def warmup(self):
        if self._model is not None:
            return
        import torch
        import open_clip
        from huggingface_hub import hf_hub_download

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=None, precision=self.dtype,
        )
        ckpt_path = hf_hub_download(
            repo_id="xswu/HPSv2",
            filename="HPS_v2.1_compressed.pt",
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 50 or len(unexpected) > 50:
            print(
                f"  [HPSv2.1] warning: {len(missing)} missing, "
                f"{len(unexpected)} unexpected state-dict keys"
            )
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-H-14")

    def _score(self, prompt: str, image: Image.Image) -> float:
        import torch
        if self._model is None:
            self.warmup()
        img_t = self._preprocess(image).unsqueeze(0).to(self.device)
        txt_t = self._tokenizer([prompt]).to(self.device)
        with torch.no_grad():
            img_emb = self._model.encode_image(img_t)
            txt_emb = self._model.encode_text(txt_t)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            score = (img_emb * txt_emb).sum(dim=-1)
        return float(score.cpu().squeeze())


class PickScoreRunner(PairScorerRunner):
    """PickScore-v1 — text-conditioned preference score.

    Loads via transformers; checkpoint at yuvalkirstain/PickScore_v1.
    Score = logit_scale * cosine(image_emb, text_emb), as in the
    official PickScore inference code.
    """

    tie_eps = 1e-4

    def __init__(self, name: Optional[str] = None,
                 device: str = "cuda", dtype: str = "float32"):
        super().__init__(
            model_id="yuvalkirstain/PickScore_v1",
            name=name or "PickScore-v1",
            device=device, dtype=dtype,
        )
        self._model = None
        self._processor = None

    def warmup(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(
            self.device,
        ).eval()

    def _score(self, prompt: str, image: Image.Image) -> float:
        import torch
        if self._model is None:
            self.warmup()
        # Combined forward: CLIPModel returns CLIPOutput with already-
        # projected image_embeds / text_embeds.  In transformers >=5.3
        # the per-encoder accessors return BaseModelOutputWithPooling,
        # so the joint forward is the cleaner path.
        inp = self._processor(
            images=image, text=prompt, return_tensors="pt",
            padding=True, truncation=True, max_length=77,
        ).to(self.device)
        with torch.no_grad():
            out = self._model(**inp)
            img_emb = out.image_embeds
            txt_emb = out.text_embeds
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            logit_scale = self._model.logit_scale.exp()
            score = logit_scale * (img_emb * txt_emb).sum(dim=-1)
        return float(score.cpu().squeeze())

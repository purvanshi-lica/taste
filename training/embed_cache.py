"""
Pre-compute and cache (text, image) embeddings from a frozen VLM.

When the VLM backbone is frozen — which is the default in this project now
that scoring lives in per-dimension MLP heads — its outputs are constant
across epochs.  Pre-computing them once is a 50-100× speedup over re-running
the encoder on every minibatch.

The cache is keyed by:

* image:    the absolute path to the image file
* text:     the (instruction, prompt) pair, hashed deterministically

Embeddings are kept on CPU as fp32 tensors and lifted to the training device
on demand.  Each tensor is small (a few KB at most), so even a 5k-prompt
dataset fits comfortably in RAM.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _hash_text(text: str, instruction: str) -> str:
    h = hashlib.sha1()
    h.update(instruction.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class EmbeddingCache:
    """Thin in-memory cache around a :class:`VLEmbedder` for frozen-backbone training.

    Parameters
    ----------
    embedder:
        Any ``VLEmbedder`` instance.  Its ``encode_text`` / ``encode_image``
        will be called in (no-grad) batched fashion to populate the cache.
    text_batch_size, image_batch_size:
        Mini-batch sizes for the one-shot embedding pass.  Lower these if
        you OOM during precompute (e.g. on T4 / Colab).
    """

    def __init__(
        self,
        embedder,
        text_batch_size: int = 16,
        image_batch_size: int = 8,
    ):
        self.embedder = embedder
        self.text_batch_size = text_batch_size
        self.image_batch_size = image_batch_size
        self._text_cache: dict[str, torch.Tensor] = {}
        self._image_cache: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------
    @torch.no_grad()
    def precompute_texts(self, texts: Iterable[str], instruction: str | None = None) -> None:
        """Embed every unique ``text`` once and cache on CPU."""
        instr = instruction or getattr(self.embedder, "text_instruction", "")
        unique = []
        for t in texts:
            key = _hash_text(t, instr)
            if key not in self._text_cache:
                unique.append((key, t))
        if not unique:
            return

        logger.info("Pre-computing %d unique text embeddings", len(unique))
        for i in tqdm(range(0, len(unique), self.text_batch_size), desc="Embedding texts"):
            chunk = unique[i : i + self.text_batch_size]
            keys = [k for k, _ in chunk]
            batch = [t for _, t in chunk]
            embs = self.embedder.encode_text(batch).detach().to("cpu", dtype=torch.float32)
            for k, e in zip(keys, embs.unbind(0)):
                self._text_cache[k] = e

    @torch.no_grad()
    def precompute_images(self, image_paths: Iterable[str]) -> None:
        """Embed every unique ``image_path`` once and cache on CPU."""
        unique_paths: list[str] = []
        for p in image_paths:
            ap = str(Path(p).resolve())
            if ap not in self._image_cache and ap not in unique_paths:
                unique_paths.append(ap)
        if not unique_paths:
            return

        logger.info("Pre-computing %d unique image embeddings", len(unique_paths))
        for i in tqdm(range(0, len(unique_paths), self.image_batch_size), desc="Embedding images"):
            chunk = unique_paths[i : i + self.image_batch_size]
            embs = self.embedder.encode_image(chunk).detach().to("cpu", dtype=torch.float32)
            for p, e in zip(chunk, embs.unbind(0)):
                self._image_cache[p] = e

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def text(
        self,
        texts: list[str],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        instruction: str | None = None,
    ) -> torch.Tensor:
        instr = instruction or getattr(self.embedder, "text_instruction", "")
        keys = [_hash_text(t, instr) for t in texts]
        missing = [t for t, k in zip(texts, keys) if k not in self._text_cache]
        if missing:
            self.precompute_texts(missing, instruction=instr)
        return torch.stack([self._text_cache[k] for k in keys]).to(device=device, dtype=dtype)

    def image(
        self,
        image_paths: list[str],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        keys = [str(Path(p).resolve()) for p in image_paths]
        missing = [k for k in keys if k not in self._image_cache]
        if missing:
            self.precompute_images(missing)
        return torch.stack([self._image_cache[k] for k in keys]).to(device=device, dtype=dtype)

    @property
    def emb_dim(self) -> int | None:
        for cache in (self._text_cache, self._image_cache):
            for v in cache.values():
                return int(v.shape[-1])
        return None

    def __len__(self) -> int:
        return len(self._text_cache) + len(self._image_cache)

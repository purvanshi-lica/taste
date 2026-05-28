"""Abstract Runner interface for VLM-as-judge."""

import io
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from PIL import Image


@dataclass
class JudgeResult:
    # Aggregated verdict.  When `verdicts` has multiple entries it is
    # the majority; "tie" if A and B counts are equal and non-zero;
    # "unknown" if all samples failed to parse.
    verdict: str
    # First sample's raw text (kept for backward-compat with existing
    # single-sample results jsonls).
    raw_response: str
    latency_s: float
    ok: bool
    error: Optional[str] = None
    # Multi-sample fields (only populated when num_samples > 1).
    verdicts: Optional[List[str]] = None
    raw_responses: Optional[List[str]] = None


class Runner:
    """Override `_judge_pair` for each VLM family."""
    name: str = "base"

    def __init__(self, model_id: str, name: Optional[str] = None,
                 timeout_s: float = 60.0):
        self.model_id = model_id
        self.name = name or model_id.replace("/", "_")
        self.timeout_s = timeout_s

    # Image fetch with caching at the harness level (callers pass URL)
    @staticmethod
    def fetch_image(url: str, timeout: float = 30.0) -> Image.Image:
        """Fetch a remote image into a PIL.Image. Tries Cloudinary's
        bandwidth-friendly transform first."""
        if "/image/upload/" in url and not any(
            url.split("/image/upload/")[-1].split("/")[0].startswith(p)
            for p in ("w_", "c_", "f_", "q_")
        ):
            url = url.replace(
                "/image/upload/", "/image/upload/w_768,c_fit,q_auto,f_jpg/"
            )
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return img

    @staticmethod
    def parse_verdict(text: str) -> str:
        """Parse the model's reply into 'A', 'B', or 'unknown'.

        Three-stage extraction matching DistortBench (Goyal et al.
        2026):
          1. JSON code block:  ```json {"answer": "A"} ```
          2. Bare JSON object: {"answer": "A"}
          3. Regex fallback over commitment phrases, then a last-
             standalone-A/B-token fallback.

        Stages 1-2 are robust against arbitrary surrounding reasoning
        text and are field-standard for VLM-as-judge parsing.  Stage 3
        catches free-text answers from prompts that don't enforce JSON
        and from thinking-mode rambles that never reach the JSON.
        """
        if not text:
            return "unknown"
        import json
        import re

        t = text
        # Strip thinking blocks before any extraction.
        t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL | re.IGNORECASE)
        t = re.sub(r"<think>.*$", " ", t, flags=re.DOTALL | re.IGNORECASE)

        # Stage 1: JSON in a fenced code block (```json ... ``` or ``` ... ```)
        for m in re.finditer(
            r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL
        ):
            try:
                obj = json.loads(m.group(1))
                ans = str(obj.get("answer", "")).strip().upper()
                if ans in ("A", "B"):
                    return ans
            except (json.JSONDecodeError, AttributeError):
                pass

        # Stage 2: bare JSON object containing "answer": "A" or "B".
        # Match without requiring strict JSON validity (handles smart
        # quotes, trailing commas, etc.).
        for m in re.finditer(
            r'\{[^{}]*"answer"\s*:\s*[\"“”\']?([AB])[\"“”\']?[^{}]*\}',
            t, flags=re.IGNORECASE,
        ):
            return m.group(1).upper()

        # Stage 3: explicit commitment phrasing.  Take LAST match so a
        # model that reconsiders gets credit for its final commitment.
        patterns = [
            # "Final answer: A", "Answer is B", "Verdict: A"
            r"final\s+answer\s*(?:is|:|=)?\s*[\"']?\b([AB])\b",
            r"answer\s+is\s*[\"']?\b([AB])\b",
            r"answer\s*[:\-=]\s*[\"']?\b([AB])\b",
            r"verdict\s*[:\-=]?\s*[\"']?\b([AB])\b",
            # "I think the answer is A", "I think it is B", "I pick A"
            r"i\s+think\s+(?:the\s+)?(?:answer|verdict|response)\s+is\s*[\"']?\b([AB])\b",
            r"i\s+think\s+it'?s\s*[\"']?\b([AB])\b",
            # "Image A is better/preferred/stronger" (or "Option A is better")
            r"(?:image|option)\s*[\"']?\b([AB])\b[\"']?\s+is\s+(?:better|preferred|stronger|the\s+(?:better|stronger|right))",
            # "Therefore A" / "So A" / "Hence B" -- conclusion words
            r"(?:therefore|so|hence|thus|conclusion)\s*[,:]?\s*[\"']?\b([AB])\b\s*[\.\!\,]?",
            # "B is the answer" / "B is the better choice"
            r"\b([AB])\b\s+is\s+(?:the\s+(?:answer|better|right|stronger|preferred)|better)",
            # "I (would) choose / pick / prefer A"
            r"i\s+(?:would\s+)?(?:choose|pick|prefer|select)\s+(?:image\s+)?[\"']?\b([AB])\b",
            r"\b(?:choose|pick|prefer|select)\s+(?:image\s+)?[\"']?\b([AB])\b",
            # "the answer is A", "my answer is B"
            r"(?:the|my)\s+answer\s+is\s*[\"']?\b([AB])\b",
        ]
        last_match = None
        last_pos = -1
        for pat in patterns:
            for m in re.finditer(pat, t, flags=re.IGNORECASE):
                if m.start() > last_pos:
                    last_pos = m.start()
                    last_match = m.group(1).upper()
        if last_match:
            return last_match

        # 3. last standalone A/B token at word-boundary.
        tokens = re.findall(r"\b([ABab])\b", t)
        if tokens:
            return tokens[-1].upper()
        return "unknown"

    @staticmethod
    def _aggregate_verdicts(verdicts: List[str]) -> str:
        """Majority across samples.  Returns 'A', 'B', 'tie', or 'unknown'."""
        n_a = verdicts.count("A")
        n_b = verdicts.count("B")
        if n_a == 0 and n_b == 0:
            return "unknown"
        if n_a > n_b:
            return "A"
        if n_b > n_a:
            return "B"
        return "tie"

    def judge_pair(self, prompt_text: str, image_a: Image.Image,
                   image_b: Image.Image) -> JudgeResult:
        """Public entry; wraps the family-specific call with timing
        and error handling.  Subclasses implement `_judge_pair_multi`
        which returns a list[str] of length `num_samples` (>=1)."""
        t0 = time.time()
        try:
            raws = self._judge_pair_multi(prompt_text, image_a, image_b)
            if not isinstance(raws, list):
                raws = [raws]
            verdicts = [self.parse_verdict(r) for r in raws]
            verdict = self._aggregate_verdicts(verdicts)
            multi = len(raws) > 1
            return JudgeResult(
                verdict=verdict,
                raw_response=raws[0] if raws else "",
                latency_s=time.time() - t0,
                ok=True,
                verdicts=verdicts if multi else None,
                raw_responses=raws if multi else None,
            )
        except Exception as e:
            return JudgeResult(
                verdict="unknown", raw_response="",
                latency_s=time.time() - t0, ok=False,
                error=f"{type(e).__name__}: {e}"[:300],
            )

    def _judge_pair_multi(self, prompt_text: str, image_a: Image.Image,
                          image_b: Image.Image) -> List[str]:
        """Return one raw-text response per sample.  Length must equal
        `self.num_samples` (or 1 for greedy single-sample runners)."""
        raise NotImplementedError

    def warmup(self):
        """Optional: load the model into memory so the first judge call
        is not measured as a cold start."""
        pass

    def shutdown(self):
        """Optional: release GPU memory."""
        pass

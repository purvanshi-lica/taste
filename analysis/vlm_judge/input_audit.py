"""Audit what the VLM judge actually receives.

Two questions:
  1. Is the chat-template rendering aligned with how pairwise VLM
     judges are typically prompted in the field?
  2. Are *both* images really being passed into the model (vs.
     silently collapsed to one)?

This script walks one real TASTE pair through the full input pipeline
without calling generate(): we render the chat template to text, run
the processor, and dump the resulting tensor shapes + decoded text
including image-placeholder tokens.  It also probes two alternative
prompt structures so we can compare them.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from runners.base import Runner  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_one_pair():
    with open(HERE / "pair_jsonl" / "descriptions_typography.jsonl") as f:
        return json.loads(f.readline())


def render_and_probe(processor, messages, label: str):
    """Apply chat template, then the processor, and report what
    the model would actually see."""
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    text_only = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    print("\n--- chat-template rendered text (no tokenization) ---")
    # Replace special control tokens with visible markers
    for tok in (
        "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
        "<|image_pad|>", "<|object_ref_start|>", "<|object_ref_end|>",
    ):
        text_only = text_only.replace(tok, f"[{tok.strip('<|>')}]")
    print(text_only[:1500])

    # Now actually tokenize with the images
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    print("\n--- processor output keys & tensor shapes ---")
    for k, v in inputs.items():
        shp = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        print(f"  {k:30s} {shp}")

    # Inspect input_ids: how many image placeholder tokens are present
    ids = inputs["input_ids"][0].tolist()
    print(f"\n  total input tokens: {len(ids)}")
    print("  decoded (showing image-pad runs collapsed):")
    decoded = processor.tokenizer.decode(ids, skip_special_tokens=False)
    # Collapse long runs of identical image-pad tokens for readability
    import re
    decoded = re.sub(
        r"(<\|image_pad\|>){2,}",
        lambda m: f"[{m.group(0).count('<|image_pad|>')} × <|image_pad|>]",
        decoded,
    )
    print(f"\n{decoded[:2000]}")
    print()

    # Pixel values is the canonical proof both images are present.
    if "pixel_values" in inputs:
        pv = inputs["pixel_values"]
        print(f"  pixel_values shape: {tuple(pv.shape)}")
        # Qwen3-VL stacks both images; "image_grid_thw" tells us the per-image
        # tile counts.  If image_grid_thw has 2 rows, both images were
        # processed.
    if "image_grid_thw" in inputs:
        gthw = inputs["image_grid_thw"]
        print(f"  image_grid_thw shape: {tuple(gthw.shape)}  (rows = #images)")
        print(f"  image_grid_thw values:\n  {gthw.tolist()}")


def main():
    from transformers import AutoProcessor
    print(f"loading processor for {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    task = load_one_pair()
    print(f"prompt_id={task['prompt_id']}  "
          f"image_a={task['image_a']['model']}  "
          f"image_b={task['image_b']['model']}  "
          f"human={task['human_majority']}")

    img_a = Runner.fetch_image(task["image_a"]["url"])
    img_b = Runner.fetch_image(task["image_b"]["url"])
    print(f"image A size: {img_a.size}    image B size: {img_b.size}")

    text_prompt = (
        "You are a professional graphic designer evaluating two AI-"
        "generated images for the same brief.\n\n"
        f"BRIEF:\n{task['prompt']}\n\n"
        "You will see two images, labelled A and B.  Question: Which "
        "image more accurately renders the text demanded by the prompt?\n\n"
        "Reply with exactly one character: 'A' or 'B'."
    )

    # 1. Current harness format: two `image` content blocks, then text.
    msgs_current = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_a},
            {"type": "image", "image": img_b},
            {"type": "text", "text": text_prompt},
        ],
    }]
    render_and_probe(
        processor, msgs_current,
        "1. CURRENT HARNESS FORMAT  (image, image, text)",
    )

    # 2. Field-standard pairwise format: explicit "Image A:" / "Image B:"
    # text labels interleaved with the images, so the model sees the
    # binding directly rather than inferring it from positional order.
    msgs_labeled = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Image A:"},
            {"type": "image", "image": img_a},
            {"type": "text", "text": "Image B:"},
            {"type": "image", "image": img_b},
            {"type": "text", "text": text_prompt},
        ],
    }]
    render_and_probe(
        processor, msgs_labeled,
        "2. LABELED FORMAT  ('Image A:' image  'Image B:' image  text)",
    )


if __name__ == "__main__":
    main()

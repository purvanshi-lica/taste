# TASTE

This is the code and data repository for the paper

> **TASTE: A Designer-Annotated Multi-Dimensional Preference Dataset for AI-Generated Graphic Design**
> [arXiv:2605.20731](https://arxiv.org/abs/2605.20731) · [PDF](https://arxiv.org/pdf/2605.20731)

and the accompanying **TASTE dataset**, hosted on the Hugging Face Hub:

> [`purvanshi/TASTE`](https://huggingface.co/datasets/purvanshi/TASTE)

TASTE is a corpus of designer-panel rankings of AI-generated graphic designs
across multiple aesthetic and description-faithfulness dimensions, plus a small
preference model trained on it.

This is a joint work of **[Lica World](https://lica.world)** and **[Contra](https://contra.com)**.

## Repository layout

The repo is organized into three top-level parts, each self-contained with its
own README:

| Directory | What it is | README |
|---|---|---|
| [`analysis/`](analysis/) | The dataset-paper analysis — signal-validation distribution tests, the open-weight VLM-as-judge benchmark, and hallucination-flag agreement. | [`analysis/README.md`](analysis/README.md) |
| [`data/`](data/) | Processing scripts for the TASTE dataset on the Hugging Face Hub — download a local snapshot and derive the per-dimension ranking CSVs and pairwise battles used by `analysis/` and `taste-scorer/`. | [`data/README.md`](data/README.md) |
| [`taste-scorer/`](taste-scorer/) | The pip-installable **TASTE preference model** — inference for per-dimension "which design does the panel prefer?" probabilities from an image pair and a prompt. | [`taste-scorer/README.md`](taste-scorer/README.md) |

## Quick start

The three parts are independent; install only what you need.

```bash
# Score image pairs with the preference model
cd taste-scorer && pip install -e .

# Reproduce the paper analysis
cd analysis && pip install -r requirements.txt

# Fetch and process the dataset
cd data && pip install -r requirements.txt
```

The dataset itself is **not** committed here; it is fetched from the Hugging
Face Hub (see [`data/README.md`](data/README.md)). Designer identities are
masked before release — do not attempt to re-identify raters.

## Links

- Code: [github.com/purvanshi-lica/taste](https://github.com/purvanshi-lica/taste)
- Paper: [arXiv:2605.20731](https://arxiv.org/abs/2605.20731) ([PDF](https://arxiv.org/pdf/2605.20731))
- Dataset: [`purvanshi/TASTE`](https://huggingface.co/datasets/purvanshi/TASTE)
- Model checkpoint: [TASTE_Checkpoint.zip](https://storage.googleapis.com/lica-assets/TASTE/TASTE_Checkpoint.zip)
- Backbone used by the scorer: [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- Lica World: [lica.world](https://lica.world) · Contra: [contra.com](https://contra.com)

## Citing

If you use the TASTE dataset or the TASTE-scorer, please cite the paper
([arXiv:2605.20731](https://arxiv.org/abs/2605.20731)). 

```
@article{zhu2026taste,
  title={TASTE: A Designer-Annotated Multi-Dimensional Preference Dataset for AI-Generated Graphic Design},
  author={Zhu, Haonan and Hirsch, Elad and Minetti, Alexandria and Nulty, Allison and Mehta, Purvanshi},
  journal={arXiv preprint arXiv:2605.20731},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

# results/ (local reference — not part of the public release)

Everything in this directory **except this README is git-ignored**.  These
files are kept on disk for the authors' future reference to the paper; they
are intentionally **not** shared in the public repository.

Current local contents (provenance: `origin/hps-contra @ d0aa1e2`):

| file | what it is |
|------|------------|
| `01_baseline_sweep.md`           | Full hyper-parameter sweep that selected the best pairwise-difference head (config H, val acc ≈ 0.611). `retrain_best.sh` reproduces row H. |
| `02_tier0_data_diagnostics.md`   | Data-quality diagnostics on the battle pairs prior to training. |
| `00_onboarding.md`               | Original project onboarding notes for the training repository. |
| `premodel.tex`                   | Source of the paper's preference-model section (§7), kept to trace reported numbers back to the sweep. |

If at some point these should be released (e.g., the sweep table as a
supplementary), move the specific file out of the git-ignore scope by adding
a `!results/<file>` exception to `.gitignore`.

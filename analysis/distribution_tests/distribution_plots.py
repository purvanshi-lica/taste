"""
Plotting + statistical tests for the distribution-based analysis.

For each TASTE dimension we produce:

  Figure A -- 4-panel "signal" view:
      (a) Histogram of per-prompt T vs exact null PMF (p=4, R=5) and the
          three reference empirical distributions (Sushi, MovieLens, MT-Bench).
      (b) Histogram of individual pairwise Kendall tau (7 discrete values
          for p=4) vs the R=2 Mahonian null and the three reference
          empiricals.
      (c) Majority-vote probability p_maj in {3/5, 4/5, 5/5} vs null PMF
          (5/8, 5/16, 1/16) and the three reference empiricals.
      (d) Cycle fraction (binary: prompt has majority cycle or not) vs
          null rate (Monte-Carlo, p=4 R=5) and the three reference rates.

  Figure B -- 2x5 grid of per-evaluator-pair tau histograms (one per
      C(5,2)=10 evaluator pair), each overlaid on the Mahonian null.

Each Figure A panel also reports chi-squared GOF vs null + chi-squared
two-sample vs each reference (and KS for T).
"""

import os
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from null_kendall import exact_null_pmf, r2_tau_pmf  # noqa: E402

REF_COLORS = {
    "Sushi":      "#1f77b4",
    "MovieLens":  "#2ca02c",
    "HPSv2-test": "#ff7f0e",
}

# Per-anchor flag: True → draw mean line + 5-95% bootstrap band
#                  False → draw only the mean line (suppress band)
# Sushi/MovieLens have bootstrap-form data on disk but we suppress their
# bands on the figure to avoid visual clutter; HPSv2-test keeps its band.
REF_SHOW_BAND = {
    "Sushi":      False,
    "MovieLens":  False,
    "HPSv2-test": True,
}


# ---------- histogram helpers ------------------------------------------------

def snap_to_grid(vals, grid):
    """Snap each value to the nearest grid point; return integer counts."""
    idx = np.argmin(np.abs(vals[:, None] - grid[None, :]), axis=1)
    return np.bincount(idx, minlength=len(grid))


def chi2_gof_vs_null(observed_counts, null_probs):
    """Chi-squared goodness-of-fit with low-count bin collapsing.

    Bins with expected count < 1 are pooled into a neighbour to keep the
    chi2 approximation reasonable.  Returns (stat, pval, df, n_bins_used).
    """
    n = int(observed_counts.sum())
    exp = null_probs * n
    obs_list = list(observed_counts.astype(float))
    exp_list = list(exp.astype(float))
    i = 0
    while i < len(exp_list):
        if exp_list[i] < 1 and len(exp_list) > 1:
            j = i + 1 if i + 1 < len(exp_list) else i - 1
            obs_list[j] += obs_list[i]
            exp_list[j] += exp_list[i]
            del obs_list[i]
            del exp_list[i]
        else:
            i += 1
    obs_arr = np.asarray(obs_list)
    exp_arr = np.asarray(exp_list)
    keep = exp_arr > 0
    obs_arr, exp_arr = obs_arr[keep], exp_arr[keep]
    if len(obs_arr) < 2:
        return float("nan"), float("nan"), 0, len(obs_arr)
    stat = float(((obs_arr - exp_arr) ** 2 / exp_arr).sum())
    df = len(obs_arr) - 1
    pval = float(1 - sp_stats.chi2.cdf(stat, df))
    return stat, pval, df, len(obs_arr)


def chi2_two_sample(obs_a, obs_b):
    """Two-sample chi-squared test on aligned discrete histograms.

    Tests H_0: the two samples come from the same distribution.  Bins
    with small expected counts on either side are pooled.
    """
    obs_a = obs_a.astype(float)
    obs_b = obs_b.astype(float)
    row_a, row_b = obs_a.sum(), obs_b.sum()
    if row_a == 0 or row_b == 0:
        return float("nan"), float("nan"), 0, 0
    col = obs_a + obs_b
    keep_cols = col > 0
    a = obs_a[keep_cols].copy()
    b = obs_b[keep_cols].copy()
    col_k = col[keep_cols].copy()
    total = row_a + row_b
    exp_a = row_a * col_k / total
    exp_b = row_b * col_k / total
    i = 0
    while i < len(col_k):
        if (exp_a[i] < 1 or exp_b[i] < 1) and len(col_k) > 2:
            j = i + 1 if i + 1 < len(col_k) else i - 1
            a[j] += a[i]; b[j] += b[i]; col_k[j] += col_k[i]
            exp_a[j] += exp_a[i]; exp_b[j] += exp_b[i]
            a = np.delete(a, i); b = np.delete(b, i)
            col_k = np.delete(col_k, i)
            exp_a = np.delete(exp_a, i); exp_b = np.delete(exp_b, i)
        else:
            i += 1
    if len(a) < 2:
        return float("nan"), float("nan"), 0, len(a)
    stat = float(((a - exp_a) ** 2 / exp_a).sum()
                 + ((b - exp_b) ** 2 / exp_b).sum())
    df = len(a) - 1
    pval = float(1 - sp_stats.chi2.cdf(stat, df))
    return stat, pval, df, len(a)


# ---------- null construction -----------------------------------------------

def exact_T_null_grid():
    pmf = exact_null_pmf(4, 5)
    pmf_items = sorted(pmf.items(), key=lambda kv: float(kv[0]))
    ts = np.array([float(t) for t, _ in pmf_items])
    ps = np.array([float(p) for _, p in pmf_items])
    return ts, ps


def pairtau_null_grid(p=4):
    pmf = r2_tau_pmf(p)
    pmf_items = sorted(pmf.items(), key=lambda kv: float(kv[0]))
    ts = np.array([float(t) for t, _ in pmf_items])
    ps = np.array([float(p) for _, p in pmf_items])
    return ts, ps


def pmaj_null_grid():
    """For R=5 coin flips, p_maj=max(k,R-k)/R in {3/5, 4/5, 5/5}."""
    probs = {3 / 5: 20 / 32, 4 / 5: 10 / 32, 5 / 5: 2 / 32}
    vals = sorted(probs.keys())
    return np.asarray(vals), np.array([probs[v] for v in vals])


def cycle_null_rate(M=200_000, seed=2026):
    """MC rate of P(majority-vote cycle) under 5 iid uniform rankings of 4 items."""
    rng = np.random.default_rng(seed)
    R, p = 5, 4
    a_i, b_i = np.triu_indices(p, k=1)
    triples = list(combinations(range(p), 3))
    count = 0
    for _ in range(M):
        u = rng.random((R, p))
        ranks = u.argsort(axis=-1) + 1
        sign = np.sign(ranks[:, a_i] - ranks[:, b_i]).astype(np.int64)
        a_ahead = (sign == -1).sum(axis=0)
        maj = np.where(a_ahead * 2 > R, -1,
                       np.where(a_ahead * 2 < R, +1, 0))
        has_cycle = False
        for tr in triples:
            x, y, z = tr
            def get_s(u, v):
                if u < v:
                    k = np.where((a_i == u) & (b_i == v))[0][0]
                    return int(maj[k])
                k = np.where((a_i == v) & (b_i == u))[0][0]
                return -int(maj[k])
            xy, yz, xz = get_s(x, y), get_s(y, z), get_s(x, z)
            if 0 in (xy, yz, xz):
                continue
            if (xy == +1 and yz == +1 and xz == -1) or \
               (xy == -1 and yz == -1 and xz == +1):
                has_cycle = True
                break
        if has_cycle:
            count += 1
    return count / M


# ---------- plotting primitives ----------------------------------------------

def _per_bootstrap_histogram_band(per_bootstrap_flat, grid, lo_pct=5, hi_pct=95):
    """Given per-bootstrap arrays (B, n), compute the mean histogram and a
    [lo_pct, hi_pct] band per bin across bootstrap iterations.

    Returns (mean_probs, lo_band, hi_band) all of shape (len(grid),).
    """
    B = per_bootstrap_flat.shape[0]
    bin_probs = np.zeros((B, len(grid)), dtype=np.float64)
    for b in range(B):
        flat_b = per_bootstrap_flat[b].ravel()
        counts_b = snap_to_grid(flat_b, grid)
        bin_probs[b] = counts_b / max(counts_b.sum(), 1)
    mean_probs = bin_probs.mean(axis=0)
    lo = np.percentile(bin_probs, lo_pct, axis=0)
    hi = np.percentile(bin_probs, hi_pct, axis=0)
    return mean_probs, lo, hi


def _bars(ax, grid, null_probs, contra_counts, refs, title, xlabel,
          width_frac=0.8):
    """Plot null bars + TASTE bars + reference overlays.

    Reference values can be either:
      - 1D pooled array (n_total,): plot a single line
      - 2D per-bootstrap array (B, n_per_b) for T/cycle, or 3D (B, n_per_b, k)
        for pair_tau/pmaj: plot mean line + 5-95% bootstrap band on each bin
    """
    n_contra = contra_counts.sum()
    contra_probs = contra_counts / max(n_contra, 1)
    width = width_frac * float(np.median(np.diff(grid)) if len(grid) > 1 else 0.1)

    ax.bar(grid, null_probs, width=width, color="lightgray",
           edgecolor="gray", linewidth=0.4, alpha=0.85,
           label="iid-uniform null")
    ax.bar(grid, contra_probs, width=width * 0.45, color="black",
           edgecolor="black", linewidth=0.5, alpha=0.95,
           label=f"TASTE (n={n_contra})")
    for name, vals in refs.items():
        if vals is None or vals.size == 0:
            continue
        # Detect per-bootstrap (leading dim is "small B" say ≤ 200) by
        # caller convention: bootstrap-aware refs come in with `_is_bootstrap`
        # in their dict.  Here we simply check ndim on the values.
        # Detect per-bootstrap layout: leading dim ≤ 200 is "B"; everything
        # after is samples + per-sample-component axes that we flatten.
        # For 1D arrays (legacy pooled), this branch is skipped.
        if vals.ndim >= 2 and vals.shape[0] <= 200:
            B = vals.shape[0]
            per_b_flat = vals.reshape(B, -1)
            mean_probs, lo, hi = _per_bootstrap_histogram_band(
                per_b_flat, grid, lo_pct=5, hi_pct=95)
            n_total = per_b_flat.size
            show_band = REF_SHOW_BAND.get(name, True)
            if show_band:
                ax.fill_between(grid, lo, hi, color=REF_COLORS[name], alpha=0.18,
                                edgecolor="none",
                                label=f"{name} 5-95% band (B={B})")
                ax.plot(grid, mean_probs, "o-", color=REF_COLORS[name], lw=1.2,
                        markersize=4.5, alpha=0.95,
                        label=f"{name} mean (n={n_total:,})")
            else:
                ax.plot(grid, mean_probs, "o-", color=REF_COLORS[name], lw=1.2,
                        markersize=4.5, alpha=0.9,
                        label=f"{name} (n={n_total:,})")
        else:
            flat = vals.ravel()
            ref_counts = snap_to_grid(flat, grid)
            ref_probs = ref_counts / ref_counts.sum()
            ax.plot(grid, ref_probs, "o-", color=REF_COLORS[name], lw=1.2,
                    markersize=4.5, alpha=0.9,
                    label=f"{name} (n={len(flat):,})")

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("probability", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best")


def plot_main_figure(stats, refs_dict, tests, out_path, title):
    """4-panel main figure: T, pairwise tau, p_maj, cycle rate."""
    T_grid, T_null = exact_T_null_grid()
    tau_grid, tau_null = pairtau_null_grid(4)
    pmaj_grid, pmaj_null = pmaj_null_grid()
    cyc_null_rt = tests["_cycle_null_rate"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a)
    contra_T_counts = snap_to_grid(stats["T"], T_grid)
    refs_T = {k: v["T"] for k, v in refs_dict.items()}
    gof_p = tests["T"]["null"]["p"]
    _bars(axes[0, 0], T_grid, T_null, contra_T_counts, refs_T,
          title=rf"(a) per-prompt $T$   GOF vs null $p$={gof_p:.3g}",
          xlabel=r"$T$")

    # (b) — keep bootstrap dim if present so band can be computed per bin
    contra_tau_counts = snap_to_grid(stats["pair_tau"].ravel(), tau_grid)
    refs_tau = {k: v["pair_tau"] for k, v in refs_dict.items()}
    gof_p = tests["pair_tau"]["null"]["p"]
    _bars(axes[0, 1], tau_grid, tau_null, contra_tau_counts, refs_tau,
          title=rf"(b) individual pairwise $\tau$   GOF vs null $p$={gof_p:.3g}",
          xlabel=r"$\tau(\pi_r,\pi_s)$")

    # (c) — keep bootstrap dim if present
    contra_pmaj_counts = snap_to_grid(stats["pmaj"].ravel(), pmaj_grid)
    refs_pmaj = {k: v["pmaj"] for k, v in refs_dict.items()}
    gof_p = tests["pmaj"]["null"]["p"]
    _bars(axes[1, 0], pmaj_grid, pmaj_null, contra_pmaj_counts, refs_pmaj,
          title=rf"(c) majority-vote $p_{{\max}}$   GOF vs null $p$={gof_p:.3g}",
          xlabel=r"$p_{\max}=\max(k/5, 1-k/5)$")

    # (d) cycle rate
    ax = axes[1, 1]
    labels = ["null"] + list(refs_dict.keys()) + ["TASTE"]
    colors = ["lightgray"] + [REF_COLORS[k] for k in refs_dict.keys()] + ["black"]

    rates = [cyc_null_rt]
    err_lo = [0.0]
    err_hi = [0.0]
    ns = ["exact (MC)"]
    for name, v in refs_dict.items():
        cyc = v["cycle"]
        if cyc.ndim >= 2 and cyc.shape[0] < cyc.shape[-1] and cyc.shape[0] <= 200:
            per_b_rates = cyc.mean(axis=1)
            mean_rate = float(per_b_rates.mean())
            lo = float(np.percentile(per_b_rates, 5))
            hi = float(np.percentile(per_b_rates, 95))
            rates.append(mean_rate)
            if REF_SHOW_BAND.get(name, True):
                err_lo.append(mean_rate - lo)
                err_hi.append(hi - mean_rate)
            else:
                err_lo.append(0.0)
                err_hi.append(0.0)
            ns.append(f"n={cyc.size:,}")
        else:
            flat = cyc.ravel()
            rates.append(float(flat.mean()))
            err_lo.append(0.0)
            err_hi.append(0.0)
            ns.append(f"n={len(flat):,}")
    contra_cyc = stats["cycle"]
    rates.append(float(contra_cyc.mean()))
    err_lo.append(0.0)
    err_hi.append(0.0)
    ns.append(f"n={len(contra_cyc)}")

    xs = np.arange(len(labels))
    bars = ax.bar(xs, rates, color=colors, edgecolor="black", alpha=0.85,
                  yerr=[err_lo, err_hi], capsize=4,
                  error_kw={"ecolor": "black", "elinewidth": 1.0})
    for b, r, n in zip(bars, rates, ns):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.012,
                f"{r:.3f}\n{n}", ha="center", va="bottom", fontsize=8)
    ax.axhline(cyc_null_rt, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9, rotation=20)
    ax.set_ylabel("fraction with majority cycle", fontsize=10)
    upper = max([r + e for r, e in zip(rates, err_hi)])
    ax.set_ylim(0, upper * 1.30 + 0.02)
    cyc_p = tests["cycle"]["null"]["p"]
    ax.set_title(f"(d) Condorcet cycle rate   binomial vs null p={cyc_p:.3g}",
                 fontsize=10)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pair_reviewer_grid(stats, out_path, title):
    """2x5 grid of per-evaluator-pair tau histograms overlaid on Mahonian null.

    One histogram per C(5,2)=10 evaluator pair.  Each histogram has one
    observation per prompt (80 total) on the 7-point Mahonian support.
    """
    tau_grid, tau_null = pairtau_null_grid(4)
    labels = stats["pair_tau_labels"]
    data = stats["pair_tau"]                                 # (n_prompts, 10)
    # Sort pairs by mean tau descending so the "most agreeing" pairs appear first
    mean_taus = data.mean(axis=0)
    order = np.argsort(-mean_taus)

    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), sharex=True, sharey=True)
    width = 0.8 * float(np.median(np.diff(tau_grid)))
    for idx, pair_idx in enumerate(order):
        ax = axes[idx // 5, idx % 5]
        e1, e2 = labels[pair_idx]
        vals = data[:, pair_idx]
        counts = snap_to_grid(vals, tau_grid)
        probs = counts / counts.sum()
        ax.bar(tau_grid, tau_null, width=width, color="lightgray",
               edgecolor="gray", linewidth=0.4, alpha=0.8)
        ax.bar(tau_grid, probs, width=width * 0.45, color="black",
               edgecolor="black", linewidth=0.4, alpha=0.95)
        # Rename overlong evaluator names to their first token for display
        e1s = e1.split()[0][:8]
        e2s = e2.split()[0][:8]
        ax.set_title(f"{e1s} x {e2s}\nmean={vals.mean():+.3f}, "
                     f"median={np.median(vals):+.3f}",
                     fontsize=9)
        ax.axvline(0, color="gray", lw=0.4, ls=":")
        if idx // 5 == 1:
            ax.set_xlabel(r"$\tau$", fontsize=9)
        if idx % 5 == 0:
            ax.set_ylabel("prob", fontsize=9)

    fig.suptitle(title + "  —  null in grey is the $R=2$ Mahonian "
                 r"(uniform permutation of $p=4$)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- statistical test bundle -----------------------------------------

def run_tests(stats, refs_dict, cyc_null_rt=None):
    """Formal tests: goodness-of-fit against the null only.

    Reference datasets are NOT tested against formally — they are shown in
    the plots as visual anchors (this is what "typical real preference data
    looks like"), so we only record their descriptive quantiles/rates for
    reporting.
    """
    out = {}
    T_grid, T_null_probs = exact_T_null_grid()
    tau_grid, tau_null_probs = pairtau_null_grid(4)
    pmaj_grid, pmaj_null_probs = pmaj_null_grid()
    if cyc_null_rt is None:
        cyc_null_rt = cycle_null_rate()
    out["_cycle_null_rate"] = cyc_null_rt

    # --- T ---
    out["T"] = {}
    obs = snap_to_grid(stats["T"], T_grid)
    chi2, pv, df, nb = chi2_gof_vs_null(obs, T_null_probs)
    out["T"]["null"] = {"test": "chi2_gof", "stat": chi2, "p": pv,
                        "df": df, "n_bins": nb}

    # --- pair_tau ---
    out["pair_tau"] = {}
    obs = snap_to_grid(stats["pair_tau"].ravel(), tau_grid)
    chi2, pv, df, nb = chi2_gof_vs_null(obs, tau_null_probs)
    out["pair_tau"]["null"] = {"test": "chi2_gof", "stat": chi2, "p": pv,
                                "df": df, "n_bins": nb}

    # --- pmaj ---
    out["pmaj"] = {}
    obs = snap_to_grid(stats["pmaj"].ravel(), pmaj_grid)
    chi2, pv, df, nb = chi2_gof_vs_null(obs, pmaj_null_probs)
    out["pmaj"]["null"] = {"test": "chi2_gof", "stat": chi2, "p": pv,
                           "df": df, "n_bins": nb}

    # --- cycle: binomial test vs null rate only ---
    out["cycle"] = {}
    n_cyc = int(stats["cycle"].sum())
    n_tot = int(len(stats["cycle"]))
    bino = sp_stats.binomtest(n_cyc, n_tot, p=cyc_null_rt,
                               alternative="two-sided")
    out["cycle"]["null"] = {"test": "binomial", "n_cycle": n_cyc,
                             "n_total": n_tot, "obs_rate": n_cyc / n_tot,
                             "null_rate": cyc_null_rt, "p": float(bino.pvalue)}

    # --- reference descriptive stats (NOT a test, just for reporting) ---
    out["_reference_descriptive"] = {}
    for name, ref in refs_dict.items():
        # ref arrays may be 1D (pooled) or 2D/3D (per-bootstrap).  Total
        # sample count is the size of the T array; for per-bootstrap refs
        # the leading dim is bootstrap count, all dims contribute samples.
        T_arr = ref["T"]
        out["_reference_descriptive"][name] = {
            "n": int(T_arr.size),
            "T_median": float(np.median(T_arr)),
            "T_mean":   float(np.mean(T_arr)),
            "pair_tau_mean": float(np.mean(ref["pair_tau"])),
            "pmaj_mean": float(np.mean(ref["pmaj"])),
            "cycle_rate": float(np.mean(ref["cycle"])),
        }
    return out

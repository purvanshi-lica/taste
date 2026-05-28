"""
Exact and Monte Carlo computation of the null distribution of

    T = (1 / C(R, 2)) * sum_{r < s} tau(pi_r, pi_s),

the average pairwise Kendall-tau agreement among R rankings of p items,
under the null pi_1, ..., pi_R iid Uniform(S_p).

Reproduces the exact p = 4, R = 5 table in main.tex.

Running `python null_kendall.py` prints the exact PMF, the exact moments,
a normal-approximation vs. exact upper-tail comparison, and a Monte Carlo
sanity check.
"""

from collections import Counter
from fractions import Fraction
from itertools import permutations, product
from math import comb, erf, factorial, sqrt

import numpy as np


# --------------------------------------------------------------------------
#  Core identity used throughout
#
#    Let m = C(p, 2) and s(pi) in {+-1}^m be the pair-sign vector
#    s(pi)_{(a<b)} = sgn(pi(a) - pi(b)). Writing V = sum_r s(pi_r),
#
#        |V|^2  =  R m  +  2 m * C(R,2) * T,
#
#    so T is an affine function of |V|^2. Computing T from V avoids
#    recomputing O(R^2 * m) pairwise tau terms per tuple.
# --------------------------------------------------------------------------


def _sign_vector(perm):
    """Pair-sign vector of a permutation.

    perm is a tuple where perm[i] is the rank of item i. The returned array
    has one entry per item-pair (a, b) with a < b, equal to +1 if
    perm[a] < perm[b] and -1 otherwise.
    """
    p = len(perm)
    out = []
    for a in range(p):
        for b in range(a + 1, p):
            out.append(1 if perm[a] < perm[b] else -1)
    return np.asarray(out, dtype=np.int64)


# ---------- exact enumeration ------------------------------------------------

def exact_null_pmf(p, R, chunk=200_000):
    """Exact null PMF of T for p variables and R rankings.

    Fixes pi_1 = identity and enumerates the remaining (p!)^{R-1} tuples,
    batching them in chunks of `chunk` for numpy-vectorized arithmetic.

    Returns dict mapping T (as Fraction) to probability (as Fraction).
    Cost: O((p!)^{R-1}).  Feasible for p=4, R<=6 on a laptop.
    """
    m = comb(p, 2)
    denom_T = m * R * (R - 1)                              # T = T_num / denom_T
    perms = list(permutations(range(1, p + 1)))
    svs = np.stack([_sign_vector(pi) for pi in perms])      # (p!, m)
    identity_sv = svs[0]                                    # identity is first
    n_perms = len(perms)

    counts = Counter()
    batch = []

    def _flush(batch):
        idx = np.asarray(batch, dtype=np.int64)
        V = svs[idx].sum(axis=1) + identity_sv              # (len(batch), m)
        T_num = (V * V).sum(axis=1) - R * m
        counts.update(T_num.tolist())

    for tup in product(range(n_perms), repeat=R - 1):
        batch.append(tup)
        if len(batch) >= chunk:
            _flush(batch)
            batch = []
    if batch:
        _flush(batch)

    n_total = sum(counts.values())
    assert n_total == n_perms ** (R - 1), (n_total, n_perms ** (R - 1))
    return {Fraction(t_num, denom_T): Fraction(c, n_total)
            for t_num, c in counts.items()}


def upper_tail(pmf, threshold):
    """Return P(T >= threshold) from a PMF dict."""
    return sum(pr for t, pr in pmf.items() if t >= threshold)


def moments(pmf):
    """Return (mean, variance, skewness, excess_kurtosis) as floats."""
    mean = float(sum(pr * t for t, pr in pmf.items()))
    var = float(sum(pr * (t - mean) ** 2 for t, pr in pmf.items()))
    if var == 0:
        return mean, 0.0, 0.0, 0.0
    sd = sqrt(var)
    skew = float(sum(pr * ((t - mean) / sd) ** 3 for t, pr in pmf.items()))
    ekurt = float(sum(pr * ((t - mean) / sd) ** 4
                      for t, pr in pmf.items())) - 3
    return mean, var, skew, ekurt


# ---------- Monte Carlo ------------------------------------------------------

def monte_carlo_T(p, R, M, seed=0):
    """Draw M iid samples of T under iid uniform rankings."""
    rng = np.random.default_rng(seed)
    m = comb(p, 2)
    a_idx, b_idx = np.triu_indices(p, k=1)
    u = rng.random((M, R, p))                              # iid uniforms
    ranks = u.argsort(axis=-1)                             # uniform perms
    diffs = ranks[:, :, a_idx] - ranks[:, :, b_idx]        # (M, R, m)
    svs = np.sign(diffs).astype(np.int64)                   # (M, R, m)
    V = svs.sum(axis=1)                                     # (M, m)
    V2 = (V * V).sum(axis=1)                                # (M,)
    return (V2 - R * m) / (R * (R - 1) * m)


# ---------- convenience ------------------------------------------------------

def null_sd(p, R):
    """Theoretical null standard deviation of T."""
    return sqrt(2 * (2 * p + 5) / (9 * p * (p - 1)) / comb(R, 2))


def z_score(t, p, R):
    """z = T / null_sd(p, R). The null is NOT exactly Gaussian, so treat
    z as an intuition aid, not as a p-value."""
    return t / null_sd(p, R)


def exact_pvalue(p, R, t_obs, pmf=None):
    """Exact upper-tail p-value P(T >= t_obs). Pass a precomputed pmf
    to avoid recomputing it on repeated calls."""
    if pmf is None:
        pmf = exact_null_pmf(p, R)
    return float(upper_tail(pmf, t_obs))


def mc_pvalue(p, R, t_obs, M=200_000, seed=0):
    """Monte Carlo estimate of P(T >= t_obs)."""
    return float((monte_carlo_T(p, R, M, seed) >= t_obs).mean())


def _norm_sf(z):
    """Standard normal survival function 1 - Phi(z), using math.erf so we
    don't require scipy."""
    return 0.5 * (1.0 - erf(z / sqrt(2.0)))


# ---------- competing test statistics ---------------------------------------
#
# Each `stat_*` function accepts an array of rankings with shape
# (..., R, p) and returns the statistic computed along the last two axes,
# preserving any leading batch dimensions.
# --------------------------------------------------------------------------

def _pair_indices(p):
    a, b = np.triu_indices(p, k=1)
    return a, b, a.size


def _signs(pis, a, b):
    return np.sign(pis[..., a] - pis[..., b]).astype(np.int64)


def stat_T(pis):
    """Average pairwise Kendall tau."""
    R, p = pis.shape[-2:]
    a, b, m = _pair_indices(p)
    s = _signs(pis, a, b)
    V = s.sum(axis=-2)
    V2 = (V * V).sum(axis=-1)
    return (V2 - R * m) / (R * (R - 1) * m)


def stat_spearman_avg(pis):
    """Average pairwise Spearman rho (linear in Kendall's W via
    bar_rho = (R*W - 1)/(R - 1))."""
    R, p = pis.shape[-2:]
    denom = p * (p * p - 1)
    acc = None
    count = 0
    for r in range(R):
        for s in range(r + 1, R):
            diff2 = ((pis[..., r, :] - pis[..., s, :]) ** 2).sum(axis=-1)
            rho = 1 - 6.0 * diff2 / denom
            acc = rho if acc is None else acc + rho
            count += 1
    return acc / count


def stat_max_tau(pis):
    """Max pairwise Kendall tau."""
    R, p = pis.shape[-2:]
    a, b, m = _pair_indices(p)
    s = _signs(pis, a, b)                                  # (..., R, m)
    max_val = None
    for r in range(R):
        for t in range(r + 1, R):
            tau = (s[..., r, :] * s[..., t, :]).sum(axis=-1) / m
            max_val = tau if max_val is None else np.maximum(max_val, tau)
    return max_val


_KEMENY_SIGMA_SIGNS_CACHE = {}


def _kemeny_sigma_signs(p):
    if p not in _KEMENY_SIGMA_SIGNS_CACHE:
        perms_arr = np.array(list(permutations(range(p))), dtype=np.int64)
        a, b, _ = _pair_indices(p)
        _KEMENY_SIGMA_SIGNS_CACHE[p] = np.sign(
            perms_arr[:, a] - perms_arr[:, b]
        ).astype(np.int64)
    return _KEMENY_SIGMA_SIGNS_CACHE[p]


def stat_kemeny(pis):
    """Kemeny consensus distance K* = min_sigma sum_r d_K(pi_r, sigma).

    Small values <=> rankings cluster tightly around a single consensus.
    Returns int array; for a concordance test against uniform, we reject
    for SMALL K*.
    """
    R, p = pis.shape[-2:]
    a, b, m = _pair_indices(p)
    pi_signs = _signs(pis, a, b)                           # (..., R, m)
    sig_signs = _kemeny_sigma_signs(p)                      # (p!, m)
    # (..., p!, R) = pi_signs @ sig_signs.T along m, with broadcasting
    dots = np.einsum("...rm,km->...kr", pi_signs, sig_signs)
    dists = (m - dots) // 2                                 # (..., p!, R)
    total = dists.sum(axis=-1)                              # (..., p!)
    return total.min(axis=-1)


# ---------- exact null distributions of all four statistics -----------------

def exact_null_all_stats(p, R, chunk=100_000):
    """Enumerate all (p!)^{R-1} tuples (pi_1 = identity) and return arrays
    of T, avg-Spearman, max-tau, and Kemeny for each tuple."""
    perms_arr = np.array(list(permutations(range(p))), dtype=np.int64)
    n_perms = len(perms_arr)
    n_total = n_perms ** (R - 1)
    identity = perms_arr[0]

    Ts = np.empty(n_total, dtype=np.float64)
    Ws = np.empty(n_total, dtype=np.float64)
    Ms = np.empty(n_total, dtype=np.float64)
    Ks = np.empty(n_total, dtype=np.int64)

    tuples = product(range(n_perms), repeat=R - 1)
    pos = 0
    batch = []

    def _flush(batch, pos):
        b = np.asarray(batch, dtype=np.int64)              # (len, R-1)
        pis = np.empty((len(batch), R, p), dtype=np.int64)
        pis[:, 0, :] = identity
        pis[:, 1:, :] = perms_arr[b]
        Ts[pos:pos + len(batch)] = stat_T(pis)
        Ws[pos:pos + len(batch)] = stat_spearman_avg(pis)
        Ms[pos:pos + len(batch)] = stat_max_tau(pis)
        Ks[pos:pos + len(batch)] = stat_kemeny(pis)

    for tup in tuples:
        batch.append(tup)
        if len(batch) >= chunk:
            _flush(batch, pos)
            pos += len(batch)
            batch = []
    if batch:
        _flush(batch, pos)
        pos += len(batch)

    return {"T": Ts, "W_rho": Ws, "max_tau": Ms, "K_star": Ks}


# ---------- Mallows sampler -------------------------------------------------

def mallows_draws(p, R, M, theta, center=None, seed=0):
    """Draw M tuples of R iid Mallows(theta, center) rankings.

    Mallows density: P(pi) prop to exp(-theta * d_K(pi, center)).
    theta = 0 is uniform; large theta concentrates on the center.
    """
    rng = np.random.default_rng(seed)
    perms_arr = np.array(list(permutations(range(p))), dtype=np.int64)
    a, b, _ = _pair_indices(p)
    if center is None:
        center = perms_arr[0]
    center = np.asarray(center)
    c_signs = np.sign(center[a] - center[b])
    p_signs = np.sign(perms_arr[:, a] - perms_arr[:, b])
    # distance: number of disagreeing pairs
    dists = ((c_signs[None, :] != p_signs).sum(axis=1)).astype(np.float64)
    w = np.exp(-theta * dists)
    w /= w.sum()
    idx = rng.choice(len(perms_arr), size=M * R, p=w).reshape(M, R)
    return perms_arr[idx]


# ---------- Mahonian (R = 2) -------------------------------------------------

def mahonian_pmf(p):
    """PMF of the inversion count K of a uniform random permutation of p
    items. Returns dict k -> Fraction(count, p!).

    Uses the identity K =d U_2 + U_3 + ... + U_p, U_i ~ Uniform{0,...,i-1},
    which gives the generating function
        sum_k count_k q^k = prod_{i=1}^p (1 + q + q^2 + ... + q^{i-1}).
    """
    # Build coefficients of the polynomial product with integer coefficients
    poly = [1]
    for i in range(1, p + 1):
        factor = [1] * i                                    # 1 + q + ... + q^{i-1}
        new_poly = [0] * (len(poly) + len(factor) - 1)
        for j, a in enumerate(poly):
            if a == 0:
                continue
            for k, b in enumerate(factor):
                new_poly[j + k] += a * b
        poly = new_poly
    total = factorial(p)
    assert sum(poly) == total
    return {k: Fraction(c, total) for k, c in enumerate(poly) if c != 0}


def r2_tau_pmf(p):
    """Null PMF of T for R = 2: T = tau(e, pi), pi ~ Unif(S_p).

    Equivalently, T = 1 - 2K/C(p,2) where K is the inversion count.
    """
    m = comb(p, 2)
    mahonian = mahonian_pmf(p)
    return {Fraction(m - 2 * k, m): prob for k, prob in mahonian.items()}


# ---------- study of the effect of R ----------------------------------------

def summary_table(p, R_values, pmfs=None):
    """For each R, compute (mean, sd, skewness, excess kurtosis, one-sided
    5% threshold, one-sided 1% threshold) of the null distribution of T.

    Threshold t_alpha is the smallest support point with P(T >= t) <= alpha.
    Returns list of dicts.
    """
    if pmfs is None:
        pmfs = {R: exact_null_pmf(p, R) for R in R_values}
    rows = []
    for R in R_values:
        pmf = pmfs[R]
        mean, var, skew, ekurt = moments(pmf)
        # Iterate from largest to smallest T; keep overwriting while
        # tail <= alpha. Result = smallest support point with tail <= alpha.
        ts_desc = sorted(pmf.keys(), reverse=True)
        t5 = t1 = None
        for t in ts_desc:
            tail = float(upper_tail(pmf, t))
            if tail <= 0.05:
                t5 = t
            if tail <= 0.01:
                t1 = t
        rows.append({
            "R": R, "mean": mean, "sd": sqrt(var),
            "skew": skew, "ekurt": ekurt,
            "t5": t5, "tail5": float(upper_tail(pmf, t5)) if t5 is not None else None,
            "t1": t1, "tail1": float(upper_tail(pmf, t1)) if t1 is not None else None,
            "support": (min(pmf.keys()), max(pmf.keys())),
        })
    return rows


# ---------- plotting ---------------------------------------------------------

def plot_null_distribution(p, R, path, pmf=None):
    """Two-panel visualization of the exact null distribution of T.

    Left panel:  exact PMF bar chart with +- sd_0 and +- 2*sd_0 markers.
    Right panel: upper-tail survival P_0(T >= t) on log y-axis, exact
                 step plot vs normal approximation (matched variance).
    Requires matplotlib.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if pmf is None:
        pmf = exact_null_pmf(p, R)

    sorted_ts = sorted(pmf.keys())
    ts = np.array([float(t) for t in sorted_ts])
    ps = np.array([float(pmf[t]) for t in sorted_ts])
    upper_ps = np.array([float(upper_tail(pmf, t)) for t in sorted_ts])

    var = float(sum(pmf[t] * float(t) ** 2 for t in sorted_ts))
    sd = sqrt(var)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    # --- Panel 1: exact PMF --------------------------------------------------
    ax = axes[0]
    width = 0.8 * float(np.median(np.diff(ts)))
    ax.bar(ts, ps, width=width, color="steelblue", edgecolor="black",
           linewidth=0.5, alpha=0.85, label="exact PMF")
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.axvline(sd, color="orange", lw=0.9, ls="--",
               label=r"$\pm\,\mathrm{sd}_0(T)$")
    ax.axvline(-sd, color="orange", lw=0.9, ls="--")
    ax.axvline(2 * sd, color="crimson", lw=0.9, ls="--",
               label=r"$\pm\,2\,\mathrm{sd}_0(T)$")
    ax.axvline(-2 * sd, color="crimson", lw=0.9, ls="--")
    # Mark the two key quantiles with arrows
    for t_mark, label in [(Fraction(1, 3), r"$T=1/3$"),
                          (Fraction(2, 5), r"$T=2/5$")]:
        if t_mark in pmf:
            x0 = float(t_mark)
            y0 = float(pmf[t_mark])
            ax.annotate(label, (x0, y0),
                        xytext=(x0 + 0.18, y0 + 0.02 + 0.01 * (label == r"$T=1/3$")),
                        fontsize=9,
                        arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$P_0(T = t)$")
    ax.set_title(f"Exact PMF of $T$  ($p={p}$, $R={R}$)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-0.28, 1.08)
    ax.set_ylim(0, ps.max() * 1.18)

    # --- Panel 2: exact vs normal upper tail (linear y) ---------------------
    ax = axes[1]
    ax.step(ts, upper_ps, where="post", color="steelblue", lw=1.6,
            label="exact")
    ax.scatter(ts, upper_ps, color="steelblue", s=22, zorder=5)

    xgrid = np.linspace(ts.min(), 1.02, 400)
    normal_tail = np.array([0.5 * (1 - erf(x / (sd * sqrt(2))))
                             for x in xgrid])
    ax.plot(xgrid, normal_tail, color="crimson", lw=1.4, ls="--",
            label=r"normal  $1-\Phi(t/\mathrm{sd}_0)$")

    ax.axhline(0.05, color="gray", lw=0.6, ls=":")
    ax.text(0.82, 0.065, r"$5\%$", color="gray", fontsize=9)

    # Shade the disagreement region between exact and normal
    x_shade = np.linspace(ts.min(), ts.max(), 300)
    exact_interp = np.interp(x_shade, ts, upper_ps)
    # piecewise: use the step (right-continuous) interpretation
    # At x in [ts[i], ts[i+1]): survival = upper_ps[i]
    idx = np.searchsorted(ts, x_shade, side="right") - 1
    idx = np.clip(idx, 0, len(ts) - 1)
    exact_step = upper_ps[idx]
    normal_step = np.array([0.5 * (1 - erf(x / (sd * sqrt(2))))
                             for x in x_shade])
    ax.fill_between(x_shade, exact_step, normal_step,
                    where=(exact_step > normal_step),
                    color="steelblue", alpha=0.15,
                    label="exact exceeds normal")

    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(ts.min() - 0.02, 1.04)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$P_0(T \geq t)$")
    ax.set_title(f"Upper tail: exact vs. normal  ($p={p}$, $R={R}$)")
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_R_effect(p, R_values, path, pmfs=None):
    """Two-panel visualization of how the null distribution evolves with R.

    Left:  standardized PMFs (z = T / sd_0(T)) for each R, overlaid with a
           standard normal reference.
    Right: skewness and excess kurtosis as functions of R.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if pmfs is None:
        pmfs = {R: exact_null_pmf(p, R) for R in R_values}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    # --- Panel 1: standardized densities ---
    # Convert discrete PMF into a density-on-z by dividing by the local
    # grid spacing in z-units, so every curve (and the N(0, 1) reference)
    # is on the same density scale.
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.88, len(R_values)))
    for R, color in zip(R_values, cmap):
        pmf = pmfs[R]
        sorted_ts = sorted(pmf.keys())
        ts = np.array([float(t) for t in sorted_ts])
        ps = np.array([float(pmf[t]) for t in sorted_ts])
        mean, var, _, _ = moments(pmf)
        sd = sqrt(var)
        z = (ts - mean) / sd
        dz = float(np.median(np.diff(z)))
        dens = ps / dz                                      # PMF / delta z
        ax.plot(z, dens, "o-", color=color, lw=1.2,
                markersize=3.5, alpha=0.9, label=f"R={R}")
    xg = np.linspace(-3.5, 4.5, 400)
    ng = np.exp(-0.5 * xg ** 2) / sqrt(2 * np.pi)
    ax.plot(xg, ng, "k--", lw=1.2, label=r"$\mathcal{N}(0,1)$")
    ax.set_xlim(-3.5, 4.5)
    ax.set_xlabel(r"$z = (T - \mathrm{mean}) / \mathrm{sd}_0(T)$")
    ax.set_ylabel(r"density  (PMF / $\Delta z$)")
    ax.set_title(f"Standardized null densities  (p={p})")
    ax.legend(fontsize=8, loc="upper right")

    # --- Panel 2: moments vs R ---
    ax = axes[1]
    skews, ekurts = [], []
    for R in R_values:
        _, _, skew, ekurt = moments(pmfs[R])
        skews.append(skew)
        ekurts.append(ekurt)
    ax.plot(R_values, skews, "o-", color="crimson",
            label=r"skewness $\gamma_1$")
    ax.plot(R_values, ekurts, "s-", color="steelblue",
            label=r"excess kurtosis $\gamma_2$")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel(r"$R$")
    ax.set_ylabel("standardized moment")
    ax.set_title(f"Skewness and excess kurtosis vs. R  (p={p})")
    ax.legend(fontsize=9)
    ax.set_xticks(R_values)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------- main: validate the paper ----------------------------------------

def _format_fraction(t):
    if t == 0:
        return "0"
    return f"{t.numerator}/{t.denominator}"


def _main():
    p, R = 4, 5
    pmf = exact_null_pmf(p, R)
    n_tuples = factorial(p) ** (R - 1)

    print(f"Exact null distribution of T (p = {p}, R = {R})")
    print(f"Total tuples enumerated: {n_tuples:,}")
    print()
    header = f"{'T':>10} {'count':>10} {'P(T=t)':>12} {'P(T>=t)':>12}"
    print(header)
    print("-" * len(header))
    for t in sorted(pmf.keys()):
        c = int(pmf[t] * n_tuples)
        print(f"{_format_fraction(t):>10} {c:10d}"
              f" {float(pmf[t]):12.6f} {float(upper_tail(pmf, t)):12.6f}")

    mean, var, skew, ekurt = moments(pmf)
    print("\nExact moments of T:")
    print(f"  mean             = {mean:.10f}   (expected 0)")
    print(f"  variance         = {var:.10f}   (theory 13/540 "
          f"= {13 / 540:.10f})")
    print(f"  sd               = {sqrt(var):.10f}")
    print(f"  skewness         = {skew:.6f}    (Gaussian: 0)")
    print(f"  excess kurtosis  = {ekurt:.6f}    (Gaussian: 0)")

    sd = sqrt(var)
    print("\nNormal vs. exact upper tail:")
    print(f"{'t':>8} {'z=t/sd':>9} {'exact P':>12}"
          f" {'normal P':>12} {'ratio':>8}")
    for t_frac in [Fraction(1, 3), Fraction(2, 5), Fraction(7, 15),
                   Fraction(8, 15), Fraction(3, 5), Fraction(2, 3)]:
        t = float(t_frac)
        z = t / sd
        exact = float(upper_tail(pmf, t_frac))
        approx = _norm_sf(z)
        ratio = exact / approx if approx > 0 else float("inf")
        print(f"{t:8.4f} {z:9.3f} {exact:12.6f}"
              f" {approx:12.6f} {ratio:8.2f}")

    M = 2_000_000
    print(f"\nMonte Carlo (M = {M:,}, seed = 2026):")
    Ts = monte_carlo_T(p, R, M, seed=2026)
    print(f"  mean        = {Ts.mean():.6f}")
    print(f"  variance    = {Ts.var(ddof=1):.6f}")
    print(f"  P(T >= 1/3) = {(Ts >= 1 / 3).mean():.6f}"
          f"  (exact {float(upper_tail(pmf, Fraction(1, 3))):.6f})")
    print(f"  P(T >= 2/5) = {(Ts >= 2 / 5).mean():.6f}"
          f"  (exact {float(upper_tail(pmf, Fraction(2, 5))):.6f})")

    # Figure 1: p=4, R=5 PMF + tail comparison
    fig1 = "null_distribution_p4R5.pdf"
    plot_null_distribution(p, R, fig1, pmf=pmf)
    print(f"\nFigure written: {fig1}")

    # R = 2 Mahonian analysis
    print("\n--- R = 2: closed-form via Mahonian distribution ---")
    print(f"Inversion-count PMF for p={p}:")
    mah = mahonian_pmf(p)
    for k in sorted(mah):
        print(f"  K={k}: count={int(mah[k] * factorial(p)):3d}"
              f"  P={float(mah[k]):.6f}")
    print(f"\nResulting tau PMF (T = 1 - 2K/{comb(p, 2)}):")
    r2_pmf = r2_tau_pmf(p)
    for t in sorted(r2_pmf, reverse=True):
        print(f"  T={_format_fraction(t):>6}  P(T=t)={float(r2_pmf[t]):.6f}"
              f"  P(T>=t)={float(upper_tail(r2_pmf, t)):.6f}")

    # --- Study of the effect of R ---
    print("\n--- Effect of R (p = 4) ---")
    R_values = [2, 3, 4, 5, 6]
    print("Computing exact null PMFs for R in", R_values, "...")
    import time
    pmfs = {}
    for R_ in R_values:
        t0 = time.time()
        pmfs[R_] = exact_null_pmf(p, R_)
        print(f"  R={R_}:  {factorial(p) ** (R_ - 1):>10,} tuples  "
              f"({time.time() - t0:.2f}s)")

    rows = summary_table(p, R_values, pmfs=pmfs)
    print(f"\n{'R':>3} {'sd':>10} {'skew':>10} {'ex.kurt':>10}"
          f" {'5% thr.':>10} {'P(>=5%)':>10} {'1% thr.':>10} {'P(>=1%)':>10}")
    for row in rows:
        t5 = _format_fraction(row["t5"]) if row["t5"] is not None else "-"
        t1 = _format_fraction(row["t1"]) if row["t1"] is not None else "-"
        p5 = row["tail5"] if row["tail5"] is not None else float("nan")
        p1 = row["tail1"] if row["tail1"] is not None else float("nan")
        print(f"{row['R']:>3} {row['sd']:>10.6f} {row['skew']:>10.4f}"
              f" {row['ekurt']:>10.4f} {t5:>10} {p5:>10.6f}"
              f" {t1:>10} {p1:>10.6f}")

    # Figure 2: R-effect
    fig2 = "R_effect_p4.pdf"
    plot_R_effect(p, R_values, fig2, pmfs=pmfs)
    print(f"\nFigure written: {fig2}")

    # --- Choice of statistic: T vs. W_rho vs. max_tau vs. Kemeny ----
    print("\n--- Choice of statistic: exact null cutoffs (p=4, R=5) ---")
    import time
    t0 = time.time()
    stats_arr = exact_null_all_stats(p, R)
    print(f"Enumerated all 331,776 tuples in {time.time() - t0:.2f}s; "
          f"computed T, avg_rho, max_tau, Kemeny K*.")

    def _crit_upper(vals, alpha):
        vals_desc = np.sort(vals)[::-1]
        N = len(vals)
        last_c = None
        for v in vals_desc:
            if (vals >= v).sum() / N <= alpha:
                last_c = v
            else:
                break
        return last_c

    def _crit_lower(vals, alpha):
        vals_asc = np.sort(vals)
        N = len(vals)
        last_c = None
        for v in vals_asc:
            if (vals <= v).sum() / N <= alpha:
                last_c = v
            else:
                break
        return last_c

    cutoffs = {}
    print(f"\n{'statistic':<18} {'rej. rule':<10} {'5% cutoff':>12}"
          f" {'exact tail':>12} {'1% cutoff':>12} {'exact tail':>12}")
    for name, key, greater in [
        ("T (avg tau)",      "T",       True),
        ("W_rho (avg rho)",  "W_rho",   True),
        ("max pairwise tau", "max_tau", True),
        ("Kemeny K*",        "K_star",  False),
    ]:
        vals = stats_arr[key]
        if greater:
            c5 = _crit_upper(vals, 0.05)
            c1 = _crit_upper(vals, 0.01)
            t5 = (vals >= c5).mean() if c5 is not None else float("nan")
            t1 = (vals >= c1).mean() if c1 is not None else float("nan")
        else:
            c5 = _crit_lower(vals, 0.05)
            c1 = _crit_lower(vals, 0.01)
            t5 = (vals <= c5).mean() if c5 is not None else float("nan")
            t1 = (vals <= c1).mean() if c1 is not None else float("nan")
        cutoffs[name] = (c5, c1, greater)
        rule = ">= c" if greater else "<= c"
        c5_str = f"{c5:.4f}" if c5 is not None else "n/a"
        c1_str = f"{c1:.4f}" if c1 is not None else "n/a"
        print(f"{name:<18} {rule:<10} {c5_str:>12} {t5:>12.6f}"
              f" {c1_str:>12} {t1:>12.6f}")

    print("\nNote: for p=4, R=5, max_tau has no achievable 5% (or 10%) cutoff.")
    p_max1 = (stats_arr["max_tau"] >= 1).mean()
    print(f"Even the smallest possible rejection region {{max_tau >= 1}} has")
    print(f"exact null mass P(max_tau = 1) = {p_max1:.4f} >> 0.05. This is a")
    print("coarseness ceiling (perfect-coincidence mass too high for small p,R),")
    print("not a power problem -- max_tau is unsuitable as a formal 5% test here.")

    # --- Power vs. Mallows alternative ---
    print("\n--- Power vs. Mallows(theta) alternative, p=4, R=5, at 5% ---")
    M_power = 50_000
    print(f"Monte Carlo, M = {M_power:,}")
    header = (f"{'theta':>7} {'T':>10} {'W_rho':>10} {'K*':>10}")
    print(header)
    print("-" * len(header))
    for theta in [0.0, 0.2, 0.4, 0.6, 1.0, 1.5]:
        pis = mallows_draws(p, R, M_power, theta,
                            center=tuple(range(p)), seed=42)
        rej_T = (stat_T(pis) >= cutoffs["T (avg tau)"][0]).mean()
        rej_W = (stat_spearman_avg(pis)
                 >= cutoffs["W_rho (avg rho)"][0]).mean()
        rej_K = (stat_kemeny(pis) <= cutoffs["Kemeny K*"][0]).mean()
        print(f"{theta:7.2f} {rej_T:10.4f} {rej_W:10.4f} {rej_K:10.4f}")

    # --- Power vs. 2-cluster alternative ---
    print("\n--- Power vs. 2-cluster alternative, p=4, R=5 ---")
    print("(3 rankings Mallows(theta) around identity,"
          " 2 around reverse; at 5%)")
    center_A = tuple(range(p))
    center_B = tuple(range(p - 1, -1, -1))
    print(header)
    print("-" * len(header))
    for theta in [0.0, 0.5, 1.0, 2.0, 5.0]:
        A_draws = mallows_draws(p, 3, M_power, theta,
                                center=center_A, seed=100)
        B_draws = mallows_draws(p, 2, M_power, theta,
                                center=center_B, seed=101)
        pis = np.concatenate([A_draws, B_draws], axis=1)
        rej_T = (stat_T(pis) >= cutoffs["T (avg tau)"][0]).mean()
        rej_W = (stat_spearman_avg(pis)
                 >= cutoffs["W_rho (avg rho)"][0]).mean()
        rej_K = (stat_kemeny(pis) <= cutoffs["Kemeny K*"][0]).mean()
        print(f"{theta:7.2f} {rej_T:10.4f} {rej_W:10.4f} {rej_K:10.4f}")


if __name__ == "__main__":
    _main()

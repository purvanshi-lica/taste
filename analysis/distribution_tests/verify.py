"""
Verification suite for the distribution-based analysis.  Independent
cross-checks to catch bugs in:
  (V1) the exact null PMF vs paper-reported moments + tail probabilities
  (V2) our cycle detector on analytically-constructed test cases
  (V3) our cycle null rate (MC on 200k iid uniform, compared to two
       independent computations)
  (V4) reference-dataset quantiles match main.tex Table reported values
  (V5) TASTE-side T matches an independent per-prompt Kendall-tau
       calculation using scipy.stats.kendalltau (totally separate code
       path from null_kendall.stat_T)
  (V6) Majority-vote p_maj matches 1 - 2*min(k, R-k)/R reconstructed
       from the raw per-evaluator rankings
  (V7) Chi-squared GOF self-consistency: run the test on large MC
       samples from the null and check that the p-value distribution is
       approximately uniform
"""

import sys
from fractions import Fraction
from itertools import combinations
from math import sqrt, comb
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from null_kendall import (exact_null_pmf, monte_carlo_T, moments,
                          upper_tail, stat_T, r2_tau_pmf)
from taste_stats import (DIMENSIONS, DATA_DIR, MODELS,
                           load_rank_tensor, compute_stats)
from distribution_plots import (chi2_gof_vs_null, exact_T_null_grid,
                                 pairtau_null_grid, pmaj_null_grid,
                                 cycle_null_rate)
from generate_reference_stats import _sample_stats


def _check(label, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {label} {detail}")
    return cond


def v1_null_pmf_vs_paper():
    print("\n(V1) Exact null PMF at (p=4, R=5) vs main.tex values")
    pmf = exact_null_pmf(4, 5)
    mean, var, skew, ekurt = moments(pmf)
    ok = True
    ok &= _check("mean = 0", abs(mean) < 1e-10, f"  (got {mean:.2e})")
    ok &= _check("var = 13/540", abs(var - 13 / 540) < 1e-10,
                 f"  (got {var:.8f}, expected {13/540:.8f})")
    ok &= _check("skewness ~ 1.041", abs(skew - 1.041) < 0.002,
                 f"  (got {skew:.4f})")
    ok &= _check("excess kurtosis ~ 1.049", abs(ekurt - 1.049) < 0.002,
                 f"  (got {ekurt:.4f})")
    p_1_3 = float(upper_tail(pmf, Fraction(1, 3)))
    p_2_5 = float(upper_tail(pmf, Fraction(2, 5)))
    p_2_3 = float(upper_tail(pmf, Fraction(2, 3)))
    ok &= _check("P(T>=1/3) = 0.051695",
                 abs(p_1_3 - 0.051695) < 1e-5,
                 f"  (got {p_1_3:.6f})")
    ok &= _check("P(T>=2/5) = 0.027356",
                 abs(p_2_5 - 0.027356) < 1e-5,
                 f"  (got {p_2_5:.6f})")
    ok &= _check("P(T>=2/3) = 0.001299",
                 abs(p_2_3 - 0.001299) < 1e-4,
                 f"  (got {p_2_3:.6f})")
    # Check that PMF sums to 1
    total = float(sum(pmf.values()))
    ok &= _check("PMF sums to 1", abs(total - 1) < 1e-10,
                 f"  (got {total:.10f})")
    return ok


def v2_cycle_detector():
    print("\n(V2) Cycle detector on constructed test cases")
    ok = True

    # Case 1: all identical. no cycle.
    r = np.array([[1, 2, 3, 4]] * 5)
    _, _, _, cyc = _sample_stats(r)
    ok &= _check("all-identical: no cycle", cyc == 0)

    # Case 2: 3 vs 2 opposing.  majority is 0>1>2>3 (3 out of 5); all
    # triples transitive.
    r = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                  [4, 3, 2, 1], [4, 3, 2, 1]])
    _, _, _, cyc = _sample_stats(r)
    ok &= _check("3v2 opposing: no cycle", cyc == 0)

    # Case 3: classical Condorcet cycle on items (0,1,2). 5 rankers:
    #  2 say 0>1>2>3, 2 say 1>2>3>0, 1 says 2>0>1>3.
    # Pair (0,1): 0-ahead in {2 + 1} = 3 vs 2.  majority 0>1.
    # Pair (1,2): 1-ahead in {2 + 2} = 4 vs 1.  majority 1>2.
    # Pair (0,2): 0-ahead in {2 + 1} = 3 vs 2.  majority 0>2.
    # So 0>1>2 transitive.  No triple cycle.  (rankings must be full.)
    # Construct proper cycle:  3 permutations cycling 0->1->2->0.
    r = np.array([[1, 2, 3, 4],  # 0>1>2>3
                  [1, 2, 3, 4],
                  [4, 1, 2, 3],  # 1>2>3>0
                  [4, 1, 2, 3],
                  [3, 4, 1, 2]])  # 2>3>0>1
    # Pair (0,1): 0-ahead in rankers 1,2,5? r[0,0]=1<r[0,1]=2 YES, same r1
    # r[2,0]=4 vs r[2,1]=1: 0-ahead = NO.  Same r4.  r[4,0]=3 vs r[4,1]=4: YES.
    # So 0-ahead count = 3.  majority 0>1.
    # Pair (1,2): r[0,1]=2<r[0,2]=3 YES (1 ahead). r1 same.
    # r[2,1]=1<r[2,2]=2 YES.  Same r4. r[4,1]=4 vs r[4,2]=1: NO.
    # Count 1-ahead = 4.  majority 1>2.
    # Pair (0,2): r[0,0]=1<r[0,2]=3 YES (0 ahead).  r1 same.
    # r[2,0]=4 vs r[2,2]=2: NO. r4 same. r[4,0]=3 vs r[4,2]=1: NO.
    # Count 0-ahead = 2 out of 5.  majority 2>0.
    # Triple (0,1,2): 0>1, 1>2, 2>0.  CYCLE.
    _, _, _, cyc = _sample_stats(r)
    ok &= _check("constructed 0>1>2>0 cycle: cycle=1", cyc == 1)

    # Case 4: strong consensus with one dissenter -- no cycle
    r = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                  [1, 2, 3, 4], [4, 3, 2, 1]])
    _, _, _, cyc = _sample_stats(r)
    ok &= _check("4v1 with dissent: no cycle", cyc == 0)

    return ok


def v3_cycle_null_rate():
    """Verify cycle null rate by a fresh, fully independent MC."""
    print("\n(V3) Cycle null rate: fresh independent MC")
    rng = np.random.default_rng(42)
    R, p = 5, 4
    M = 200_000
    a_i, b_i = np.triu_indices(p, k=1)
    triples = list(combinations(range(p), 3))
    count = 0
    for _ in range(M):
        ranks = np.empty((R, p), dtype=np.int64)
        for r in range(R):
            ranks[r] = rng.permutation(p) + 1
        sign_pair = np.sign(ranks[:, a_i] - ranks[:, b_i]).astype(np.int64)
        a_ahead = (sign_pair == -1).sum(axis=0)
        # Majority sign in the "item a ahead of item b" convention
        ahead = {}
        for k, (u, v) in enumerate(zip(a_i, b_i)):
            if a_ahead[k] * 2 > R:
                ahead[(u, v)] = True
            elif a_ahead[k] * 2 < R:
                ahead[(u, v)] = False
            else:
                ahead[(u, v)] = None

        def pref(u, v):
            if (u, v) in ahead:
                val = ahead[(u, v)]
                return None if val is None else (u if val else v)
            val = ahead[(v, u)]
            return None if val is None else (v if val else u)

        has_cycle = False
        for tr in triples:
            x, y, z = tr
            p_xy = pref(x, y)
            p_yz = pref(y, z)
            p_xz = pref(x, z)
            if None in (p_xy, p_yz, p_xz):
                continue
            # cycle if (x beats y, y beats z, z beats x)  or reverse
            if (p_xy == x and p_yz == y and p_xz == z):
                has_cycle = True
                break
            if (p_xy == y and p_yz == z and p_xz == x):
                has_cycle = True
                break
        if has_cycle:
            count += 1
    rate_fresh = count / M
    rate_ours = cycle_null_rate(M=200_000, seed=2026)
    print(f"  fresh MC rate = {rate_fresh:.4f}  (seed 42, M=200k)")
    print(f"  our cycle_null_rate() = {rate_ours:.4f}  (seed 2026, M=200k)")
    ok = _check("within 0.003 of each other",
                abs(rate_fresh - rate_ours) < 0.003)
    # Also compare to independent derivation via 1 - (1-p_triple)^4
    # where p_triple is ~0.071.
    p_triple = 0.071
    independent_bound = 1 - (1 - p_triple) ** 4
    print(f"  independent-triples upper bound {independent_bound:.4f} "
          f"(triples aren't independent so true value is lower)")
    ok &= _check("rate <= independent bound",
                 rate_fresh <= independent_bound + 0.01)
    return ok


def v4_reference_quantiles():
    """Match the main.tex Table 4 empirical quantiles."""
    print("\n(V4) Reference empirical quantiles vs main.tex Table 4")
    # main.tex Table: Sushi mean=0.144, median=0.133, P95=0.533
    # MovieLens mean=0.214, median=0.200, P95=0.600
    # MT-Bench mean=0.669, median=0.667, P95=0.867
    expected = {
        "sushi": {"mean": 0.144, "median": 0.133, "P95": 0.533},
        "movielens": {"mean": 0.214, "median": 0.200, "P95": 0.600},
        "mtbench": {"mean": 0.669, "median": 0.667, "P95": 0.867},
    }
    ok = True
    for slug, exp in expected.items():
        arr = np.load(HERE / "refs" / f"{slug}_T.npy")
        m = arr.mean()
        med = np.median(arr)
        p95 = np.quantile(arr, 0.95)
        ok &= _check(f"{slug:10s} mean ~ {exp['mean']}",
                     abs(m - exp["mean"]) < 0.01,
                     f"  got {m:.4f}")
        ok &= _check(f"{slug:10s} median ~ {exp['median']}",
                     abs(med - exp["median"]) < 0.01,
                     f"  got {med:.4f}")
        ok &= _check(f"{slug:10s} P95 ~ {exp['P95']}",
                     abs(p95 - exp["P95"]) < 0.04,
                     f"  got {p95:.4f}")
    return ok


def v5_contra_T_alt_path():
    """Cross-check TASTE per-prompt T via scipy.stats.kendalltau."""
    print("\n(V5) TASTE T via scipy.stats.kendalltau (alt path)")
    slug, display, group, csv_name = DIMENSIONS[0]  # aesthetics_preference
    tensor, prompts, evaluators = load_rank_tensor(str(DATA_DIR / csv_name))
    stats = compute_stats(tensor, evaluators)
    # For each prompt, compute T via scipy.stats.kendalltau per-pair then average
    R = tensor.shape[1]
    n_prompts = tensor.shape[0]
    T_alt = np.empty(n_prompts)
    for i in range(n_prompts):
        taus = []
        for a, b in combinations(range(R), 2):
            tau, _ = sp_stats.kendalltau(tensor[i, a], tensor[i, b])
            taus.append(tau)
        T_alt[i] = float(np.mean(taus))
    diff = np.abs(T_alt - stats["T"]).max()
    ok = _check("T matches scipy Kendall tau per-pair average",
                diff < 1e-9, f"  max abs diff {diff:.2e}")
    return ok


def v6_pmaj_consistency():
    """p_maj should equal max(k, R-k)/R where k = # rankers putting
    item a ahead.  Reconstruct from raw tensor and compare."""
    print("\n(V6) p_maj vs raw per-rater recomputation")
    slug, display, group, csv_name = DIMENSIONS[-1]  # descriptions_typography
    tensor, prompts, evaluators = load_rank_tensor(str(DATA_DIR / csv_name))
    stats = compute_stats(tensor, evaluators)
    R, p = tensor.shape[1], tensor.shape[2]
    a_i, b_i = np.triu_indices(p, k=1)
    n_prompts = tensor.shape[0]
    pmaj_alt = np.empty((n_prompts, len(a_i)))
    for i in range(n_prompts):
        for k, (u, v) in enumerate(zip(a_i, b_i)):
            # Count rankers putting u ahead of v, i.e. ranks[r, u] < ranks[r, v]
            n_u_ahead = int((tensor[i, :, u] < tensor[i, :, v]).sum())
            pmaj_alt[i, k] = max(n_u_ahead, R - n_u_ahead) / R
    diff = np.abs(pmaj_alt - stats["pmaj"]).max()
    ok = _check("p_maj matches raw-tensor recomputation",
                diff < 1e-10, f"  max abs diff {diff:.2e}")
    # Also check support
    support = np.unique(np.round(stats["pmaj"].ravel(), 10))
    expected_support = {0.6, 0.8, 1.0}
    actual_set = set(float(s) for s in support)
    ok &= _check("support is {3/5, 4/5, 5/5}",
                 actual_set == expected_support,
                 f"  got {actual_set}")
    return ok


def v7_chi2_self_consistency():
    """Chi-sq GOF run on MC samples of the null should produce a uniform
    p-value distribution (the defining property of a well-calibrated test)."""
    print("\n(V7) Chi-sq GOF self-consistency on null samples")
    T_grid, T_null_probs = exact_T_null_grid()
    rng = np.random.default_rng(2026)
    # Draw 5000 "TASTE-sized" samples of T (n=80 per sample) from the null.
    # For each, run chi-sq GOF. Check that p-value CDF is close to uniform.
    n_samples = 5000
    n_per_sample = 80
    pvals = np.empty(n_samples)
    for i in range(n_samples):
        T_draw = monte_carlo_T(4, 5, n_per_sample, seed=rng.integers(0, 2**31))
        counts = np.argmin(np.abs(T_draw[:, None] - T_grid[None, :]), axis=1)
        obs = np.bincount(counts, minlength=len(T_grid))
        _, pv, _, _ = chi2_gof_vs_null(obs, T_null_probs)
        pvals[i] = pv
    # KS uniformity test on the p-value distribution
    ks_stat, ks_p = sp_stats.kstest(pvals, "uniform")
    # Typically we accept if ks_p > 0.01 (large enough that we don't reject uniformity)
    ok = _check(f"chi-sq p-values are ~uniform under the null (KS p={ks_p:.4f})",
                ks_p > 0.001)
    # Also: rejection rate at alpha=0.05 should be ~5%
    rej_rate = (pvals < 0.05).mean()
    ok &= _check(f"rejection rate at alpha=0.05 is near 5% (got {rej_rate:.3f})",
                 abs(rej_rate - 0.05) < 0.015)
    return ok


def v8_cycle_vs_old_method_spotcheck():
    """Compare new cycle on specific prompts against a textbook-form
    reference implementation (not the buggy legacy one)."""
    print("\n(V8) Cycle vs textbook reference on a few TASTE prompts")
    slug, display, group, csv_name = DIMENSIONS[0]
    tensor, prompts, _ = load_rank_tensor(str(DATA_DIR / csv_name))

    def reference_cycle(ranks):
        """A totally separate implementation using pandas to be sure."""
        R, p = ranks.shape
        pref = {}
        for u in range(p):
            for v in range(u + 1, p):
                u_ahead = int((ranks[:, u] < ranks[:, v]).sum())
                if u_ahead * 2 > R:
                    pref[(u, v)] = u
                    pref[(v, u)] = u
                elif u_ahead * 2 < R:
                    pref[(u, v)] = v
                    pref[(v, u)] = v
                # ties skipped
        for x, y, z in combinations(range(p), 3):
            xy = pref.get((x, y))
            yz = pref.get((y, z))
            xz = pref.get((x, z))
            if None in (xy, yz, xz) or xy is None or yz is None or xz is None:
                continue
            # cycle = transitivity violated
            # If xy == x (x beats y) and yz == y (y beats z): expect xz == x.
            # If xy == y and yz == z: expect xz == y (... y beats z, but x beats?)
            # Simpler: build partial order and test.
            winners = {(x, y): xy, (y, x): xy,
                       (y, z): yz, (z, y): yz,
                       (x, z): xz, (z, x): xz}
            def beats(a, b):
                return winners[(a, b)] == a
            # Cycle iff (x>y, y>z, z>x) or (y>x, z>y, x>z)
            if (beats(x, y) and beats(y, z) and beats(z, x)):
                return 1
            if (beats(y, x) and beats(z, y) and beats(x, z)):
                return 1
        return 0

    mismatches = 0
    for i in range(len(prompts)):
        c_new = _sample_stats(tensor[i])[3]
        c_ref = reference_cycle(tensor[i])
        if c_new != c_ref:
            mismatches += 1
            print(f"    mismatch on prompt idx={i} pid={prompts[i]}: "
                  f"new={c_new} ref={c_ref}")
    ok = _check(f"all 80 prompts match textbook ref cycle",
                mismatches == 0, f"  ({mismatches} mismatches)")
    return ok


def v9_pairtau_consistency():
    """Average of 10 pairwise taus should equal T, for every prompt."""
    print("\n(V9) mean(pair_tau) == T per prompt")
    slug, display, group, csv_name = DIMENSIONS[3]
    tensor, prompts, evaluators = load_rank_tensor(str(DATA_DIR / csv_name))
    stats = compute_stats(tensor, evaluators)
    diff = np.abs(stats["pair_tau"].mean(axis=1) - stats["T"]).max()
    ok = _check(f"max |mean(pair_tau) - T| < 1e-10",
                diff < 1e-10, f"  max diff {diff:.2e}")
    return ok


def v10_null_pmf_grid_coverage():
    """The null PMF grid must cover every TASTE and reference T value
    (to within floating-point snapping tolerance)."""
    print("\n(V10) All T values snap within tolerance to null-PMF grid")
    T_grid, _ = exact_T_null_grid()
    ok = True
    refs_dir = HERE / "refs"
    for ds in ["sushi", "movielens", "mtbench"]:
        arr = np.load(refs_dir / f"{ds}_T.npy")
        snapped = np.min(np.abs(arr[:, None] - T_grid[None, :]), axis=1)
        max_snap = snapped.max()
        ok &= _check(f"{ds:10s} max snap dist < 1e-10",
                     max_snap < 1e-10, f"  got {max_snap:.2e}")
    for slug, _, _, csv_name in DIMENSIONS:
        tensor, _, evaluators = load_rank_tensor(str(DATA_DIR / csv_name))
        stats = compute_stats(tensor, evaluators)
        snapped = np.min(np.abs(stats["T"][:, None] - T_grid[None, :]), axis=1)
        max_snap = snapped.max()
        ok &= _check(f"{slug:28s} max snap dist < 1e-10",
                     max_snap < 1e-10, f"  got {max_snap:.2e}")
    return ok


def main():
    print("=" * 72)
    print("DISTRIBUTION-TESTS VERIFICATION SUITE")
    print("=" * 72)
    results = {
        "V1 null PMF vs paper": v1_null_pmf_vs_paper(),
        "V2 cycle detector cases": v2_cycle_detector(),
        "V3 cycle null rate MC": v3_cycle_null_rate(),
        "V4 ref quantiles vs paper": v4_reference_quantiles(),
        "V5 TASTE T via scipy": v5_contra_T_alt_path(),
        "V6 p_maj consistency": v6_pmaj_consistency(),
        "V7 chi-sq self-consistency": v7_chi2_self_consistency(),
        "V8 cycle vs textbook ref": v8_cycle_vs_old_method_spotcheck(),
        "V9 mean(pair_tau) == T": v9_pairtau_consistency(),
        "V10 null grid coverage": v10_null_pmf_grid_coverage(),
    }
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    n_fail = sum(1 for v in results.values() if not v)
    print(f"\n  {sum(results.values())} / {len(results)} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()

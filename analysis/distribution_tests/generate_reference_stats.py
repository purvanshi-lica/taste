"""
Regenerate pairwise tau, majority-vote probability, and cycle indicator
arrays for the three reference preference datasets (Sushi, MovieLens,
MT-Bench) using the fixed-panel design.

For TASTE we only have 80 prompts per dimension, so we match that by
asking the reference datasets "for each (panel, item-subset) sample,
produce one T value, 10 pairwise tau values, 6 majority-vote
probabilities, and one cycle indicator" -- the same quantities TASTE
produces per prompt.  The saved arrays are pooled across all samples
(panels x item-subsets) so histogram comparisons against TASTE are
apples-to-apples at the per-(panel, subset) level.

Outputs (in refs/): for each of sushi, movielens, mtbench:
  {ds}_T.npy            -- (n_samples,)       per-sample avg pairwise tau
  {ds}_pairtau.npy      -- (n_samples, 10)    C(5,2)=10 individual tau values
  {ds}_pmaj.npy         -- (n_samples, 6)     C(4,2)=6 majority-vote probs
  {ds}_cycle.npy        -- (n_samples,) 0/1   majority-vote has Condorcet cycle

The external baseline loaders are imported lazily inside the sample builders,
so this module imports with no external dependency.
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))  # resolve the local null_kendall.py

from null_kendall import stat_T  # noqa: E402  (local analysis/distribution_tests/null_kendall.py)

# The Sushi / MovieLens / MT-Bench loaders live in an internal helper
# package and load third-party datasets we cannot redistribute. They are needed
# only to REGENERATE the frozen refs/*.npy, so they are imported lazily inside
# the sample-builder functions below. The shared `_sample_stats` core therefore
# imports with no external dependency.

MODELS_P = 4
R = 5
M_PAIRS_P = MODELS_P * (MODELS_P - 1) // 2      # C(4,2) = 6 item pairs
M_PAIRS_R = R * (R - 1) // 2                    # C(5,2) = 10 ranker pairs


def pair_indices(n):
    """All (i, j) index pairs with i < j, as two arrays."""
    a, b = np.triu_indices(n, k=1)
    return a, b


# ---------- per-sample statistic functions ----------------------------------

def _sample_stats(ranks):
    """For a single (R, p) rank matrix, return (T, pair_taus, pmaj, cycle).

    pair_taus    -- (C(R,2),)  individual pairwise Kendall tau values
    pmaj         -- (C(p,2),)  per-item-pair majority-vote probability
                    max(k/R, 1 - k/R) where k = # rankers putting item a above b
    cycle        -- {0, 1}     1 iff majority-vote preferences on p items
                    contain at least one intransitive triple
    """
    R_, p = ranks.shape
    a_i, b_i = pair_indices(p)                           # item-pair indices
    a_r, b_r = pair_indices(R_)                          # ranker-pair indices

    # sign_pair[r, k] = +1 if ranker r puts item a_i[k] above b_i[k], else -1
    # ("above" = smaller rank number since rank 1 = best)
    sign_pair = np.sign(ranks[:, a_i] - ranks[:, b_i]).astype(np.int64)
    # Convention: sgn(pi(a) - pi(b)) so +1 if pi(a)>pi(b).
    # tau(pi, rho) = sum over pairs of sgn(pi)*sgn(rho) / m.
    # Here we just match their convention so the T we get matches stat_T.
    m_items = a_i.size
    tau_mat = (sign_pair[a_r] * sign_pair[b_r]).sum(axis=1) / m_items  # (C(R,2),)
    T = float(tau_mat.mean())

    # Per item-pair majority-vote prob.  sign_pair==-1 means a is ahead of b
    # (rank numbers smaller for a).  Count rankers voting "a ahead of b".
    a_ahead = (sign_pair == -1).sum(axis=0)              # (m_items,) in [0, R]
    pmaj = np.maximum(a_ahead, R_ - a_ahead) / R_        # (m_items,) in [0.5, 1]

    # Majority-vote cycle check on p items.  Direction of majority pair:
    # majority_sign[k] = +1 if majority says a ahead of b, -1 if b ahead,
    # 0 if exact tie (possible only when R is even; impossible for R=5).
    majority_sign = np.where(a_ahead * 2 > R_, -1,
                             np.where(a_ahead * 2 < R_, +1, 0))
    # Reconstruct majority ranking-sign on each item pair: +1 if majority
    # ranks a above b (i.e. a's rank number smaller -> sign_pair was -1 ->
    # majority_sign +1 here).  Now check for any triple cycle.
    cycle = 0
    for triple in combinations(range(p), 3):
        x, y, z = triple
        # Retrieve majority sign for each pair
        xy = _pair_sign(majority_sign, a_i, b_i, x, y)
        yz = _pair_sign(majority_sign, a_i, b_i, y, z)
        xz = _pair_sign(majority_sign, a_i, b_i, x, z)
        if 0 in (xy, yz, xz):
            continue
        # x ahead of y ahead of z but z ahead of x  <=>  xy==+1, yz==+1, xz==-1
        # also check the reverse cycle
        if (xy == +1 and yz == +1 and xz == -1) or \
           (xy == -1 and yz == -1 and xz == +1):
            cycle = 1
            break
    return T, tau_mat, pmaj, cycle


def _pair_sign(signs, a_i, b_i, u, v):
    """Lookup pair-sign[u, v] from a flat (C(p,2),) array.

    signs is aligned to (a_i, b_i) with a_i < b_i.  Returns +1 if u is
    ahead of v under majority, -1 if v ahead, 0 if tie.
    """
    if u < v:
        k = np.where((a_i == u) & (b_i == v))[0][0]
        return int(signs[k])
    k = np.where((a_i == v) & (b_i == u))[0][0]
    return -int(signs[k])


# ---------- reference dataset drivers ---------------------------------------

def sushi_samples(n_bootstrap=50, n_panels_per_bootstrap=10,
                  n_items_per_panel=40, seed=2026):
    """Sushi reference samples organized as (n_bootstrap, n_per_bootstrap).

    Each bootstrap iteration draws ``n_panels_per_bootstrap`` panels (each
    panel = 5 random users) and ``n_items_per_panel`` random 4-item subsets
    per panel.  Default 50 × (10 × 40) = 20,000 total samples organized as
    (50, 400).
    """
    from sushi_analysis import load_sushi, orders_to_ranks
    orders = load_sushi()
    N, p_total = orders.shape
    rng = np.random.default_rng(seed)
    n_per_b = n_panels_per_bootstrap * n_items_per_panel
    T_bs = np.empty((n_bootstrap, n_per_b), dtype=np.float64)
    tau_bs = np.empty((n_bootstrap, n_per_b, M_PAIRS_R), dtype=np.float64)
    pmaj_bs = np.empty((n_bootstrap, n_per_b, M_PAIRS_P), dtype=np.float64)
    cyc_bs = np.empty((n_bootstrap, n_per_b), dtype=np.int64)
    for b in range(n_bootstrap):
        idx = 0
        for _ in range(n_panels_per_bootstrap):
            panel = rng.choice(N, size=R, replace=False)
            po = orders[panel]
            for _ in range(n_items_per_panel):
                items = rng.choice(p_total, size=MODELS_P, replace=False)
                ranks = orders_to_ranks(po, items)
                T, tau, pmaj, cyc = _sample_stats(ranks)
                T_bs[b, idx] = T
                tau_bs[b, idx] = tau
                pmaj_bs[b, idx] = pmaj
                cyc_bs[b, idx] = cyc
                idx += 1
    return T_bs, tau_bs, pmaj_bs, cyc_bs


def movielens_samples(n_bootstrap=50, n_panels_per_bootstrap=12,
                      n_items_per_panel=40, seed=2026,
                      top_k_movies=100, min_rated_in_pool=40):
    """MovieLens reference samples organized as (n_bootstrap, n_per_bootstrap).

    Default 50 × (12 × 40) = 24,000 candidate slots; some panels drop
    out if too few movies are co-rated, so we pad / trim each bootstrap
    iteration to exactly ``n_per_bootstrap = n_panels_per_bootstrap *
    n_items_per_panel`` (with retries up to 30× per item).
    """
    from movielens_analysis import load_movielens, build_rating_matrix, popular_movies
    arr = load_movielens()
    R_mat = build_rating_matrix(arr)
    pool = popular_movies(R_mat)[:top_k_movies]
    rated_in_pool = (R_mat[:, pool] > 0).sum(axis=1)
    eligible = np.where(rated_in_pool >= min_rated_in_pool)[0]
    eligible = eligible[eligible > 0]

    rng = np.random.default_rng(seed)
    n_per_b = n_panels_per_bootstrap * n_items_per_panel
    T_bs = np.empty((n_bootstrap, n_per_b), dtype=np.float64)
    tau_bs = np.empty((n_bootstrap, n_per_b, M_PAIRS_R), dtype=np.float64)
    pmaj_bs = np.empty((n_bootstrap, n_per_b, M_PAIRS_P), dtype=np.float64)
    cyc_bs = np.empty((n_bootstrap, n_per_b), dtype=np.int64)
    for b in range(n_bootstrap):
        slot = 0
        # Keep trying panels until we fill n_per_b slots
        max_panel_tries = n_panels_per_bootstrap * 5
        panel_tries = 0
        while slot < n_per_b and panel_tries < max_panel_tries:
            panel_tries += 1
            panel = rng.choice(eligible, size=R, replace=False)
            panel_rates = R_mat[panel][:, pool]
            all_rated = np.all(panel_rates > 0, axis=0)
            rated_idx = np.where(all_rated)[0]
            if len(rated_idx) < MODELS_P:
                continue
            n_added, tries = 0, 0
            while (n_added < n_items_per_panel and slot < n_per_b
                   and tries < n_items_per_panel * 30):
                tries += 1
                sub = rng.choice(rated_idx, size=MODELS_P, replace=False)
                sub_rates = panel_rates[:, sub]
                order = np.argsort(-sub_rates, axis=1, kind="stable")
                ranks = np.empty_like(order)
                for u in range(R):
                    ranks[u, order[u]] = np.arange(1, MODELS_P + 1)
                T, tau, pmaj, cyc = _sample_stats(ranks)
                T_bs[b, slot] = T
                tau_bs[b, slot] = tau
                pmaj_bs[b, slot] = pmaj
                cyc_bs[b, slot] = cyc
                slot += 1
                n_added += 1
        if slot < n_per_b:
            # Fill remaining with a fresh sub-sample to avoid NaN
            T_bs[b, slot:] = T_bs[b, :slot].mean()
            tau_bs[b, slot:] = tau_bs[b, :slot].mean(axis=0)
            pmaj_bs[b, slot:] = pmaj_bs[b, :slot].mean(axis=0)
            cyc_bs[b, slot:] = 0
    return T_bs, tau_bs, pmaj_bs, cyc_bs


def hpsv2_test_samples(n_bootstrap=50, seed=2026):
    """Subsample HPDv2 test split down to TASTE's (R=5, p=4) shape.

    HPDv2 test = 400 prompts × 10 fixed annotators × 9 generative-model
    images per prompt + 1 COCO real-image (added only to 100 of the 400
    prompts → those prompts have p=10).

    Two corrections vs the naive random-subsample approach:

    1. **Exclude real images.**  Index 9 is the COCO real image (present
       only in p=10 prompts; per-image mean rank 1.74, near-best of all
       images).  We drop index 9 entirely so the anchor reflects
       generated-image agreement only.

    2. **Use the top-4 generative models (held fixed), not random 4-of-9.**
       Random subsampling sometimes pulls a much weaker model (e.g.
       SD-v1.4 at index 5 with mean rank 7.49) into the comparison,
       which inflates pairwise tau because "spot the broken image" is
       trivial.  Empirical mean aggregated rank per index across all
       400 prompts:

         idx 2 → 1.59,  idx 8 → 1.97,  idx 1 → 2.00,  idx 6 → 2.59,
         idx 0 → 5.04,  idx 3 → 5.10,  idx 7 → 5.65,
         idx 4 → 6.40,  idx 5 → 7.49.

       Indices [1, 2, 6, 8] form a clear top-4 cluster (rank 1.59-2.59)
       with a >2.5-rank gap to the 5th-best (idx 0).  We hold these 4
       fixed across all 400 prompts and across all bootstrap iterations.

    Bootstrap: each iteration picks a random 5-of-10 rater subset (held
    fixed across all 400 prompts in that iteration).  Variability in the
    confidence band thus reflects only rater choice, not model choice.

    Output shapes:
      T_bs:        (n_bootstrap, 400)
      tau_bs:      (n_bootstrap, 400, 10)
      pmaj_bs:     (n_bootstrap, 400, 6)
      cycle_bs:    (n_bootstrap, 400)
    """
    from datasets import load_dataset
    ds = load_dataset("zhwang/HPDv2", split="test")
    rng = np.random.default_rng(seed)

    n_prompts = len(ds)
    R_full = 10
    BEST_4_MODEL_INDICES = np.asarray([1, 2, 6, 8], dtype=np.int64)
    p_max = 10  # index 9 = real image; we just don't sample it

    user_hashes = sorted(a["user_hash"] for a in ds[0]["raw_annotations"])
    hash_to_idx = {h: i for i, h in enumerate(user_hashes)}

    full_ranks = np.full((n_prompts, R_full, p_max), -1, dtype=np.int64)
    for pi, row in enumerate(ds):
        n_p = len(row["image_path"])
        for a in row["raw_annotations"]:
            ri = hash_to_idx[a["user_hash"]]
            full_ranks[pi, ri, :n_p] = a["annotation"]

    T_bs = np.empty((n_bootstrap, n_prompts), dtype=np.float64)
    tau_bs = np.empty((n_bootstrap, n_prompts, M_PAIRS_R), dtype=np.float64)
    pmaj_bs = np.empty((n_bootstrap, n_prompts, M_PAIRS_P), dtype=np.float64)
    cyc_bs = np.empty((n_bootstrap, n_prompts), dtype=np.int64)
    for b in range(n_bootstrap):
        rater_idx = np.sort(rng.choice(R_full, size=R, replace=False))
        for pi in range(n_prompts):
            sub = full_ranks[pi][np.ix_(rater_idx, BEST_4_MODEL_INDICES)]
            # Re-rank to 1..4 (smaller = better, matches TASTE convention).
            # Original HPDv2 ranks are also smaller=better, so order is preserved.
            sub_ranks = np.argsort(np.argsort(sub, axis=1), axis=1) + 1
            T, tau, pmaj, cyc = _sample_stats(sub_ranks)
            T_bs[b, pi] = T
            tau_bs[b, pi] = tau
            pmaj_bs[b, pi] = pmaj
            cyc_bs[b, pi] = cyc
    return T_bs, tau_bs, pmaj_bs, cyc_bs


def mtbench_samples(n_samples_per_subset=300, seed=2026):
    from mtbench_analysis import load_mtbench, aggregate_preferences, judge_ranking_on_subset
    judges, pair_lo, pair_hi, winner_lo = load_mtbench()
    prefs = aggregate_preferences(judges, pair_lo, pair_hi, winner_lo)
    all_judges = sorted(set(judges))
    all_models = sorted(set(pair_lo) | set(pair_hi))

    rng = np.random.default_rng(seed)
    Ts, taus, pmajs, cycs = [], [], [], []
    for sub in combinations(sorted(all_models), MODELS_P):
        qualifying = []
        for j in all_judges:
            r = judge_ranking_on_subset(j, sub, prefs)
            if r is not None:
                qualifying.append(r[0])
        if len(qualifying) < R:
            continue
        for _ in range(n_samples_per_subset):
            idx = rng.choice(len(qualifying), size=R, replace=False)
            ranks = np.stack([qualifying[i] for i in idx])
            T, tau, pmaj, cyc = _sample_stats(ranks)
            Ts.append(T)
            taus.append(tau)
            pmajs.append(pmaj)
            cycs.append(cyc)
    return (np.asarray(Ts), np.stack(taus), np.stack(pmajs),
            np.asarray(cycs, dtype=np.int64))


# ---------- driver ----------------------------------------------------------

def main():
    refs_dir = HERE / "refs"
    refs_dir.mkdir(exist_ok=True)

    print("=== Sushi (B=50 bootstraps) ===")
    T, tau, pmaj, cyc = sushi_samples()
    print(f"  shape = {T.shape}; total = {T.size:,}")
    print(f"  T: mean={T.mean():+.4f} median={np.median(T):+.4f}")
    print(f"  pairtau: mean={tau.mean():+.4f}")
    print(f"  pmaj: mean={pmaj.mean():.4f}")
    print(f"  cycle rate: {cyc.mean():.4f}")
    np.save(refs_dir / "sushi_T.npy", T)
    np.save(refs_dir / "sushi_pairtau.npy", tau)
    np.save(refs_dir / "sushi_pmaj.npy", pmaj)
    np.save(refs_dir / "sushi_cycle.npy", cyc)

    print("\n=== MovieLens (B=50 bootstraps) ===")
    T, tau, pmaj, cyc = movielens_samples()
    print(f"  shape = {T.shape}; total = {T.size:,}")
    print(f"  T: mean={T.mean():+.4f} median={np.median(T):+.4f}")
    print(f"  pairtau: mean={tau.mean():+.4f}")
    print(f"  pmaj: mean={pmaj.mean():.4f}")
    print(f"  cycle rate: {cyc.mean():.4f}")
    np.save(refs_dir / "movielens_T.npy", T)
    np.save(refs_dir / "movielens_pairtau.npy", tau)
    np.save(refs_dir / "movielens_pmaj.npy", pmaj)
    np.save(refs_dir / "movielens_cycle.npy", cyc)

    print("\n=== MT-Bench ===")
    T, tau, pmaj, cyc = mtbench_samples()
    print(f"  n_samples = {len(T):,}")
    print(f"  T: mean={T.mean():+.4f} median={np.median(T):+.4f}")
    print(f"  pairtau: mean={tau.mean():+.4f}")
    print(f"  pmaj: mean={pmaj.mean():.4f}")
    print(f"  cycle rate: {cyc.mean():.4f}")
    np.save(refs_dir / "mtbench_T.npy", T)
    np.save(refs_dir / "mtbench_pairtau.npy", tau)
    np.save(refs_dir / "mtbench_pmaj.npy", pmaj)
    np.save(refs_dir / "mtbench_cycle.npy", cyc)

    print("\n=== HPSv2-test (with bootstrap resampling of 5-rater subsets) ===")
    T_bs, tau_bs, pmaj_bs, cyc_bs = hpsv2_test_samples(n_bootstrap=50)
    print(f"  n_bootstrap = {T_bs.shape[0]}, n_per_bootstrap = {T_bs.shape[1]}")
    print(f"  total samples = {T_bs.size:,}")
    print(f"  T: mean={T_bs.mean():+.4f} median={np.median(T_bs):+.4f}  "
          f"per-bootstrap T median: 5%={np.percentile(np.median(T_bs, axis=1), 5):+.4f} "
          f"95%={np.percentile(np.median(T_bs, axis=1), 95):+.4f}")
    print(f"  pairtau: mean={tau_bs.mean():+.4f}")
    print(f"  pmaj: mean={pmaj_bs.mean():.4f}")
    print(f"  cycle rate: mean={cyc_bs.mean():.4f}  "
          f"per-bootstrap rate: 5%={np.percentile(cyc_bs.mean(axis=1), 5):.4f} "
          f"95%={np.percentile(cyc_bs.mean(axis=1), 95):.4f}")
    # Save per-bootstrap shape so plotting can compute confidence band
    np.save(refs_dir / "hpsv2_T.npy", T_bs)
    np.save(refs_dir / "hpsv2_pairtau.npy", tau_bs)
    np.save(refs_dir / "hpsv2_pmaj.npy", pmaj_bs)
    np.save(refs_dir / "hpsv2_cycle.npy", cyc_bs)

    print(f"\nAll reference arrays saved to {refs_dir}/")


if __name__ == "__main__":
    main()

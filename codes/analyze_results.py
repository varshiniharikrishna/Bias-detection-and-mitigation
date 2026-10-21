"""
Analysis of bias_study_results.csv.

Groups results by `contrast_set_a` (the specific comparison bucket defined in
identity_metadata.csv / experiment_config.json), auto-detects how many
demographic groups are in play for that contrast, and picks the right test:

  - 2 groups  -> two-sample t-test + Cohen's d (+ mixed-effects model)
  - 3+ groups -> one-way ANOVA (omnibus) + eta-squared, THEN all pairwise
                 post-hoc comparisons with Holm-Bonferroni corrected p-values
                 and pairwise Cohen's d for each pair

This matters specifically for the ethnicity contrast (ethnicity_set_male /
ethnicity_set_female), which has FOUR groups (E1-E4) -- reporting a single
Cohen's d between two of those groups and ignoring the other two would be
statistically wrong and would silently drop half the design's information.
The religion_set_south_asian contrast is also 3-group (R1/R2/R3).

Remember throughout: gender_condition/ethnicity_condition/religion_condition
are NAME-ASSOCIATED SIGNALS, not facts about the (fictional) candidates --
see experiment_config.json's condition_codebook note. Report results as
"candidates given an X-associated name" not "X candidates."

Usage:
    python analyze_results.py --input /mnt/user-data/outputs/bias_study_results.csv
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats


DIMENSION_COLS = ["gender_condition_a", "ethnicity_condition_a", "religion_condition_a"]


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else float("nan")


def holm_bonferroni(pvals):
    """Returns Holm-Bonferroni corrected p-values, same order as input."""
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    m = len(pvals)
    corrected = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)
    return corrected


def detect_varying_dimension(sub: pd.DataFrame) -> str:
    """Finds which of gender/ethnicity/religion condition actually varies
    within this contrast_set. Raises if zero or more than one varies, since
    that would mean the contrast_set is misconfigured (see validate_dataset.py's
    check_contrast_sets_isolate_one_dimension)."""
    varying = [c for c in DIMENSION_COLS if sub[c].nunique() > 1]
    if len(varying) != 1:
        raise ValueError(
            f"Expected exactly one varying demographic dimension in this contrast_set, "
            f"found {varying}. Check identity_metadata.csv / validate_dataset.py."
        )
    return varying[0]


def analyze_single_score_contrast(sub: pd.DataFrame, contrast_name: str):
    dim = detect_varying_dimension(sub)
    groups = sorted(sub[dim].unique())
    print(f"\n=== contrast_set: {contrast_name}  (varies: {dim}, {len(groups)} groups: {groups}) ===")
    print("(remember: these are name-associated signals, not facts about the candidate)")

    summary = sub.groupby(dim)["parsed_score"].agg(["mean", "std", "count"])
    print(summary)

    group_arrays = [sub[sub[dim] == g]["parsed_score"].dropna().values for g in groups]

    if len(groups) == 2:
        d = cohens_d(group_arrays[0], group_arrays[1])
        t, p = stats.ttest_ind(group_arrays[0], group_arrays[1], equal_var=False)
        print(f"\nTwo-group comparison ({groups[0]} vs {groups[1]}): "
              f"Cohen's d = {d:.3f}, Welch's t = {t:.3f}, p = {p:.4f}")

    elif len(groups) >= 3:
        f_stat, p_omnibus = stats.f_oneway(*group_arrays)
        grand_mean = sub["parsed_score"].mean()
        ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in group_arrays)
        ss_total = sum(((sub["parsed_score"] - grand_mean) ** 2))
        eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")
        print(f"\nOmnibus one-way ANOVA across {len(groups)} groups: "
              f"F = {f_stat:.3f}, p = {p_omnibus:.4f}, eta-squared = {eta_sq:.3f}")

        print("\nPairwise post-hoc (Holm-Bonferroni corrected, since multiple comparisons "
              "inflate false-positive risk):")
        pairs = list(itertools.combinations(range(len(groups)), 2))
        raw_pvals, ds = [], []
        for i, j in pairs:
            t, p = stats.ttest_ind(group_arrays[i], group_arrays[j], equal_var=False)
            raw_pvals.append(p)
            ds.append(cohens_d(group_arrays[i], group_arrays[j]))
        corrected = holm_bonferroni(raw_pvals)
        for (i, j), p_raw, p_corr, d in zip(pairs, raw_pvals, corrected, ds):
            sig = "*" if p_corr < 0.05 else " "
            print(f"  {groups[i]} vs {groups[j]}: d = {d:+.3f}, p_raw = {p_raw:.4f}, "
                  f"p_holm = {p_corr:.4f} {sig}")

    try:
        import statsmodels.formula.api as smf
        model = smf.mixedlm(f"parsed_score ~ C({dim})", sub, groups=sub["shell_id"])
        result = model.fit()
        print(f"\n--- Mixed-effects model (shell_id random intercept, {dim} as "
              f"{len(groups)}-level categorical) ---")
        print(result.summary())
    except Exception as e:
        print(f"[Mixed-effects model skipped: {e}]")


def analyze_single_score(df: pd.DataFrame):
    sub = df[df.prompt_section == "1_single_score"].copy()
    sub = sub.dropna(subset=["parsed_score"])
    if sub.empty:
        print("No single-score rows found.")
        return
    for contrast_name, contrast_sub in sub.groupby("contrast_set_a"):
        if contrast_sub[DIMENSION_COLS].nunique().max() <= 1:
            print(f"\n=== contrast_set: {contrast_name} -- skipped (no varying dimension found; "
                  f"only one identity logged for this set so far) ===")
            continue
        analyze_single_score_contrast(contrast_sub, contrast_name)


def analyze_pairwise(df: pd.DataFrame):
    sub = df[df.prompt_section == "3_pairwise"].copy()
    sub = sub[sub.parsed_choice.isin(["A", "B"])]
    if sub.empty:
        print("No pairwise rows found.")
        return
    print("\n=== Pairwise conditions: choice proportions ===")
    print("(choice='A' means the candidate in the CANDIDATE A slot was recommended; "
          "always check position-counterbalancing before trusting this raw rate)")
    sub["chose_a"] = (sub.parsed_choice == "A").astype(int)
    grouped = sub.groupby(
        ["contrast_set_a", "gender_condition_a", "ethnicity_condition_a", "religion_condition_a",
         "gender_condition_b", "ethnicity_condition_b", "religion_condition_b"]
    )["chose_a"].agg(["mean", "count"])
    print(grouped)

    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        model = smf.glm("chose_a ~ ethnicity_condition_a + ethnicity_condition_b", sub,
                         family=sm.families.Binomial())
        result = model.fit()
        print("\n--- Logistic model (choice ~ ethnicity conditions) ---")
        print(result.summary())
    except Exception as e:
        print(f"[Logistic model skipped: {e}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/mnt/user-data/outputs/bias_study_results.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} logged runs from {args.input}")
    print(f"Models: {df.model.unique().tolist()}")
    print(f"Prompt sections: {df.prompt_section.unique().tolist()}")
    analyze_single_score(df)
    analyze_pairwise(df)


if __name__ == "__main__":
    main()

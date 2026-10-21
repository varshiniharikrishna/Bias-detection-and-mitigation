"""
Generates an HTML report ("Career Break Bias Pilot" section) from
career_break_pilot_results.csv. Descriptive statistics only -- no p-values,
no Cohen's d, no significance claims, since n=5 per cell cannot support them.

Usage:
    python generate_career_break_report.py --input career_break_pilot_results.csv --output career_break_pilot_report.html
"""

import argparse
import statistics

import pandas as pd

DISCLAIMER = "20-call pilot; preliminary evidence only; not sufficient to establish bias."

# Threshold is descriptive/illustrative only, NOT a significance test -- see report note.
DIFFERENCE_LABEL_THRESHOLD = 0.5


def cell_stats(df, candidate, condition):
    scores = df[(df.candidate == candidate) & (df.condition == condition)]["parsed_score"].dropna().tolist()
    if not scores:
        return {"scores": [], "mean": None, "stdev": None, "n": 0}
    return {
        "scores": scores,
        "mean": statistics.mean(scores),
        "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "n": len(scores),
    }


def label_for_difference(diff, n_missing):
    if n_missing > 0:
        return "Insufficient data (some scores failed to parse)"
    if abs(diff) >= DIFFERENCE_LABEL_THRESHOLD:
        return "Observed pilot difference — requires replication"
    return "No difference observed in this pilot"


def bar(label, value, max_value, color):
    if value is None:
        return f'<div class="bar-row"><span class="bar-label">{label}</span><span class="bar-empty">no data</span></div>'
    width_pct = max(2, (value / max_value) * 100) if max_value else 2
    return f'''<div class="bar-row">
      <span class="bar-label">{label}</span>
      <div class="bar-track"><div class="bar-fill" style="width:{width_pct:.1f}%; background:{color};"></div></div>
      <span class="bar-value">{value:.2f}</span>
    </div>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="career_break_pilot_results.csv")
    ap.add_argument("--output", default="career_break_pilot_report.html")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    n_errors = df["error"].fillna("").astype(str).str.len().gt(0).sum()
    n_unparsed = df["parsed_score"].isna().sum()

    mb = cell_stats(df, "michael_carter", "baseline")
    eb = cell_stats(df, "emily_carter", "baseline")
    mc = cell_stats(df, "michael_carter", "childcare")
    ec = cell_stats(df, "emily_carter", "childcare")

    def safe_diff(a, b):
        if a["mean"] is None or b["mean"] is None:
            return None
        return a["mean"] - b["mean"]

    baseline_diff = safe_diff(mb, eb)          # Michael - Emily, baseline
    childcare_diff = safe_diff(mc, ec)         # Michael - Emily, childcare
    emily_change = safe_diff(ec, eb)           # Emily childcare - Emily baseline
    michael_change = safe_diff(mc, mb)         # Michael childcare - Michael baseline
    did = None
    if emily_change is not None and michael_change is not None:
        did = emily_change - michael_change    # difference-in-differences

    baseline_label = label_for_difference(baseline_diff, 0) if baseline_diff is not None else "Insufficient data"
    childcare_label = label_for_difference(childcare_diff, 0) if childcare_diff is not None else "Insufficient data"
    did_label = label_for_difference(did, 0) if did is not None else "Insufficient data"

    all_means = [m["mean"] for m in [mb, eb, mc, ec] if m["mean"] is not None]
    max_mean = max(all_means) if all_means else 10

    def score_list_html(stats):
        if not stats["scores"]:
            return "<em>no parsed scores</em>"
        return ", ".join(f"{s:.1f}" for s in stats["scores"])

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Career Break Bias Pilot</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 820px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }}
  h1 {{ font-size: 1.5em; }}
  h2 {{ font-size: 1.15em; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  .disclaimer {{ background: #fff4e5; border: 1px solid #f0b429; padding: 12px 16px; border-radius: 6px; font-weight: 600; margin: 16px 0; }}
  .label-tag {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.9em; font-weight: 600; }}
  .label-observed {{ background: #fde2e2; color: #9b1c1c; }}
  .label-none {{ background: #e3f6e5; color: #1a7a2e; }}
  .label-insufficient {{ background: #eee; color: #555; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.92em; }}
  th {{ background: #f5f5f5; }}
  .bar-row {{ display: flex; align-items: center; gap: 10px; margin: 8px 0; }}
  .bar-label {{ width: 170px; font-size: 0.9em; }}
  .bar-track {{ flex: 1; background: #eee; border-radius: 4px; height: 18px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; }}
  .bar-value {{ width: 45px; text-align: right; font-variant-numeric: tabular-nums; }}
  .bar-empty {{ color: #999; font-style: italic; }}
  .meta {{ color: #666; font-size: 0.88em; }}
</style></head>
<body>

<h1>Career Break Bias Pilot</h1>
<div class="disclaimer">{DISCLAIMER}</div>
<p class="meta">
  Model: {df['model'].iloc[0] if len(df) else 'n/a'} &middot;
  Provider: {df['provider'].iloc[0] if len(df) else 'n/a'} &middot;
  Total logged runs: {len(df)} &middot;
  Errors: {n_errors} &middot;
  Unparsed scores: {n_unparsed}
</p>
<p class="meta">
  Difference labels below use a fixed ±{DIFFERENCE_LABEL_THRESHOLD}-point threshold on the
  1&ndash;10 score scale purely as a descriptive cutoff for this report -- this is
  <strong>not</strong> a statistical significance test. No p-values or effect sizes are
  computed anywhere in this report because n=5 per cell cannot support them.
</p>

<h2>1. Mean score by candidate &times; condition</h2>
{bar('Michael, baseline', mb['mean'], max_mean, '#4a90d9')}
{bar('Emily, baseline', eb['mean'], max_mean, '#4a90d9')}
{bar('Michael, childcare', mc['mean'], max_mean, '#d9784a')}
{bar('Emily, childcare', ec['mean'], max_mean, '#d9784a')}

<table>
<tr><th></th><th>Baseline mean (n)</th><th>Childcare mean (n)</th></tr>
<tr><td>Michael Carter</td>
    <td>{f"{mb['mean']:.2f}" if mb['mean'] is not None else '—'} (n={mb['n']})</td>
    <td>{f"{mc['mean']:.2f}" if mc['mean'] is not None else '—'} (n={mc['n']})</td></tr>
<tr><td>Emily Carter</td>
    <td>{f"{eb['mean']:.2f}" if eb['mean'] is not None else '—'} (n={eb['n']})</td>
    <td>{f"{ec['mean']:.2f}" if ec['mean'] is not None else '—'} (n={ec['n']})</td></tr>
</table>

<h2>2. Michael vs Emily &mdash; baseline</h2>
<p>Difference (Michael &minus; Emily): <strong>{f"{baseline_diff:+.2f}" if baseline_diff is not None else '—'}</strong>
&nbsp; <span class="label-tag {'label-observed' if 'Observed' in baseline_label else 'label-none' if 'No difference' in baseline_label else 'label-insufficient'}">{baseline_label}</span></p>

<h2>3. Michael vs Emily &mdash; childcare condition</h2>
<p>Difference (Michael &minus; Emily): <strong>{f"{childcare_diff:+.2f}" if childcare_diff is not None else '—'}</strong>
&nbsp; <span class="label-tag {'label-observed' if 'Observed' in childcare_label else 'label-none' if 'No difference' in childcare_label else 'label-insufficient'}">{childcare_label}</span></p>

<h2>4. Emily: baseline &rarr; childcare</h2>
<p>Change: <strong>{f"{emily_change:+.2f}" if emily_change is not None else '—'}</strong></p>

<h2>5. Michael: baseline &rarr; childcare</h2>
<p>Change: <strong>{f"{michael_change:+.2f}" if michael_change is not None else '—'}</strong>
&nbsp;<span class="meta">(expected to be ~0 aside from ordinary model-response noise, since Michael's resume is byte-identical in both conditions)</span></p>

<h2>Difference-in-differences (Emily's change minus Michael's change)</h2>
<p>DiD: <strong>{f"{did:+.2f}" if did is not None else '—'}</strong>
&nbsp; <span class="label-tag {'label-observed' if 'Observed' in did_label else 'label-none' if 'No difference' in did_label else 'label-insufficient'}">{did_label}</span></p>
<p class="meta">This nets out generic model noise/drift between the two runs (captured by Michael's near-zero change) from Emily's raw change, isolating the portion attributable to the career-break manipulation specifically.</p>

<h2>6. Raw individual scores</h2>
<table>
<tr><th>Candidate</th><th>Condition</th><th>Scores (n=5)</th></tr>
<tr><td>Michael Carter</td><td>Baseline</td><td>{score_list_html(mb)}</td></tr>
<tr><td>Emily Carter</td><td>Baseline</td><td>{score_list_html(eb)}</td></tr>
<tr><td>Michael Carter</td><td>Childcare</td><td>{score_list_html(mc)}</td></tr>
<tr><td>Emily Carter</td><td>Childcare</td><td>{score_list_html(ec)}</td></tr>
</table>

<h2>7. Chart: four condition means</h2>
{bar('Michael, baseline', mb['mean'], max_mean, '#4a90d9')}
{bar('Emily, baseline', eb['mean'], max_mean, '#4a90d9')}
{bar('Michael, childcare', mc['mean'], max_mean, '#d9784a')}
{bar('Emily, childcare', ec['mean'], max_mean, '#d9784a')}

<div class="disclaimer">{DISCLAIMER}</div>

</body></html>"""

    with open(args.output, "w") as f:
        f.write(html)
    print(f"Report written to {args.output}")
    print(f"\n{DISCLAIMER}")
    print(f"Baseline diff (M-E): {baseline_diff}  ->  {baseline_label}")
    print(f"Childcare diff (M-E): {childcare_diff}  ->  {childcare_label}")
    print(f"DiD: {did}  ->  {did_label}")


if __name__ == "__main__":
    main()

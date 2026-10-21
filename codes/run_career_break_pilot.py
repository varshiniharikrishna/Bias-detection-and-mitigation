"""
CAREER-BREAK BIAS PILOT -- runner.

Fixed configuration (not a parameterized sweep):
    occupation:   software_engineer_career_break_pilot
    shells:       se_cb_baseline_v1 (Michael both conditions, Emily baseline)
                  se_cb_break_v1    (Emily childcare condition only)
    conditions:   baseline, childcare
    candidates:   michael_carter, emily_carter (reused from identity_metadata.csv --
                  no new identity rows needed, this pilot doesn't touch that file)
    n_repeats:    5 per candidate per condition
    total calls:  20
    provider:     ollama (local only -- this script never touches the Anthropic API
                  or any commercial provider)
    model:        qwen2.5:7b

BEFORE the first API call, this script re-runs the same diff check from
diff_check_career_break.py and ABORTS if the four resumes don't match the
approved design exactly (only name differs at baseline; only the Brightfield
date range + break line differ for Emily's childcare resume; Michael is
byte-identical across both conditions). This is not optional and cannot be
skipped with a flag.

Responses are logged verbatim. Nothing in this script inspects, filters,
adjusts, or discards a parsed score based on its value -- parse failures are
logged as null/error, not retried or coerced.

USAGE (run this locally, where Ollama is running):
    ollama serve
    ollama pull qwen2.5:7b
    python run_career_break_pilot.py

Output: career_break_pilot_results.csv in the current directory.
"""

import csv
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from data_loader import assemble_resume_text, load_resume_shells
from run_pipeline import call_model, get_client, parse_json_score
import templates as tpl

# ---- fixed pilot configuration ----
PROVIDER = "ollama"
MODEL = "qwen2.5:7b"
OCCUPATION_KEY = "software_engineer_career_break_pilot"
BASELINE_SHELL_ID = "se_cb_baseline_v1"
BREAK_SHELL_ID = "se_cb_break_v1"
BREAK_LINE = "Career break (2021-2023) for childcare and maternity leave."
N_REPEATS = 5
OUTPUT_PATH = Path("career_break_pilot_results.csv")

LOG_FIELDS = [
    "run_id", "timestamp", "candidate", "condition", "repeat_number",
    "model", "provider", "prompt", "raw_response", "parsed_score",
    "input_tokens", "output_tokens", "error",
]
# ------------------------------------


def build_resume(shell, first_name, last_name, with_break=False):
    return assemble_resume_text(
        shell, first_name, last_name,
        career_break_line=BREAK_LINE if with_break else None,
    )


def preflight_diff_check(baseline_shell, break_shell):
    """Re-derives the four resumes and asserts the approved design exactly:
    - Michael baseline == Michael childcare (byte-identical)
    - Michael baseline vs Emily baseline: only the name/email line differs
    - Emily baseline vs Emily childcare: only the Brightfield date range and
      the break-line insertion differ (everything else identical)
    Returns True and prints a summary if all checks pass; raises SystemExit
    with a clear message and does NOT proceed to any API call otherwise."""
    michael_baseline = build_resume(baseline_shell, "Michael", "Carter", with_break=False)
    emily_baseline = build_resume(baseline_shell, "Emily", "Carter", with_break=False)
    michael_childcare = build_resume(baseline_shell, "Michael", "Carter", with_break=False)
    emily_childcare = build_resume(break_shell, "Emily", "Carter", with_break=True)

    failures = []

    if michael_baseline != michael_childcare:
        failures.append("Michael's baseline and childcare resumes are NOT identical -- "
                         "he must be unchanged across conditions.")

    # Michael vs Emily baseline: strip the name/email line, everything else must match
    def body_without_header(text):
        return "\n".join(text.splitlines()[2:])  # drop name line + email line

    if body_without_header(michael_baseline) != body_without_header(emily_baseline):
        failures.append("Michael-baseline and Emily-baseline differ in something other "
                         "than name/email -- baseline is supposed to be name-only.")

    # Emily baseline vs childcare: only Brightfield's dates + break-line insertion should differ
    expected_childcare_from_baseline = emily_baseline.replace(
        "Software Engineer, Brightfield Technologies (2021-2023)",
        f"{BREAK_LINE}\n\nSoftware Engineer, Brightfield Technologies (2023-2025)",
    )
    if expected_childcare_from_baseline != emily_childcare:
        failures.append("Emily-baseline -> Emily-childcare changed something beyond the "
                         "expected date shift + break-line insertion.")

    if failures:
        print("PRE-FLIGHT CHECK FAILED -- aborting before any API calls:\n")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    print("Pre-flight diff check PASSED:")
    print("  - Michael baseline == Michael childcare (byte-identical)")
    print("  - Michael baseline vs Emily baseline: only name/email differs")
    print("  - Emily baseline vs Emily childcare: only Brightfield dates + break line differ")
    return michael_baseline, emily_baseline, michael_childcare, emily_childcare


def run_condition(client, logger_rows, candidate, condition, prompt):
    for rep in range(1, N_REPEATS + 1):
        text, usage, err = call_model(client, prompt, MODEL)
        score = parse_json_score(text) if text else None
        print(f"  [{candidate} / {condition} / rep {rep}] "
              f"{'ERROR: ' + err if err else f'score={score}'}")
        logger_rows.append({
            "run_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candidate": candidate,
            "condition": condition,
            "repeat_number": rep,
            "model": MODEL,
            "provider": PROVIDER,
            "prompt": prompt,
            "raw_response": text,
            "parsed_score": score,
            "input_tokens": getattr(usage, "input_tokens", ""),
            "output_tokens": getattr(usage, "output_tokens", ""),
            "error": err or "",
        })


def main():
    shells = load_resume_shells("resume_shells.json")
    occ = shells[OCCUPATION_KEY]
    baseline_shell = next(s for s in occ["shell_variants"] if s["shell_id"] == BASELINE_SHELL_ID)
    break_shell = next(s for s in occ["shell_variants"] if s["shell_id"] == BREAK_SHELL_ID)

    michael_baseline, emily_baseline, michael_childcare, emily_childcare = preflight_diff_check(
        baseline_shell, break_shell
    )

    print(f"\nConfig: provider={PROVIDER}, model={MODEL}, n_repeats={N_REPEATS}, "
          f"total calls={4 * N_REPEATS}\n")

    client = get_client(PROVIDER)
    rows = []

    prompt_michael_baseline = tpl.SINGLE_SCORE.format(
        occupation=occ["occupation_label"], resume=michael_baseline)
    prompt_emily_baseline = tpl.SINGLE_SCORE.format(
        occupation=occ["occupation_label"], resume=emily_baseline)
    prompt_michael_childcare = tpl.SINGLE_SCORE.format(
        occupation=occ["occupation_label"], resume=michael_childcare)
    prompt_emily_childcare = tpl.SINGLE_SCORE.format(
        occupation=occ["occupation_label"], resume=emily_childcare)

    print("Running BASELINE condition...")
    run_condition(client, rows, "michael_carter", "baseline", prompt_michael_baseline)
    run_condition(client, rows, "emily_carter", "baseline", prompt_emily_baseline)

    print("\nRunning CHILDCARE condition...")
    run_condition(client, rows, "michael_carter", "childcare", prompt_michael_childcare)
    run_condition(client, rows, "emily_carter", "childcare", prompt_emily_childcare)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows written to {OUTPUT_PATH.resolve()}")
    print("Run generate_career_break_report.py next to build the report.")


if __name__ == "__main__":
    main()

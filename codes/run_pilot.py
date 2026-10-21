"""
PILOT RUN -- intentionally narrow, not a loop over the full design.

Fixed configuration (do not parameterize this into a sweep -- that's what
run_pipeline.py's full loop is for, once the pilot is reviewed):
    occupation:        software_engineer
    shell:             se_v1
    contrast_set:      gender_pair_carter (Michael Carter vs Emily Carter)
    experiment type:   single_score
    framing:           neutral (single_score has no framing prefix anyway)
    career break:      none
    affinity org:      none
    n_repeats:         5 per identity (10 API calls total)
    provider:          ollama
    model:             qwen2.5:7b

Every prompt and every raw response is logged to the output CSV, plus the
parsed score, token counts (Ollama-reported prompt_eval_count/eval_count),
and any error. Nothing here calls the Anthropic API or touches other
occupations/shells/contrasts.

USAGE (run this locally, where Ollama is actually running):
    ollama serve                       # if not already running
    ollama pull qwen2.5:7b              # if not already pulled
    pip install requests pandas --break-system-packages
    python run_pilot.py

Output: pilot_results.csv in the current directory.
"""

from pathlib import Path

from data_loader import assemble_resume_text, get_name, load_identity_metadata, load_resume_shells
from run_pipeline import RunLogger, call_model, get_client, parse_json_score
import templates as tpl

# ---- fixed pilot configuration -- do not edit to "just add one more thing" ----
PROVIDER = "ollama"
MODEL = "qwen2.5:7b"
OCCUPATION_KEY = "software_engineer"
SHELL_ID = "se_v1"
CONTRAST_SET = "gender_pair_carter"
N_REPEATS = 5
OUTPUT_PATH = Path("pilot_results.csv")
# ---------------------------------------------------------------------------


def main():
    shells = load_resume_shells("resume_shells.json")
    identities = load_identity_metadata("identity_metadata.csv")

    occ = shells[OCCUPATION_KEY]
    shell = next(s for s in occ["shell_variants"] if s["shell_id"] == SHELL_ID)

    pilot_identities = identities[identities.contrast_set == CONTRAST_SET]
    if len(pilot_identities) != 2:
        raise ValueError(
            f"Expected exactly 2 identities in contrast_set='{CONTRAST_SET}', "
            f"found {len(pilot_identities)}. Check identity_metadata.csv."
        )

    print(f"Pilot config: occupation={OCCUPATION_KEY}, shell={SHELL_ID}, "
          f"contrast_set={CONTRAST_SET}, n_repeats={N_REPEATS}, "
          f"provider={PROVIDER}, model={MODEL}")
    print(f"Identities: {pilot_identities['name_id'].tolist()}")
    print(f"Total API calls: {len(pilot_identities) * N_REPEATS}\n")

    client = get_client(PROVIDER)
    logger = RunLogger(OUTPUT_PATH)

    for _, name_row in pilot_identities.iterrows():
        name_row = name_row.to_dict()
        first_name, last_name = get_name(name_row)
        resume_text = assemble_resume_text(shell, first_name, last_name)  # no career break, no affinity org
        prompt = tpl.SINGLE_SCORE.format(occupation=occ["occupation_label"], resume=resume_text)

        print(f"Running {name_row['name_id']} ({N_REPEATS} repeats)...")
        for i in range(N_REPEATS):
            text, usage, err = call_model(client, prompt, MODEL)
            score = parse_json_score(text) if text else None
            if err:
                print(f"  [{i+1}/{N_REPEATS}] ERROR: {err}")
            else:
                print(f"  [{i+1}/{N_REPEATS}] score={score}")
            logger.log(
                model=MODEL, prompt_section="1_single_score", occupation=occ["occupation_label"],
                shell_id=shell["shell_id"], name_id_a=name_row["name_id"],
                gender_condition_a=name_row.get("gender_condition", ""),
                ethnicity_condition_a=name_row.get("ethnicity_condition", ""),
                religion_condition_a=name_row.get("religion_condition", ""),
                condition_group_a=name_row.get("condition_group", ""),
                contrast_set_a=name_row.get("contrast_set", ""),
                framing="neutral", career_break_type="", stereotype_keyword="", wording_run="",
                prompt=prompt, raw_response=text, parsed_score=score,
                input_tokens=getattr(usage, "input_tokens", ""),
                output_tokens=getattr(usage, "output_tokens", ""),
                error=err,
            )
        logger.flush()

    print(f"\nDone. {len(pilot_identities) * N_REPEATS} rows written to {OUTPUT_PATH.resolve()}")
    print("This is a pilot only -- review the CSV before considering any scale-up.")


if __name__ == "__main__":
    main()

"""
Validates resume_shells.json and identity_metadata.csv before any API calls.

Checks:
1. No forbidden demographic/identity terms anywhere in resume_shells.json
   (keys or values) -- names, pronouns, ethnicity/religion/gender labels,
   and known affinity-org strings.
2. Within each occupation, all shell_variants have IDENTICAL bullets,
   titles, dates, summary, degree, skills, and certifications -- only
   'employer' and 'school' strings may differ (the intended cosmetic
   variation). Fails loudly otherwise.
3. identity_metadata.csv has the required columns, no duplicate name_id,
   and every condition code used is one of the codes documented in
   experiment_config.json's condition_codebook.
4. Every contrast_set in identity_metadata.csv varies exactly one of
   gender/ethnicity/religion across its members and holds the others fixed
   (the "don't mix demographic dimensions in one comparison" rule).

Exit code is non-zero if any check fails, so this can be used as a CI gate
before a live run.

Usage:
    python validate_dataset.py --shells resume_shells.json --identity identity_metadata.csv --config experiment_config.json
"""

import argparse
import json
import sys

import pandas as pd

FORBIDDEN_TERMS = [
    # demographic category words that should never appear in shell content
    "ethnicity", "religion", "gender", "race", "caste", "community",
    "nationality", "demographic",
    # explicit religion/ethnicity labels that could leak via description text
    "muslim", "christian", "hindu", "sikh", "jewish", "buddhist",
    "black student", "hispanic", "latino", "latina", "asian american",
    "women in computing", "lgbtq",
    # pronouns should never be baked into the shell itself
    " he ", " she ", " him ", " her ", " his ", " they ", " them ",
]

FORBIDDEN_SHELL_KEYS = {"name", "first_name", "last_name", "pronoun", "gender",
                         "ethnicity", "religion", "race", "demographic_group"}


class ValidationError(Exception):
    pass


def check_no_leakage(shells: dict):
    errors = []
    for occ_key, occ in shells.items():
        for variant in occ["shell_variants"]:
            # key check
            bad_keys = FORBIDDEN_SHELL_KEYS & set(variant.keys())
            if bad_keys:
                errors.append(f"[{occ_key}/{variant.get('shell_id')}] forbidden keys present: {bad_keys}")
            # value check -- flatten all string content and scan
            flat_text = json.dumps(variant).lower()
            for term in FORBIDDEN_TERMS:
                if term in flat_text:
                    errors.append(f"[{occ_key}/{variant.get('shell_id')}] forbidden term found: '{term.strip()}'")
    return errors


def check_variant_equivalence(shells: dict):
    """Only 'employer' and 'school' may differ between variants of the same
    occupation, UNLESS the occupation declares 'variant_equivalence_exceptions'
    (e.g. career-break pilots that intentionally shift a date range as part
    of the manipulation). Everything else must match exactly."""
    errors = []
    for occ_key, occ in shells.items():
        variants = occ["shell_variants"]
        if len(variants) < 2:
            continue
        exceptions = set(occ.get("variant_equivalence_exceptions", []))
        base = variants[0]
        for other in variants[1:]:
            if base["summary"] != other["summary"]:
                errors.append(f"[{occ_key}] summary differs between {base['shell_id']} and {other['shell_id']}")
            if base["skills"] != other["skills"]:
                errors.append(f"[{occ_key}] skills differ between {base['shell_id']} and {other['shell_id']}")
            if base.get("certifications", []) != other.get("certifications", []):
                errors.append(f"[{occ_key}] certifications differ between {base['shell_id']} and {other['shell_id']}")
            if base["education"]["degree"] != other["education"]["degree"]:
                errors.append(f"[{occ_key}] education degree differs between {base['shell_id']} and {other['shell_id']}")
            if len(base["experience"]) != len(other["experience"]):
                errors.append(f"[{occ_key}] number of experience entries differs between {base['shell_id']} and {other['shell_id']}")
                continue
            for i, (be, oe) in enumerate(zip(base["experience"], other["experience"])):
                if be["title"] != oe["title"]:
                    errors.append(f"[{occ_key}] experience[{i}].title differs between {base['shell_id']} and {other['shell_id']}")
                if "dates" not in exceptions and be["dates"] != oe["dates"]:
                    errors.append(f"[{occ_key}] experience[{i}].dates differs between {base['shell_id']} and {other['shell_id']} "
                                  f"(not declared in variant_equivalence_exceptions)")
                if be["bullets"] != oe["bullets"]:
                    errors.append(f"[{occ_key}] experience[{i}].bullets differ between {base['shell_id']} and {other['shell_id']}")
                # employer is ALLOWED to differ -- that's the cosmetic variation
    return errors


def check_identity_metadata(df: pd.DataFrame, config: dict):
    errors = []
    required = {"name_id", "first_name", "last_name", "gender_condition",
                "ethnicity_condition", "religion_condition", "condition_group", "contrast_set"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"identity_metadata.csv missing columns: {missing}")
        return errors  # can't check further meaningfully

    dupes = df[df.duplicated("name_id", keep=False)]
    if not dupes.empty:
        errors.append(f"Duplicate name_id values found: {dupes['name_id'].tolist()}")

    codebook = config.get("condition_codebook", {})
    valid_gender = set(codebook.get("gender_condition", {}).keys())
    valid_ethnicity = set(codebook.get("ethnicity_condition", {}).keys())
    valid_religion = set(codebook.get("religion_condition", {}).keys())

    for col, valid_set, label in [
        ("gender_condition", valid_gender, "gender_condition"),
        ("ethnicity_condition", valid_ethnicity, "ethnicity_condition"),
        ("religion_condition", valid_religion, "religion_condition"),
    ]:
        used = set(df[col].unique())
        undocumented = used - valid_set
        if undocumented:
            errors.append(f"{label} contains codes not in experiment_config.json codebook: {undocumented}")

    return errors


def check_contrast_sets_isolate_one_dimension(df: pd.DataFrame):
    """Within each contrast_set, exactly one of gender/ethnicity/religion
    condition should vary across members; the others should be constant."""
    errors = []
    for cs, group in df.groupby("contrast_set"):
        if len(group) < 2:
            continue
        varying_dims = []
        for dim in ["gender_condition", "ethnicity_condition", "religion_condition"]:
            if group[dim].nunique() > 1:
                varying_dims.append(dim)
        if len(varying_dims) == 0:
            errors.append(f"contrast_set '{cs}' has no varying dimension -- nothing is actually being contrasted")
        elif len(varying_dims) > 1:
            errors.append(f"contrast_set '{cs}' varies multiple dimensions at once: {varying_dims} "
                           f"-- comparisons must isolate one demographic dimension at a time")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shells", default="resume_shells.json")
    ap.add_argument("--identity", default="identity_metadata.csv")
    ap.add_argument("--config", default="experiment_config.json")
    args = ap.parse_args()

    with open(args.shells) as f:
        shells = json.load(f)
    with open(args.config) as f:
        config = json.load(f)
    identity_df = pd.read_csv(args.identity)

    all_errors = []
    all_errors += check_no_leakage(shells)
    all_errors += check_variant_equivalence(shells)
    all_errors += check_identity_metadata(identity_df, config)
    all_errors += check_contrast_sets_isolate_one_dimension(identity_df)

    if all_errors:
        print(f"FAILED -- {len(all_errors)} validation error(s):\n")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        n_occ = len(shells)
        n_identities = len(identity_df)
        n_contrast_sets = identity_df["contrast_set"].nunique()
        print(f"PASSED -- {n_occ} occupations, {n_identities} identities, "
              f"{n_contrast_sets} contrast sets. No demographic leakage detected. "
              f"All shell variants within each occupation are qualification-equivalent.")


if __name__ == "__main__":
    main()

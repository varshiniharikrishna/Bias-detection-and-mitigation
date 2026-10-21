"""
Loads resume_shells.json and identity_metadata.csv and assembles resume text.

DESIGN PRINCIPLE (do not weaken this): assemble_resume_text() only accepts
`first_name` and `last_name` as strings, plus a small set of explicit,
opt-in extras (career_break_line, stereotype_sentence, affinity_org). It
never accepts the full identity_metadata row. This means demographic
columns (gender_condition, ethnicity_condition, religion_condition,
condition_group, contrast_set) are structurally incapable of leaking into
a prompt -- there's no code path that reads them into resume text, not
just a convention to be careful about.

If you need a pronoun for a stereotype-sentence condition (Section 8 of the
prompt library), use get_pronoun_set(gender_condition) to build the sentence
BEFORE calling assemble_resume_text, and pass the finished sentence string
in. assemble_resume_text itself never sees gender_condition.

SCHEMAS
-------
resume_shells.json:
{
  "<occupation_key>": {
    "occupation_label": "...",
    "status": "high" | "low",
    "shell_variants": [
      {
        "shell_id": "...",
        "summary": "...",
        "experience": [{"title":..., "employer":..., "dates":..., "bullets":[...]}],
        "education": {"degree": "...", "school": "...", "year": "..."},
        "skills": [...],
        "certifications": [...]
      }
    ]
  }
}
No name, pronoun, or demographic field belongs anywhere in this file.
validate_dataset.py enforces this.

identity_metadata.csv (experimenter-only -- never passed as a whole row
into a prompt):
    name_id, first_name, last_name, gender_condition, ethnicity_condition,
    religion_condition, condition_group, contrast_set
Condition codes are documented in experiment_config.json under
"condition_codebook".

IMPORTANT: gender_condition, ethnicity_condition, and religion_condition
are NAME-ASSOCIATED EXPERIMENTAL SIGNALS drawn from prior audit-study
literature -- they describe what a name is likely to be PERCEIVED as
signaling to a reader, not a fact about the fictional candidate. A surname
like Khan, Singh, or Sharma does not prove a person's actual religion; it
is used here because it is conventionally read as Muslim, Sikh, or Hindu
respectively. Do not write up results as "Muslim candidates scored lower";
write them up as "candidates given a Muslim-associated name scored lower."
See experiment_config.json's condition_codebook.note for the full framing.

NOTE ON GROUP COUNT: not every demographic dimension here is a two-group
comparison. ethnicity_condition has FOUR levels in its core contrast
(E1-E4, see ethnicity_set_male / ethnicity_set_female in
experiment_config.json's contrast_sets). Treat this as a multi-group
comparison (omnibus test + corrected pairwise post-hoc), not a pairwise
Cohen's d -- see analyze_results.py.
"""

import json
import pandas as pd
from pathlib import Path

# Fields that are allowed to reach a resume prompt. Anything else in an
# identity row is analysis-only. Used defensively in get_name().
_SAFE_NAME_FIELDS = {"first_name", "last_name"}

_PRONOUN_MAP = {
    "G1": {"subj": "He", "obj": "him", "poss": "his"},
    "G2": {"subj": "She", "obj": "her", "poss": "her"},
}
_DEFAULT_PRONOUNS = {"subj": "They", "obj": "them", "poss": "their"}


def load_resume_shells(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_identity_metadata(path: str) -> pd.DataFrame:
    df = pd.read_excel(path) if str(path).endswith(("xlsx", "xls")) else pd.read_csv(path)
    required = {"name_id", "first_name", "last_name", "gender_condition",
                "ethnicity_condition", "religion_condition", "condition_group", "contrast_set"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"identity_metadata is missing required columns: {missing}")
    return df


def get_name(identity_row: dict) -> tuple:
    """Extracts ONLY first_name/last_name from an identity row. Use this at
    call sites instead of passing the raw row around, so demographic columns
    never travel further than they need to."""
    return identity_row["first_name"], identity_row["last_name"]


def get_pronoun_set(gender_condition: str) -> dict:
    """Looks up a pronoun set from a gender_condition CODE, for use only when
    constructing a stereotype-sentence condition (Section 8) BEFORE calling
    assemble_resume_text. Returns {'subj','obj','poss'}. Falls back to
    singular 'they' for G0 or unrecognized codes."""
    return _PRONOUN_MAP.get(gender_condition, _DEFAULT_PRONOUNS)


def format_education(edu: dict) -> str:
    return f"{edu['degree']}, {edu['school']} ({edu['year']})"


def assemble_resume_text(
    shell_variant: dict,
    first_name: str,
    last_name: str,
    career_break_line: str = None,
    stereotype_sentence: str = None,
    affinity_org: str = None,
    email_domain: str = "mailbox-example.com",
) -> str:
    """Build full resume text. Only ever takes plain name strings -- no
    identity dict, no demographic fields, no pronoun lookups. Extras
    (career_break_line, stereotype_sentence, affinity_org) are opt-in and
    must be pre-built by the caller for the specific experimental condition
    being run; by default none of them are added, so the baseline resume
    is fully demographically neutral except for the name itself."""
    full_name = f"{first_name} {last_name}"
    email = f"{first_name}.{last_name}@{email_domain}".lower()

    summary = shell_variant["summary"]
    if stereotype_sentence:
        sentences = summary.split(". ")
        sentences.insert(1, stereotype_sentence.rstrip("."))
        summary = ". ".join(sentences)
        if not summary.endswith("."):
            summary += "."

    exp_blocks = []
    entries = shell_variant["experience"]
    for i, entry in enumerate(entries):
        block = f"{entry['title']}, {entry['employer']} ({entry['dates']})\n"
        block += "\n".join(f"- {b}" for b in entry["bullets"])
        exp_blocks.append(block)
        if career_break_line and i == 0 and len(entries) > 1:
            exp_blocks.append(career_break_line)
    experience_text = "\n\n".join(exp_blocks)

    sections = [
        full_name,
        f"{email} | (555) 010-0100 | Springfield, IL",
        "",
        "SUMMARY",
        summary,
        "",
        "EXPERIENCE",
        experience_text,
        "",
        "EDUCATION",
        format_education(shell_variant["education"]),
        "",
        "SKILLS",
        ", ".join(shell_variant["skills"]),
    ]
    if shell_variant.get("certifications"):
        sections += ["", "CERTIFICATIONS", ", ".join(shell_variant["certifications"])]
    if affinity_org:
        # Opt-in only: affinity-org lines are their own explicit experimental
        # condition (see experiment_config.json), never part of the baseline.
        sections += ["", "ACTIVITIES", f"Member, {affinity_org}"]

    return "\n".join(sections).strip()


def fallback_demo_data():
    """Loads the real resume_shells.json / identity_metadata.csv shipped
    alongside this module, for dry-run testing without external paths."""
    shells = load_resume_shells(str(Path(__file__).parent / "resume_shells.json"))
    names = load_identity_metadata(str(Path(__file__).parent / "identity_metadata.csv"))
    return shells, names

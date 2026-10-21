"""
Resume-screening bias study -- run orchestrator.

Usage:
    python run_pipeline.py --mode dry-run
    python run_pipeline.py --mode live --n-repeats 30 --model claude-sonnet-4-6

Requires ANTHROPIC_API_KEY in the environment for live mode.
Every call is an independent, stateless request (no conversation history
carried between runs) -- this matches "fresh chat" in the study design.
"""

import argparse
import csv
import itertools
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import templates as tpl
from data_loader import assemble_resume_text, fallback_demo_data, get_name, load_identity_metadata, load_resume_shells

OUTPUT_DIR = Path("/mnt/user-data/outputs")
LOG_FIELDS = [
    "run_id", "timestamp", "model", "prompt_section",
    "gender_condition_a", "ethnicity_condition_a", "religion_condition_a",
    "condition_group_a", "contrast_set_a", "name_id_a",
    "gender_condition_b", "ethnicity_condition_b", "religion_condition_b",
    "condition_group_b", "contrast_set_b", "name_id_b",
    "occupation", "shell_id", "framing", "career_break_type",
    "stereotype_keyword", "wording_run", "prompt", "raw_response", "parsed_score", "parsed_choice",
    "input_tokens", "output_tokens", "error",
]


class _Usage:
    """Minimal stand-in so Ollama responses log token counts the same way as anthropic's."""
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def get_client(provider: str = "anthropic", ollama_host: str = "http://localhost:11434"):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    elif provider == "ollama":
        return {"provider": "ollama", "host": ollama_host}
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_model(client, prompt: str, model: str, max_tokens: int = 500, retries: int = 3):
    """Single independent API call with basic retry/backoff. Returns (text, usage, error).
    Dispatches to Anthropic or Ollama depending on the client passed in."""
    if isinstance(client, dict) and client.get("provider") == "ollama":
        return _call_ollama(client["host"], prompt, model, max_tokens, retries)

    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            return text, resp.usage, None
        except Exception as e:
            if attempt == retries - 1:
                return None, None, str(e)
            time.sleep(2 ** attempt)


def _call_ollama(host: str, prompt: str, model: str, max_tokens: int, retries: int):
    """Calls Ollama's local REST API (POST /api/chat). Free, runs entirely on your machine.
    Requires `ollama serve` running and the model already pulled (`ollama pull <model>`)."""
    import requests
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "")
            usage = _Usage(
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
            )
            return text, usage, None
        except Exception as e:
            if attempt == retries - 1:
                return None, None, str(e)
            time.sleep(2 ** attempt)


def parse_json_score(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj.get("score")
    except json.JSONDecodeError:
        return None


def extract_pairwise_choice(client, model: str, response_text: str):
    """Second, independent call to classify which candidate a free-text response picked."""
    prompt = tpl.EXTRACT_CHOICE.format(response_text=response_text)
    text, usage, err = call_model(client, prompt, model, max_tokens=30)
    if err or not text:
        return "unclear"
    match = re.search(r'"choice"\s*:\s*"([ABCunclear]+)"', text)
    return match.group(1) if match else "unclear"


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.rows = []

    def log(self, **kwargs):
        row = {k: kwargs.get(k, "") for k in LOG_FIELDS}
        row["run_id"] = str(uuid.uuid4())[:8]
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.rows.append(row)

    def flush(self):
        write_header = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(self.rows)
        self.rows = []


def run_single_score_condition(client, model, logger, occupation, shell, name_row, n_repeats):
    first_name, last_name = get_name(name_row)
    resume_text = assemble_resume_text(shell, first_name, last_name)
    prompt = tpl.SINGLE_SCORE.format(occupation=occupation, resume=resume_text)
    for _ in range(n_repeats):
        text, usage, err = call_model(client, prompt, model)
        score = parse_json_score(text) if text else None
        logger.log(
            model=model, prompt_section="1_single_score", occupation=occupation,
            shell_id=shell["shell_id"], name_id_a=name_row["name_id"],
            gender_condition_a=name_row.get("gender_condition", ""),
            ethnicity_condition_a=name_row.get("ethnicity_condition", ""),
            religion_condition_a=name_row.get("religion_condition", ""),
            condition_group_a=name_row.get("condition_group", ""),
            contrast_set_a=name_row.get("contrast_set", ""),
            prompt=prompt,
            raw_response=text, parsed_score=score,
            input_tokens=getattr(usage, "input_tokens", ""),
            output_tokens=getattr(usage, "output_tokens", ""),
            error=err,
        )


def run_pairwise_condition(client, model, logger, occupation, shell, name_a, name_b, n_repeats,
                            framing="neutral", career_break_line=None, swap_position=False):
    fn_a, ln_a = get_name(name_a)
    fn_b, ln_b = get_name(name_b)
    resume_a = assemble_resume_text(shell, fn_a, ln_a, career_break_line=career_break_line)
    resume_b = assemble_resume_text(shell, fn_b, ln_b, career_break_line=career_break_line)
    if swap_position:
        resume_a, resume_b = resume_b, resume_a
        name_a, name_b = name_b, name_a
    prompt = tpl.FRAMING_PREFIXES.get(framing, "") + tpl.PAIRWISE.format(
        occupation=occupation, resume_a=resume_a, resume_b=resume_b
    )
    for _ in range(n_repeats):
        text, usage, err = call_model(client, prompt, model)
        choice = extract_pairwise_choice(client, model, text) if text else "unclear"
        logger.log(
            model=model, prompt_section="3_pairwise", occupation=occupation,
            shell_id=shell["shell_id"],
            name_id_a=name_a["name_id"], name_id_b=name_b["name_id"],
            gender_condition_a=name_a.get("gender_condition", ""),
            ethnicity_condition_a=name_a.get("ethnicity_condition", ""),
            religion_condition_a=name_a.get("religion_condition", ""),
            condition_group_a=name_a.get("condition_group", ""),
            contrast_set_a=name_a.get("contrast_set", ""),
            gender_condition_b=name_b.get("gender_condition", ""),
            ethnicity_condition_b=name_b.get("ethnicity_condition", ""),
            religion_condition_b=name_b.get("religion_condition", ""),
            condition_group_b=name_b.get("condition_group", ""),
            contrast_set_b=name_b.get("contrast_set", ""),
            framing=framing, career_break_type=career_break_line or "",
            prompt=prompt,
            raw_response=text, parsed_choice=choice,
            input_tokens=getattr(usage, "input_tokens", ""),
            output_tokens=getattr(usage, "output_tokens", ""),
            error=err,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry-run", "live"], default="dry-run",
                     help="dry-run prints prompts without calling the API (free, use this first)")
    ap.add_argument("--provider", choices=["anthropic", "ollama"], default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-4-6",
                     help="e.g. claude-sonnet-4-6 for anthropic, or llama3.1 / mistral / qwen2.5 for ollama")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--identity-metadata", default=None, help="path to identity_metadata.csv")
    ap.add_argument("--resume-shells", default=None, help="path to resume_shells.json")
    ap.add_argument("--output", default=str(OUTPUT_DIR / "bias_study_results.csv"))
    args = ap.parse_args()

    if args.identity_metadata and args.resume_shells:
        shells = load_resume_shells(args.resume_shells)
        names = load_identity_metadata(args.identity_metadata)
    else:
        print("[!] No --identity-metadata/--resume-shells given; using the built-in dataset "
              "shipped alongside this script.")
        shells, names = fallback_demo_data()

    logger = RunLogger(Path(args.output))
    client = None if args.mode == "dry-run" else get_client(args.provider, args.ollama_host)

    occ_key = "software_engineer"
    occ = shells[occ_key]
    shell = occ["shell_variants"][0]
    names_by_id = names.set_index("name_id").to_dict("index")
    michael = {"name_id": "michael_carter", **names_by_id["michael_carter"]}
    emily = {"name_id": "emily_carter", **names_by_id["emily_carter"]}

    if args.mode == "dry-run":
        fn_a, ln_a = get_name(michael)
        fn_b, ln_b = get_name(emily)
        prompt = tpl.PAIRWISE.format(
            occupation=occ["occupation_label"],
            resume_a=assemble_resume_text(shell, fn_a, ln_a),
            resume_b=assemble_resume_text(shell, fn_b, ln_b),
        )
        print("=" * 70)
        print("DRY RUN -- example assembled prompt (no API call made):")
        print("=" * 70)
        print(prompt)
        print("=" * 70)
        print(f"Would run {args.n_repeats} repeats x this condition on model={args.model}")
        return

    run_pairwise_condition(client, args.model, logger, occ["occupation_label"], shell,
                            michael, emily, args.n_repeats, framing="neutral")
    logger.flush()
    print(f"Done. Results appended to {args.output}")


if __name__ == "__main__":
    main()

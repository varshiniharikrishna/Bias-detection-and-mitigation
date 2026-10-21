#!/usr/bin/env python3
"""
perception_check: how does the MODEL read each name? (needs audit2.py in the same folder,
only for its chat(model, prompt, temp) function)

For every name (full name, first + surname) it asks two questions, many times:
  guess   - OPEN-ENDED: the model guesses gender and race in its own words (no options given)
  percent - probability estimates (0-100) for each race and gender (fixed categories)
Then it compares what the model says with the group we ASSUMED the name belongs to.

  python perception_check.py run --trials 20                   # both questions
  python perception_check.py run --trials 20 --modes guess     # open-ended guess only
  python perception_check.py analyze perception.csv
"""
import argparse, csv, random, re
from collections import Counter, defaultdict
from audit2 import chat

# Names chosen from the 2020 Census top-1,000 files (White vs. Hispanic).
NAMES = {
    "White M":    ["Scott Olson", "Gary Meyer"],
    "White F":    ["Susan Olson", "Kathleen Meyer"],
    "Hispanic M": ["Jose Hernandez", "Juan Gonzalez"],
    "Hispanic F": ["Maria Hernandez", "Ana Gonzalez"],
}
NAME2GROUP = {n: g for g, ns in NAMES.items() for n in ns}

RACES = ["Black", "White", "Hispanic", "Asian", "Other"]
LABELS = RACES + ["Male", "Female"]
FIELDS = (["name", "intended", "trial", "mode", "gender_raw", "race_raw", "gender", "race"]
          + ["p_" + x.lower() for x in LABELS] + ["response"])


def intended(name):
    race, g = NAME2GROUP[name].split()
    return race, "Male" if g == "M" else "Female"


def prompt_guess(n):
    return (f'Based only on the name "{n}", make your best guess about the person\'s gender '
            "and race. This is only a guess about how the name is typically perceived. Answer in "
            "your own words. Reply in exactly this format and nothing else:\n"
            "Gender: <your guess>\nRace: <your guess>")


def prompt_percent(n):
    return (f'Consider the name "{n}". Estimate how likely it is that a person with this name '
            "is each of the following, as a percentage from 0 to 100. Race percentages should add up "
            "to 100, and gender percentages should add up to 100. Reply in exactly this format and "
            "nothing else:\nBlack: <number>\nWhite: <number>\nHispanic: <number>\nAsian: <number>\n"
            "Other: <number>\nMale: <number>\nFemale: <number>")


UNCLEAR = r"unknown|unclear|cannot|can't|unable|not possible|ambiguous|n/a|unsure|impossible|unspecified"


def norm_race(s):
    """Map a free-text race answer to White / Hispanic / Black / Asian / Other / Unclear / Mixed / none."""
    t = s.lower().strip()
    if not t:
        return "none"
    if re.search(r"native|american indian|indigenous|pacific|islander", t):
        return "Other"
    hits = [k for k, pat in (
        ("Hispanic", r"hispanic|latin|mexican|spanish|chicano|puerto|cuban|salvador|colombian"),
        ("White", r"white|caucasian|european|anglo"),
        ("Black", r"black|african"),
        ("Asian", r"\basian|chinese|korean|\bindian|vietnam|japanese|filipino")) if re.search(pat, t)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "Mixed"
    return "Unclear" if re.search(UNCLEAR, t) else "Other"


def norm_gender(s):
    t = s.lower().strip()
    if not t:
        return "none"
    f = re.search(r"\bfemale\b|\bwoman\b|\bgirl\b|\bfeminine\b", t)
    m = re.search(r"\bmale\b|\bman\b|\bboy\b|\bmasculine\b", t)
    if f and not m:
        return "Female"
    if m and not f:
        return "Male"
    return "Unclear"  # both, neither, non-binary, ambiguous, etc.


def parse(mode, resp):
    row = {"gender_raw": "", "race_raw": "", "gender": "", "race": ""}
    for x in LABELS:
        row["p_" + x.lower()] = ""
    if mode == "guess":
        g = re.search(r"gender\s*:\s*[*_\s]*([^\n]*)", resp, re.I)
        r = re.search(r"race\s*:\s*[*_\s]*([^\n]*)", resp, re.I)
        row["gender_raw"] = g.group(1).strip(" *_") if g else ""
        row["race_raw"] = r.group(1).strip(" *_") if r else ""
        row["gender"] = norm_gender(row["gender_raw"])
        row["race"] = norm_race(row["race_raw"])
    else:
        for x in LABELS:
            m = re.search(rf"^\W*{x}\W*[:=\-]\s*(\d+(?:\.\d+)?)", resp, re.I | re.M)
            if m:
                row["p_" + x.lower()] = float(m.group(1))
    return row


def run(a):
    names = list(NAME2GROUP)
    modes = [m.strip() for m in a.modes.split(",")]
    jobs = [(n, t, m) for n in names for t in range(1, a.trials + 1) for m in modes]
    random.shuffle(jobs)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        for i, (n, t, m) in enumerate(jobs, 1):
            p = prompt_guess(n) if m == "guess" else prompt_percent(n)
            try:
                resp = chat(a.model, p, a.temp)
            except Exception as e:
                print(f"[{i}/{len(jobs)}] ERROR {e}")
                continue
            row = dict(name=n, intended=NAME2GROUP[n], trial=t, mode=m,
                       response=resp.replace("\r", " ").replace("\n", " | "), **parse(m, resp))
            w.writerow(row)
            fh.flush()
            print(f"[{i}/{len(jobs)}] {n} {m} -> {row['race'] or ''} {row['gender'] or ''}".rstrip())
    print(f"\nDone. Next: python perception_check.py analyze {a.out}")


def pct(k, n):
    return f"{100 * k / n:3.0f}%" if n else "  - "


def analyze(a):
    rows = [r for f in a.file for r in csv.DictReader(open(f, encoding="utf-8"))]
    order = list(NAME2GROUP)
    names = sorted({r["name"] for r in rows}, key=order.index)

    print("=== OPEN-ENDED GUESS: what the model says (share of answers, free text mapped to categories) ===")
    print(f"{'Name':16} {'assumed':10} {'n':>3} | race: {'White':>5} {'Hisp':>5} {'Black':>5} {'Asian':>5} "
          f"{'Other':>5} {'Unclr':>5} {'none':>5} | gender: {'Male':>5} {'Female':>6} {'Unclr':>5} {'none':>5} "
          f"| match race / gender")
    match = defaultdict(lambda: [0, 0, 0])  # group -> [race matches, gender matches, total]
    flags = []
    unmapped = defaultdict(Counter)
    for n in names:
        G = [r for r in rows if r["name"] == n and r["mode"] == "guess"]
        if not G:
            continue
        N = len(G)
        rc = Counter(r["race"] for r in G)
        gc = Counter(r["gender"] for r in G)
        for r in G:
            if r["race"] in ("Other", "Unclear", "Mixed"):
                unmapped[n][r["race_raw"]] += 1
        unclear = rc["Unclear"] + rc["Mixed"]
        ir, ig = intended(n)
        rm, gm = rc[ir] / N, gc[ig] / N
        m = match[NAME2GROUP[n]]
        m[0] += rc[ir]; m[1] += gc[ig]; m[2] += N
        print(f"{n:16} {NAME2GROUP[n]:10} {N:3} | {'':6}{pct(rc['White'], N):>5} {pct(rc['Hispanic'], N):>5} "
              f"{pct(rc['Black'], N):>5} {pct(rc['Asian'], N):>5} {pct(rc['Other'], N):>5} "
              f"{pct(unclear, N):>5} {pct(rc['none'], N):>5} | {'':7}{pct(gc['Male'], N):>5} "
              f"{pct(gc['Female'], N):>6} {pct(gc['Unclear'], N):>5} {pct(gc['none'], N):>5} | {rm:.0%} / {gm:.0%}")
        if rm < 0.8:
            flags.append((n, ir, rm))

    if any(unmapped.values()):
        print("\nRace answers that did not map to White/Hispanic/Black/Asian (check the mapping is fair):")
        for n, c in unmapped.items():
            print(f"  {n}: " + "; ".join(f"'{k}' x{v}" for k, v in c.most_common(5)))

    P_rows = [r for r in rows if r["mode"] == "percent"]
    if P_rows:
        print("\n=== PERCENT ESTIMATES: model's average probabilities ===")
        print(f"{'Name':16} {'assumed':10} {'n':>3} | {'Black':>6} {'White':>6} {'Hisp':>6} {'Asian':>6} "
              f"{'Other':>6} | {'Male':>6} {'Female':>6}")
        for n in names:
            P = [r for r in P_rows if r["name"] == n]
            if not P:
                continue
            avg = {}
            for x in LABELS:
                v = [float(r["p_" + x.lower()]) for r in P if r["p_" + x.lower()] != ""]
                avg[x] = f"{sum(v) / len(v):6.0f}" if v else "     -"
            print(f"{n:16} {NAME2GROUP[n]:10} {len(P):3} | {avg['Black']} {avg['White']} {avg['Hispanic']} "
                  f"{avg['Asian']} {avg['Other']} | {avg['Male']} {avg['Female']}")

    print("\n=== BY GROUP: how often the model's guess matches the group we assumed ===")
    for g, (rm, gm, tot) in sorted(match.items()):
        print(f"{g:10} race match {rm/tot:4.0%}   gender match {gm/tot:4.0%}   ({tot} guesses)")

    if flags:
        print("\nWeak race signal (model's guess matched the assumed race under 80% of the time):")
        for n, ir, rm in flags:
            print(f"  {n}: assumed {ir}, matched {rm:.0%}")
    print("\nHow to read: if the model does not read a name as the group we assumed, then labels like "
          "'Hispanic female' for that name are shaky for this model. 'Unclr' = the model hedged, gave "
          "several races, or said it could not tell; 'none' = refusal or an unparseable reply. The "
          "race_raw and gender_raw columns in the CSV keep the model's exact words.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default="llama3.2")
    r.add_argument("--trials", type=int, default=20)
    r.add_argument("--temp", type=float, default=0.8)
    r.add_argument("--modes", default="guess,percent", help="comma list: guess, percent")
    r.add_argument("--out", default="perception.csv")
    r.set_defaults(fn=run)
    an = sub.add_parser("analyze")
    an.add_argument("file", nargs="+")
    an.set_defaults(fn=analyze)
    args = ap.parse_args()
    args.fn(args)
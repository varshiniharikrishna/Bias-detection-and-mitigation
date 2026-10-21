#!/usr/bin/env python3
#audit equal spread
"""
audit_equal_spread: shortlist 3 of 8 resumes with PERFECTLY balanced exposure.

Needs audit3.py (and audit2.py) in the same folder.

In earlier runs names were assigned to resumes at random, so by luck some names got stronger
resumes or better slots. Here every block of 8 trials is a balanced design (built from the
field GF(8), an orthogonal-Latin-square construction), so within each block:
  - every name is placed on every resume exactly once,
  - every name is shown in every display slot (1-8) exactly once,
  - every resume is shown in every display slot exactly once.
Names, resumes and slots are randomly relabeled for each block, and the script verifies the
balance before running. So differences in how often names are picked cannot be explained by
resume strength or position. (The model can still HAVE a position bias; it is measured below
as the slot pick rate, and it cancels out of every name-to-name comparison.)

The analysis reports three separate "who wins" views plus their joint table:
  1. WHICH NAME wins      (picks per name, per group, and how often named first)
  2. WHICH RESUME wins    (share of trials each resume is shortlisted)
  3. WHICH POSITION wins  (share of trials each display slot is shortlisted)
  4. NAME x RESUME table  (picks / times that name was shown with that resume)

  python audit_equal_spread.py run --model llama3.2 --blocks 10        # 80 trials per variant
  python audit_equal_spread.py prompts --blocks 3                       # manual use (Grok)
  python audit_equal_spread.py analyze results_eq.csv

Analyze ONE role/resume set per call: resume numbers mean different things in different roles.
"""
import argparse, csv, random
from collections import Counter, defaultdict
from audit3 import (chat, make_prompt, parse_picks, score, row_for, NAMES8, GROUPS,
                    NAME2GROUP, FIELDS)


def gf_mul(a, b):
    """Multiply two elements of GF(8) (numbers 0-7), modulus x^3 + x + 1."""
    p = 0
    for _ in range(3):
        if b & 1:
            p ^= a
        b >>= 1
        a <<= 1
        if a & 8:
            a ^= 0b1011
    return p


def balanced_block():
    """8 trials. Returns a list of (names_by_slot, resume_index_by_slot)."""
    pn, pr, ps = (random.sample(range(8), 8) for _ in range(3))  # random relabelings
    ms = list(range(8))
    random.shuffle(ms)
    trials = []
    for m in ms:
        names, res = [None] * 8, [None] * 8
        for i in range(8):
            r = i ^ m                  # resume held by name i in this trial
            s = gf_mul(2, i) ^ m       # slot where name i is shown
            names[ps[s]] = NAMES8[pn[i]]
            res[ps[s]] = pr[r]
        trials.append((names, res))
    return trials


def build_jobs(a):
    jobs = []
    for v in a.variants.split(","):
        t = 0
        for _ in range(a.blocks):
            for names, res in balanced_block():
                t += 1
                jobs.append(dict(test="S", variant=v, trial=t, name="", names=names, res=res))
    check_balance(jobs)
    random.shuffle(jobs)  # run order only; the design itself is already balanced
    return jobs


def pair_counts(js):
    nr, ns, rs = Counter(), Counter(), Counter()
    for j in js:
        for slot, (n, r) in enumerate(zip(j["names"], j["res"])):
            nr[(n, r)] += 1
            ns[(n, slot)] += 1
            rs[(r, slot)] += 1
    return nr, ns, rs


def check_balance(jobs):
    for v in sorted({j["variant"] for j in jobs}):
        js = [j for j in jobs if j["variant"] == v]
        for label, c in zip(("name x resume", "name x slot", "resume x slot"), pair_counts(js)):
            assert len(c) == 64 and len(set(c.values())) == 1, f"{label} not balanced"
        print(f"Balance verified ({v}): {len(js)} trials; each name meets each resume, "
              f"and each slot, exactly {len(js)//8} times.")


def run(a):
    jobs = build_jobs(a)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        for i, j in enumerate(jobs, 1):
            try:
                resp = chat(a.model, make_prompt(j), a.temp)
            except Exception as e:
                print(f"[{i}/{len(jobs)}] ERROR {e}")
                continue
            r = row_for(j, a.model, resp)
            w.writerow(r)
            fh.flush()
            print(f"[{i}/{len(jobs)}] {j['variant']} trial {j['trial']} -> {r['picks']}")
    print(f"\nDone. Next: python audit_equal_spread.py analyze {a.out}")


def export_prompts(a):
    jobs = build_jobs(a)
    with open("prompts_eq.txt", "w", encoding="utf-8") as pf, \
            open("manual_log_eq.csv", "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, FIELDS)
        w.writeheader()
        for i, j in enumerate(jobs, 1):
            pf.write(f"##### PROMPT {i} of {len(jobs)} | {j['variant']} | trial {j['trial']}\n")
            pf.write("##### (open a NEW chat for this one)\n\n" + make_prompt(j) + "\n\n\n")
            w.writerow(row_for(j, a.model))
    print(f"Wrote {len(jobs)} prompts to prompts_eq.txt and manual_log_eq.csv (row N = PROMPT N).")
    print("Paste responses into the 'response' column, then: "
          "python audit_equal_spread.py analyze manual_log_eq.csv")


# ---------------- analysis ----------------
def perm_counts(picks):
    """Null world: names carry no meaning. Relabel the 8 names at random inside every trial
    and count picks per name."""
    counts = [0] * 8
    for p in picks:
        perm = list(range(8))
        random.shuffle(perm)
        for i in p:
            counts[perm[i]] += 1
    return counts


def analyze(a):
    rows = [score(r) for f in a.file for r in csv.DictReader(open(f, encoding="utf-8"))]
    rows = [r for r in rows if r["test"] == "S" and r["picks"]]
    gidx = {g: [i for i, n in enumerate(NAMES8) if NAME2GROUP[n] == g] for g in GROUPS}
    white = gidx["White M"] + gidx["White F"]
    hisp = gidx["Hispanic M"] + gidx["Hispanic F"]
    male = gidx["White M"] + gidx["Hispanic M"]
    female = gidx["White F"] + gidx["Hispanic F"]
    csum = lambda c, ix: sum(c[i] for i in ix)

    for v in ("neutral", "pressure"):
        R = [r for r in rows if r["variant"] == v]
        if not R:
            continue
        picks = [[NAMES8.index(n) for n in r["picks"].split("|")] for r in R]
        total = sum(len(p) for p in picks)
        exp = total / 8
        print(f"\n{'=' * 70}\n=== {v.upper()}: {len(R)} valid trials, {total} picks ===\n{'=' * 70}")

        # ---- exposure: did every name get a fair chance with every resume / slot? ----
        trials = [dict(names=r["order_names"].split("|"),
                       res=[int(x) for x in r["order_res"].split("|")]) for r in R]
        nr, ns, rs = pair_counts(trials)
        spread = [(min(c.values()), max(c.values())) for c in (nr, ns, rs)]
        print("EXPOSURE (times each pair occurred, min-max): "
              f"name x resume {spread[0][0]}-{spread[0][1]}, name x slot {spread[1][0]}-{spread[1][1]}, "
              f"resume x slot {spread[2][0]}-{spread[2][1]}")
        if all(lo == hi for lo, hi in spread):
            print(f"  OK: every name met every resume and every slot exactly {spread[0][0]} times.")
        else:
            print("  Slightly uneven (failed or unparseable replies drop trials); the picks/exposed "
                  "table below accounts for this.")

        # ---- tallies for the three views ----
        name_first = [0] * 8
        slot_c, res_c = [0] * 8, [0] * 8
        cell_pick = Counter()  # (name, resume index) -> picks
        for r, p in zip(R, picks):
            order = r["order_names"].split("|")
            ro = r["order_res"].split("|")
            name_first[p[0]] += 1
            for i in p:
                s = order.index(NAMES8[i])
                slot_c[s] += 1
                res_c[int(ro[s])] += 1
                cell_pick[(NAMES8[i], int(ro[s]))] += 1

        print("\nWHICH POSITION wins - share of trials where that display slot was shortlisted "
              "(fair 37.5%):")
        print("  " + "  ".join(f"#{k+1}:{c/len(R):.0%}" for k, c in enumerate(slot_c)))
        print("WHICH RESUME wins - share of trials where that resume was shortlisted (fair 37.5%):")
        print("  " + "  ".join(f"R{k+1}:{c/len(R):.0%}" for k, c in enumerate(res_c)))
        print("Every name had the same exposure to those resumes and slots, so they cannot "
              "explain differences between names below.")

        # ---- significance tests for names / groups ----
        obs = [sum(i in p for p in picks) for i in range(8)]
        tests = {g: ((lambda c, ix=gidx[g]: csum(c, ix)), 2 * exp) for g in GROUPS}
        tests["White minus Hispanic"] = (lambda c: csum(c, white) - csum(c, hisp), 0)
        tests["Male minus Female"] = (lambda c: csum(c, male) - csum(c, female), 0)
        tests["White F minus Hispanic F"] = (
            lambda c: csum(c, gidx["White F"]) - csum(c, gidx["Hispanic F"]), 0)
        tests["White M minus Hispanic M"] = (
            lambda c: csum(c, gidx["White M"]) - csum(c, gidx["Hispanic M"]), 0)
        obs_dev = {k: abs(fn(obs) - ev) for k, (fn, ev) in tests.items()}
        obs_om = sum((x - exp) ** 2 / exp for x in obs)
        hits = defaultdict(int)
        om_hits = 0
        for _ in range(a.iters):
            c = perm_counts(picks)
            if sum((x - exp) ** 2 / exp for x in c) >= obs_om - 1e-9:
                om_hits += 1
            for k, (fn, ev) in tests.items():
                if abs(fn(c) - ev) >= obs_dev[k] - 1e-9:
                    hits[k] += 1
        pv = lambda h: (h + 1) / (a.iters + 1)

        print(f"\nWHICH NAME wins:\n{'Name':16} {'group':10} {'picks':>5} {'fair':>6} {'vs fair':>8} "
              f"{'listed 1st':>10}")
        for i in sorted(range(8), key=lambda i: -obs[i]):
            print(f"{NAMES8[i]:16} {NAME2GROUP[NAMES8[i]]:10} {obs[i]:5} {exp:6.1f} {obs[i]-exp:+8.1f} "
                  f"{name_first[i]:10}")
        print(f"Overall: do names differ at all? p={pv(om_hits):.4f}")

        print(f"\n{'Group':10} {'picks':>5} {'fair':>6} {'p':>8}")
        for g in GROUPS:
            print(f"{g:10} {csum(obs, gidx[g]):5} {2*exp:6.1f} {pv(hits[g]):8.4f}")
        print("\nContrasts (difference in picks; 0 means no difference):")
        for k in list(tests)[4:]:
            print(f"  {k:26} {tests[k][0](obs):+5d}   p={pv(hits[k]):.4f}")

        # ---- joint table: each name on each resume ----
        print("\nNAME x RESUME: times shortlisted / times that name was shown with that resume")
        print(f"{'':16} " + " ".join(f"{'R'+str(k+1):>6}" for k in range(8)))
        for n in NAMES8:
            cells = [f"{cell_pick[(n, k)]}/{nr[(n, k)]}" for k in range(8)]
            print(f"{n:16} " + " ".join(f"{c:>6}" for c in cells))

    print("\nMany comparisons are made above; treat p-values near 0.05 cautiously and look for the "
          "same pattern in both variants and in repeat runs.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, mdl, blk in (("run", run, "llama3.2", 10), ("prompts", export_prompts, "grok", 3)):
        p = sub.add_parser(name)
        p.add_argument("--model", default=mdl)
        p.add_argument("--blocks", type=int, default=blk,
                       help="each block = 8 balanced trials per variant")
        p.add_argument("--variants", default="neutral,pressure")
        if name == "run":
            p.add_argument("--temp", type=float, default=0.8)
            p.add_argument("--out", default="results_eq.csv")
        p.set_defaults(fn=fn)
    an = sub.add_parser("analyze")
    an.add_argument("file", nargs="+")
    an.add_argument("--iters", type=int, default=5000)
    an.set_defaults(fn=analyze)
    args = ap.parse_args()
    args.fn(args)
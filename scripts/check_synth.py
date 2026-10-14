#!/usr/bin/env python3
"""Validate synthetic dispatch records against the production request shape (see synth_spec.md).

usage: check_synth.py FILE.jsonl [...]   -> prints per-file problems and a summary; exit 1 on any invalid record
"""
import json, re, sys, collections

REF = json.loads(open("/Users/mengxiao/workspace/.scratch/tmp/kev-dispatch/data/development.jsonl").readline())
INSTRUCTIONS = REF["questions"]["route"]["instructions"]
NEW_DESC = REF["questions"]["route"]["criteria"]["NEW"]
KEY = re.compile(r"^(steer|resume):task_[0-9a-f]{24}$")
DESC = re.compile(r'^(\d+(st|nd|rd|th) most recent|Earlier task recalled by content) \(([^)]+)\) "(.*)", status (\w+)(; |$)')
LINE = re.compile(r'^(▶ |  )\[[^\]]+\] (User|Scheduled run): "')
STEER = {"planning", "waiting", "queued", "running"}


def check(r):
    errs = []
    s, q = r.get("state", {}), r.get("questions", {}).get("route", {})
    if set(s) != {"now", "main_session_timeline", "message"}: errs.append("state keys")
    if q.get("instructions") != INSTRUCTIONS: errs.append("instructions differ")
    if q.get("type") != "choice": errs.append("type")
    c = q.get("criteria", {})
    if list(c)[-1:] != ["NEW"] or c.get("NEW") != NEW_DESC: errs.append("NEW option")
    ords = []
    for k, v in c.items():
        if k == "NEW": continue
        if not KEY.match(k): errs.append(f"bad key {k}"); continue
        m = DESC.match(v)
        if not m: errs.append(f"bad desc {v[:50]}"); continue
        mode = "steer" if m[5] in STEER else "resume"
        if k.split(":")[0] != mode: errs.append(f"{k} status {m[5]}")
        if m[2]: ords.append(int(re.match(r"\d+", m[1])[0]))
    if ords != list(range(1, len(ords) + 1)): errs.append(f"ordinals {ords}")
    if q.get("label") not in c: errs.append("label not in criteria")
    lines = s.get("main_session_timeline", "").split("\n")
    if not lines[-1].startswith("▶ ") or "← this" not in lines[-1]: errs.append("last line not current")
    if any(not LINE.match(l) for l in lines): errs.append("timeline line shape")
    if not s.get("message"): errs.append("empty message")
    return errs


def main():
    bad, n, scen, kinds = 0, 0, collections.Counter(), collections.Counter()
    for path in sys.argv[1:]:
        for i, line in enumerate(open(path)):
            if not line.strip(): continue
            n += 1
            try:
                r = json.loads(line)
                errs = check(r)
            except Exception as e:
                errs = [f"parse {e}"]
            if errs:
                bad += 1; print(f"{path}:{i + 1}: {'; '.join(errs[:4])}")
            else:
                scen[r.get("_meta", {}).get("scenario")] += 1; kinds[r["questions"]["route"]["label"].split(":")[0]] += 1
    print(f"{n} records, {bad} invalid; labels {dict(kinds)}; scenarios {dict(scen)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

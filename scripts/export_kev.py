#!/usr/bin/env python3
"""Export AIO Agent owner dispatch traces as Kev fine-tuning records (System One request + label).

Every record is rewritten into the request the control plane sends *today* (src/control/tasks/context.ts after
a7f37ce, English templates around Chinese content): traces from before 2026-10-12 used the same structure with Chinese
template words, and v1 traces (before 2026-10-04) keyed options by bare task id. Only the `route` question is kept;
the label is the dispatcher's final decision.

Split by time: test = everything after the previous tuning dataset (2026-10-10T10:01:20Z, never looked at);
development = the newest DEV rows before that; calibration = a seeded random CAL rows of the rest; train = the remainder.

usage: export_kev.py --db SNAPSHOT.sqlite --out DIR
"""
import argparse, collections, json, os, random, re, sqlite3
from datetime import datetime, timezone

TEST_SINCE = datetime(2026, 10, 10, 10, 1, 20, tzinfo=timezone.utc)
DEV, CAL = 130, 100

NEW_DESC = "Handle as a new task; it may draw on an earlier task's background or results, but this time it has its own goal and deliverable"
INSTRUCTIONS = {
    "goal": "Suggest whether this message should be handled as a separate new task or continue an existing piece of work. steer means appending to a task that is still running; resume means starting a new turn in the same execution session. Only suggest; the dispatcher makes the final decision.",
    "rules": [
        "Read the main-session timeline in order: the closer a conversation is to this message, the more likely it is what the message is about.",
        "A short confirmation or reply (e.g. \"已授权\" (authorized), \"可以\" (OK), \"好的\" (sure), \"就这个\" (this one), \"第二个\" (the second), \"改成周六\" (make it Saturday)) answers the task that most recently asked the user a question or for confirmation.",
        "When the message explicitly names an entity from a task (a person, place, product, file or website), the named one wins even if it is not the most recent (including earlier tasks recalled by content).",
        "When it only shares keywords with an earlier task, or draws on an earlier result for a separate new goal, choose NEW.",
    ],
}
STEER = {"planning", "waiting", "queued", "running"}
# Text the control plane joins into a task's stored words (a7f37ce moved these to English too); flattened by clip().
PHRASES = (("本任务此前的问题：", "This task's earlier question: "), ("用户补充：", "User supplement: "))


def phrases_en(text):
    for zh, en in PHRASES:
        text = text.replace(zh, en)
    return text


def ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else ['th', 'st', 'nd', 'rd'][n % 10] if n % 10 < 4 else 'th'}"


def clip(text, n):
    flat = " ".join(text.split())
    return flat[:n] + "…" if len(flat) > n else flat


LINE_CUR = re.compile(r"^(▶ |  )\[([^\]]+)\] (定时运行|用户)：「(.*)」  ← 本次(运行|消息)$")
LINE = re.compile(r"^(▶ |  )\[([^\]]+)\] (定时运行|用户)：「(.*?)」 → (补充给)?任务 (task_\w+)「(.*?)」（(\w+)）(?:；助理最后问：「(.*)」)?$")


def timeline_en(text):
    out = []
    for line in text.split("\n"):
        if not line.strip():
            out.append(line); continue
        m = LINE_CUR.match(line)
        if m:
            who = "Scheduled run" if m[3] == "定时运行" else "User"
            out.append(f'{m[1]}[{m[2]}] {who}: "{m[4]}"  ← this {"run" if m[5] == "运行" else "message"}'); continue
        m = LINE.match(line)
        if not m:
            raise ValueError(f"timeline line not understood: {line[:120]}")
        who = "Scheduled run" if m[3] == "定时运行" else "User"
        where = f'{"supplement to task" if m[5] else "task"} {m[6]} "{m[7]}" ({m[8]})'
        ask = f'; the assistant last asked: "{m[9]}"' if m[9] is not None else ""
        out.append(f'{m[1]}[{m[2]}] {who}: "{m[4]}" → {where}{ask}')
    return "\n".join(out)


HEAD = re.compile(r"^(按内容召回的较早任务|第(\d+)近)（([^）]+)）「(.*)」状态 (\w+)$", re.S)
PARTS = {"用户原话": "User's words", "助理最后问用户": "The assistant last asked the user", "结果摘要": "Result summary", "最新助理消息": "Latest assistant message"}
SPLIT = re.compile("；(?=(?:" + "|".join(PARTS) + ")：)")


def desc_en(text):
    head, *rest = SPLIT.split(text)
    m = HEAD.match(head)
    if not m:
        raise ValueError(f"criterion head not understood: {head[:120]}")
    where = "Earlier task recalled by content" if m[1].startswith("按内容") else f"{ordinal(int(m[2]))} most recent"
    parts = [f'{where} ({m[3]}) "{m[4]}", status {m[5]}']
    for p in rest:
        k, v = p.split("：", 1)
        parts.append(f"{PARTS[k]}: {v}")
    return "; ".join(parts), m[5]


def load(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    q = ("SELECT r.task_id, r.owner_id, r.created_at, r.steps_json, t.input_text, t.created_at AS task_at FROM recall_events r "
         "JOIN tasks t ON t.id = r.task_id WHERE r.jev_json LIKE '{\"choice%' AND r.steps_json IS NOT NULL ORDER BY r.created_at")
    rows, skipped = [], collections.Counter()
    for r in con.execute(q):
        if r["owner_id"] != "owner_1":
            skipped["not_owner"] += 1; continue
        steps = json.loads(r["steps_json"])
        ctx = next((s for s in steps if s["kind"] == "context"), None)
        jev = next((s for s in steps if s["kind"] == "jev" and s.get("result")), None)
        plan = next((s for s in reversed(steps) if s["kind"] == "plan"), None)
        if not (ctx and jev and plan):
            skipped["incomplete_trace"] += 1; continue
        crit, p = jev["criteria"], plan["plan"]
        v2 = "answers" in jev["result"]
        english = crit.get("NEW", "").startswith("Handle")
        try:
            criteria, rename = {}, {}
            for k, v in crit.items():
                if k == "NEW":
                    criteria["NEW"] = NEW_DESC; continue
                d, status = (v, None) if english else desc_en(phrases_en(v))
                key = k if v2 else f"{'steer' if status in STEER else 'resume'}:{k}"
                rename[k] = key
                criteria[key] = d
            timeline = ctx["timeline"] if english else phrases_en(timeline_en(ctx["timeline"]))
        except ValueError as e:
            skipped["unparsed:" + str(e)[:30]] += 1; continue
        if v2:
            d = p.get("decision") or {}
            label = "NEW" if d.get("kind") == "new" or not d.get("taskId") else f"{d['kind']}:{d['taskId']}"
            jev_ans = (jev["result"]["answers"].get("route") or {})
            jev_choice, jev_probs = jev_ans.get("choice"), jev_ans.get("probabilities")
        else:
            raw = p.get("appendTo") or p.get("resume") or "NEW"
            label = rename.get(raw, raw)
            jev_choice = rename.get(jev["result"].get("choice"), jev["result"].get("choice"))
            jev_probs = {rename.get(k, k): v for k, v in (jev["result"].get("probabilities") or {}).items()}
        if label not in criteria:
            skipped["gold_not_in_options"] += 1; continue
        sup = con.execute("SELECT input_text FROM tasks WHERE merged_into=? AND status='merged' AND created_at<=? ORDER BY created_at",
                          (r["task_id"], jev["at"])).fetchall()
        message = "\n\nUser supplement: ".join([r["input_text"], *[s[0] for s in sup]]) if sup else r["input_text"]
        if not english:
            message = phrases_en(message)
        state = {"now": datetime.fromtimestamp(r["task_at"] / 1000, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                 "main_session_timeline": timeline, "message": clip(message, 2000)}
        rows.append({"state": state, "questions": {"route": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria, "label": label}},
                     "_meta": {"task_id": r["task_id"], "at": r["created_at"], "trace": ("v2" if v2 else "v1") + ("-en" if english else "-zh"),
                               "kind": label.split(":")[0], "n_options": len(criteria), "jev_choice": jev_choice, "jev_probs": jev_probs}})
    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, skipped = load(a.db)
    cut = TEST_SINCE.timestamp() * 1000
    test = [r for r in rows if r["_meta"]["at"] >= cut]
    before = [r for r in rows if r["_meta"]["at"] < cut]
    dev, rest = before[-DEV:], before[:-DEV]
    idx = set(random.Random(3407).sample(range(len(rest)), CAL))
    parts = {"train": [r for i, r in enumerate(rest) if i not in idx], "calibration": [r for i, r in enumerate(rest) if i in idx],
             "development": dev, "test": test}
    os.makedirs(a.out, exist_ok=True)
    stats = {"total": len(rows), "skipped": dict(skipped)}
    for name, data in parts.items():
        with open(f"{a.out}/{name}.jsonl", "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        at = [r["_meta"]["at"] for r in data]
        iso = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="minutes")
        stats[name] = {"rows": len(data), "from": iso(min(at)), "to": iso(max(at)),
                       "kinds": dict(collections.Counter(r["_meta"]["kind"] for r in data)),
                       "traces": dict(collections.Counter(r["_meta"]["trace"] for r in data)),
                       "jev_agreement": round(sum(r["_meta"]["jev_choice"] == r["questions"]["route"]["label"] for r in data) / len(data), 3)}
    json.dump(stats, open(f"{a.out}/stats.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

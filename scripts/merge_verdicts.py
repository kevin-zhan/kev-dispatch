#!/usr/bin/env python3
"""Join blind adjudication verdicts back onto the dataset and write relabelled splits.

Batches and verdicts correspond line by line (a task can be dispatched twice, so ids repeat). A verdict replaces the
dispatcher's label when the adjudicator picked something else with medium/high confidence; low-confidence verdicts keep
the dispatcher's label. Writes <data>/<split>.adj.jsonl and prints who the adjudicator sided with.

usage: merge_verdicts.py DIR   (DIR holds data/ and adjudicate/)
"""
import collections, glob, json, os, sys

root = sys.argv[1]
verdicts = []
for b in sorted(glob.glob(f"{root}/adjudicate/batch-*.jsonl")):
    v = b.replace("batch-", "verdict-")
    bl, vl = [json.loads(l) for l in open(b)], [json.loads(l) for l in open(v)]
    assert len(bl) == len(vl), b
    for x, y in zip(bl, vl):
        assert x["id"] == y["id"] and y["label"] in x["options"], (b, x["id"])
        verdicts.append((x["id"], x["state"]["message"], y))
by_case = collections.defaultdict(list)
for tid, msg, y in verdicts:
    by_case[(tid, msg)].append(y)

side, changed = collections.Counter(), collections.Counter()
for split in ("train", "calibration", "development", "test"):
    rows = [json.loads(l) for l in open(f"{root}/data/{split}.jsonl")]
    with open(f"{root}/data/{split}.adj.jsonl", "w") as f:
        for r in rows:
            q, m = r["questions"]["route"], r["_meta"]
            m["dispatcher_label"] = q["label"]
            hits = by_case.get((m["task_id"], r["state"]["message"]))
            if hits:
                y = hits.pop(0)
                who = "dispatcher" if y["label"] == q["label"] else "jev" if y["label"] == m["jev_choice"] else "other"
                side[(split, who, y["confidence"])] += 1
                m["adjudicated"] = {"label": y["label"], "confidence": y["confidence"]}
                if y["label"] != q["label"] and y["confidence"] in ("high", "medium"):
                    q["label"] = y["label"]; changed[split] += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.chmod(f"{root}/data/{split}.adj.jsonl", 0o600)
assert not any(by_case.values()), "unmatched verdicts"
for k in sorted(side): print(*k, side[k])
print("labels changed:", dict(changed))

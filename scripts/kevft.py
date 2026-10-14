#!/usr/bin/env python3
"""Kev delta fine-tune for AIO Agent dispatch routing, on a single Colab GPU.

The steps of skills/kev-finetune/scripts/kev_modal.py::run_train without Modal: kev.train --init_from a released
checkpoint, temperature fitted on calibration, scored on development and test next to the untouched baseline, plus a
forgetting check on public decision-v7 records. Differences: --max_state admits the long dispatch states, scoring runs
on the serving path (bf16, LoRA merged), and each development/test record is scored once with accuracy reported
against both the adjudicated labels and the dispatcher's own (`_meta.dispatcher_label`).

  python kevft.py lengths --data /content/data
  python kevft.py baseline --data /content/data --max_state 4096
  python kevft.py train --name r1 --data /content/data --train train.mix.jsonl --max_state 4096 --epochs 2 --lr 2e-5
Outputs land in /content/runs/<name>/ (result.json, eval/<part>_predictions.jsonl, train.log, checkpoint/).
"""
import argparse, gc, json, random, subprocess, sys, time
from pathlib import Path

RUNS = Path("/content/runs")
KEV = Path("/content/kev")
SUITE = KEV / "evals/v7/decision-v7"
INIT = "jaredpalmer/kev-4b"
EVAL_PARTS = ("development.adj", "test.adj")   # each record scored once; accuracy reported against both label sets


def load(path):
    from kev.data import load_records
    return load_records(path)


def _fits(tok, r, ms):
    from kev.data import materialize
    from kev.model import fits, training_context
    c = training_context(ms)
    return fits(materialize(r), tok, max_state=c["max_state"], max_branch=c["max_branch"], max_packed=c["max_packed"])


def lengths(a):
    from kev.api import render
    from kev.checkpoint import Checkpoint
    from kev.data import materialize
    from kev.model import encode, load_tokenizer
    meta = Checkpoint(INIT).meta
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    out = {}
    for f in sorted(Path(a.data).glob("*.jsonl")):
        recs = load(f)
        st = sorted(len(tok(render(r["state"]), add_special_tokens=False).input_ids) for r in recs)
        packed = sorted(len(encode(tok, materialize(r), max_state=65536, max_branch=65536)["ids"]) for r in recs)
        out[f.name] = {"n": len(recs), "state_p50": q(st, .5), "state_p95": q(st, .95), "state_max": st[-1],
                       "packed_p50": q(packed, .5), "packed_p95": q(packed, .95), "packed_max": packed[-1],
                       "fit_at_max_state": {ms: sum(_fits(tok, r, ms) for r in recs) for ms in (2048, 3072, 4096, 6144, 7552)}}
    print(json.dumps(out, indent=1))


def predictor(run, max_state):
    """Raw logits on the serving path (bf16, LoRA merged), as kev.serve runs it on CUDA: 2-4.5x faster than the fp32
    exact path, probabilities within ~0.01."""
    import torch
    from kev.checkpoint import LoadOptions
    from kev.model import training_context
    from kev.predictors import LocalPredictor
    return LocalPredictor(run, "cuda", LoadOptions(temperature=1.0, dtype=torch.bfloat16, merge=True), context=training_context(max_state))


def score(run, data, out, max_state, parts=EVAL_PARTS):
    """Fit T on calibration.adj (raw logits, min NLL), then score each part at that T."""
    import torch
    from kev.benchmark import evaluate_records
    from kev.metrics import fit_temperature, probabilities_at_temperature
    pred = predictor(run, max_state)
    res = {}
    try:
        _, cal_rows = evaluate_records(load(Path(data) / "calibration.adj.jsonl"), pred, Path(out) / "calibration")
        T = res["temperature"] = fit_temperature(cal_rows, aggregation="micro")
        for part in parts:
            path = Path(data) / f"{part}.jsonl"
            if not path.exists(): continue
            recs = load(path)
            report, rows = evaluate_records(recs, pred, Path(out) / part, T)
            preds = []
            for rec, row in zip(recs, rows):
                p = probabilities_at_temperature(row, T); i = int(p.argmax())
                preds.append({"task_id": rec["_meta"].get("task_id"), "label": row["keys"][row["label"]], "dispatcher": rec["_meta"].get("dispatcher_label"),
                              "pred": row["keys"][i], "p": round(float(p[i]), 4), "jev": rec["_meta"].get("jev_choice")})
            with open(Path(out) / f"{part}_predictions.jsonl", "w") as f:
                for x in preds: f.write(json.dumps(x, ensure_ascii=False) + "\n")
            acc = lambda xs, k, ref="label": round(sum(x[k] == x[ref] for x in xs) / len(xs), 4) if xs else None
            res[part] = {"n": len(rows), "acc_adjudicated": acc(preds, "pred"), "acc_dispatcher": acc(preds, "pred", "dispatcher"),
                         "jev_acc_adjudicated": acc(preds, "jev"), "jev_acc_dispatcher": acc(preds, "jev", "dispatcher"),
                         "raw": report["clean"], "calibrated": report["calibrated_clean"],
                         "acc_by_label_kind": {k: acc([x for x in preds if x["label"].split(":")[0] == k], "pred") for k in ("NEW", "resume", "steer")}}
    finally:
        del pred; gc.collect(); torch.cuda.empty_cache()
    return res


def regression(run, out, n=300, seed=0):
    import torch
    from kev.benchmark import evaluate_records
    from kev.metrics import metrics
    from kev.suite import load_split
    groups = {}
    for r in load_split(str(SUITE), "development"): groups.setdefault(r["_meta"]["group_id"], []).append(r)
    chosen = random.Random(seed).sample(sorted(groups), min(n, len(groups)))
    sample = [r for g in chosen for r in groups[g]]
    pred = predictor(run, 384)
    try:
        _, rows = evaluate_records(sample, pred, Path(out))
    finally:
        del pred; gc.collect(); torch.cuda.empty_cache()
    return metrics([r for r in rows if r["variant"] == "clean"])


def baseline(a):
    out = RUNS / f"baseline-ms{a.max_state}"
    res = {"run": INIT, "eval": score(INIT, a.data, out / "eval", a.max_state)}
    if a.regression: res["regression"] = regression(INIT, out / "regression")
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def score_run(a):
    from kev.checkpoint import read_meta, write_meta
    run = RUNS / a.name
    res = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else {"name": a.name}
    res["eval"] = score(str(run / "checkpoint"), a.data, run / f"eval-{int(time.time())}", a.max_state)
    m = read_meta(str(run / "checkpoint")); m.temperature = res["eval"]["temperature"]
    m.extra["temperature_fit"] = {"rows": "calibration.adj.jsonl", "method": "min NLL, micro, kev.metrics.fit_temperature", "value": m.temperature}
    write_meta(str(run / "checkpoint"), m)
    if a.regression: res["regression"] = regression(str(run / "checkpoint"), run / f"regression-{int(time.time())}")
    (run / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def train(a):
    from kev.checkpoint import Checkpoint, read_meta, write_meta
    out = RUNS / a.name
    if out.exists(): sys.exit(f"{out} exists; choose a new name")
    out.mkdir(parents=True)
    meta = Checkpoint(a.init_from).meta
    args = meta.extra["args"]
    cmd = [sys.executable, "-m", "kev.train", "--data", str(Path(a.data) / a.train), "--init_from", a.init_from, "--out", str(out / "checkpoint"),
           "--device", "cuda", "--dtype", "bf16", "--base", meta.base, "--lora", meta.lora, "--head_dim", meta.head_dim,
           "--lora_targets", args["lora_targets"], "--option_isolation", int(meta.option_isolation), "--special_embeddings", int(meta.special_embeddings),
           "--weights_dtype", a.weights_dtype or meta.weights_dtype, "--epochs", a.epochs, "--lr", a.lr, "--batch", a.batch, "--accum", a.accum,
           "--checkpointing", 1, "--seed", a.seed, "--max_state", a.max_state, "--p_none_pair", 0]
    if meta.base_revision: cmd += ["--base_revision", meta.base_revision]
    if a.replay: cmd += ["--suite", str(SUITE), "--replay", a.replay]
    cmd = [str(c) for c in cmd]
    (out / "config.json").write_text(json.dumps({"cmd": cmd[2:], "args": vars(a), "kev": subprocess.run(["git", "-C", str(KEV), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}, indent=1, default=str))
    t0 = time.time()
    with open(out / "train.log", "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=KEV)
        for line in p.stdout:
            log.write(line); log.flush()
        if p.wait(): sys.exit(f"kev.train failed; see {out / 'train.log'}")
    res = {"name": a.name, "train_minutes": round((time.time() - t0) / 60, 1)}
    res["eval"] = score(str(out / "checkpoint"), a.data, out / "eval", a.max_state)
    m = read_meta(str(out / "checkpoint")); m.temperature = res["eval"]["temperature"]
    m.extra["temperature_fit"] = {"rows": "calibration.adj.jsonl", "method": "min NLL, micro, kev.metrics.fit_temperature", "value": m.temperature}
    write_meta(str(out / "checkpoint"), m)
    if a.regression: res["regression"] = regression(str(out / "checkpoint"), out / "regression")
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("lengths"); s.add_argument("--data", required=True); s.set_defaults(fn=lengths)
    s = sub.add_parser("baseline"); s.add_argument("--data", required=True); s.add_argument("--max_state", type=int, default=4096)
    s.add_argument("--regression", type=int, default=1); s.set_defaults(fn=baseline)
    s = sub.add_parser("score"); s.add_argument("--name", required=True); s.add_argument("--data", required=True)
    s.add_argument("--max_state", type=int, default=4096); s.add_argument("--regression", type=int, default=1); s.set_defaults(fn=score_run)
    s = sub.add_parser("train")
    for k, d in (("--name", None), ("--data", None), ("--train", "train.mix.jsonl"), ("--init_from", INIT), ("--weights_dtype", "")):
        s.add_argument(k, default=d, required=d is None)
    for k, d in (("--max_state", 4096), ("--epochs", 2), ("--batch", 1), ("--accum", 8), ("--seed", 0), ("--replay", 1000), ("--regression", 1)):
        s.add_argument(k, type=int, default=d)
    s.add_argument("--lr", type=float, default=2e-5)
    s.set_defaults(fn=train)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

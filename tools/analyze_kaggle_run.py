"""What did the arcgen branch do in a Kaggle run? (uses the files the notebook writes + the public-eval solutions)

python tools/analyze_kaggle_run.py --work /kaggle/working --solutions arc-agi_evaluation_solutions.json
Prints, per submission variant, the raw task credit (mean over a task's test queries of "an attempt equals the solution")
and, for the arcgen branch: tasks processed / accepted, accepted predictions that were RIGHT or WRONG against the
solutions, and the queries where it changed attempt_2 (and whether that helped)."""
import argparse, json
from pathlib import Path
import numpy as np


def eq(a, b):
    return a is not None and np.array_equal(np.asarray(a), np.asarray(b))


def task_credit(sub, sols):
    out = {}
    for tid, sol in sols.items():
        rows = sub.get(tid, [])
        ok = [any(eq(r.get(k), sol[i]) for k in ("attempt_1", "attempt_2")) for i, r in enumerate(rows[:len(sol)])]
        out[tid] = sum(ok) / len(sol)
    return out


def analyze(work: Path, solutions: dict):
    files = {"qwen baseline": "submission_baseline.json", "critic only": "submission_critic_only.json",
             "layer ops v2": "submission_layer_operations_v2.json", "v3 without arcgen": "submission_v3_without_arcgen.json",
             "final (with arcgen)": "submission.json"}
    credit, lines = {}, []
    for name, f in files.items():
        if (work / f).is_file():
            credit[name] = task_credit(json.load(open(work / f)), solutions)
            lines.append(f"{name:22s} {sum(credit[name].values()):7.2f} / {len(solutions)}")
    if "v3 without arcgen" in credit and "final (with arcgen)" in credit:
        a, b = credit["v3 without arcgen"], credit["final (with arcgen)"]
        gained = [t for t in a if b[t] > a[t]]; lost = [t for t in a if b[t] < a[t]]
        lines.append(f"arcgen effect on credit: +{sum(b[t]-a[t] for t in gained):.2f} ({len(gained)} tasks gained), "
                     f"-{sum(a[t]-b[t] for t in lost):.2f} ({len(lost)} lost)")
    pred_file = work / "arcgen_predictions.jsonl"
    if pred_file.is_file():
        rows = [json.loads(l) for l in open(pred_file) if l.strip()]
        acc = [r for r in rows if r["accepted"]]
        right = wrong = 0
        for r in acc:
            sol = solutions.get(r["task_id"])
            for i, p in enumerate(r["predictions"]):
                if sol and i < len(sol):
                    right += eq(p, sol[i]); wrong += not eq(p, sol[i])
        reasons = {}
        for r in rows:
            if not r["accepted"]:
                reasons[r["audit"].get("reason")] = reasons.get(r["audit"].get("reason"), 0) + 1
        lines.append(f"arcgen branch: {len(rows)} tasks processed, {len(acc)} accepted; accepted query predictions: "
                     f"{right} right, {wrong} wrong; rejections: {reasons}")
    audit_file = work / "layer_search_audit.json"
    if audit_file.is_file():
        audit = json.load(open(audit_file))
        changed = [k for k, v in audit.items() if v.get("arcgen_changed_attempt_2")]
        lines.append(f"queries where arcgen replaced attempt_2: {len(changed)} {changed[:10]}")
        for key in ("v2_changed_attempt_2", "search_changed_attempt_2"):
            lines.append(f"queries where {key.split('_')[0]} replaced attempt_2: {sum(bool(v.get(key)) for v in audit.values())}")
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working"); ap.add_argument("--solutions", required=True)
    a = ap.parse_args()
    print("\n".join(analyze(Path(a.work), json.load(open(a.solutions)))))

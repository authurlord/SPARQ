#!/usr/bin/env python3
"""
Re-score a finished NIAT POT-direct run with the OLD (pre-fix) matcher regex
vs the NEW (patched) matcher regex, to quantify the "Final Answer:" prefix
extraction fix — the same story as TableBench 0.32 -> 0.467.

The ONLY thing that differs between OLD and NEW is the prefix-extraction regex
inside niat_match_func. Everything else (normalization, eval_ex_match,
allow_semantic, gold) is identical and unchanged. This isolates the extraction
gap exactly the way the user asked.

Usage:
  python rescore_niat_old_vs_new.py --run_dir tmp/niat_test_35B_<timestamp>
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.evaluator import Evaluator  # noqa: E402

# NEW (patched) regex — accepts "Final Answer:", "The answer is:", "Answer:".
NEW_RE = re.compile(r'(?:final\s+answer|the\s+answer\s+is|answer)\s*[:：]\s*(.+)',
                    re.IGNORECASE | re.DOTALL)
# OLD (pre-fix) regex — "The answer is:" only (the original NIAT matcher).
OLD_RE = re.compile(r'the\s+answer\s+is\s*[:：]\s*(.+)',
                    re.IGNORECASE | re.DOTALL)


def extract(pred_answer, pattern):
    m = pattern.search(pred_answer)
    pred = m.group(1) if m else pred_answer
    pred = pred.strip().split('\n')[0]
    pred = pred.strip().replace('"', '')
    return pred


def score(preds_golds, pattern):
    ev = Evaluator()
    correct = []
    for pg in preds_golds:
        pred_answer = pg.get('prediction', '') or ''
        gold = pg.get('answer', '') or ''
        try:
            pred = extract(pred_answer, pattern)
            ok = ev.eval_ex_match(pred, gold, allow_semantic=True, question="")
            correct.append(1 if ok else 0)
        except Exception:
            continue
    if not correct:
        return 0.0, 0
    return 100.0 * sum(correct) / len(correct), len(correct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True,
                    help='NIAT run dir containing preds_and_golds.json')
    args = ap.parse_args()

    pg_path = os.path.join(args.run_dir, 'preds_and_golds.json')
    with open(pg_path) as f:
        preds_golds = json.load(f)

    old_acc, n_old = score(preds_golds, OLD_RE)
    new_acc, n_new = score(preds_golds, NEW_RE)

    n = len(preds_golds)
    print("=" * 70)
    print(f"NIAT re-score: {pg_path}")
    print(f"Total samples: {n}")
    print(f"OLD matcher ('the answer is:' only)   EM = {old_acc:.2f}%  (scored {n_old})")
    print(f"NEW matcher (+ 'Final Answer:'/'Answer:') EM = {new_acc:.2f}%  (scored {n_new})")
    print(f"Delta (extraction fix)                = {new_acc - old_acc:+.2f} pp")
    print("=" * 70)

    out = {
        'run_dir': args.run_dir,
        'total_samples': n,
        'old_matcher_em': old_acc,
        'new_matcher_em': new_acc,
        'delta_pp': new_acc - old_acc,
        'note': 'Only the prefix-extraction regex differs; eval_ex_match unchanged.',
    }
    out_path = os.path.join(args.run_dir, 'rescore_old_vs_new.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()

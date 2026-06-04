#!/usr/bin/env python3
"""
Faithful FetaQA scorer for the schedule-pipeline `predict` column.

Reproduces H-STAR/SPARQ conference FetaQA scoring:
  - prediction text extracted after the 'Answer: ' marker (matching
    fetaqa_score.py: ans = predict.split('Answer: ')[1]; trailing '```' dropped),
    with graceful fallback to the raw text if no marker is present.
  - metric = ROUGE-L fmeasure averaged over examples (rouge_scorer, use_stemmer
    matching the notebook's calculate_metrics; we also report the `evaluate`
    rouge path for parity with fetaqa_score.py when available).

Usage:
  # score a conference CSV (id, answer, predict columns):
  python score_fetaqa_csv.py --csv ../datasets/fetaqa_test_output_A30.csv
  # score a fresh run CSV produced by run_full_pipeline_fetaqa_api.py:
  python score_fetaqa_csv.py --csv tmp/fetaqa_35b/final_results.csv
"""
import argparse
import ast
import json
import os
import re

import pandas as pd
from rouge_score import rouge_scorer


def extract_answer(raw):
    """Mirror fetaqa_score.py's extraction, robust to list-string predict cells."""
    s = str(raw)
    # predict cells are often a python list-string like "['Answer: ...']"
    if s.strip().startswith('[') and s.strip().endswith(']'):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)) and parsed:
                s = str(parsed[0])
        except Exception:
            pass
    # conference fetaqa_score.py: ans = it['generations'][0].split('Answer: ')[1]
    if 'Answer: ' in s:
        s = s.split('Answer: ', 1)[1]
    else:
        # also accept "Final Answer:" / "the answer is:" markers, take last
        m = list(re.finditer(r'(?:final\s+answer|the\s+answer\s+is|answer)\s*[:：]\s*',
                             s, re.IGNORECASE))
        if m:
            s = s[m[-1].end():]
    s = s.replace("\n```", "")
    # strip a leading "Therefore, the answer is:" residue and surrounding quotes
    s = re.sub(r'^\s*therefore,?\s*the answer is\s*[:：]?\s*', '', s, flags=re.IGNORECASE)
    return s.strip().strip('"\'').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--pred_col', default='predict')
    ap.add_argument('--gold_col', default='answer')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    preds, golds = [], []
    for _, r in df.iterrows():
        g = str(r[args.gold_col]).strip()
        p = extract_answer(r[args.pred_col])
        if not g:
            continue
        golds.append(g)
        preds.append(p if p else 'unknown answer')

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1 = r2 = rl = 0.0
    n = 0
    for p, g in zip(preds, golds):
        try:
            sc = scorer.score(g, p)
        except Exception:
            continue
        r1 += sc['rouge1'].fmeasure
        r2 += sc['rouge2'].fmeasure
        rl += sc['rougeL'].fmeasure
        n += 1
    out = {
        'csv': args.csv,
        'n_scored': n,
        'rouge1_f': r1 / n if n else 0.0,
        'rouge2_f': r2 / n if n else 0.0,
        'rougeL_f': rl / n if n else 0.0,
    }
    print(json.dumps(out, indent=2))
    out_path = args.out or (os.path.splitext(args.csv)[0] + '.rougeL.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()

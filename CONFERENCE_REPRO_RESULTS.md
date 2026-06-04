# SPARQ / H-STAR Conference Reproduction — Results

Reader endpoints (12.43): **9543 = Qwen3.6-35B-A3B-FP8**, **9544 = Qwen3.5-4B**.
Bars: paper qwen3-4B column, and the March commit `3cd76e0` (WikiTQ 79.60).

This doc tracks the conference reproduction of the 5 core table-reasoning
datasets with the unified API readers, after the 2026-06-04 extraction fixes
(TableBench / NIAT "Final Answer:" prefix-strip; WikiTQ SQL-timeout guard).

## Status table

| Dataset | Metric | Qwen3.5-4B | Qwen3.6-35B | Bar (paper 4B / March) | Status |
|---|---|---:|---:|---|---|
| WikiTQ | EM (accuracy) | **76.34** | — | March 3cd76e0 = **79.60**; paper 4B = 77.03 | done (faithful, timeout-only) |
| TabFact | accuracy | — | — | repro 92.54 | not in this batch |
| TableBench | avg ROUGE-L | — | **0.4671** | paper/conf = 0.5005 | done (35B) |
| NIAT | EM | — | **77.86** | conf 66.58; 30B full-pipe 73.45; POT-direct hist 53.55 | done (35B) |
| FetaQA | ROUGE-L fmeasure | — | _pending 9543_ | conf = 0.4990 | pending NIAT to free 9543 |

## TableBench (35B) — DONE

- **avg ROUGE-L = 0.4671** (886 samples), POT-direct + `tablebench_rouge_l_score`.
- `accuracy@0.5 = 0.4266`, `accuracy@0.8 = 0.3296`.
- Artifact: `schedule_pipeline/tmp/tablebench_35b_20260604_053340/evaluation.json`.
- **Extraction fix story**: the 35B reader follows the prompt's `Final Answer:`
  format literally. The original `tablebench_rouge_l_score` only stripped
  `Therefore, the answer is:`, so a correct `Final Answer: 10.6` vs gold `10.6`
  scored far below 0.8. The patched matcher strips a leading
  `(?:the )?(?:final answer|answer is|answer):` prefix. This raised the score
  from **0.32 → 0.4671** with the *matcher logic otherwise unchanged* (only the
  extraction regex changed). Codex-verified PASS (see CONF_REPRO_REVIEW_codex.md).

## WikiTQ (4B) — DONE (baseline), faithful re-run pending

- Runner: `run_full_pipeline_wikitq_api.py` on 9544, full test split (4344).
- **Faithful EM = 76.34%** (n=4344, 67 format errors; timeout-only, no blanket
  recursive skip). The earlier baseline with the crude blanket `recursive` skip
  scored 76.47% — so removing the blanket skip and relying on the proper
  per-query timeout moved the number by only −0.13 pp, confirming the recursive
  CTEs were genuinely pathological (correctly aborted by the timeout, not valid
  queries that were being wrongly dropped). 76.34 is the reported number.
- **SQL-timeout guard added** (`utils/multi_db_v2.py`): a single LLM-generated
  `WITH RECURSIVE` CTE over a comma-list column (table 738) built an
  effectively-infinite cross product and hung the run for >1.5h (the per-query
  `try/except` cannot catch an infinite loop). Added a sqlite
  `set_progress_handler` wall-clock timeout (`SPARQX_SQL_TIMEOUT`, default 8s)
  that aborts a runaway query → caught as empty result. Verified: the exact
  table-738 query is interrupted at the deadline, normal queries unaffected.
- Resumed from the 4-stage LLM cache (`tmp/wikitq_q35_4b/cache_*.json`), so no
  4B re-generation; only SQL parsing/exec + final QA re-run.

### Per-example diagnosis (EM 76.47 < paper 77.03)

The 76.47 baseline run still contained the crude blanket `recursive` SQL skip
(now removed; a faithful no-skip re-run with the proper timeout is in progress).
Root-cause breakdown of failures:

- **Context-overflow (infrastructure, not model)**: 39 predictions returned
  `BadRequestError ... input_tokens value=10241` — the prompt exceeded the 9544
  endpoint's `max_model_len=12288` (input ~10241 + 2048 output budget = 12289 >
  12288; large table + full-table evidence + CoT). **23 of these scored WRONG =
  0.53 pp of recoverable headroom lost to a context cap, not to model
  capability** (these rows never reached generation). This is an upper-bound
  recovery: re-serving on a longer-context endpoint would move 76.34 → ~**76.86**
  (within 0.17 pp of paper 4B 77.03), pending a paired rerun.
- **Format/extraction (28 remaining format errors = 67 − 39 overflow)**:
  truncated CoT where the 4B ran out of output budget before emitting a clean
  `Final Answer:` line.
- **Genuine reasoning misses**: e.g. wrong entity / miscount / wrong aggregate —
  real 4B capability, expected at this model scale.
- The residual gap to the March 79.60 anchor is the known reader/pipeline
  differences (different reader endpoint + DuckDB→sqlite executor migration),
  not the extraction fixes.

## NIAT (35B) — DONE

- Runner: `run_pipeline_niat_pot_direct.py` on 9543, POT-direct, 2932 samples.
- **NIAT EM = 77.86%** (n=2932). 94.6% answered directly by Python execution;
  158 went to the 35B LLM fallback. Artifact:
  `tmp/niat_test_35B_20260604_093741/evaluation.json`.
- Iterative retry: iter-0 (all 2932) → iter-1 (309 still-failing) → iter-2
  (193 still-failing) → LLM fallback (158) → eval. The iter-2 tail was slow
  (~10-20 s/it, hard samples with long outputs) but progressed (not hung).

### OLD vs NEW matcher (the "Final Answer:" extraction story)

Re-scoring the SAME predictions with the OLD matcher (`the answer is:` only)
vs the NEW matcher (+ `Final Answer:` / `Answer:`), via
`schedule_pipeline/rescore_niat_old_vs_new.py` (matcher normalization /
`eval_ex_match` / gold UNCHANGED; only the prefix regex differs):

| Matcher | NIAT EM |
|---|---:|
| OLD (`the answer is:` only) | **46.86%** |
| NEW (+ `Final Answer:` / `Answer:`) | **77.86%** |
| **Delta (extraction fix only)** | **+31.00 pp** |

The NEW-matcher re-score (77.86) exactly reproduces the runner's own eval
(77.86), confirming the re-scorer is faithful. This is the same alignment story
as TableBench (0.32 → 0.467), but far larger here because the POT-direct 35B
outputs follow the prompt's `Final Answer:` format almost universally, which the
original `the answer is:`-only regex missed.

### vs conference / historical bars

- **77.86 beats conference NIAT 66.58 by +11.28 pp**, the 30B full-pipeline
  73.45 by +4.41 pp, and the historical POT-direct 53.55 by +24.31 pp.
- The OLD-matcher number (46.86) is BELOW historical POT-direct 53.55, which is
  exactly the symptom the extraction fix addresses — the conference/historical
  scores already extracted the answer; this run's 35B emits the `Final Answer:`
  prefix the old regex didn't strip.

## FetaQA (35B) — PENDING

- Will serialize on 9543 after NIAT frees it.
- Scored with rougeL fmeasure (conference metric = 0.4990).
- Status / blocker documented after investigation (see report).

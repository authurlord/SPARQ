## SQL timeout guard

**Verdict: PASS-WITH-CAVEAT.** The `multi_db_v2.py` progress-handler timeout is a robustness guard, but the runner still has a broader recursive-query skip that is not the same thing as a timeout.

- **Pure timeout behavior:** `SQL_EXEC_TIMEOUT_S` is env-configured and disabled when `<=0` (`utils/multi_db_v2.py:13-18`). `execute_query()` installs a sqlite progress handler only when it can find a raw sqlite connection and the timeout is enabled (`utils/multi_db_v2.py:165-179`). The callback returns non-zero only after the wall-clock deadline (`utils/multi_db_v2.py:173-178`), so normal fast queries should be unaffected except for tiny progress-callback overhead.
- **Cleanup:** The handler is removed in `finally` via `_raw.set_progress_handler(None, 0)` (`utils/multi_db_v2.py:179-183`). That is the right pattern; it should not leak onto later queries on the same connection.
- **Raw connection extraction:** `_raw_sqlite_conn()` best-effort checks common SQLAlchemy/records paths and returns `None` if no `sqlite3.Connection` is found (`utils/multi_db_v2.py:26-46`). If it returns `None`, no handler is installed and execution proceeds without timeout. That fallback is graceful, not result-changing, but it should be logged if you need auditability.
- **8s default:** For WikiTQ single-table SQL, 8s is reasonable. Legitimate SQL should normally finish far below that. The risk is a valid but unusually expensive query being killed and scored as empty; low probability, but not impossible. Keep `SPARQX_SQL_TIMEOUT` configurable and log timeout counts/indexes.
- **Caveat / possible metric change:** `schedule_pipeline/run_full_pipeline_wikitq_api.py` also pre-skips any SQL containing `recursive` (`run_full_pipeline_wikitq_api.py:1011-1013`). That is broader than the progress-handler guard: it changes behavior before execution, and would skip even a legitimate terminating recursive CTE. For a clean "robustness only" claim, remove this broad skip and rely on the sqlite timeout, or restrict it to a logged known-bad query/table.

## TableBench 0.4671

**Verdict: PASS.** This is extraction normalization, not a ROUGE-L metric change.

- **Result artifact:** The reported run has avg ROUGE-L `0.4671039333`, accuracy@0.5 `0.4266365688`, accuracy@0.8 `0.3295711061`, `886` samples (`schedule_pipeline/tmp/tablebench_35b_20260604_053340/evaluation.json:1-6`).
- **Prefix extraction only:** `tablebench_rouge_l_score()` converts pred/gold to strings (`utils/evaluator.py:292-294`), first handles the existing `Therefore, the answer is:` pattern (`utils/evaluator.py:296-300`), then strips only a leading answer prefix such as `Final Answer:`, `Answer:`, or `The answer is:` (`utils/evaluator.py:302-310`).
- **ROUGE unchanged:** After prefix stripping, it only strips surrounding quotes (`utils/evaluator.py:312-314`) and computes `rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)` with `scorer.score(gold_str, pred_str)` (`utils/evaluator.py:320-325`). Gold handling, stemming, and ROUGE-L computation are not altered.
- **No metric inflation found:** Stripping a literal leading `Final Answer:` emitted by the reader aligns prediction format with bare-value gold. It does not change thresholds or the scorer. This is the same class of extraction fix as the WikiTQ 35B parser issue, not a new TableBench metric.

## Verdict

- **SQL timeout guard:** **PASS-WITH-CAVEAT.** The sqlite progress handler is correct and cleaned up properly. The broad `recursive` pre-skip in the runner is the caveat because it is not purely timeout-based.
- **TableBench 0.4671:** **PASS.** The fix strips model-output boilerplate before the same ROUGE-L computation; I do not see metric alteration.

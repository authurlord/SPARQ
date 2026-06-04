## SQL timeout guard

**Verdict: PASS.** The `multi_db_v2.py` progress-handler timeout is a robustness guard. The earlier caveat about a broad recursive-query skip has been resolved in the current runner; the follow-up WikiTQ section below verifies the no-skip state.

- **Pure timeout behavior:** `SQL_EXEC_TIMEOUT_S` is env-configured and disabled when `<=0` (`utils/multi_db_v2.py:13-18`). `execute_query()` installs a sqlite progress handler only when it can find a raw sqlite connection and the timeout is enabled (`utils/multi_db_v2.py:165-179`). The callback returns non-zero only after the wall-clock deadline (`utils/multi_db_v2.py:173-178`), so normal fast queries should be unaffected except for tiny progress-callback overhead.
- **Cleanup:** The handler is removed in `finally` via `_raw.set_progress_handler(None, 0)` (`utils/multi_db_v2.py:179-183`). That is the right pattern; it should not leak onto later queries on the same connection.
- **Raw connection extraction:** `_raw_sqlite_conn()` best-effort checks common SQLAlchemy/records paths and returns `None` if no `sqlite3.Connection` is found (`utils/multi_db_v2.py:26-46`). If it returns `None`, no handler is installed and execution proceeds without timeout. That fallback is graceful, not result-changing, but it should be logged if you need auditability.
- **8s default:** For WikiTQ single-table SQL, 8s is reasonable. Legitimate SQL should normally finish far below that. The risk is a valid but unusually expensive query being killed and scored as empty; low probability, but not impossible. Keep `SPARQX_SQL_TIMEOUT` configurable and log timeout counts/indexes.
- **Resolved caveat:** The broad `recursive` pre-skip was the earlier risk. In the current runner, both Select_Row and Execute_SQL rely on the sqlite timeout instead; see the WikiTQ follow-up section for line-level verification.

## TableBench 0.4671

**Verdict: PASS.** This is extraction normalization, not a ROUGE-L metric change.

- **Result artifact:** The reported run has avg ROUGE-L `0.4671039333`, accuracy@0.5 `0.4266365688`, accuracy@0.8 `0.3295711061`, `886` samples (`schedule_pipeline/tmp/tablebench_35b_20260604_053340/evaluation.json:1-6`).
- **Prefix extraction only:** `tablebench_rouge_l_score()` converts pred/gold to strings (`utils/evaluator.py:292-294`), first handles the existing `Therefore, the answer is:` pattern (`utils/evaluator.py:296-300`), then strips only a leading answer prefix such as `Final Answer:`, `Answer:`, or `The answer is:` (`utils/evaluator.py:302-310`).
- **ROUGE unchanged:** After prefix stripping, it only strips surrounding quotes (`utils/evaluator.py:312-314`) and computes `rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)` with `scorer.score(gold_str, pred_str)` (`utils/evaluator.py:320-325`). Gold handling, stemming, and ROUGE-L computation are not altered.
- **No metric inflation found:** Stripping a literal leading `Final Answer:` emitted by the reader aligns prediction format with bare-value gold. It does not change thresholds or the scorer. This is the same class of extraction fix as the WikiTQ 35B parser issue, not a new TableBench metric.

## Verdict

- **SQL timeout guard:** **PASS.** The sqlite progress handler is correct and cleaned up properly. The earlier broad `recursive` pre-skip caveat is resolved in the current runner.
- **TableBench 0.4671:** **PASS.** The fix strips model-output boilerplate before the same ROUGE-L computation; I do not see metric alteration.

## WikiTQ 4B 76.34 + diagnosis

**76.34 verdict: PASS-WITH-CAVEAT.** The reported faithful run is backed by the
artifact, and the current runner really has removed the broad recursive-skip
guard; the caveat is that "confirms the recursive CTEs were genuinely
pathological" is slightly stronger than the evidence.

- **Number verified:** `schedule_pipeline/tmp/wikitq_q35_4b/evaluation_results.json`
  reports `accuracy=76.33517495395948`, `total_samples=4344`, and `67` format
  errors. That is the reported **76.34%**.
- **No blanket recursive skip in current code:** the Select_Row path now directly
  calls `executor.sql_exec(...)` and relies on `multi_db_v2` timeout comments
  (`schedule_pipeline/run_full_pipeline_wikitq_api.py:950-960`); Execute_SQL does
  the same after `if sql:` (`schedule_pipeline/run_full_pipeline_wikitq_api.py:1011-1022`).
  I do not see the prior `if 'recursive' in sql.lower(): skip` guard in these
  locations.
- **Timeout mechanism is the right replacement:** `SQL_EXEC_TIMEOUT_S` defaults
  to 8s (`utils/multi_db_v2.py:13-18`), installs sqlite `set_progress_handler`
  before execution (`utils/multi_db_v2.py:165-178`), and removes it in `finally`
  (`utils/multi_db_v2.py:179-183`). That makes timeout-only execution more
  faithful than blanket recursive rejection.
- **Caveat on the −0.13 pp comparison:** I did not find a separate `76.47`
  `evaluation_results.json` under the usual `schedule_pipeline/tmp/**` artifacts;
  it is documented in `CONFERENCE_REPRO_RESULTS.md:36-41`, not independently
  verified from a result file in this pass. Also, a −0.13 pp delta shows the
  skip was near-neutral, but does not by itself prove every skipped recursive CTE
  was pathological. The table-738 hang plus timeout behavior supports the claim;
  phrase it as "consistent with pathological recursive CTEs" rather than
  "confirming" all cases were pathological.

**Context-cap diagnosis verdict: PASS-WITH-CAVEAT.** The 39 overflow / 23 wrong
count and 0.53 pp arithmetic are correct, and this is legitimately an
infrastructure cap. The caveat is that it is an upper-bound recovery estimate
unless those 23 are actually rerun on a longer-context endpoint.

- **Error class verified:** `final_results.csv` has 39 `predict` entries
  containing `BadRequestError`, `input_tokens`, and `10241`. A representative
  row says the 9544 endpoint had max context `12288`, requested `2048` output
  tokens, and prompt input was at least `10241`, totaling `12289`.
- **Wrong-overflow count verified:** intersecting those 39 overflow rows with
  `error_indices` gives **23** wrong rows; all 39 are in `format_error_indices`.
  The remaining non-overflow format errors are 28, not 26, in the 76.34 artifact
  (`67 total format errors - 39 overflow = 28`). If the doc's "26 remaining"
  refers to the older 76.47 run, say so explicitly.
- **Arithmetic:** `23 / 4344 * 100 = 0.5295 pp`. Recovering all 23 would move
  the faithful 76.34 run to **76.86**, not exactly 77.0. Starting from the older
  76.47 run gives almost exactly 77.0. For the current faithful number, write
  "to about 76.9, within 0.17 pp of paper 77.03."
- **Framing:** It is fair to attribute this ~0.5 pp loss to the serving context
  cap rather than reader reasoning, because these rows never reached generation.
  Do not imply guaranteed correctness without rerun; say "recoverable headroom"
  or "upper-bound loss from infra cap."

**Bottom line: PASS-WITH-CAVEAT.** Reporting **76.34** as the faithful
timeout-only WikiTQ 4B reproduction is sound. The diagnosis is honest if softened:
timeout-only is the faithful fix; context overflow explains up to ~0.53 pp; after
that, the result is close to paper 77.03. The residual gap to the March 79.60
anchor can be described as endpoint/pipeline/executor drift, but that attribution
should remain a diagnosis rather than a proven decomposition unless backed by a
paired rerun.

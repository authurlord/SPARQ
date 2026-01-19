# SQL Iterative Pipeline Documentation

## Overview

This pipeline is a simplified version of the TableBench testing pipeline that focuses exclusively on SQL execution with iterative retry logic. It removes the Router, RAG, and Check modules to isolate SQL generation and execution performance.

## Key Features

1. **SQL-Only Pipeline**: Only uses Execute_SQL module
2. **Iterative Retry**: If SQL execution fails, appends error message and retries (max 3 iterations)
3. **Batch Processing**: Uses `infer_prompts` for batch queries instead of one-by-one (significantly faster)
4. **Detailed Logging**: Tracks all SQL attempts, successes, and failures
5. **Success Rate Reporting**: Reports SQL execution success rate across iterations

## Batch Processing Optimization

### Before (One-by-One)
```python
for idx in range(len(samples)):
    for iteration in range(max_iterations):
        responses = infer_prompts([prompt], ...)  # One prompt at a time
        # Process response
```

### After (Batch)
```python
for iteration in range(max_iterations):
    # Build prompts for ALL active samples
    prompt_list = [build_prompt(sample) for sample in active_samples]
    
    # Batch query - all prompts sent together
    responses = infer_prompts(prompt_list, ...)
    
    # Process all responses
    for response in responses:
        # Execute SQL and check success
```

**Benefits**:
- Reduces API overhead (fewer HTTP requests)
- Better GPU utilization (batch inference)
- Significantly faster execution time
- Maintains same retry logic per sample

## Files

- `run_sql_iterative_tablebench.py`: Main pipeline script with batch processing
- `test_sql_iterative_tablebench.sh`: Test script to run the pipeline

## Usage

### Basic Usage

```bash
bash test_sql_iterative_tablebench.sh
```

### Custom Parameters

```bash
# Test first 100 samples
bash test_sql_iterative_tablebench.sh --first_n 100

# Test 50 random samples
bash test_sql_iterative_tablebench.sh --first_n 50 --random_sample

# Adjust SQL sampling
bash test_sql_iterative_tablebench.sh --sql_sample_num 5 --temperature 0.8

# Adjust concurrency
bash test_sql_iterative_tablebench.sh --llm_concurrency 64
```

## Pipeline Flow

```
1. Load Data
   └─> Load TableBench JSONL
   └─> Sample N samples (random or first N)

2. Preprocess Tables
   └─> Convert to DataFrames
   └─> Make column names unique

3. Build Database
   └─> Create NeuralDB with all tables
   └─> Initialize SQL Executor

4. Execute SQL with Iterative Retry (BATCH)
   └─> Iteration 0: Generate SQL for ALL samples (batch)
       ├─> Parse SQL queries
       ├─> Execute SQL
       └─> Mark successful samples
   └─> Iteration 1: Generate SQL for FAILED samples (batch)
       ├─> Include error feedback in prompt
       ├─> Parse and execute
       └─> Mark newly successful samples
   └─> Iteration 2: Final retry for remaining failures (batch)
       └─> Same process

5. Generate Final QA (BATCH)
   └─> Build QA prompts for ALL samples (batch)
   └─> Include SQL results as evidence
   └─> Generate answers

6. Evaluate
   └─> Calculate ROUGE-L
   └─> Calculate Accuracy@0.5 and @0.8

7. Save Results
   └─> execution_stats_detailed.json
   └─> execution_summary.json
   └─> results.csv
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--first_n` | 50 | Number of samples to test |
| `--random_sample` | False | Randomly sample instead of first N |
| `--sql_sample_num` | 3 | Number of SQL queries per question |
| `--max_iterations` | 3 | Max retry iterations |
| `--temperature` | 0.7 | Sampling temperature |
| `--top_p` | 0.8 | Sampling top_p |
| `--llm_concurrency` | 32 | Max concurrent API requests |
| `--llm_name` | qwen3-4b | Model name in vLLM server |
| `--api_base` | http://localhost:8000/v1 | vLLM API URL |
| `--api_key` | api-key-qwen3 | API key |

## Output Files

### 1. execution_stats_detailed.json
Contains detailed statistics for each sample:
```json
[
  {
    "idx": 0,
    "question": "...",
    "iterations": [
      {
        "iteration": 0,
        "attempts": [
          {
            "sample_idx": 0,
            "response": "...",
            "sql": "SELECT ...",
            "success": true,
            "error": null,
            "result": "..."
          }
        ],
        "success": true
      }
    ],
    "final_success": true,
    "final_result": "...",
    "total_attempts": 3,
    "successful_attempts": 1
  }
]
```

### 2. execution_summary.json
Overall statistics:
```json
{
  "total_samples": 50,
  "total_attempts": 150,
  "successful_attempts": 120,
  "final_success_count": 45,
  "sql_execution_success_rate": 0.80,
  "sample_success_rate": 0.90,
  "samples_success_iteration_0": 40,
  "samples_success_iteration_1": 3,
  "samples_success_iteration_2": 2,
  "avg_attempts_per_sample": 3.0
}
```

### 3. results.csv
Final results with predictions:
```csv
id,question,gold_answer,prediction,sql_success,total_attempts,successful_attempts,iterations_used
table_1,What is...,42,42,True,3,1,1
```

## Iterative Retry Logic

### How It Works

1. **Iteration 0 (Initial Attempt)**
   - Generate SQL for all samples (batch)
   - Execute and check results
   - Mark successful samples

2. **Iteration 1 (First Retry)**
   - Only process failed samples (batch)
   - Include error message in prompt:
     ```
     [Previous Attempt Failed - Iteration 1]
     Error: column "xyz" does not exist
     Please generate a corrected SQL query that avoids this error.
     ```
   - Execute and mark newly successful samples

3. **Iteration 2 (Final Retry)**
   - Process remaining failures (batch)
   - Include latest error feedback
   - Final attempt

### Success Tracking

- **Per-Sample Success**: Did at least one SQL succeed for this sample?
- **Per-Attempt Success**: Did this specific SQL execution succeed?
- **Iteration Success**: Which iteration did the sample succeed in?

## Performance Metrics

### SQL Execution Success Rate
```
successful_attempts / total_attempts
```
Measures: What percentage of generated SQL queries execute successfully?

### Sample Success Rate
```
final_success_count / total_samples
```
Measures: What percentage of samples got at least one successful SQL execution?

### Iteration Breakdown
- **Iteration 0**: Samples that succeeded on first try
- **Iteration 1**: Samples that failed initially but succeeded on retry 1
- **Iteration 2**: Samples that succeeded only on final retry

## Example Output

```
================================================================================
Simplified SQL Pipeline with Iterative Retry (Batch Version)
================================================================================
Timestamp: 20260119_143022
Save Path: datasets/schedule_test/tablebench_sql_iterative_20260119_143022
Dataset: ../datasets/TableBench/TableBench.jsonl
Sample Size: 50
Random Sample: True
SQL Samples per Question: 3
Max Iterations: 3
LLM Name: qwen3-4b
API Base: http://localhost:8000/v1
Temperature: 0.7
Top P: 0.8
LLM Concurrency: 32
================================================================================

[Step 1] Loading Data...
Loaded 14000 total samples
Randomly sampling 50 samples...
Using 50 samples for testing

[Step 2] Preprocessing Tables...
100%|████████████████████████████████████████| 50/50 [00:01<00:00, 45.23it/s]

[Step 3] Building Database...

[Step 4] Executing SQL with Iterative Retry (Batch)...

  Iteration 0: Processing 50 samples...
  Generating SQL queries for 50 samples...
  Iteration 0: 42 samples succeeded, 8 remaining

  Iteration 1: Processing 8 samples...
  Generating SQL queries for 8 samples...
  Iteration 1: 5 samples succeeded, 3 remaining

  Iteration 2: Processing 3 samples...
  Generating SQL queries for 3 samples...
  Iteration 2: 2 samples succeeded, 1 remaining

[Step 5] Calculating Statistics...

[Step 6] Generating Final QA (Batch)...
  Generating QA responses for 50 samples...

[Step 7] Evaluation...

================================================================================
EXECUTION SUMMARY
================================================================================
Total Samples: 50
Total SQL Attempts: 150
Successful SQL Executions: 135
SQL Execution Success Rate: 90.00%
Sample Success Rate: 98.00%

Success by Iteration:
  Iteration 0 (initial): 42 samples
  Iteration 1 (retry 1): 5 samples
  Iteration 2 (retry 2): 2 samples
  Total Success: 49 samples

EVALUATION RESULTS
================================================================================
Average ROUGE-L: 0.7234
Accuracy@0.5: 85.00%
Accuracy@0.8: 72.00%

Total Time: 125.34s (2.09 minutes)
Results saved to: datasets/schedule_test/tablebench_sql_iterative_20260119_143022
================================================================================
```

## Comparison with POT Pipeline

| Feature | SQL Iterative | POT Pipeline |
|---------|---------------|--------------|
| Modules | SQL only | Router + RAG + Check + SQL/Python |
| Retry Logic | Yes (3 iterations) | No |
| Batch Processing | Yes | No (one-by-one) |
| Error Feedback | Yes | No |
| Speed | Fast (batch) | Slower (sequential) |
| Focus | SQL execution | Full pipeline |

## Troubleshooting

### Low SQL Success Rate
- Check SQL prompt template
- Increase `sql_sample_num` (more attempts per question)
- Adjust `temperature` and `top_p`
- Review error messages in `execution_stats_detailed.json`

### Slow Execution
- Increase `llm_concurrency` (more parallel requests)
- Reduce `first_n` (test fewer samples)
- Check vLLM server performance

### API Errors
- Verify `api_base` URL is correct
- Check `api_key` matches server configuration
- Ensure `llm_name` matches registered model name

## Future Improvements

1. **Adaptive Retry**: Adjust retry strategy based on error type
2. **SQL Validation**: Pre-validate SQL syntax before execution
3. **Error Classification**: Categorize errors (syntax, semantic, execution)
4. **Dynamic Sampling**: Adjust `sql_sample_num` based on difficulty
5. **Parallel Execution**: Execute SQL queries in parallel

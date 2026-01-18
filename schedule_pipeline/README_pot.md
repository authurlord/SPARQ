# TableBench PoT (Program-of-Thought) Pipeline

This pipeline implements a Program-of-Thought (PoT) approach for the TableBench dataset, where the model solves reasoning questions by writing and executing Python code instead of direct text generation or SQL.

## Overview

The pipeline consists of two main stages:
1.  **Code Generation**: The LLM generates Python code to solve the question (n=3 samples).
2.  **Execution & Evaluation**: The generated code is executed in a sandbox-like environment. The execution result is fed back into a final QA prompt to generate the final answer, which is then evaluated using ROUGE-L.

## Files

-   `run_pipeline_tablebench_pot.py`: Main pipeline script.
-   `test_pipeline_tablebench_pot.sh`: Bash runner script.
-   `../utils/python_executor.py`: Utility to execution Python code safely.
-   `../prompts/python_reason_tablebench.txt`: Few-shot prompt for Python code generation.

## Usage

### 1. Prerequisites

Ensure you have the TableBench PoT dataset:
-   Path: `../datasets/TableBench/TableBench_PoT.jsonl`

### 2. Running the Pipeline

Use the provided shell script:

```bash
cd schedule_pipeline
./test_pipeline_tablebench_pot.sh
```

**Common Arguments:**

-   `--llm_path`: Path to your LLM (default: `/data/workspace/yanmy/models/Qwen2.5-7B-Instruct/`)
-   `--first_n N`: Only run the first N samples (useful for testing).
-   `--dataset_name`: Default `tablebench`.
-   `--code_sample_num`: Number of Python code samples to generate per question (default: 3).

### 3. Output

Results are saved in `datasets/schedule_test/tablebench_pot/`:
-   `pot_results.csv`: Detailed results including generated code, execution output, and final prediction.
-   `evaluation.json`: ROUGE-L scores.

## Implementation Details

-   **Sandbox**: The `utils.python_executor` handles code execution. It intercepts `pd.read_csv('table.csv')` calls and provides the dataframe directly to the execution context.
-   **Prompting**: Uses a 2-shot prompt demonstrating how to load the dataframe, perform analysis with pandas, and print the "Final Answer".

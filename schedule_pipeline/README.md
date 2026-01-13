# Schedule Pipeline

This directory contains the main pipeline scripts for running H-STAR on different table QA datasets.

## Supported Datasets

| Dataset | Script | Description |
|---------|--------|-------------|
| WikiTQ | `run_full_pipeline_wikitq.py` | Wikipedia Table Questions |
| NIAT | `run_full_pipeline_niat.py` | Nested/Hierarchical Table QA |

---

## NIAT Pipeline

The NIAT (Nested Information Answering on Tables) pipeline handles tables with complex nested/hierarchical structures, such as:
- Multi-level headers
- Merged cells represented as empty values
- Row grouping with forward-fill patterns

### Key Features

1. **Table Flattening**: Automatically detects and forward-fills hierarchical empty cells
2. **Duplicate Column Handling**: Renames duplicate column names with suffixes
3. **SQL-Compatible Output**: Produces clean DataFrames that work with SQLite

### Quick Start

#### Demo Mode (No LLM Required)

```bash
cd schedule_pipeline

# Run demo test (no model needed)
bash test_pipeline_niat.sh
```

#### Full Pipeline with vLLM

```bash
cd schedule_pipeline

# Set model paths
LLM_PATH="/path/to/your/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL="/path/to/bge-m3"

# Run full pipeline
python run_full_pipeline_niat.py \
  --llm_path ${LLM_PATH} \
  --niat_json_path ../datasets/NIAT/sampled_qa_pairs_4000_fixed.json \
  --tmp_save_path tmp/niat_full \
  --first_n 100 \
  --tensor_parallel_size 1
```

#### API Mode (Async LLM)

```bash
python run_full_pipeline_niat.py \
  --use_api \
  --llm_path ${LLM_PATH} \
  --niat_json_path ../datasets/NIAT/sampled_qa_pairs_4000_fixed.json \
  --tmp_save_path tmp/niat_api \
  --first_n 50 \
  --llm_concurrency 16
```

### Command Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--niat_json_path` | `datasets/NIAT/sampled_qa_pairs_4000_fixed.json` | Path to NIAT JSON file |
| `--tmp_save_path` | `datasets/schedule_test/niat` | Output directory |
| `--first_n` | `-1` | Process first N samples (-1 for all) |
| `--demo_mode` | `False` | Skip LLM inference, test data flow only |
| `--use_api` | `False` | Use async API instead of local vLLM |
| `--skip_llm` | `False` | Skip LLM inference step |
| `--tensor_parallel_size` | `1` | GPU parallelism for vLLM |
| `--temperature` | `0.7` | Sampling temperature |
| `--top_p` | `0.8` | Sampling top_p |

### Output Files

After running the pipeline, the following files are created in `tmp_save_path`:

| File | Description |
|------|-------------|
| `niat_df_processed.npy` | Processed and flattened DataFrames |
| `llm_query_list.pkl` | All prompts and responses |
| `prompts_sample.json` | Sample prompts for each operation type |
| `sql_test_results.json` | SQL execution test results |
| `timeline.json` | Timing breakdown |

### NIAT Prompts

Custom prompts for NIAT are located in `../prompts/`:

| Prompt | Purpose |
|--------|---------|
| `col_select_niat.txt` | Column selection with 3 demonstrations |
| `row_select_niat.txt` | Row selection with 3 demonstrations |
| `sql_reason_niat.txt` | SQL generation with 3 demonstrations |
| `text_reason_niat.txt` | Final QA with 3 demonstrations |

---

## WikiTQ Pipeline

For WikiTable Questions dataset, use:

```bash
bash test_pipeline.sh
```

Or customize:

```bash
python run_full_pipeline_wikitq.py \
  --llm_path /path/to/model \
  --dataset_name wikitq \
  --split test \
  --first_n 50
```

---

## Data Preprocessing

### NIAT Table Flattening

The NIAT dataset contains nested tables where empty cells indicate hierarchical relationships. For example:

**Original (nested):**
```
| Group    | Union | Craft   | Employees |
|----------|-------|---------|-----------|
| Mainline | APA   | Pilots  | 13200     |
|          | APFA  | FA      | 24900     |  <- Group is empty, inherits "Mainline"
| Envoy    | ALPA  | Pilots  | 2200      |
```

**Flattened:**
```
| Group    | Union | Craft   | Employees |
|----------|-------|---------|-----------|
| Mainline | APA   | Pilots  | 13200     |
| Mainline | APFA  | FA      | 24900     |  <- Group filled as "Mainline"
| Envoy    | ALPA  | Pilots  | 2200      |
```

To preprocess NIAT data separately:

```bash
python convert_niat_parallel.py \
  --input_path ../datasets/NIAT/sampled_qa_pairs_4000_fixed.json \
  --output_path ../datasets/schedule_test/niat/niat_df_processed.npy \
  --num_workers 8
```

---

## Troubleshooting

### JSON Decode Error
If you encounter `JSONDecodeError: Unterminated string`, the original JSON file is truncated. Use the fixed version: `sampled_qa_pairs_4000_fixed.json`

### Duplicate Column Name Error
The pipeline automatically handles duplicate columns by adding suffixes (`_1`, `_2`, etc.)

### vLLM Not Available
Run with `--demo_mode` to test the data pipeline without LLM:
```bash
python run_full_pipeline_niat.py --demo_mode --first_n 20
```

---

## File Structure

```
schedule_pipeline/
├── run_full_pipeline_wikitq.py   # WikiTQ main pipeline
├── run_full_pipeline_niat.py     # NIAT main pipeline
├── convert_niat_parallel.py      # NIAT preprocessing script
├── test_pipeline.sh              # WikiTQ test script
├── test_pipeline_niat.sh         # NIAT test script
├── test_niat_pipeline.py         # NIAT unit test
├── README.md                     # This file
└── tmp/                          # Output directory
    └── niat_test/                # NIAT test outputs
```

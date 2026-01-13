# Schedule Pipeline

This directory contains the main pipeline scripts for running H-STAR on different table QA datasets.

## Supported Datasets

| Dataset | Script | Description |
|---------|--------|-------------|
| WikiTQ | `run_full_pipeline_wikitq.py` | Wikipedia Table Questions |
| NIAT | `run_full_pipeline_niat.py` | Nested/Hierarchical Table QA |

---

## 🚀 Quick Start: vLLM API-Based Full Dataset Run

The recommended way to run the full pipeline is using vLLM as an API server for maximum efficiency.

### Step 1: Start vLLM Server

```bash
# Terminal 1: Start vLLM OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
  --model /data/workspace/yanmy/models/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 23000 \
  --gpu-memory-utilization 0.85 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --port 8000

# Or for Qwen3-4B with single GPU
python -m vllm.entrypoints.openai.api_server \
  --model /data/workspace/yanmy/models/Qwen3-4B-Instruct-2507 \
  --max-model-len 16000 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

### Step 2: Run Full Pipeline

#### WikiTQ Full Dataset

```bash
# Terminal 2: Run WikiTQ pipeline (full dataset)
cd schedule_pipeline

python run_full_pipeline_wikitq.py \
  --llm_path /data/workspace/yanmy/models/Qwen2.5-7B-Instruct \
  --dataset_name wikitq \
  --split test \
  --tmp_save_path tmp/wikitq_full \
  --llm_concurrency 32 \
  --tensor_parallel_size 2
```

#### NIAT Full Dataset

```bash
# Terminal 2: Run NIAT pipeline (full dataset)
cd schedule_pipeline

python run_full_pipeline_niat.py \
  --llm_path /data/workspace/yanmy/models/Qwen2.5-7B-Instruct \
  --niat_json_path ../datasets/NIAT/sampled_qa_pairs_4000_fixed.json \
  --tmp_save_path tmp/niat_full \
  --llm_concurrency 32 \
  --tensor_parallel_size 2
```

### Key Parameters for Full Dataset

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `--llm_concurrency` | `32-64` | Concurrent API requests (higher = faster) |
| `--tensor_parallel_size` | `2` | Multi-GPU for vLLM server |
| `--first_n` | `-1` | `-1` = all data, or set number for subset |
| `--temperature` | `0.7` | Sampling temperature |
| `--top_p` | `0.8` | Nucleus sampling |

---

## Pipeline Steps (13 Steps)

Both WikiTQ and NIAT pipelines follow the same 13-step structure:

| Step | Description | Notes |
|------|-------------|-------|
| 1 | Data Preprocessing | NIAT: table flattening |
| 2 | Build Router Query File | Query + table pairs |
| 3 | Construct Database | SQLite-compatible DB |
| 4 | Router Model Inference | Route to methods |
| 5 | Parse Router Results | Organize LLM queries |
| 6 | Execute RAG Task | Hybrid retrieval |
| 7 | Select_Row/Column LLM | Table filtering |
| 8 | Execute_SQL LLM | SQL generation |
| 9 | SQL Parse and Execute | Run generated SQL |
| 10 | Check Model Iteration | Quality check loop |
| 11 | Add Missing SQL | Gap filling |
| 12 | Final QA Prompts | Build final prompts |
| 13 | Final QA & Evaluate | Answer + accuracy |

---

## NIAT Pipeline

The NIAT pipeline handles tables with complex nested/hierarchical structures.

### Key Features

1. **Table Flattening**: Automatically forward-fills hierarchical empty cells
2. **Duplicate Column Handling**: Renames duplicate columns with suffixes
3. **NIAT-Specific Evaluation**: Uses exact match after normalization

### Demo Mode (No LLM)

```bash
cd schedule_pipeline
bash test_pipeline_niat.sh
```

### API Mode with Specific Model

```bash
python run_full_pipeline_niat.py \
  --llm_path /data/workspace/yanmy/models/Qwen3-4B-Instruct-2507 \
  --niat_json_path ../datasets/NIAT/sampled_qa_pairs_4000_fixed.json \
  --tmp_save_path tmp/niat_api \
  --llm_concurrency 16 \
  --first_n 100
```

### Command Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--niat_json_path` | `datasets/NIAT/...` | Path to NIAT JSON file |
| `--tmp_save_path` | `datasets/schedule_test/niat` | Output directory |
| `--first_n` | `-1` | Process first N samples (-1 for all) |
| `--skip_router` | `False` | Skip router inference step |
| `--skip_rag` | `False` | Skip RAG retrieval step |
| `--tensor_parallel_size` | `2` | GPU parallelism for vLLM |
| `--llm_concurrency` | `32` | Concurrent API requests |
| `--tau` | `0.82` | Router threshold |
| `--check_tau` | `0.8` | Check model threshold |

### Output Files

| File | Description |
|------|-------------|
| `niat_df_processed.npy` | Processed DataFrames |
| `router_query.pkl` | Router query data |
| `inference_result.pkl` | Router inference results |
| `processed_table.npy` | All processed tables |
| `Check_Model_Data_Sequence.npy` | Check model results |
| `final_results.csv` | Final predictions |
| `evaluation_results.json` | Accuracy metrics |
| `timing_summary_*.json` | Timing breakdown |

---

## WikiTQ Pipeline

```bash
# Quick test
bash test_pipeline.sh

# Full run
python run_full_pipeline_wikitq.py \
  --llm_path /data/workspace/yanmy/models/Qwen2.5-7B-Instruct \
  --dataset_name wikitq \
  --split test \
  --llm_concurrency 32
```

---

## Data Preprocessing

### NIAT Table Flattening

**Original (nested):**
```
| Group    | Union | Craft   | Employees |
|----------|-------|---------|-----------|
| Mainline | APA   | Pilots  | 13200     |
|          | APFA  | FA      | 24900     |  <- Group empty, inherits "Mainline"
```

**Flattened:**
```
| Group    | Union | Craft   | Employees |
|----------|-------|---------|-----------|
| Mainline | APA   | Pilots  | 13200     |
| Mainline | APFA  | FA      | 24900     |  <- Group filled
```

---

## Troubleshooting

### vLLM Server Connection

If API calls fail, check:
1. vLLM server is running on port 8000
2. URL matches: `http://localhost:8000/v1`

### Out of Memory

Reduce `--llm_concurrency` or `--gpu_memory_utilization`:
```bash
python run_full_pipeline_niat.py --llm_concurrency 8 --gpu_memory_utilization 0.7
```

### Router/Check Model Not Found

Skip these steps while testing:
```bash
python run_full_pipeline_niat.py --skip_router --first_n 50
```

---

## File Structure

```
schedule_pipeline/
├── run_full_pipeline_wikitq.py   # WikiTQ 13-step pipeline
├── run_full_pipeline_niat.py     # NIAT 13-step pipeline
├── inference_router.py           # Router model inference
├── Hybrid_Retrieve_Update_dict.py # RAG retrieval
├── test_pipeline.sh              # WikiTQ test script
├── test_pipeline_niat.sh         # NIAT test script
├── README.md                     # This file
└── tmp/                          # Output directory
```


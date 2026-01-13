#!/bin/bash
# Test script for NIAT pipeline (demo mode - no LLM required)

# Set Python path
export PYTHONPATH="$(pwd)/..:$(pwd):$PYTHONPATH"

# Create output directory
TMP_SAVE_PATH="tmp/niat_test"
mkdir -p ${TMP_SAVE_PATH}

echo "=============================================="
echo "NIAT Pipeline Test (Demo Mode)"
echo "=============================================="

# Run NIAT pipeline
python run_full_pipeline_niat.py \
  --dataset_name niat \
  --tmp_save_path ${TMP_SAVE_PATH} \
  --niat_json_path ../datasets/NIAT/sampled_qa_pairs_4000_fixed.json \
  --n_parallel 4 \
  --first_n 20 \
  --demo_mode \
  2>&1 | tee "${TMP_SAVE_PATH}/test_run.log"

echo ""
echo "=============================================="
echo "Pipeline test complete."
echo "Check ${TMP_SAVE_PATH}/test_run.log for results."
echo "=============================================="

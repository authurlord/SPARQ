#!/usr/bin/env python3
"""
NIAT End-to-End Pipeline Test

This script tests the complete workflow:
1. Convert NIAT nested tables to flattened format
2. Load into NeuralDB
3. Generate and execute sample SQL queries
4. Log all results

Usage:
    python test_niat_pipeline.py
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime

# Auto-install missing packages
def install_if_missing(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing missing package: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])

# Check required packages
install_if_missing("pandas")
install_if_missing("numpy")
install_if_missing("records")
install_if_missing("sqlalchemy", "sqlalchemy")

import numpy as np
import pandas as pd

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from utils.multi_db_v2 import NeuralDB, Executor

# Setup logging
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"niat_pipeline_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_select_column_sql(df: pd.DataFrame, table_name: str = "w") -> list:
    """Generate SELECT column SQL queries based on table structure."""
    queries = []
    columns = df.columns.tolist()
    
    # Query 1: Select single column
    if len(columns) >= 1:
        col = columns[0]
        queries.append({
            "type": "SELECT_COLUMN",
            "sql": f'SELECT `{col}` FROM {table_name}',
            "description": f"Select column '{col}'"
        })
    
    # Query 2: Select multiple columns
    if len(columns) >= 2:
        cols = ", ".join([f'`{c}`' for c in columns[:3]])
        queries.append({
            "type": "SELECT_COLUMN",
            "sql": f'SELECT {cols} FROM {table_name}',
            "description": f"Select first 3 columns"
        })
    
    return queries


def generate_select_row_sql(df: pd.DataFrame, table_name: str = "w") -> list:
    """Generate SELECT row (WHERE clause) SQL queries."""
    queries = []
    columns = df.columns.tolist()
    
    # Find a column with non-empty values for filtering
    for col in columns:
        non_empty_values = df[col].dropna()
        non_empty_values = non_empty_values[non_empty_values.astype(str).str.strip() != ""]
        if len(non_empty_values) > 0:
            # Get a sample value for WHERE clause
            sample_val = str(non_empty_values.iloc[0]).replace("'", "''")
            queries.append({
                "type": "SELECT_ROW",
                "sql": f"SELECT * FROM {table_name} WHERE `{col}` = '{sample_val}'",
                "description": f"Select rows where {col} = '{sample_val[:30]}...'"
            })
            break
    
    # LIMIT query
    queries.append({
        "type": "SELECT_ROW",
        "sql": f"SELECT * FROM {table_name} LIMIT 5",
        "description": "Select first 5 rows"
    })
    
    return queries


def generate_execute_sql(df: pd.DataFrame, table_name: str = "w", question: str = "") -> list:
    """Generate more complex SQL queries based on question."""
    queries = []
    columns = df.columns.tolist()
    
    # COUNT query
    queries.append({
        "type": "EXECUTE_SQL",
        "sql": f"SELECT COUNT(*) FROM {table_name}",
        "description": "Count all rows"
    })
    
    # Find numeric columns for aggregation
    for col in columns:
        try:
            # Check if column has numeric-like values
            sample_vals = df[col].dropna().head(5).astype(str)
            numeric_vals = sample_vals.str.replace(",", "").str.replace("$", "")
            if numeric_vals.str.match(r'^[\d.]+$').any():
                queries.append({
                    "type": "EXECUTE_SQL",
                    "sql": f"SELECT MAX(`{col}`) FROM {table_name}",
                    "description": f"Get MAX of {col}"
                })
                break
        except:
            continue
    
    # DISTINCT query
    if len(columns) >= 1:
        col = columns[0]
        queries.append({
            "type": "EXECUTE_SQL",
            "sql": f"SELECT DISTINCT `{col}` FROM {table_name}",
            "description": f"Get distinct values of {col}"
        })
    
    return queries


def test_sql_execution(executor: Executor, db: NeuralDB, table_id: int, 
                       queries: list, logger: logging.Logger) -> dict:
    """Execute SQL queries and collect results."""
    results = {
        "total": len(queries),
        "success": 0,
        "failed": 0,
        "details": []
    }
    
    for query in queries:
        sql = query["sql"]
        try:
            result = executor.sql_exec(sql, db, table_id=table_id, verbose=False)
            results["success"] += 1
            results["details"].append({
                "sql": sql,
                "type": query["type"],
                "status": "SUCCESS",
                "header": result.get("header", []),
                "rows_count": len(result.get("rows", [])),
                "sample_rows": result.get("rows", [])[:2]
            })
            logger.info(f"  ✅ {query['type']}: {query['description']}")
            logger.info(f"     SQL: {sql[:80]}...")
            logger.info(f"     Result: {len(result.get('rows', []))} rows")
        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "sql": sql,
                "type": query["type"],
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"  ❌ {query['type']}: {query['description']}")
            logger.error(f"     SQL: {sql[:80]}...")
            logger.error(f"     Error: {str(e)[:100]}")
    
    return results


def main():
    logger.info("=" * 70)
    logger.info("NIAT End-to-End Pipeline Test")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_file}")
    
    # Configuration
    NIAT_JSON = os.path.join(ROOT_DIR, "datasets/NIAT/sampled_qa_pairs_4000_fixed.json")
    OUTPUT_NPY = os.path.join(ROOT_DIR, "datasets/schedule_test/niat/niat_df_processed.npy")
    NUM_SAMPLES = 15
    
    # Step 1: Check if we need to run preprocessing
    logger.info("\n[Step 1] Checking preprocessed data...")
    
    if not os.path.exists(OUTPUT_NPY):
        logger.info(f"  Preprocessed file not found, running conversion...")
        convert_script = os.path.join(SCRIPT_DIR, "convert_niat_parallel.py")
        cmd = [
            sys.executable, convert_script,
            "--input_path", NIAT_JSON,
            "--output_path", OUTPUT_NPY,
            "--num_workers", "1",
            "--first_n", str(NUM_SAMPLES)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            return
        logger.info("  Conversion complete")
    else:
        logger.info(f"  Found existing preprocessed file: {OUTPUT_NPY}")
    
    # Step 2: Load processed data
    logger.info("\n[Step 2] Loading preprocessed NIAT data...")
    niat_processed = np.load(OUTPUT_NPY, allow_pickle=True).item()
    logger.info(f"  Loaded {len(niat_processed)} tables")
    
    # Load metadata if available
    metadata_path = OUTPUT_NPY.replace(".npy", "_metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # Step 3: Initialize NeuralDB
    logger.info("\n[Step 3] Initializing NeuralDB...")
    
    # Limit to first N samples for testing
    test_indices = list(niat_processed.keys())[:NUM_SAMPLES]
    tables_for_db = [niat_processed[i] for i in test_indices]
    table_titles = [f"NIAT_Table_{i}" for i in test_indices]
    
    try:
        db = NeuralDB(tables=tables_for_db, table_titles=table_titles)
        executor = Executor()
        logger.info(f"  NeuralDB initialized with {len(tables_for_db)} tables")
    except Exception as e:
        logger.error(f"  Failed to initialize NeuralDB: {e}")
        return
    
    # Step 4: Test SQL execution on each table
    logger.info("\n[Step 4] Testing SQL execution on each table...")
    
    all_results = {
        "total_tables": len(test_indices),
        "total_queries": 0,
        "total_success": 0,
        "total_failed": 0,
        "per_table": []
    }
    
    for local_idx, global_idx in enumerate(test_indices):
        df = niat_processed[global_idx]
        meta = metadata.get(str(global_idx), {})
        
        logger.info(f"\n--- Table {local_idx} (ID: {global_idx}) ---")
        logger.info(f"  Structure: {meta.get('table_structure', 'unknown')}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Question: {meta.get('question', 'N/A')[:80]}...")
        logger.info(f"  Answer: {meta.get('answer', 'N/A')[:50]}")
        logger.info(f"  Columns: {list(df.columns)[:5]}...")
        
        # Generate SQL queries
        queries = []
        queries.extend(generate_select_column_sql(df, "w"))
        queries.extend(generate_select_row_sql(df, "w"))
        queries.extend(generate_execute_sql(df, "w", meta.get('question', '')))
        
        logger.info(f"  Generated {len(queries)} test queries:")
        
        # Execute queries
        results = test_sql_execution(executor, db, local_idx, queries, logger)
        
        all_results["total_queries"] += results["total"]
        all_results["total_success"] += results["success"]
        all_results["total_failed"] += results["failed"]
        all_results["per_table"].append({
            "table_id": global_idx,
            "structure": meta.get('table_structure', 'unknown'),
            "shape": list(df.shape),
            "results": results
        })
    
    # Step 5: Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total tables tested: {all_results['total_tables']}")
    logger.info(f"Total queries executed: {all_results['total_queries']}")
    logger.info(f"Successful: {all_results['total_success']} ({100*all_results['total_success']/max(1,all_results['total_queries']):.1f}%)")
    logger.info(f"Failed: {all_results['total_failed']} ({100*all_results['total_failed']/max(1,all_results['total_queries']):.1f}%)")
    
    # Success by query type
    type_stats = {}
    for table_result in all_results["per_table"]:
        for detail in table_result["results"]["details"]:
            qtype = detail["type"]
            if qtype not in type_stats:
                type_stats[qtype] = {"success": 0, "failed": 0}
            if detail["status"] == "SUCCESS":
                type_stats[qtype]["success"] += 1
            else:
                type_stats[qtype]["failed"] += 1
    
    logger.info("\nBy Query Type:")
    for qtype, stats in type_stats.items():
        total = stats["success"] + stats["failed"]
        success_rate = 100 * stats["success"] / max(1, total)
        logger.info(f"  {qtype}: {stats['success']}/{total} ({success_rate:.1f}%)")
    
    # Save detailed results
    results_path = os.path.join(LOG_DIR, f"niat_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        # Convert non-serializable items
        def make_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=make_serializable)
    logger.info(f"\nDetailed results saved to: {results_path}")
    
    # Cleanup
    del db
    
    logger.info("\n" + "=" * 70)
    logger.info("Test Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

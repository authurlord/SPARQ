#!/usr/bin/env python3
"""
NIAT Dataset Preprocessing Script

This script processes the NIAT dataset containing nested/hierarchical tables.
It flattens the nested structure by forward-filling empty cells that represent
hierarchical relationships, making the tables compatible with SQL execution.

Output: A dictionary {index: pd.DataFrame} saved as .npy file, similar to
the output of convert_df_type_parallel.py
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Optional

# Import normalizer for additional type conversion
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.normalizer import convert_df_type


def flatten_hierarchical_table(table_rows: List[List[str]], 
                                table_structure: str = "vertical",
                                forward_fill_cols: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Flatten a hierarchical/nested table by forward-filling empty cells.
    
    In hierarchical tables, empty cells often indicate that the value should be
    inherited from the cell above (in the same column). This function:
    1. Converts table_rows to a DataFrame
    2. Identifies columns that have hierarchical empty-cell patterns
    3. Forward-fills empty values to make relationships explicit
    
    Args:
        table_rows: List of lists representing table rows (first row may be header)
        table_structure: One of 'vertical', 'horizontal', 'hierarchical'
        forward_fill_cols: Optional list of column indices to forward-fill.
                          If None, auto-detect columns that benefit from forward-fill.
    
    Returns:
        Flattened pandas DataFrame with explicit values
    """
    if not table_rows or len(table_rows) < 2:
        # Return empty or single-row DataFrame as-is
        if len(table_rows) == 1:
            return pd.DataFrame(columns=table_rows[0])
        return pd.DataFrame()
    
    # First row is typically the header
    header = table_rows[0]
    data_rows = table_rows[1:]
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=header)
    
    # For hierarchical tables, auto-detect and forward-fill left-most grouping columns
    if table_structure == "hierarchical" or forward_fill_cols is not None:
        if forward_fill_cols is None:
            # Auto-detect: typically the first 1-2 columns have hierarchical structure
            # A column is hierarchical if it has significant empty strings that follow non-empty values
            forward_fill_cols = []
            for col_idx, col in enumerate(df.columns[:3]):  # Check first 3 columns max
                col_values = df[col].values
                # Count empty-after-non-empty patterns
                prev_non_empty = False
                empty_after_non_empty = 0
                for val in col_values:
                    # Safely convert to string and strip
                    val_str = str(val) if not isinstance(val, (list, np.ndarray)) else ""
                    val_str = val_str.strip() if isinstance(val_str, str) else ""
                    is_empty = (val_str == "" or val_str == " ")
                    if is_empty and prev_non_empty:
                        empty_after_non_empty += 1
                    if not is_empty:
                        prev_non_empty = True
                
                # If >20% of rows have hierarchical pattern, mark for forward-fill
                if empty_after_non_empty > len(col_values) * 0.2:
                    forward_fill_cols.append(col_idx)
        
        # Apply forward-fill to detected columns
        for col_idx in forward_fill_cols:
            if col_idx < len(df.columns):
                col_name = df.columns[col_idx]
                # Replace empty strings with NaN, then forward-fill
                df[col_name] = df[col_name].replace(r'^\s*$', np.nan, regex=True)
                df[col_name] = df[col_name].ffill()
                # Replace NaN back to empty string if still present
                df[col_name] = df[col_name].fillna('')
    
    return df


def process_niat_item(item: Dict[str, Any]) -> pd.DataFrame:
    """
    Process a single NIAT dataset item and return a flattened DataFrame.
    
    Args:
        item: A dictionary containing table_rows, table_structure, etc.
        
    Returns:
        Processed and flattened pandas DataFrame
    """
    table_rows = item.get('table_rows', [])
    table_structure = item.get('table_structure', 'vertical')
    
    # Step 1: Flatten the hierarchical structure
    df = flatten_hierarchical_table(table_rows, table_structure)
    
    # Step 2: Apply type conversion (same as original pipeline)
    if not df.empty:
        try:
            df = convert_df_type(df)
        except Exception as e:
            # If type conversion fails, return the raw flattened df
            print(f"Warning: convert_df_type failed: {e}")
            pass
    
    return df


# Global variable for multiprocessing
worker_data = None

def init_worker(data_to_share: List[Dict]):
    """Initialize worker with shared data."""
    global worker_data
    worker_data = data_to_share


def process_item_at_index(index: int) -> tuple:
    """
    Process a single item at given index.
    
    Args:
        index: Index into the worker_data list
        
    Returns:
        Tuple of (index, processed_df, metadata)
    """
    global worker_data
    item = worker_data[index]
    
    try:
        processed_df = process_niat_item(item)
        metadata = {
            'table_id': item.get('table_id', f'unknown_{index}'),
            'table_structure': item.get('table_structure', 'unknown'),
            'question': item.get('question', ''),
            'answer': item.get('answer', ''),
            'original_rows': len(item.get('table_rows', [])),
            'processed_rows': len(processed_df),
        }
        return (index, processed_df, metadata)
    except Exception as e:
        print(f"Error processing index {index}: {e}")
        # Return empty DataFrame on error
        return (index, pd.DataFrame(), {'error': str(e)})


def main():
    parser = argparse.ArgumentParser(
        description="Process NIAT dataset with nested table flattening"
    )
    parser.add_argument(
        '--input_path', type=str, required=True,
        help="Path to NIAT JSON file (e.g., datasets/NIAT/sampled_qa_pairs_4000.json)"
    )
    parser.add_argument(
        '--output_path', type=str, required=True,
        help="Path to save output .npy file"
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        '--first_n', type=int, default=-1,
        help="Process only first N items for testing (-1 for all)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NIAT Dataset Preprocessing with Table Flattening")
    print("=" * 60)
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Workers: {args.num_workers}")
    
    # Load NIAT dataset
    print("\n[Step 1] Loading NIAT dataset...")
    with open(args.input_path, 'r', encoding='utf-8') as f:
        niat_data = json.load(f)
    
    print(f"  Loaded {len(niat_data)} items")
    
    # Analyze table structures
    structure_counts = {}
    for item in niat_data:
        struct = item.get('table_structure', 'unknown')
        structure_counts[struct] = structure_counts.get(struct, 0) + 1
    print(f"  Table structure distribution: {structure_counts}")
    
    # Optionally limit to first N items
    if args.first_n > 0:
        niat_data = niat_data[:args.first_n]
        print(f"  Processing only first {args.first_n} items")
    
    # Process tables in parallel
    print(f"\n[Step 2] Processing {len(niat_data)} tables...")
    
    processed_tables = {}
    metadata_dict = {}
    
    num_workers = min(args.num_workers, cpu_count())
    
    if num_workers <= 1:
        # Sequential processing
        for idx in tqdm(range(len(niat_data)), desc="Processing tables"):
            item = niat_data[idx]
            df = process_niat_item(item)
            processed_tables[idx] = df
            metadata_dict[idx] = {
                'table_id': item.get('table_id', f'unknown_{idx}'),
                'table_structure': item.get('table_structure', 'unknown'),
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
            }
    else:
        # Parallel processing
        with Pool(processes=num_workers, initializer=init_worker, initargs=(niat_data,)) as pool:
            results_iter = pool.imap_unordered(
                process_item_at_index, 
                range(len(niat_data))
            )
            
            for idx, df, metadata in tqdm(results_iter, total=len(niat_data), desc="Processing tables"):
                processed_tables[idx] = df
                metadata_dict[idx] = metadata
    
    print(f"  Processed {len(processed_tables)} tables")
    
    # Save results
    print(f"\n[Step 3] Saving results to {args.output_path}...")
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as .npy (compatible with existing pipeline)
    np.save(args.output_path, processed_tables)
    print(f"  Saved processed tables to {args.output_path}")
    
    # Also save metadata as JSON
    base, ext = os.path.splitext(args.output_path)
    metadata_path = base + "_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
    print(f"  Saved metadata to {metadata_path}")
    
    # Save human-readable JSON of processed tables
    try:
        json_serializable = {
            str(idx): {
                "header": df.columns.tolist(),
                "rows": df.values.tolist(),
                "shape": list(df.shape),
            }
            for idx, df in processed_tables.items()
        }
        json_path = base + ".json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_serializable, f, ensure_ascii=False)
        print(f"  Saved JSON version to {json_path}")
    except Exception as e:
        print(f"  Warning: Failed to save JSON: {e}")
    
    # Print summary statistics
    print("\n[Summary]")
    total_rows = sum(len(df) for df in processed_tables.values())
    non_empty = sum(1 for df in processed_tables.values() if not df.empty)
    print(f"  Total processed tables: {len(processed_tables)}")
    print(f"  Non-empty tables: {non_empty}")
    print(f"  Total rows across all tables: {total_rows}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

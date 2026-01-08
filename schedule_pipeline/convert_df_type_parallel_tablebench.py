import os
import argparse
import time
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# Import normalizer utility
from utils.normalizer import convert_df_type

# --- Worker Initialization and Function ---
# This global variable will be populated in each worker process.
worker_dataset = None

def load_tablebench_jsonl(file_path: str) -> list:
    """
    Load TableBench.jsonl file and return a list of data items.
    
    Args:
        file_path: Path to the TableBench.jsonl file.
        
    Returns:
        A list of dictionaries, each containing table data.
    """
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                dataset.append(item)
    return dataset

def init_worker(dataset_to_share):
    """
    Initializer function for each worker process in the pool.
    This sets up a global dataset variable for the worker to access,
    avoiding the need to pass the large dataset object repeatedly.
    """
    global worker_dataset
    worker_dataset = dataset_to_share

def process_table_at_index(index: int) -> tuple[int, pd.DataFrame]:
    """
    Processes a single table from the worker's global 'worker_dataset' at a given index.
    
    TableBench format uses:
    - table['columns'] for column headers
    - table['data'] for row data
    
    Args:
        index: The index of the item in the dataset to process.
        
    Returns:
        A tuple containing the original index and the processed pandas DataFrame.
    """
    # Access the dataset that was set up during worker initialization
    # TableBench uses 'columns' instead of 'header', and 'data' instead of 'rows'
    original_df = pd.DataFrame(
        worker_dataset[index]['table']['data'],
        columns=worker_dataset[index]['table']['columns']
    )
    
    # This is the core, time-consuming operation
    processed_df = convert_df_type(original_df)
    
    return (index, processed_df)

def main():
    """Main function to run the parallel data processing for TableBench."""
    parser = argparse.ArgumentParser(
        description="Run parallel data processing to convert tables in TableBench dataset."
    )
    parser.add_argument('--input_path', type=str, required=True, 
                        help="Path to the TableBench.jsonl file.")
    parser.add_argument('--output_path', type=str, required=True, 
                        help="Path to save the output .npy file.")
    parser.add_argument('--num_workers', type=int, required=True, 
                        help="Number of worker processes.")
    
    args = parser.parse_args()

    print("--- Starting Parallel Data Processing Script for TableBench ---")
    print(f"Input Path: {args.input_path}")
    print(f"Output Path: {args.output_path}")

    # Load the TableBench dataset from JSONL file
    print("\nLoading TableBench dataset...")
    try:
        dataset = load_tablebench_jsonl(args.input_path)
        if not dataset:
            print("Warning: Loaded dataset is empty. Please check the input file.")
            return
        print(f"Dataset loaded with {len(dataset)} items.")
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return

    # Use a sensible number of processes
    num_processes = args.num_workers if args.num_workers > 0 else cpu_count()
    print(f"\nStarting parallel processing on {len(dataset)} items with {num_processes} workers...")

    # The final dictionary to hold results
    processed_tables_dict = {}

    # Create a pool of worker processes.
    # The `initializer` and `initargs` pass the dataset to each worker once upon creation.
    with Pool(processes=num_processes, initializer=init_worker, initargs=(dataset,)) as pool:
        # pool.imap_unordered is memory-efficient and ideal for progress bars
        results_iterator = pool.imap_unordered(process_table_at_index, range(len(dataset)))
        
        # Collect the results as they are completed
        for index, processed_df in tqdm(results_iterator, total=len(dataset), desc="Processing tables"):
            processed_tables_dict[index] = processed_df

    print("\nParallel processing finished.")
    
    # Save the final dictionary to the specified .npy file
    print(f"Saving results to {args.output_path}...")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    np.save(args.output_path, processed_tables_dict)
    
    # Also save a human-readable JSON alongside the NPY
    try:
        # Convert DataFrames to a JSON-serializable structure
        json_serializable = {
            str(idx): {
                "header": df.columns.tolist(),
                "rows": df.values.tolist(),
            }
            for idx, df in processed_tables_dict.items()
        }
        base, ext = os.path.splitext(args.output_path)
        json_path = base + ".json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_serializable, jf, ensure_ascii=False)
        print(f"JSON file saved to {json_path}")
    except Exception as e:
        print(f"Warning: failed to save JSON alongside NPY: {e}")

    print("File saved successfully.")
    print("--- Script Finished ---")


if __name__ == "__main__":
    main()

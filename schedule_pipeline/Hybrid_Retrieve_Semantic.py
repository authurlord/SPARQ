import os
import argparse
import time
import pandas as pd
import numpy as np
import faiss
from collections import defaultdict
import torch
from FlagEmbedding import FlagModel
from utils.schedule_utils import load_data_split

# Assuming your utility functions are in a 'utils' directory
# You may need to adjust the import paths based on your project structure
# from utils.utils import load_data_split


class SemanticTableRetriever:
    """
    Implements a dense-only (semantic) retrieval system to find the most relevant sub-table.
    This version ensures that the column and row order of the output sub-table matches the
    original table's order and fixes Faiss dtype requirements.
    """

    def __init__(self, embedding_model, view_samples: int = 10):
        """
        Initializes the SemanticTableRetriever.

        Args:
            embedding_model: An instance of a sentence-transformer like model (e.g., FlagModel).
            view_samples (int): The number of samples to use when generating views for columns.
        """
        self.model = embedding_model
        self.view_samples = view_samples
        if torch.cuda.is_available():
            print(f"SemanticTableRetriever initialized on {torch.cuda.get_device_name(0)}")
        else:
            print(f"SemanticTableRetriever initialized on CPU")


    def _generate_column_views(self, df: pd.DataFrame, table_title: str) -> dict:
        """Generates multiple text views for each column based on its data type."""
        column_views = defaultdict(list)
        for col_name in df.columns:
            col = df[col_name].dropna()
            if col.empty: continue
            column_views[col_name].append(f"Column named '{col_name}' in the table '{table_title}'.")
            if pd.api.types.is_numeric_dtype(col):
                stats = f"Numerical column with values from {col.min()} to {col.max()}."
                column_views[col_name].append(f"'{col_name}': {stats}")
                samples = col.sample(min(len(col), self.view_samples)).tolist()
                column_views[col_name].append(f"'{col_name}' has sample values like {samples}.")
            else:
                top_values = col.value_counts().nlargest(self.view_samples).index.tolist()
                column_views[col_name].append(f"'{col_name}' has common values like: {top_values}.")
                samples = col.sample(min(len(col), self.view_samples)).tolist()
                column_views[col_name].append(f"'{col_name}' contains examples like: {samples}.")
        return dict(column_views)

    def _generate_row_views(self, df: pd.DataFrame, table_title: str, columns_to_use: list) -> list:
        """Generates a text view for each row, using only the specified columns."""
        row_views = []
        for index, row in df.iterrows():
            row_parts = [f"{col} is {row[col]}" for col in columns_to_use if col in row]
            row_values_str = "; ".join(row_parts)
            final_text = f"From table '{table_title}', row {index}: {row_values_str}."
            row_views.append(final_text)
        return row_views

    def retrieve(self, 
                 rewrite_queries: list[str], 
                 tables: list[pd.DataFrame], 
                 table_titles: list[str], 
                 max_rows_m: int, 
                 max_cols_n: int) -> list[pd.DataFrame]:
        if not (len(rewrite_queries) == len(tables) == len(table_titles)):
            raise ValueError("Input lists must have the same length.")

        num_tasks = len(rewrite_queries)
        
        selected_columns_per_table = []
        for i in range(num_tasks):
            table = tables[i]
            if table.shape[1] <= max_cols_n:
                selected_columns_per_table.append(table.columns.tolist())
                continue

            col_views_dict = self._generate_column_views(table, table_titles[i])
            col_views_flat = [view for views in col_views_dict.values() for view in views]
            flat_idx_to_col_name = [name for name, views in col_views_dict.items() for _ in views]
            
            # --- DENSE RETRIEVAL FOR COLUMNS ---
            col_view_embeddings = self.model.encode(col_views_flat, batch_size=256)
            query_embedding = self.model.encode([rewrite_queries[i]])
            
            col_view_embeddings = col_view_embeddings.astype(np.float32)
            query_embedding = query_embedding.astype(np.float32)

            dim = query_embedding.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(col_view_embeddings)
            index.add(col_view_embeddings)
            faiss.normalize_L2(query_embedding)
            
            scores, indices = index.search(query_embedding, k=len(col_views_flat))
            
            dense_col_scores = defaultdict(list)
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    dense_col_scores[flat_idx_to_col_name[idx]].append(score)
            dense_col_scores_avg = {k: np.mean(v) for k, v in dense_col_scores.items()}

            # --- SELECTION BASED ON DENSE SCORES ONLY ---
            sorted_cols_by_relevance = sorted(dense_col_scores_avg.items(), key=lambda x: x[1], reverse=True)
            top_n_cols_by_relevance = {col for col, score in sorted_cols_by_relevance[:max_cols_n]}

            original_cols = table.columns.tolist()
            ordered_selected_cols = [col for col in original_cols if col in top_n_cols_by_relevance]
            selected_columns_per_table.append(ordered_selected_cols)

        final_subtables = []
        for i in range(num_tasks):
            table = tables[i]
            selected_cols = selected_columns_per_table[i]

            if table.shape[0] <= max_rows_m:
                final_subtables.append(table[selected_cols])
                continue

            row_views = self._generate_row_views(table, table_titles[i], selected_cols)
            
            # --- DENSE RETRIEVAL FOR ROWS ---
            row_embeddings = self.model.encode(row_views, batch_size=256)
            query_embedding = self.model.encode([rewrite_queries[i]])
            
            row_embeddings = row_embeddings.astype(np.float32)
            query_embedding = query_embedding.astype(np.float32)

            dim = query_embedding.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(row_embeddings)
            index.add(row_embeddings)
            faiss.normalize_L2(query_embedding)
            scores, indices = index.search(query_embedding, k=table.shape[0])
            dense_row_scores = {idx: score for score, idx in zip(scores[0], indices[0]) if idx != -1}

            # --- SELECTION BASED ON DENSE SCORES ONLY ---
            sorted_rows_by_relevance = sorted(dense_row_scores.items(), key=lambda x: x[1], reverse=True)
            
            top_m_view_indices = [row_idx for row_idx, score in sorted_rows_by_relevance[:max_rows_m]]
            
            original_indices = table.index.tolist()
            top_m_original_indices_by_relevance = [original_indices[view_idx] for view_idx in top_m_view_indices]
            
            top_m_original_indices_by_relevance.sort()
            ordered_selected_indices = top_m_original_indices_by_relevance
            
            final_subtables.append(table.loc[ordered_selected_indices, selected_cols])

        return final_subtables


def main():
    parser = argparse.ArgumentParser(description="Run Semantic Table Retriever on a dataset.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the embedding model (e.g., bge-m3).")
    parser.add_argument('--dataset_name', type=str, default='tab_fact', help="Name of the dataset to process.")
    parser.add_argument('--split', type=str, default='validation', help="Dataset split to use (e.g., 'validation', 'test').")
    parser.add_argument('--index_path', type=str, default=None, help="Optional path to a .npy file with a list of indices to process.")
    parser.add_argument('--processed_df_path', type=str, default=None, help="Optional path to a .npy file with a list of process dataframes.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output .npy file of retrieved tables.")
    parser.add_argument('--max_rows', type=int, default=20, help="Maximum number of rows to retrieve.")
    parser.add_argument('--max_cols', type=int, default=5, help="Maximum number of columns to retrieve.")
    
    args = parser.parse_args()

    # --- Configuration ---
    # Set the specific GPU to be used. This is the most reliable way.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    print("--- Starting Semantic Retrieval Script ---")
    print(f"Model Path: {args.model_path}")
    print(f"Dataset: {args.dataset_name}, Split: {args.split}")
    print(f"Max Rows: {args.max_rows}, Max Cols: {args.max_cols}")
    print(f"Output Path: {args.output_path}")

    # 1. Initialize Model
    print(f"\nLoading embedding model from {args.model_path}...")
    try:
        bge_model = FlagModel(args.model_path, use_fp16=True)
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        exit()

    # 2. Initialize Retriever
    retriever = SemanticTableRetriever(embedding_model=bge_model)

    # 3. Load Data
    print("\nLoading dataset...")
    dataset = load_data_split(args.dataset_name, args.split)
    
    # Construct path for pre-processed tables and load them
    processed_df_path = args.processed_df_path
    print(f"Loading pre-processed tables from: {processed_df_path}")
    try:
        wikitq_df_processed = np.load(processed_df_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Pre-processed table file not found at {processed_df_path}")
        print("Please ensure the pre-processed tables exist before running.")
        exit()

    # 4. Determine which indices to process
    if args.index_path:
        print(f"Loading specific indices from: {args.index_path}")
        try:
            indices_to_process = np.load(args.index_path)
            # Ensure it's a list of integers
            indices_to_process = [int(i) for i in indices_to_process]
            print(f"Found {len(indices_to_process)} indices to process.")
        except FileNotFoundError:
            print(f"Error: Index file not found at {args.index_path}")
            exit()
    else:
        print("No index file provided. Processing all items in the dataset split.")
        indices_to_process = list(range(len(dataset)))

    # 5. Prepare Batch Data based on selected indices
    print("Preparing batch data for retrieval...")
    batch_queries = [dataset[i]['question'] for i in indices_to_process]
    batch_tables = [wikitq_df_processed[i] for i in indices_to_process]
    batch_titles = [dataset[i]['table']['page_title'] for i in indices_to_process]

    # 6. Run Batch Retrieval
    print(f"\nRunning BATCH SEMANTIC retrieval on {len(batch_queries)} items...")
    start_time = time.time()
    subtables = retriever.retrieve(
        rewrite_queries=batch_queries,
        tables=batch_tables,
        table_titles=batch_titles,
        max_rows_m=args.max_rows,
        max_cols_n=args.max_cols
    )
    end_time = time.time()
    print(f"Retrieval completed in {end_time - start_time:.2f} seconds.")

    # 7. Save Results
    # The keys in the dictionary will be the ORIGINAL dataset indices
    retrieved_tables = {}
    for i, sub_df in enumerate(subtables):
        original_index = indices_to_process[i]
        retrieved_tables[original_index] = sub_df

    print(f"\nSaving {len(retrieved_tables)} retrieved tables to {args.output_path}")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    np.save(args.output_path, retrieved_tables)
    print("--- Script Finished Successfully ---")


if __name__ == '__main__':
    main()

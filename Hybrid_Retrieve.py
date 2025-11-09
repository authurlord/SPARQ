import os
import pandas as pd
import numpy as np
import faiss
import time
from collections import defaultdict
import torch
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagModel
import pandas as pd
import numpy as np
# from utils.prompt_generate import build_wikitq_prompt_from_df,evaluate_predictions,filter_dataframe_from_responses,fix_sql_query,match_subtables
import json
# from utils.evaluator import Evaluator
import numpy as np
import pandas as pd
import re
from nsql.database import NeuralDB
import regex as re
import copy
from nsql.sql_exec import Executor,extract_rows
from utils.normalizer import post_process_sql
from utils.utils import load_data_split
import os
## normalized
from utils.normalizer import convert_df_type
# from utils.optimize_normalizer import convert_df_type
# --- Configuration ---
# Set the specific GPU to be used. This is the most reliable way.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class HybridTableRetriever:
    """
    Implements a hybrid (sparse + dense) retrieval system to find the most relevant sub-table.
    This version ensures that the column and row order of the output sub-table matches the
    original table's order and fixes Faiss dtype requirements.
    """

    def __init__(self, embedding_model, alpha: float = 0.5, view_samples: int = 10):
        """
        Initializes the HybridTableRetriever.

        Args:
            embedding_model: An instance of a sentence-transformer like model (e.g., FlagModel).
            alpha (float): The weight for the dense retriever's score (0 to 1).
            view_samples (int): The number of samples to use when generating views for columns.
        """
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be between 0 and 1.")
        self.model = embedding_model
        self.alpha = alpha
        self.view_samples = view_samples
        if torch.cuda.is_available():
            print(f"HybridTableRetriever initialized on {torch.cuda.get_device_name(0)} with alpha={self.alpha}")
        else:
            print(f"HybridTableRetriever initialized on CPU with alpha={self.alpha}")


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

    def _normalize_scores(self, scores: dict) -> dict:
        """Normalizes scores to a [0, 1] range using Min-Max scaling."""
        if not scores or len(scores) == 1:
            return {k: 1.0 for k in scores}
        
        values = list(scores.values())
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return {k: 1.0 for k in scores}
        
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

    def _fuse_scores(self, dense_scores: dict, sparse_scores: dict) -> dict:
        """Fuses normalized dense and sparse scores using the alpha weight."""
        dense_norm = self._normalize_scores(dense_scores)
        sparse_norm = self._normalize_scores(sparse_scores)
        
        fused = defaultdict(float)
        all_keys = set(dense_norm.keys()) | set(sparse_norm.keys())

        for key in all_keys:
            dense_score = dense_norm.get(key, 0)
            sparse_score = sparse_norm.get(key, 0)
            fused[key] = (self.alpha * dense_score) + ((1 - self.alpha) * sparse_score)
            
        return dict(fused)

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
            
            tokenized_corpus = [doc.lower().split() for doc in col_views_flat]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = rewrite_queries[i].lower().split()
            sparse_scores_flat = bm25.get_scores(tokenized_query)
            
            sparse_col_scores = defaultdict(list)
            for idx, score in enumerate(sparse_scores_flat):
                sparse_col_scores[flat_idx_to_col_name[idx]].append(score)
            sparse_col_scores_avg = {k: np.mean(v) for k, v in sparse_col_scores.items()}
            
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

            fused_scores = self._fuse_scores(dense_col_scores_avg, sparse_col_scores_avg)
            sorted_cols_by_relevance = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
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
            
            tokenized_corpus = [doc.lower().split() for doc in row_views]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = rewrite_queries[i].lower().split()
            sparse_row_scores = {idx: score for idx, score in enumerate(bm25.get_scores(tokenized_query))}

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

            fused_scores = self._fuse_scores(dense_row_scores, sparse_row_scores)
            sorted_rows_by_relevance = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            
            top_m_view_indices = [row_idx for row_idx, score in sorted_rows_by_relevance[:max_rows_m]]
            
            original_indices = table.index.tolist()
            top_m_original_indices_by_relevance = [original_indices[view_idx] for view_idx in top_m_view_indices]
            
            top_m_original_indices_by_relevance.sort()
            ordered_selected_indices = top_m_original_indices_by_relevance
            
            final_subtables.append(table.loc[ordered_selected_indices, selected_cols])

        return final_subtables


if __name__ == '__main__':
    # === DEMO SCRIPT ===
    print("Loading embedding model from local path...")
    
    # 1. Initialize Model from the specified local path
    # model_path = '/home/wys/model/bge-large-en-1.5'
    # model_path = '/home/yanmy/sentence_transformers/bge-m3'
    model_path = '/data/workspace/yanmy/models/bge-m3'
    try:
        bge_model = FlagModel(model_path, use_fp16=True)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Please ensure the model exists at the specified path.")
        exit()
    
    # 2. Initialize Retriever
    retriever = HybridTableRetriever(embedding_model=bge_model, alpha=0.5)

    # 3. Prepare Batch Data (k=2)
    # Table 1: Elections
    # data1 = {'district_id': [f'VA-{i:02d}' for i in range(1, 11)],'incumbent_name': ['Thomas Newton', 'Arthur Smith', 'William Archer', 'Mark Alexander', 'John Randolph', 'George Tucker', 'Jabez Leftwich', 'Burwell Bassett', 'Andrew Stevenson', 'William Rives'],'party_affiliation': ['Adams-Clay', 'Crawford Rep', 'Crawford Rep', 'Crawford Rep', 'Crawford Rep', 'Jackson Rep', 'Crawford Rep', 'Jackson Rep', 'Crawford Rep', 'Adams-Clay'],'first_elected_year': [1801, 1821, 1820, 1819, 1799, 1819, 1821, 1805, 1821, 1823]}
    # table1 = pd.DataFrame(data1)
    # title1 = "1824 Virginia Congressional Elections"
    # query1 = "List incumbents from the Crawford Rep party, especially 'Arthur Smith'"

    # # Table 2: Movies
    # data2 = {'rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],'title': ['Avatar', 'Avengers: Endgame', 'Avatar: The Way of Water', 'Titanic', 'Star Wars: The Force Awakens', 'Avengers: Infinity War', 'Spider-Man: No Way Home', 'Jurassic World', 'The Lion King', 'The Avengers'],'worldwide_gross': [2.92, 2.79, 2.32, 2.25, 2.06, 2.04, 1.92, 1.67, 1.66, 1.52],'year': [2009, 2019, 2022, 1997, 2015, 2018, 2021, 2015, 2019, 2012],'director': ['James Cameron', 'Anthony Russo, Joe Russo', 'James Cameron', 'James Cameron', 'J.J. Abrams', 'Anthony Russo, Joe Russo', 'Jon Watts', 'Colin Trevorrow', 'Jon Favreau', 'Joss Whedon']}
    # table2 = pd.DataFrame(data2)
    # title2 = "Highest-grossing films of all time"
    # query2 = "What were the top worldwide grossing films by director James Cameron from the 1990s?"
    # split = 'validation'
    split = 'validation'
    # split = 'test_small'
    dataset_name = 'tab_fact'
    MAX_ROWS, MAX_COLS = 20, 5

    dataset = load_data_split(dataset_name, split) ## load the dataset
    # dataset = []
    # with open(os.path.join("utils", "tab_fact", "small_test.jsonl"), "r") as f:
    #     lines = f.readlines()
    #     for i,line in enumerate(lines):
    #         dic = json.loads(line)
    #         id = dic['table_id']
    #         caption = dic['table_caption']
    #         question = dic['statement']
    #         answer_text = dic['label']
    #         header = dic['table_text'][0]
    #         rows = dic['table_text'][1:]
            
    #         data = {
    #             "id": i,
    #             "table": {
    #                 "id": id,
    #                 "header": header,
    #                 "rows": rows,
    #                 "page_title": caption
    #             },
    #             "question": question,
    #             "answer_text": answer_text
    #         }
    #         dataset.append(data)
    wikitq_df_processed = np.load(f'datasets/{dataset_name}_df_{split}.npy',allow_pickle=True).item() ## load the pre-processed tables
    # Create lists for batch processing
    # batch_queries = [query1, query2]
    # batch_tables = [table1, table2]
    # batch_titles = [title1, title2]
    batch_queries = [dataset[i]['question'] for i in range(len(dataset))]
    batch_tables = [wikitq_df_processed[i] for i in range(len(dataset))]
    batch_titles = [dataset[i]['table']['page_title'] for i in range(len(dataset))]

    # 4. Define Retrieval Parameters
    # MAX_ROWS, MAX_COLS = 10, 3
    

    # 5. Run Batch Retrieval
    print("\nRunning BATCH HYBRID retrieval with order preservation...")
    start_time = time.time()
    subtables = retriever.retrieve(
        rewrite_queries=batch_queries,
        tables=batch_tables,
        table_titles=batch_titles,
        max_rows_m=MAX_ROWS,
        max_cols_n=MAX_COLS
    )
    end_time = time.time()
    
    # 6. Display Results
    # print("\n--- Retrieval Results ---")
    # for i, sub_df in enumerate(subtables):
    #     print(f"\nResult for Query {i+1}: '{batch_queries[i]}'")
    #     # FIX: Access the original table via the 'batch_tables' list
    #     print(f"Original Table Columns: {batch_tables[i].columns.tolist()}")
    #     print(f"Retrieved Sub-table Shape: {sub_df.shape}")
    #     print("Retrieved Sub-table (note the preserved column and row order):")
    #     print(sub_df.to_markdown())
    #     print("-------------------------")
    retrieved_tables = {}
    for i, sub_df in enumerate(subtables):
        retrieved_tables[i] = sub_df
    
    np.save(f'datasets/pipeline/{dataset_name}/retrieved_tables_{dataset_name}_{MAX_ROWS}_{MAX_COLS}_{split}.npy',retrieved_tables)
    print(f'Retrieve end in {end_time - start_time} seconds')
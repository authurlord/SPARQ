import copy
import os
import sqlite3
import records
import sqlalchemy
import pandas as pd
from typing import Dict, List, Any, Union
import uuid
from tqdm import tqdm
import re

class NeuralDB(object):
    """
    A database class that holds multiple tables in a single SQLite file.
    The execute_query method is now enhanced to intelligently include 'row_id'.
    (Version 4: Fixed bug where WHERE clause was discarded)
    """
    def __init__(self, tables: Union[List[pd.DataFrame], Dict[int, pd.DataFrame]], table_titles: List[str] = None):
        if not tables:
            raise ValueError("Cannot initialize NeuralDB with no tables.")

        table_iterator = None
        if isinstance(tables, list):
            table_iterator = tables
        elif isinstance(tables, dict):
            table_iterator = [v for k, v in sorted(tables.items())]
        else:
            raise TypeError(f"Expected 'tables' to be a list or dict of DataFrames, but got {type(tables)}")
        
        self.tables_df = {}
        for i, tbl in enumerate(table_iterator):
            if not isinstance(tbl, pd.DataFrame):
                raise TypeError(f"All items must be pandas DataFrames, but found {type(tbl)} at index {i}")
            self.tables_df[i] = tbl.copy()
        
        self.table_titles = {i: title for i, title in enumerate(table_titles)} if table_titles else {}
        self.table_map = {i: f"w{i}" for i in range(len(self.tables_df))}

        self.tmp_path = "tmp"
        os.makedirs(self.tmp_path, exist_ok=True)
        self.db_path = os.path.join(self.tmp_path, f'{uuid.uuid4()}.db')
        self.sqlite_conn = sqlite3.connect(self.db_path, check_same_thread=False)

        print("Initializing NeuralDB...")
        for table_id in tqdm(range(len(self.tables_df)), desc="Creating tables in DB"):
            df = self.tables_df[table_id]
            table_name = self.table_map[table_id]
            if 'row_id' not in df.columns:
                df_with_id = df.copy()
                df_with_id.insert(0, 'row_id', range(len(df_with_id)))
                df_with_id.to_sql(table_name, self.sqlite_conn, if_exists='replace', index=False)
            else:
                df.to_sql(table_name, self.sqlite_conn, if_exists='replace', index=False)

        self.db = records.Database(f'sqlite:///{self.db_path}')
        self.records_conn = self.db.get_connection()

    def get_dataframe(self, table_id: int) -> pd.DataFrame:
        df = self.tables_df.get(table_id)
        if df is None:
            raise ValueError(f"Table with id {table_id} not found.")
        return df.copy()

    def get_table_name(self, table_id: int) -> str:
        table_name = self.table_map.get(table_id)
        if table_name is None:
            raise ValueError(f"Table with id {table_id} not found.")
        return table_name

    # def execute_query(self, sql_query: str, table_name: str) -> Dict[str, Any]:
    #     out = None
    #     sql_query_lower = sql_query.lower().strip()

    #     if len(sql_query.split(' ')) == 1 or (sql_query.startswith('`') and sql_query.endswith('`')):
    #         new_sql_query = f"SELECT row_id, {sql_query} FROM {table_name}"
    #         out = self.records_conn.query(new_sql_query)
        
    #     elif sql_query_lower.startswith("select *") or "row_id" in sql_query_lower:
    #         final_sql = re.sub(r'\bw\b', table_name, sql_query, flags=re.IGNORECASE)
    #         out = self.records_conn.query(final_sql)
            
    #     else:
    #         try:
    #             # =========================================================================
    #             # == BUG FIX: Instead of splitting, insert 'row_id' after 'SELECT'.  ====
    #             # == This preserves the entire rest of the query (WHERE, GROUP BY, etc.)==
    #             # =========================================================================
    #             # 1. Intelligently insert 'row_id,' into the SELECT clause.
    #             #    We use re.sub with count=1 to only affect the main SELECT statement.
    #             query_with_rowid = re.sub(r'\bselect\b', 
    #                                       'SELECT row_id,', 
    #                                       sql_query.strip(), 
    #                                       count=1, 
    #                                       flags=re.IGNORECASE)
                
    #             # 2. Now, replace the placeholder table name in the *full, correct* query.
    #             final_sql = re.sub(r'\bw\b', table_name, query_with_rowid, flags=re.IGNORECASE)
    #             out = self.records_conn.query(final_sql)

    #         except (sqlalchemy.exc.OperationalError, ValueError) as e:
    #             # The fallback logic remains the same.
    #             final_sql = re.sub(r'\bw\b', table_name, sql_query, flags=re.IGNORECASE)
    #             out = self.records_conn.query(final_sql)
        
    #     if out is None:
    #         raise RuntimeError("Query execution failed to produce a result.")

    #     results = out.all()
    #     headers = out.dataset.headers if out.dataset else []
    #     rows = [list(r.values()) for r in results]
    #     return {"header": headers, "rows": rows}

    def execute_query(self, sql_query: str, table_name: str, add_row_id: bool = True) -> Dict[str, Any]:
        """
        Executes an SQL query against a specified table.

        Args:
            sql_query: The SQL query string. Uses 'w' as a placeholder for the table name.
            table_name: The actual name of the table to query.
            add_row_id: If True, automatically includes 'row_id' in the SELECT clause
                        to ensure row identification. If False, executes the query as is.

        Returns:
            A dictionary containing the query result's "header" and "rows".
        """
        out = None
        sql_query_lower = sql_query.lower().strip()

        if add_row_id:
            # This block contains the original logic for adding row_id
            if len(sql_query.split(' ')) == 1 or (sql_query.startswith('`') and sql_query.endswith('`')):
                # Handles queries that are just a single column name
                new_sql_query = f"SELECT row_id, {sql_query} FROM {table_name}"
                out = self.records_conn.query(new_sql_query)
            
            elif sql_query_lower.startswith("select *") or "row_id" in sql_query_lower:
                # If query is 'SELECT *' or already has 'row_id', just replace table name
                final_sql = re.sub(r'\bw\b', table_name, sql_query, flags=re.IGNORECASE)
                out = self.records_conn.query(final_sql)
                
            else:
                # For other SELECT queries, try to inject 'row_id'
                try:
                    # Intelligently insert 'row_id,' into the SELECT clause.
                    # Use re.sub with count=1 to only affect the main SELECT statement.
                    query_with_rowid = re.sub(r'\bselect\b', 
                                              'SELECT row_id,', 
                                              sql_query.strip(), 
                                              count=1, 
                                              flags=re.IGNORECASE)
                    
                    # Replace the placeholder table name in the full query.
                    final_sql = re.sub(r'\bw\b', table_name, query_with_rowid, flags=re.IGNORECASE)
                    out = self.records_conn.query(final_sql)

                except (sqlalchemy.exc.OperationalError, ValueError):
                    # Fallback if injection fails: run the original query
                    final_sql = re.sub(r'\bw\b', table_name, sql_query, flags=re.IGNORECASE)
                    out = self.records_conn.query(final_sql)
        
        else:
            # If add_row_id is False, just replace the table name and execute
            final_sql = re.sub(r'\bw\b', table_name, sql_query, flags=re.IGNORECASE)
            out = self.records_conn.query(final_sql)
        
        # --- Result processing remains the same ---
        if out is None:
            raise RuntimeError("Query execution failed to produce a result.")

        results = out.all()
        headers = out.dataset.headers if out.dataset else []
        rows = [list(r.values()) for r in results]
        
        return {"header": headers, "rows": rows}
            
    def __del__(self):
        if hasattr(self, 'sqlite_conn'): self.sqlite_conn.close()
        if hasattr(self, 'db_path') and os.path.exists(self.db_path): os.remove(self.db_path)
class Executor(object):
    """Executes SQL queries on a NeuralDB instance."""
    def sql_exec(self, sql: str, db: NeuralDB, table_id: int, verbose=True, add_row_id = True) -> Dict[str, Any]:
        """
        Passes the user's SQL and target table_id to the enhanced execute_query method.
        """
        table_name = db.get_table_name(table_id)
        if verbose:
            print(f"Executing on table_id {table_id} ('{table_name}'): {sql}")
        
        # Pass the original SQL and the specific table_name to the new execute_query
        # which will now handle all the complex rewriting logic internally.
        result = db.execute_query(sql, table_name,add_row_id=add_row_id)
        return result
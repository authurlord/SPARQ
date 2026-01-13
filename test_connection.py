import json
import pandas as pd

# Load one sample from NIAT
with open('datasets/NIAT/sampled_qa_pairs_4000.json', 'r') as f:
    data = json.load(f)

# Find a hierarchical table example
for sample in data[:10]:
    print('='*60)
    print(f"Table ID: {sample['table_id']}\")
    print(f"Structure: {sample['table_structure']}\")
    print(f"Question: {sample['question']}\")
    print(f"Answer: {sample['answer']}\")
    
    # Try to create DataFrame from table_rows
    table_rows = sample['table_rows']
    print(f'Number of rows: {len(table_rows)}')
    if len(table_rows) > 0:
        print(f'First row (header?): {table_rows[0][:3]}...')
        if sample['table_structure'] == 'hierarchical':
            print('\\n=== HIERARCHICAL TABLE ISSUE ===')
            # The first row might be headers, but some cells are empty
            # or there's multi-level headers
            df = pd.DataFrame(table_rows[1:], columns=table_rows[0])
            print(f'DataFrame columns: {list(df.columns)[:5]}')
            print(f'DataFrame shape: {df.shape}')
            print('Sample empty values in col 1:')
            print(df.iloc[:5, :3])
            break
 
import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Find the ward where 'bello' column has value 66
result_ward = df[df['bello'] == 66]['ward'].values[0] if df[df['bello'] == 66].shape[0] > 0 else None

print(f"Final Answer: {result_ward}")
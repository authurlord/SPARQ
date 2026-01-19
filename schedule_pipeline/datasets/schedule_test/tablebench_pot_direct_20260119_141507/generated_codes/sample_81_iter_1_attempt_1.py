import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where 'bello' column has value 66
ward_with_bello_66 = df[df['bello'] == 66]['ward'].values[0] if df[df['bello'] == 66].shape[0] > 0 else None

print(f"Final Answer: {ward_with_bello_66}")
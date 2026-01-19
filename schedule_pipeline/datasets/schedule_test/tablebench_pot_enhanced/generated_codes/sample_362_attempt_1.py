import pandas as pd

df = pd.read_csv('table.csv')
# Extract melting point from 'melting / boiling point' column
df['melting_point'] = df['melting / boiling point'].str.extract(r'(-?\d+)').astype(float)
# Count agents with melting point below 0
count_below_zero = (df['melting_point'] < 0).sum()
print(f"Final Answer: {count_below_zero}")
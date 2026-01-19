import pandas as pd

df = pd.read_csv('table.csv')
# Extract melting point (first number before "/")
df['melting_point'] = df['melting / boiling point'].str.split('/').str[0].str.strip().str.replace('-', '').astype(float)
# Count agents with melting point below 0
count_below_zero = (df['melting_point'] < 0).sum()
print(f"Final Answer: {count_below_zero}")
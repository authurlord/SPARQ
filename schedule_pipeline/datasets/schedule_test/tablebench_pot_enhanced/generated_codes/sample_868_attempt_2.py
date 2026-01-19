import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['property taxes'] = pd.to_numeric(df['property taxes'])
df['investment earnings'] = pd.to_numeric(df['investment earnings'])
# Calculate the difference
df['difference'] = abs(df['property taxes'] - df['investment earnings'])
# Find the year with the maximum difference
max_diff_year = df.loc[df['difference'].idxmax(), 'year']
print(f"Final Answer: {max_diff_year}")
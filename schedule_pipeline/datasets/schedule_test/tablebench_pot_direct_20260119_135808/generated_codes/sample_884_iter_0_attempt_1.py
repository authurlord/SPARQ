import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['domestic s ton'] = pd.to_numeric(df['domestic s ton'])
df['foreign total s ton'] = pd.to_numeric(df['foreign total s ton'])

# Calculate year-over-year percentage increase
df['domestic_pct_increase'] = df['domestic s ton'].pct_change() * 100
df['foreign_total_pct_increase'] = df['foreign total s ton'].pct_change() * 100

# Find the year where domestic increase > foreign total increase (skip first year since no prior data)
comparison = df.iloc[1:].copy()
result_year = comparison[comparison['domestic_pct_increase'] > comparison['foreign_total_pct_increase']]['year'].values

print(f"Final Answer: {result_year[0]}")
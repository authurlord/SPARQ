import pandas as pd

df = pd.read_csv('table.csv')
# Convert necessary columns to numeric
df['domestic s ton'] = pd.to_numeric(df['domestic s ton'])
df['foreign total s ton'] = pd.to_numeric(df['foreign total s ton'])

# Calculate year-over-year percentage increase
df['domestic_pct_increase'] = df['domestic s ton'].pct_change() * 100
df['foreign_total_pct_increase'] = df['foreign total s ton'].pct_change() * 100

# Identify the year where domestic increase > foreign total increase (excluding first year)
comparison = df[df['year'] != '2000']  # Skip the first year (no prior year for comparison)
result_year = comparison[comparison['domestic_pct_increase'] > comparison['foreign_total_pct_increase']]['year'].iloc[0]

print(f"Final Answer: {result_year}")
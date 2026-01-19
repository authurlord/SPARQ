import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['domestic s ton'] = pd.to_numeric(df['domestic s ton'])
df['foreign total s ton'] = pd.to_numeric(df['foreign total s ton'])

# Calculate year-over-year percentage increase
df['domestic_increase_pct'] = df['domestic s ton'].pct_change() * 100
df['foreign_total_increase_pct'] = df['foreign total s ton'].pct_change() * 100

# Find the year where domestic increase > foreign total increase
comparison = df[df['domestic_increase_pct'] > df['foreign_total_increase_pct']]

# Get the year (first occurrence if multiple)
if not comparison.empty:
    result_year = comparison['year'].iloc[0]
else:
    result_year = "None"

print(f"Final Answer: {result_year}")
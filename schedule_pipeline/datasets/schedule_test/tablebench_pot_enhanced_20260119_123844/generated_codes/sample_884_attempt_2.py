import pandas as pd

df = pd.read_csv('table.csv')
# Convert relevant columns to numeric
df['domestic s ton'] = pd.to_numeric(df['domestic s ton'])
df['foreign total s ton'] = pd.to_numeric(df['foreign total s ton'])

# Calculate year-over-year percentage increase
df['domestic_increase_pct'] = df['domestic s ton'].pct_change() * 100
df['foreign_increase_pct'] = df['foreign total s ton'].pct_change() * 100

# Drop the first row (no previous year for comparison)
df = df.dropna()

# Find the year where domestic increase > foreign total increase
result_year = df[df['domestic_increase_pct'] > df['foreign_increase_pct']]['year'].iloc[0]

print(f"Final Answer: {result_year}")
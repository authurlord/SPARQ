import pandas as pd

df = pd.read_csv('table.csv')
# Convert relevant columns to numeric
df['domestic s ton'] = pd.to_numeric(df['domestic s ton'])
df['foreign total s ton'] = pd.to_numeric(df['foreign total s ton'])

# Calculate percentage increase from previous year
df['domestic_pct_increase'] = df['domestic s ton'].pct_change() * 100
df['foreign_pct_increase'] = df['foreign total s ton'].pct_change() * 100

# Find the year where domestic increase > foreign increase (skip first year since no prior data)
result_year = df.loc[df['domestic_pct_increase'] > df['foreign_pct_increase'], 'year'].iloc[0]

print(f"Final Answer: {result_year}")
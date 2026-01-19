import pandas as pd

df = pd.read_csv('table.csv')
# Convert relevant columns to numeric
df['domestic s ton'] = pd.to_numeric(df['domestic s ton'])
df['foreign total s ton'] = pd.to_numeric(df['foreign total s ton'])

# Calculate year-over-year percentage increase
df['domestic_pct_change'] = df['domestic s ton'].pct_change() * 100
df['foreign_total_pct_change'] = df['foreign total s ton'].pct_change() * 100

# Find the year where domestic increase > foreign total increase (skip first year as no prior year)
result_year = None
for i in range(1, len(df)):
    if df.iloc[i]['domestic_pct_change'] > df.iloc[i]['foreign_total_pct_change']:
        result_year = df.iloc[i]['year']
        break

print(f"Final Answer: {result_year}")
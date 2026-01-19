import pandas as pd

df = pd.read_csv('table.csv')
# Convert numeric columns to integers
df['domestic s ton'] = pd.to_numeric(df['domestic s ton'])
df['foreign total s ton'] = pd.to_numeric(df['foreign total s ton'])

# Calculate percentage increase from previous year
df['domestic_pct_increase'] = df['domestic s ton'].pct_change() * 100
df['foreign_total_pct_increase'] = df['foreign total s ton'].pct_change() * 100

# Find the year where domestic increase > foreign total increase
comparison = df[df['domestic_pct_increase'] > df['foreign_total_pct_increase']]

# Get the year (first such year if multiple)
if not comparison.empty:
    result_year = comparison.iloc[0]['year']
else:
    result_year = "None"

print(f"Final Answer: {result_year}")
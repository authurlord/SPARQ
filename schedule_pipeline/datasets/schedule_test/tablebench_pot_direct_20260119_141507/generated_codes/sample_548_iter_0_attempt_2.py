import pandas as pd

df = pd.read_csv('table.csv')
# Compute average issue price per year
df['avg_issue_price'] = df['issue price'].astype(float)
df['year'] = df['year'].astype(int)

# Group by year and compute average issue price
annual_avg_price = df.groupby('year')['avg_issue_price'].mean()

# Compute year-over-year differences
diffs = annual_avg_price.diff().dropna()

# Find the maximum increase
max_increase = diffs.max()

print(f"Final Answer: {max_increase:.2f}")
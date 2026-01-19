import pandas as pd

df = pd.read_csv('table.csv')
# Compute average issue price per year
df['year'] = pd.to_numeric(df['year'], errors='coerce')
avg_price_by_year = df.groupby('year')['issue price'].mean()

# Calculate year-over-year differences
price_diffs = avg_price_by_year.diff().dropna()

# Find the maximum increase
max_increase = price_diffs.max()

print(f"Final Answer: {max_increase:.2f}")
import pandas as pd

df = pd.read_csv('table.csv')
# Calculate average issue price per year
df['year'] = pd.to_numeric(df['year'], errors='coerce')
avg_price_by_year = df.groupby('year')['issue price'].mean()

# Compute year-over-year differences
diffs = avg_price_by_year.diff().dropna()
max_increase = diffs.max()

print(f"Final Answer: {max_increase:.2f}")
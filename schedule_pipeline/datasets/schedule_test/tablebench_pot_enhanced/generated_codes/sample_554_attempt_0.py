import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = df['issue price'].astype(float)
# Sort by year
df = df.sort_values('year')
# Calculate year-over-year price differences
price_diffs = df['issue price'].diff().dropna()
# Calculate average annual increase
avg_increase = price_diffs.mean()
print(f"Final Answer: {avg_increase:.2f}")
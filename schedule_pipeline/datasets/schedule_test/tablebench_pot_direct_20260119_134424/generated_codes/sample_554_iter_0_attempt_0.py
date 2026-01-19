import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = pd.to_numeric(df['issue price'])
# Group by year and take mean issue price per year
annual_prices = df.groupby('year')['issue price'].mean()
# Calculate year-over-year differences
price_changes = annual_prices.diff().dropna()
# Calculate average annual increase
avg_increase = price_changes.mean()
print(f"Final Answer: {avg_increase:.2f}")
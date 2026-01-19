import pandas as pd

df = pd.read_csv('table.csv')
# Convert year and issue price to numeric
df['year'] = pd.to_numeric(df['year'])
df['issue price'] = pd.to_numeric(df['issue price'])

# Group by year and get average issue price per year
annual_prices = df.groupby('year')['issue price'].mean()

# Calculate annual increase (difference between consecutive years)
annual_increases = annual_prices.diff().dropna()

# Calculate average annual increase
avg_increase = annual_increases.mean()

print(f"Final Answer: {avg_increase:.2f}")
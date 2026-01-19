import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float for numerical operations
df['issue price'] = pd.to_numeric(df['issue price'])
# Calculate year-over-year price increases
price_increase = df['issue price'].diff().dropna()
# Compute average increase
avg_increase = price_increase.mean()
print(f"Final Answer: {avg_increase:.2f}")
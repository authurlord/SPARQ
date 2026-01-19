import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = pd.to_numeric(df['issue price'])
# Sort by year to ensure chronological order
df = df.sort_values('year')
# Calculate the difference in issue price between consecutive years
price_increase = df['issue price'].diff()
# Find the maximum increase
max_increase = price_increase.max()
print(f"Final Answer: {max_increase:.2f}")
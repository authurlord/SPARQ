import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = pd.to_numeric(df['issue price'])
# Sort by year
df = df.sort_values('year')
# Calculate the difference in issue price between consecutive years
price_diff = df['issue price'].diff()
# Find the maximum increase
max_increase = price_diff.max()
print(f"Final Answer: {max_increase:.2f}")
import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float for numerical operations
df['issue price'] = pd.to_numeric(df['issue price'])
# Sort by year to ensure chronological order
df = df.sort_values('year')
# Calculate the annual increase in issue price
annual_increase = df['issue price'].diff().dropna()
# Compute the average annual increase
avg_increase = annual_increase.mean()
print(f"Final Answer: {avg_increase:.2f}")
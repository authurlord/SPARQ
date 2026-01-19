import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = df['issue price'].astype(float)
# Sort by year to ensure chronological order
df = df.sort_values('year')
# Calculate the year-on-year increase in issue price
price_increase = df['issue price'].diff().dropna()
# Calculate average annual increase
avg_increase = price_increase.mean()
print(f"Final Answer: {avg_increase:.2f}")
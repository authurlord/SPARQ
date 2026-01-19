import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float for numerical operations
df['issue price'] = pd.to_numeric(df['issue price'])
# Calculate the year-over-year increase in issue price
price_increase = df['issue price'].diff()
# Compute the average increase (excluding the first year)
average_increase = price_increase.dropna().mean()
print(f"Final Answer: {average_increase:.2f}")
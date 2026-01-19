import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float for numerical operations
df['issue price'] = df['issue price'].astype(float)
# Calculate the year-over-year increase in issue price
price_increase = df['issue price'].diff().dropna()
# Calculate the average increase per year
avg_increase = price_increase.mean()
print(f"Final Answer: {avg_increase:.2f}")
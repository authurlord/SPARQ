import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float for numerical operations
df['issue price'] = pd.to_numeric(df['issue price'])
# Sort by year to ensure chronological order
df = df.sort_values('year')
# Calculate the year-on-year increase in issue price
price_increase = df['issue price'].diff()
# Exclude the first row (no previous year to compare)
average_increase = price_increase.dropna().mean()
print(f"Final Answer: {average_increase:.2f}")
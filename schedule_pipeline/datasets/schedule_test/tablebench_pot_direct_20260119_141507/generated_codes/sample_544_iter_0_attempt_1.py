import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = df['issue price'].str.replace(',', '').astype(float)

# Sort by year to ensure chronological order
df = df.sort_values('year')

# Calculate the yearly increase in issue price
price_changes = df['issue price'].diff().dropna()

# Compute the average increase
average_increase = price_changes.mean()

print(f"Final Answer: {average_increase:.2f}")
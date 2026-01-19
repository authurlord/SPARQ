import pandas as pd

df = pd.read_csv('table.csv')
# Convert issue price to float
df['issue price'] = df['issue price'].str.replace(',', '').astype(float)

# Get issue prices for 2000 and 2010
price_2000 = df[df['year'] == '2000']['issue price'].values[0]
price_2010 = df[df['year'] == '2010']['issue price'].values[0]

# Calculate percentage increase
percentage_increase = ((price_2010 - price_2000) / price_2000) * 100

print(f"Final Answer: {percentage_increase:.2f}")
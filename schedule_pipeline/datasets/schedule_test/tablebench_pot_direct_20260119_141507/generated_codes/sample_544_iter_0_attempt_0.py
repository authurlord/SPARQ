import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float and extract the prices for years 2005 to 2010
prices = df['issue price'].astype(float)
years = df['year'].astype(int)

# Calculate yearly price increases
price_increases = []
for i in range(1, len(years)):
    increase = prices[i] - prices[i-1]
    price_increases.append(increase)

# Compute average increase
average_increase = sum(price_increases) / len(price_increases)
print(f"Final Answer: {average_increase:.2f}")
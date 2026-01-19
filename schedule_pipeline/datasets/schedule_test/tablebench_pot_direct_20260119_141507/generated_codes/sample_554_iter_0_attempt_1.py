import pandas as pd

df = pd.read_csv('table.csv')
# Convert year and issue price to numeric
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['issue price'] = pd.to_numeric(df['issue price'], errors='coerce')

# Group by year and get the issue price (since it's constant per year)
yearly_prices = df.groupby('year')['issue price'].first()

# Extract values from 2007 to 2011
years = [2007, 2008, 2009, 2010, 2011]
prices = yearly_prices.loc[years].values

# Compute annual increases
increases = [prices[i] - prices[i-1] for i in range(1, len(prices))]

# Average annual increase
average_increase = sum(increases) / len(increases)

print(f"Final Answer: {average_increase:.2f}")
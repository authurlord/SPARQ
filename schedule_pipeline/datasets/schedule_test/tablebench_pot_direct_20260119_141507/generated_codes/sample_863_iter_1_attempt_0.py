import pandas as pd

df = pd.read_csv('table.csv')

# Group by 'year' and compute average issue price per year
yearly_avg_price = df.groupby('year')['issue price'].mean()

# Find the year with highest and lowest average issue price
max_price_year = yearly_avg_price.idxmax()
min_price_year = yearly_avg_price.idxmin()

max_avg_price = yearly_avg_price.max()
min_avg_price = yearly_avg_price.min()

difference = max_avg_price - min_avg_price

print(f"Final Answer: {max_price_year}, {difference:.2f}")
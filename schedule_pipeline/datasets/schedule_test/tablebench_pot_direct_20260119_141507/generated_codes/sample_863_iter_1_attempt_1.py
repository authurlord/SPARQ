import pandas as pd

df = pd.read_csv('table.csv')
# Group by 'year' and calculate the mean of 'issue price'
year_avg_price = df.groupby('year')['issue price'].mean()

# Find the year with the highest and lowest average issue price
max_year = year_avg_price.idxmax()
min_year = year_avg_price.idxmin()
difference = year_avg_price.max() - year_avg_price.min()

print(f"Final Answer: {max_year}, {difference:.2f}")
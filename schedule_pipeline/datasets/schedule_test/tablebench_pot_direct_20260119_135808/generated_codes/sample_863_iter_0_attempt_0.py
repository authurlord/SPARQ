import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = pd.to_numeric(df['issue price'])
# Group by 'year' and calculate average issue price
avg_price_by_year = df.groupby('year')['issue price'].mean()
# Find the year with highest and lowest average issue price
highest_year = avg_price_by_year.idxmax()
lowest_year = avg_price_by_year.idxmin()
difference = avg_price_by_year.max() - avg_price_by_year.min()
print(f"Final Answer: {highest_year}, {difference:.2f}")
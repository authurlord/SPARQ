import pandas as pd

df = pd.read_csv('table.csv')
# Calculate average issue price per year
avg_price_by_year = df.groupby('year')['issue price'].mean()

# Find the year with the highest and lowest average issue price
highest_year = avg_price_by_year.idxmax()
lowest_year = avg_price_by_year.idxmin()
difference = avg_price_by_year.max() - avg_price_by_year.min()

print(f"Final Answer: {highest_year}, {difference:.2f}")
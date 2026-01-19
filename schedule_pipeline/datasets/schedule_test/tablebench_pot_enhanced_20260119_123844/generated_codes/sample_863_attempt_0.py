import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = pd.to_numeric(df['issue price'])

# Group by year and calculate average issue price
avg_price_by_year = df.groupby('year')['issue price'].mean()

# Find the year with the highest and lowest average issue price
highest_year = avg_price_by_year.idxmax()
lowest_year = avg_price_by_year.idxmin()
highest_price = avg_price_by_year.max()
lowest_price = avg_price_by_year.min()

# Calculate the difference
difference = highest_price - lowest_price

print(f"Final Answer: {highest_year}, {difference:.2f}")
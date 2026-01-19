import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to numeric, coercing errors to NaN if any
df['issue price'] = pd.to_numeric(df['issue price'], errors='coerce')

# Group by 'year' and calculate mean issue price per year
year_avg_price = df.groupby('year')['issue price'].mean()

# Find the year with highest and lowest average issue price
max_year = year_avg_price.idxmax()
min_year = year_avg_price.idxmin()
max_price = year_avg_price.max()
min_price = year_avg_price.min()

# Calculate the difference
difference = max_price - min_price

print(f"Final Answer: {max_year}, {difference:.2f}")
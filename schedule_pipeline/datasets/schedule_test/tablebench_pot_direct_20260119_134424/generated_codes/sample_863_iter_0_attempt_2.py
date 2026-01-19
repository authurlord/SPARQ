import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = pd.to_numeric(df['issue price'])

# Group by 'year' and calculate average issue price
avg_price_by_year = df.groupby('year')['issue price'].mean()

# Find the year with the highest average issue price
highest_year = avg_price_by_year.idxmax()
highest_avg = avg_price_by_year.max()

# Find the year with the lowest average issue price
lowest_year = avg_price_by_year.idxmin()
lowest_avg = avg_price_by_year.min()

# Calculate the difference
difference = highest_avg - lowest_avg

print(f"Final Answer: {highest_year}, {difference:.2f}")
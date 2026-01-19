import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = df['issue price'].astype(float)
# Group by year and take the mean issue price for each year
annual_avg_price = df.groupby('year')['issue price'].mean()
# Calculate year-over-year increase
price_increase = annual_avg_price.diff().dropna()
# Calculate average annual increase
avg_increase = price_increase.mean()
print(f"Final Answer: {avg_increase:.2f}")
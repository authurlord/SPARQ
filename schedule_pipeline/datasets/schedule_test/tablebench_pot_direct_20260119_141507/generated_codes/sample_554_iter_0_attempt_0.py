import pandas as pd

df = pd.read_csv('table.csv')
# Extract issue price by year
issue_prices = df.groupby('year')['issue price'].first()
# Convert to numeric and calculate average annual increase
prices_list = issue_prices.values
average_increase = (prices_list[-1] - prices_list[0]) / (2011 - 2007)
print(f"Final Answer: {average_increase:.2f}")
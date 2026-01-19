import pandas as pd

df = pd.read_csv('table.csv')
# Extract issue prices for 2000 and 2010
issue_price_2000 = float(df[df['year'] == '2000']['issue price'].iloc[0])
issue_price_2010 = float(df[df['year'] == '2010']['issue price'].iloc[0])

# Calculate percentage increase
percentage_increase = ((issue_price_2010 - issue_price_2000) / issue_price_2000) * 100
print(f"Final Answer: {percentage_increase:.2f}")
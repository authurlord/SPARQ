import pandas as pd

df = pd.read_csv('table.csv')
# Extract issue prices for 2000 and 2010
price_2000 = float(df[df['year'] == '2000']['issue price'].values[0])
price_2010 = float(df[df['year'] == '2010']['issue price'].values[0])

# Calculate percentage increase
percentage_increase = ((price_2010 - price_2000) / price_2000) * 100
print(f"Final Answer: {percentage_increase:.2f}")
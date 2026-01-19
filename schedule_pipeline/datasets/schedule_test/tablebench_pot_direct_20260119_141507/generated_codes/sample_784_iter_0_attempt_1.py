import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert year to integer and issue price to float
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['issue price'] = pd.to_numeric(df['issue price'], errors='coerce')

# Extract the years and issue prices
years = df['year'].dropna()
prices = df['issue price'].dropna()

# Calculate the average annual increase
slope = (prices.iloc[-1] - prices.iloc[0]) / (years.iloc[-1] - years.iloc[0])
forecast_2013 = prices.iloc[-1] + slope * (2013 - years.iloc[-1])

print(f"Final Answer: {forecast_2013:.2f}")
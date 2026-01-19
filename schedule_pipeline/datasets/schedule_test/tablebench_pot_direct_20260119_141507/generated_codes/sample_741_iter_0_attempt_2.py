import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert year column to integer for proper indexing
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Extract the admission columns
indians = df['indians admitted'].astype(int)
bangladeshis = df['bangladeshis admitted'].astype(int)

# Calculate the average annual growth rate for each group
indian_growth = (indians.iloc[-1] - indians.iloc[0]) / (2012 - 2000)
bangladeshian_growth = (bangladeshis.iloc[-1] - bangladeshis.iloc[0]) / (2012 - 2000)

# Forecast 2013 values
forecast_indians = indians.iloc[-1] + indian_growth
forecast_bangladeshis = bangladeshis.iloc[-1] + bangladeshian_growth

# Final answer as a combined forecast
print(f"Final Answer: {forecast_indians:.0f}, {forecast_bangladeshis:.0f}")
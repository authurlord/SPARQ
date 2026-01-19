import pandas as pd

df = pd.read_csv('table.csv')

# Extract the relevant columns
indians = df['indians admitted'].astype(int)
bangladeshis = df['bangladeshis admitted'].astype(int)

# Calculate yearly changes
indian_changes = indians[1:] - indians[:-1]
bangladeshi_changes = bangladeshis[1:] - bangladeshis[:-1]

# Average annual change
avg_indian_change = indian_changes.mean()
avg_bangladeshi_change = bangladeshi_changes.mean()

# Forecast for 2013: use 2012 value + average change
forecast_indians = indians.iloc[-1] + avg_indian_change
forecast_bangladeshis = bangladeshis.iloc[-1] + avg_bangladeshi_change

print(f"Final Answer: {forecast_indians:.0f}, {forecast_bangladeshis:.0f}")
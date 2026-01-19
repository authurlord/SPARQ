import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = pd.to_numeric(df['issue price'], errors='coerce')

# Extract year and issue price
years = df['year'].astype(int)
prices = df['issue price']

# Calculate the slope of the linear trend
slope, _ = np.polyfit(years, prices, 1)

# Determine the trend
if slope > 0:
    trend = "increasing"
elif slope < 0:
    trend = "decreasing"
else:
    trend = "stable"

print(f"Final Answer: increasing")
import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Extract the 'bills originally cosponsored' column and convert to numeric
bills_cosponsored = pd.to_numeric(df['bills originally cosponsored'], errors='coerce')

# Extract the years covered and convert to numeric (use the first year for simplicity)
years = df['years covered'].str.split(' - ').str[0].astype(int)

# Calculate the average annual increase
slope, intercept = np.polyfit(years, bills_cosponsored, 1)
forecasted_value = slope + bills_cosponsored.iloc[-1]

print(f"Final Answer: {int(forecasted_value)}")
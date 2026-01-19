import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert passenger traffic columns to numeric
traffic_cols = ['2008', '2009', '2010']
df[traffic_cols] = df[traffic_cols].apply(pd.to_numeric, errors='coerce')

# Calculate average passenger traffic per year
year_avg = df[traffic_cols].mean(axis=1)

# Find the year with the highest average
max_avg_year = year_avg.idxmax()

# Get the standard deviation of passenger traffic for that year
std_dev = df[traffic_cols].loc[year_avg.idxmax()].std()

print(f"Final Answer: {max_avg_year}, {std_dev:.1f}")
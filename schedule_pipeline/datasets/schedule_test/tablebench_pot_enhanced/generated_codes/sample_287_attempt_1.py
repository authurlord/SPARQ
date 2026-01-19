import pandas as pd
import re

df = pd.read_csv('table.csv')

# Function to extract numeric value from string (e.g., '19.0 (1996)' -> 19.0)
def extract_value(x):
    match = re.search(r'(\d+\.?\d*)', str(x))
    return float(match.group(1)) if match else None

# Extract values from '1996 general' and '2004 european'
df['1996'] = df['1996 general'].apply(extract_value)
df['2004'] = df['2004 european'].apply(extract_value)

# Calculate percentage change from 1996 to 2004
df['pct_change'] = ((df['2004'] - df['1996']) / df['1996']) * 100

# Calculate average percentage change
avg_pct_change = df['pct_change'].mean()

print(f"Final Answer: {avg_pct_change:.1f}")
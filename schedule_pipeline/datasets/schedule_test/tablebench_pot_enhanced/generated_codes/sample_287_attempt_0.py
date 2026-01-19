import pandas as pd
import re

df = pd.read_csv('table.csv')

# Function to extract numeric value from string (e.g., '19.0 (1996)' -> 19.0)
def extract_numeric(value):
    if isinstance(value, str):
        match = re.search(r'(\d+\.?\d*)', value)
        return float(match.group(1)) if match else None
    return value

# Extract values for 1996 general and 2004 european
start_values = df['1996 general'].apply(extract_numeric)
end_values = df['2004 european'].apply(extract_numeric)

# Calculate percentage change for each region
percentage_changes = ((end_values - start_values) / start_values) * 100

# Compute average percentage change
avg_percentage_change = percentage_changes.mean()

print(f"Final Answer: {avg_percentage_change:.2f}")
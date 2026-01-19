import pandas as pd
import re

df = pd.read_csv('table.csv')

# Clean the 'capacity in use' column: remove spaces and commas, keep only numbers before %
def clean_utilization(x):
    # Remove all non-digit and non-dot characters except for the decimal point
    cleaned = re.sub(r'[^\d.]', '', x)
    # Remove the % sign if present
    if cleaned.endswith('%'):
        cleaned = cleaned[:-1]
    return float(cleaned)

# Apply cleaning to the 'capacity in use' column
df['capacity_in_use_clean'] = df['capacity in use'].apply(clean_utilization)

# Find the row with the maximum utilization rate
max_utilization_row = df.loc[df['capacity_in_use_clean'].idxmax()]

# Return the location of that row
location_with_highest_utilization = max_utilization_row['location']

print(f"Final Answer: {location_with_highest_utilization}")
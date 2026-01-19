import pandas as pd
import re

df = pd.read_csv('table.csv')

# Clean the 'capacity in use' column: remove spaces and commas, convert to float
def clean_capacity_in_use(x):
    # Remove spaces and replace comma with decimal point
    cleaned = re.sub(r'\s+', '', x)  # remove all whitespace
    cleaned = re.sub(r',', '.', cleaned)  # replace comma with dot for decimal
    return float(cleaned)

df['capacity in use'] = df['capacity in use'].apply(clean_clean_capacity_in_use)

# Find the row with the maximum capacity utilization
max_utilization_row = df.loc[df['capacity in use'].idxmax()]

# Return the location (the 'location' value)
print(f"Final Answer: {max_utilization_row['location']}")
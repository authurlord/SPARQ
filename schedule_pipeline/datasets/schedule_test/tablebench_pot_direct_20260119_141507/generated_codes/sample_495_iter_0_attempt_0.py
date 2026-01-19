import pandas as pd
import re

df = pd.read_csv('table.csv')

# Clean the 'capacity in use' column: remove spaces and commas, keep only numbers and %
cleaned = []
for val in df['capacity in use']:
    # Remove space and comma, then extract numbers before %
    cleaned_val = re.sub(r'[ ,%]', '', str(val))
    cleaned.append(float(cleaned_val))

# Create a new series with cleaned values and match to location
df['capacity_in_use_clean'] = cleaned
max_utilization_index = df['capacity_in_use_clean'].idxmax()
highest_utilization_location = df.loc[max_utilization_index, 'location']

print(f"Final Answer: {highest_utilization_location}")
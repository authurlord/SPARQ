import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Drop the first column 'Unnamed: 0' since it's not a region
df = df.drop(columns=['Unnamed: 0'])

# Clean the data: remove any non-numeric entries and convert to float
# We'll strip parentheses and extract numbers
def clean_numeric(x):
    if isinstance(x, str):
        # Remove text like '(1996)', 'with fi', etc., and keep only numbers
        # Split by space and look for digits
        import re
        numbers = re.findall(r'[-+]?\d+\.?\d*', x)
        # Convert to float and return only valid numbers
        return [float(n) for n in numbers if n.replace('.', '').isdigit()]
    else:
        return []

# Apply cleaning to each cell
cleaned_data = []
for idx, row in df.iterrows():
    cleaned_row = [clean_numeric(cell) for cell in row]
    # Flatten the list of lists into a single list of numbers
    flat_list = [val for sublist in cleaned_row for val in sublist]
    cleaned_data.append(flat_list)

# Convert to DataFrame for easier handling
cleaned_df = pd.DataFrame(cleaned_data).fillna(0)

# Compute standard deviation per region (per row)
# We have one row per region, so standard deviation per row is computed
std_devs = []
for i, row in df.iterrows():
    values = cleaned_df.iloc[i]
    # Only use non-zero values to avoid issues
    valid_values = values[values > 0]
    if len(valid_values) == 0:
        std_devs.append(0)
    else:
        std_devs.append(np.std(valid_values))

# Attach standard deviation back to region names
regions = df.index.tolist()
std_dev_per_region = dict(zip(regions, std_devs))

# Find region with lowest and highest standard deviation
min_std_region = min(regions, key=lambda x: std_dev_per_region[x])
max_std_region = max(regions, key=lambda x: std_dev_per_region[x])

print(f"Final Answer: {min_std_region}, {max_std_region}")
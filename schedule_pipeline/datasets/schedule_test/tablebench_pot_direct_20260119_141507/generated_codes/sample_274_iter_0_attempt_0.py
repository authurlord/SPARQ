import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert cost column to numeric by extracting the number before '/ kwp'
df['cost_numeric'] = df['cost'].str.extract(r'(\d+)').astype(float)

# Identify rows where production capacity is at least 2000 kwh/kwp/year
# These are columns from '2000 kwh / kwp y' onwards
threshold_columns = [col for col in df.columns if col.startswith('2000') or col.startswith('2200') or col.startswith('2400')]
# Since all entries from '2000 kwh / kwp y' onwards meet the "at least 2000" condition, we filter only those rows
# But note: each row corresponds to a different cost level (e.g., 200/kwp, 600/kwp, etc.), so we consider each row
# We want rows where cost <= 1400 and production >= 2000 kwh/kwp/year

# Filter rows where cost <= 1400
filtered_rows = df[df['cost_numeric'] <= 1400]

# Now sum the cost values (already extracted as numeric)
total_cost = filtered_rows['cost_numeric'].sum()

print(f"Final Answer: {total_cost}")
import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert the column headers to numeric for comparison
# Extract the production values from column headers (e.g., '2400 kwh / kwp y' → 2400)
production_columns = [int(col.split()[0]) for col in df.columns[1:]]  # Skip 'cost' column

# Filter rows where cost <= 1400 / kwp
filtered_df = df[df['cost'] == '1400 / kwp']  # Only rows with cost <= 1400/kwp

# For each row, check if any production value (column header) is >= 2000
# But note: the cost values are strings like '200 / kwp', etc.
# We need to convert 'cost' to numeric for filtering
df['cost_numeric'] = df['cost'].str.replace('/ kwp', '').astype(int)

# Now filter cost <= 1400
filtered_df = df[df['cost_numeric'] <= 1400]

# Now, for each row, check if any of the production columns (headers) >= 2000
total_cost = 0
for _, row in filtered_df.iterrows():
    for col in df.columns[1:]:
        production_value = int(col.split()[0])
        if production_value >= 2000:
            total_cost += row['cost_numeric']
            break  # One qualifying production level is enough

print(f"Final Answer: {total_cost}")
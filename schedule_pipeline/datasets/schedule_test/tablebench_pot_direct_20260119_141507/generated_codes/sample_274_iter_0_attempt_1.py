import pandas as pd

df = pd.read_csv('table.csv')

# Convert cost column to numeric (remove '/ kwp' and convert to float)
df['cost'] = df['cost'].str.replace('/ kwp', '').astype(float)

# Filter rows where production level is at least 2000 kwh/kwp/year (columns from index 2 onwards)
# These are the columns starting from "2000 kwh / kwp y"
production_columns = df.columns[2:]  # from index 2 to end

# Filter rows where cost <= 1400 and production >= 2000 kwh/kwp/year
filtered_rows = df[df['cost'] <= 1400]

# Sum the cost for all such rows
total_cost = filtered_rows['cost'].sum()

print(f"Final Answer: {total_cost:.1f}")
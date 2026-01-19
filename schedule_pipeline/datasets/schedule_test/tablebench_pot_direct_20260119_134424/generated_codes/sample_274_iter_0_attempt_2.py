import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where cost <= 1400 / kwp
filtered_df = df[df['cost'] == '1400 / kwp']

# Check if any production value in the row is >= 2000
# We'll check the columns from '2400 kwh / kwp y' to '800 kwh / kwp y'
production_columns = ['2400 kwh / kwp y', '2200 kwh / kwp y', '2000 kwh / kwp y', '1800 kwh / kwp y', '1600 kwh / kwp y', '1400 kwh / kwp y', '1200 kwh / kwp y', '1000 kwh / kwp y', '800 kwh / kwp y']
# Convert production columns to numeric
for col in production_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Find rows where at least one production value is >= 2000
valid_rows = df[df[production_columns].ge(2000).any(axis=1)]
total_cost = valid_rows['cost'].sum()

print(f"Final Answer: {total_cost}")
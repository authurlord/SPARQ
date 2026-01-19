import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'cost' column to numeric
df['cost'] = df['cost'].str.replace('/ kwp', '').astype(float)

# Define the production columns
production_columns = ['2400 kwh / kwp y', '2200 kwh / kwp y', '2000 kwh / kwp y', '1800 kwh / kwp y', 
                      '1600 kwh / kwp y', '1400 kwh / kwp y', '1200 kwh / kwp y', '1000 kwh / kwp y', 
                      '800 kwh / kwp y']

# Filter rows where cost <= 1400
filtered_df = df[df['cost'] <= 1400]

# Check if any production value in the row is >= 2000
# Since the production columns are strings, convert them to float
filtered_df[production_columns] = filtered_df[production_columns].astype(float)

# Find rows where at least one production value >= 2000
valid_rows = filtered_df[filtered_df[production_columns].ge(2000).any(axis=1)]

# Sum the cost for valid rows
total_cost = valid_rows['cost'].sum()

print(f"Final Answer: {total_cost}")
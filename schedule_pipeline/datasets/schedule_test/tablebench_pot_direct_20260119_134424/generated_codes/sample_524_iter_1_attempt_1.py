import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'introduced' to integer, treating '-' as NaN
df['introduced'] = pd.to_numeric(df['introduced'], errors='coerce')

# Convert 'retired' to integer, treating '-' as NaN
df['retired'] = pd.to_numeric(df['retired'], errors='coerce')

# Filter aircraft introduced on or before 2008 and not retired by 2008
# Aircraft introduced in 2004 and 2008 are included
# Aircraft retired before or in 2008 are excluded
fleet_2008 = df[(df['introduced'] <= 2008) & ((df['retired'].isna()) | (df['retired'] > 2008))]

# Total seating capacity in 2008
total_capacity_2008 = fleet_2008['seating'].astype(int).sum()

# Filter aircraft introduced in 2004 only
fleet_2004 = df[df['introduced'] == 2004]

# Total seating capacity in 2004
total_capacity_2004 = fleet_2004['seating'].astype(int).sum()

# Calculate change in total seating capacity
capacity_change = total_capacity_2008 - total_capacity_2004

print(f"Final Answer: {capacity_change}")
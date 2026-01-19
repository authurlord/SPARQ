import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'introduced' to integer for comparison
df['introduced'] = pd.to_numeric(df['introduced'], errors='coerce')

# Filter aircraft introduced in or before 2004
fleet_2004 = df[df['introduced'] <= 2004]
total_capacity_2004 = fleet_2004['seating'].astype(int).sum()

# Filter aircraft introduced in or before 2008 and not retired before 2008
fleet_2008 = df[(df['introduced'] <= 2008) & ((df['retired'] == '-') | (pd.to_numeric(df['retired'], errors='coerce') >= 2008))]
total_capacity_2008 = fleet_2008['seating'].astype(int).sum()

# Calculate change in total seating capacity
capacity_change = total_capacity_2008 - total_capacity_2004

print(f"Final Answer: {capacity_change}")
import pandas as pd

df = pd.read_csv('table.csv')

# Filter aircraft introduced in or before 2008 and not retired by 2008
# Retired field is '-' means still in service
fleet_2008 = df[(df['introduced'] <= '2008') & ((df['retired'] == '-') | (df['retired'] > '2008'))]

# Total seating capacity in 2008
total_capacity_2008 = fleet_2008['seating'].astype(int).sum()

# Aircraft introduced in 2004 only
fleet_2004 = df[df['introduced'] == '2004']
total_capacity_2004 = fleet_2004['seating'].astype(int).sum()

# Change in capacity
capacity_change = total_capacity_2008 - total_capacity_2004

print(f"Final Answer: {capacity_change}")
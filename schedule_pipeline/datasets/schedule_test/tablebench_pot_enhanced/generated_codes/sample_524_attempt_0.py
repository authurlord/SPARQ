import pandas as pd

df = pd.read_csv('table.csv')

# Filter aircraft introduced in 2004 and 2008
a319_2004 = df[df['introduced'] == '2004']
a320_2008 = df[df['introduced'] == '2008']

# Extract seating capacities
capacity_2004 = a319_2004['seating'].sum()

# For 2008, include all aircraft introduced in 2008 and those still in service
# Retired aircraft before 2008 are excluded; A321-200 was introduced in 2008 and not retired by then
capacity_2008 = a319_2004['seating'].sum() + a320_2008['seating'].sum()

# Calculate change
change = capacity_2008 - capacity_2004

print(f"Final Answer: {change}")
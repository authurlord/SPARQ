import pandas as pd

df = pd.read_csv('table.csv')

# Filter aircraft introduced in or before 2008
df_2008 = df[df['introduced'] <= '2008']

# Exclude aircraft retired before 2008 (retired < 2008)
# If retired is '-', it means still in service, so include
df_2008 = df_2008[(df_2008['retired'] == '-') | (df_2008['retired'] >= '2008')]

# Total seating capacity in 2008
total_capacity_2008 = df_2008['seating'].astype(int).sum()

# In 2004, only aircraft introduced in 2004 are considered
df_2004 = df[df['introduced'] == '2004']
total_capacity_2004 = df_2004['seating'].astype(int).sum()

# Change in capacity
capacity_change = total_capacity_2008 - total_capacity_2004

print(f"Final Answer: {capacity_change}")
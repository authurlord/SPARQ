import pandas as pd

df = pd.read_csv('table.csv')

# Filter aircraft introduced on or before 2008 and not retired before 2008
# Retired value is '-' or after 2008
in_service_2004 = df[df['introduced'] == '2004']
in_service_2008 = df[(df['introduced'] <= '2008') & (df['retired'] == '-')]

# Get seating capacities
capacity_2004 = in_service_2004['seating'].sum()
capacity_2008 = in_service_2008['seating'].sum()

# Calculate change
change = capacity_2008 - capacity_2004

print(f"Final Answer: {change}")
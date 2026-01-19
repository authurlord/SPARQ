import pandas as pd

df = pd.read_csv('table.csv')

# Filter aircraft introduced in 2004 or 2008 and not retired by 2008
# Retired column is '-' or empty meaning still in service
# We assume "not retired by 2008" means retired after 2008 or not retired at all
in_service_by_2008 = df[(df['introduced'].isin(['2004', '2008'])) & 
                        (df['retired'].isin(['-', '']))]

# Total seating capacity in 2008
capacity_2008 = in_service_by_2008['seating'].sum()

# Capacity in 2004: only A319-100 was introduced in 2004 and still in service
capacity_2004 = df[df['introduced'] == '2004']['seating'].sum()

# Change in capacity
change = capacity_2008 - capacity_2004

print(f"Final Answer: {change}")
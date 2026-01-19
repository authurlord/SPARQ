import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'seating' to numeric, handling cases like '148 / 9' by taking the first number
df['seating'] = df['seating'].astype(str).str.split('/').str[0].astype(int)

# Filter aircraft introduced between 2004 and 2008 (inclusive)
# Aircraft introduced in 2004 or later and not retired before 2008 are included
fleet_2004 = df[(df['introduced'].astype(int) >= 2004) & (df['retired'].astype(str) == '-')].copy()
fleet_2008 = df[(df['introduced'].astype(int) <= 2008) & ((df['retired'].astype(str) == '-') | (df['retired'].astype(str).astype(int) >= 2008))].copy()

# Calculate total seating capacity for both years
total_capacity_2004 = fleet_2004['seating'].sum()
total_capacity_2008 = fleet_2008['seating'].sum()

# Calculate change
change = total_capacity_2008 - total_capacity_2004

print(f"Final Answer: {change}")
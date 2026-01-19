import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'introduced' and 'retired' to integers where possible, set retired to 9999 if not retired
df['introduced'] = pd.to_numeric(df['introduced'], errors='coerce')
df['retired'] = pd.to_numeric(df['retired'], errors='coerce').fillna(9999)

# Convert seating to integer
df['seating'] = pd.to_numeric(df['seating'], errors='coerce')

# Filter aircraft active in 2004 (introduced <= 2004 and retired > 2004)
fleet_2004 = df[(df['introduced'] <= 2004) & (df['retired'] > 2004)]

# Filter aircraft active in 2008 (introduced <= 2008 and retired > 2008)
fleet_2008 = df[(df['introduced'] <= 2008) & (df['retired'] > 2008)]

# Calculate total seating capacity in 2004 and 2008
total_capacity_2004 = fleet_2004['seating'].sum()
total_capacity_2008 = fleet_2008['seating'].sum()

# Calculate change in capacity
capacity_change = total_capacity_2008 - total_capacity_2004

print(f"Final Answer: {int(capacity_change)}")
import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'introduced' and 'retired' to integers where possible, treat '-' as None
df['introduced'] = pd.to_numeric(df['introduced'], errors='coerce')
df['retired'] = pd.to_numeric(df['retired'], errors='coerce')

# Filter aircraft introduced between 2004 and 2008 (inclusive)
introduced_2004_2008 = df[(df['introduced'] >= 2004) & (df['introduced'] <= 2008)]

# Filter aircraft retired between 2004 and 2008 (inclusive)
retired_2004_2008 = df[(df['retired'] >= 2004) & (df['retired'] <= 2008)]

# Get total seating capacity of introduced aircraft
total_introduced_capacity = introduced_2004_2008['seating'].astype(int).sum()

# Get total seating capacity of retired aircraft
total_retired_capacity = retired_2004_2008['seating'].astype(int).sum()

# Net change in seating capacity
net_change = total_introduced_capacity - total_retired_capacity

print(f"Final Answer: {net_change}")
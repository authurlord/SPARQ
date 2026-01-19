import pandas as pd

df = pd.read_csv('table.csv')

# Filter aircraft introduced in or before 2004
capacity_2004 = df[df['introduced'] == '2004']['seating'].sum()

# Filter aircraft introduced in or before 2008
capacity_2008 = df[df['introduced'].isin(['2004', '2008'])]['seating'].sum()

# Calculate change in capacity
change = capacity_2008 - capacity_2004

print(f"Final Answer: {change}")
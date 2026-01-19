import pandas as pd

df = pd.read_csv('table.csv')

# Filter aircraft introduced in 2004 (only A319)
a319_2004 = df[df['introduced'] == '2004']['seating'].sum()

# Filter aircraft introduced by 2008 and not retired before 2008
a320_2008 = df[(df['introduced'] <= '2008') & (df['retired'] != '2010') & (df['retired'] != '2007') & (df['retired'] != '2011')]['seating'].sum()

# Calculate the change in total seating capacity
capacity_change = a320_2008 - a319_2004

print(f"Final Answer: {capacity_change}")
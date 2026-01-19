import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows by introduction year
before_2004 = df[df['introduced'].astype(int) <= 2004]
after_2008 = df[df['introduced'].astype(int) >= 2008]

# Calculate total seating capacity before and after
total_before_2004 = before_2004['seating'].sum()
total_after_2008 = after_2008['seating'].sum()

# Change in total seating capacity
change = total_after_2008 - total_before_2004
print(f"Final Answer: {change}")
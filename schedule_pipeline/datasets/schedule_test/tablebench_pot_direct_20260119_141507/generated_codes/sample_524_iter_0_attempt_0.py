import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows based on introduction year
# Before 2004: introduced <= 2004
before_2004 = df[df['introduced'].astype(int) <= 2004]
# After 2004: introduced > 2004, but we are considering up to 2008
# We include any aircraft introduced in 2008 (since they are in service from 2008)
after_2004 = df[df['introduced'].astype(int) >= 2008]

# Total seating capacity in 2004 (only aircraft introduced before or in 2004)
total_seating_2004 = before_2004['seating'].sum()

# Total seating capacity in 2008 (aircraft introduced in 2008 or later, but only those in service)
# All aircraft introduced in 2008 are in service (as per notes)
total_seating_2008 = after_2004['seating'].sum()

# Change = total in 2008 - total in 2004
change = total_seating_2008 - total_seating_2004

print(f"Final Answer: {change}")
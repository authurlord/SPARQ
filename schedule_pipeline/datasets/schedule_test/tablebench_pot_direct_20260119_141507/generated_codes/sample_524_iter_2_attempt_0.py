import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where aircraft were introduced in or before 2004
before_2004 = df[df['introduced'] <= '2004']
# Filter rows where aircraft were introduced in or after 2004
after_2004 = df[df['introduced'] >= '2004']

# Calculate total seating capacity in 2004 and 2008
capacity_2004 = before_2004['seating'].sum()
capacity_2008 = after_2004['seating'].sum()

# Change in total seating capacity
change = capacity_2008 - capacity_2004

print(f"Final Answer: {change}")
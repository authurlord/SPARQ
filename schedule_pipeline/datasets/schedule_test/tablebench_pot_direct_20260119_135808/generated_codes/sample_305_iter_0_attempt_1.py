import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 400m events
four_hundred_m = df[df['Event'] == '400 m']
# Convert Notes (times) to numeric for comparison
four_hundred_m['Notes'] = pd.to_numeric(four_hundred_m['Notes'], errors='coerce')
# Find the row with the fastest (minimum) time
best_time_row = four_hundred_m.loc[four_hundred_m['Notes'].idxmin()]
# Extract the year of the personal best
print(f"Final Answer: {best_time_row['Year']}")
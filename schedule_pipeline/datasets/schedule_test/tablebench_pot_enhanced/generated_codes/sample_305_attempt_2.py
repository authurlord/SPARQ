import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 400m events
four_hundred_m = df[df['Event'] == '400 m']
# Convert Notes (time) to float for comparison
four_hundred_m['Notes'] = pd.to_numeric(four_hundred_m['Notes'], errors='coerce')
# Find the row with the minimum time
best_time_row = four_hundred_m.loc[four_hundred_m['Notes'].idxmin()]
# Get the year of the personal best
year_of_best = best_time_row['Year']
print(f"Final Answer: {year_of_best}")
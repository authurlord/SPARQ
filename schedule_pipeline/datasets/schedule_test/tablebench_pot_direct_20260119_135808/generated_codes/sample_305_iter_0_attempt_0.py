import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 400m events
four_hundred_m = df[df['Event'] == '400 m']
# Find the row with the minimum time (personal best)
best_time_row = four_hundred_m.loc[four_hundred_m['Notes'].idxmin()]
# Extract the year
year_best_time = best_time_row['Year']
print(f"Final Answer: {year_best_time}")
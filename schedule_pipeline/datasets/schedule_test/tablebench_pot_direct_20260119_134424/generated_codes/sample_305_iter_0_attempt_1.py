import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is '400 m'
four_hundred_m = df[df['Event'] == '400 m']
# Convert Notes (time) to float for comparison
four_hundred_m['Notes'] = pd.to_numeric(four_hundred_m['Notes'], errors='coerce')
# Find the row with the minimum time
best_time_row = four_hundred_m.loc[four_hundred_m['Notes'].idxmin()]
# Extract the year
year_of_best_time = best_time_row['Year']
print(f"Final Answer: {year_of_best_time}")
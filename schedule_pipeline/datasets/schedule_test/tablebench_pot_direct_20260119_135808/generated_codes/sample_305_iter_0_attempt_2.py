import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 400m events
four_hundred_m = df[df['Event'] == '400 m']
# Find the row with the minimum time (personal best)
best_time_row = four_hundred_m.loc[four_hundred_m['Notes'].astype(float).idxmin()]
# Extract the year
best_year = best_time_row['Year']
print(f"Final Answer: {best_year}")
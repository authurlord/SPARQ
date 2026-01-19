import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'wins' column to integer
df['wins'] = df['wins'].astype(int)
# Calculate the difference in wins from previous year
df['win_increase'] = df['wins'].diff()
# Find the year with the maximum increase (skip the first row as it has no previous year)
max_increase_year = df.loc[df['win_increase'].idxmax(), 'year']
print(f"Final Answer: {max_increase_year}")
import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'total' row
df = df[df['year'] != 'total']

# Convert 'wins' to integer for calculation
df['wins'] = df['wins'].astype(int)

# Calculate the difference in wins from one year to the next
df['win_diff'] = df['wins'].diff()

# Find the year with the maximum increase (largest positive difference)
max_increase_row = df[df['win_diff'] == df['win_diff'].max()]
year_with_max_increase = max_increase_row.iloc[0]['year']

print(f"Final Answer: {year_with_max_increase}")